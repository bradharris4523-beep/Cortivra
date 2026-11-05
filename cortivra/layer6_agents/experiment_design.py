"""
Fisher-Information active experiment design and multi-horizon (1x, 2x, 4x)
stability verification (`cortivra.layer6_agents.experiment_design`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick
from cortivra.layer4_engine.rollout_arena import rollout_trajectories
from cortivra.observability.flight_recorder import (
    AlternativeBranchRecord,
    emit_trace_node,
)


@dataclass(frozen=True)
class DisambiguatingExperimentReport:
    """
    Result of active experimental design between two competing hypotheses.
    """

    optimal_initial_state: torch.Tensor
    optimal_control_stimulus: torch.Tensor
    initial_divergence_mse: float
    maximized_divergence_mse: float
    divergence_amplification_ratio: float
    trajectory_a: torch.Tensor
    trajectory_b: torch.Tensor
    winning_hypothesis: str
    error_mse_a: float
    error_mse_b: float
    discrimination_log_ratio: float


@dataclass(frozen=True)
class MultiHorizonStabilityReport:
    """
    Multi-horizon (1x T, 2x T, 4x T) rollout and safety-margin report.
    """

    base_steps: int
    horizon_1x_viable_fraction: float
    horizon_2x_viable_fraction: float
    horizon_4x_viable_fraction: float
    min_safety_margin_4x: float
    late_horizon_energy_growth: float
    asymptotic_contraction_ratio: float
    all_horizons_finite: bool
    passed_multi_horizon: bool


def _rollout_with_additive_stimulus(
    brick: LawBrick,
    x0: torch.Tensor,
    params: torch.Tensor,
    stimulus: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    Differentiable RK4 rollout of `brick` under additive state-space stimulus `u(t)` (`[T, D]`).
    Returns `[T+1, D]` trajectory tensor.
    """
    num_steps = stimulus.shape[0]
    p_row = params if params.ndim == 2 else params.unsqueeze(0)
    curr = x0 if x0.ndim == 2 else x0.unsqueeze(0)
    traj: list[torch.Tensor] = [curr.squeeze(0)]

    for t in range(num_steps):
        u_t = stimulus[t : t + 1]
        k1 = brick.flow(curr, p_row) + u_t
        k2 = brick.flow(curr + 0.5 * dt * k1, p_row) + u_t
        k3 = brick.flow(curr + 0.5 * dt * k2, p_row) + u_t
        k4 = brick.flow(curr + dt * k3, p_row) + u_t
        curr = curr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj.append(curr.squeeze(0))

    return torch.stack(traj, dim=0)


def design_disambiguating_experiment(
    candidate_a: LawBrick,
    params_a: torch.Tensor,
    candidate_b: LawBrick,
    params_b: torch.Tensor,
    baseline_x0: torch.Tensor,
    dt: float = 0.02,
    num_steps: int = 30,
    max_stimulus_amplitude: float = 2.0,
    optimize_initial_state: bool = True,
    optimize_stimulus: bool = True,
    optimization_steps: int = 25,
    learning_rate: float = 0.15,
    oracle_brick: LawBrick | None = None,
    oracle_params: torch.Tensor | None = None,
    observed_trajectory: torch.Tensor | None = None,
) -> DisambiguatingExperimentReport:
    """
    Actively design an experimental initial condition `x0*` and external stimulus waveform `u*(t)`
    within homeostatic safety bounds that maximizes trajectory divergence between `candidate_a`
    and `candidate_b`.
    """
    if candidate_a.state_dim != candidate_b.state_dim:
        raise ValueError(
            f"Cannot design disambiguating experiment across mismatched state dimensions "
            f"({candidate_a.state_dim} vs {candidate_b.state_dim})."
        )

    state_dim = candidate_a.state_dim
    x0_base = baseline_x0.detach().to(torch.float64)
    if x0_base.ndim == 2:
        x0_base = x0_base[0]

    pa = params_a.detach().to(torch.float64)
    pb = params_b.detach().to(torch.float64)
    if pa.ndim == 1:
        pa = pa.unsqueeze(0)
    if pb.ndim == 1:
        pb = pb.unsqueeze(0)

    h_min = torch.tensor(
        [sp.homeostatic_min for sp in candidate_a.state_specs],
        dtype=torch.float64,
        device=x0_base.device,
    )
    h_max = torch.tensor(
        [sp.homeostatic_max for sp in candidate_a.state_specs],
        dtype=torch.float64,
        device=x0_base.device,
    )
    scale_vec = torch.tensor(
        [max(abs(sp.default_value), abs(float(x0_base[i].item())), 1.0) for i, sp in enumerate(candidate_a.state_specs)],
        dtype=torch.float64,
        device=x0_base.device,
    )
    x0_low = torch.maximum(h_min + 0.1 * (h_max - h_min), x0_base - 2.5 * scale_vec)
    x0_high = torch.minimum(h_max - 0.1 * (h_max - h_min), x0_base + 2.5 * scale_vec)

    zero_stim = torch.zeros((num_steps, state_dim), dtype=torch.float64, device=x0_base.device)
    with torch.no_grad():
        init_traj_a = _rollout_with_additive_stimulus(candidate_a, x0_base, pa, zero_stim, dt)
        init_traj_b = _rollout_with_additive_stimulus(candidate_b, x0_base, pb, zero_stim, dt)
        init_div_mse = float(torch.mean((init_traj_a - init_traj_b) ** 2).item())

    best_x0 = x0_base.clone()
    best_u = zero_stim.clone()
    best_div = init_div_mse

    seed_offsets = [0.0, 0.45, -0.45, 0.85, -0.85, 1.2]
    t_grid = torch.linspace(0.0, float(num_steps) * dt, num_steps, dtype=torch.float64, device=x0_base.device)
    with torch.no_grad():
        x0_seeds_list: list[torch.Tensor] = []
        u_seeds_list: list[torch.Tensor] = []
        for idx_seed, offset in enumerate(seed_offsets):
            x0_s = torch.clamp(x0_base + offset * scale_vec, min=x0_low, max=x0_high).clone().detach()
            u_s = (
                0.30
                * max_stimulus_amplitude
                * float((idx_seed % 3) - 1)
                * torch.sin(2.0 * math.pi * (1.0 + 0.5 * idx_seed) * t_grid).unsqueeze(-1).expand(num_steps, state_dim)
                if optimize_stimulus and max_stimulus_amplitude > 0.0
                else torch.zeros_like(zero_stim)
            ).clone().detach()
            x0_seeds_list.append(x0_s)
            u_seeds_list.append(u_s)
            tr_a_s = _rollout_with_additive_stimulus(candidate_a, x0_s, pa, u_s, dt)
            tr_b_s = _rollout_with_additive_stimulus(candidate_b, x0_s, pb, u_s, dt)
            if torch.isfinite(tr_a_s).all() and torch.isfinite(tr_b_s).all():
                in_b_s = bool(
                    (tr_a_s >= h_min).all()
                    and (tr_a_s <= h_max).all()
                    and (tr_b_s >= h_min).all()
                    and (tr_b_s <= h_max).all()
                )
                s_div = float(torch.mean((tr_a_s - tr_b_s) ** 2).item())
                if in_b_s and s_div > best_div:
                    best_div = s_div
                    best_x0 = x0_s.clone()
                    best_u = u_s.clone()

    x0_var = best_x0.clone().detach().requires_grad_(optimize_initial_state)
    u_var = best_u.clone().detach().requires_grad_(optimize_stimulus and max_stimulus_amplitude > 0.0)
    opt_vars = [v for v in (x0_var, u_var) if v.requires_grad]
    if opt_vars:
        optimizer = torch.optim.Adam(opt_vars, lr=learning_rate)
        steps_refine = min(max(optimization_steps // 4, 4), 6)
        for _ in range(steps_refine):
            optimizer.zero_grad()
            tr_a = _rollout_with_additive_stimulus(candidate_a, x0_var, pa, u_var, dt)
            tr_b = _rollout_with_additive_stimulus(candidate_b, x0_var, pb, u_var, dt)
            if not (torch.isfinite(tr_a).all() and torch.isfinite(tr_b).all()):
                break
            with torch.no_grad():
                in_bounds_pre = bool(
                    (tr_a >= h_min).all()
                    and (tr_a <= h_max).all()
                    and (tr_b >= h_min).all()
                    and (tr_b <= h_max).all()
                )
                cur_div_pre = float(torch.mean((tr_a - tr_b) ** 2).item())
                if in_bounds_pre and cur_div_pre > best_div:
                    best_div = cur_div_pre
                    best_x0 = x0_var.detach().clone()
                    best_u = u_var.detach().clone()

            div_mse = torch.mean((tr_a - tr_b) ** 2)
            viol_a = torch.relu(h_min.unsqueeze(0) - tr_a) + torch.relu(tr_a - h_max.unsqueeze(0))
            viol_b = torch.relu(h_min.unsqueeze(0) - tr_b) + torch.relu(tr_b - h_max.unsqueeze(0))
            barrier = 100.0 * (torch.mean(viol_a ** 2) + torch.mean(viol_b ** 2))
            ctrl_reg = 1e-3 * torch.mean(u_var ** 2)

            loss = -div_mse + barrier + ctrl_reg
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if optimize_initial_state:
                    x0_var.clamp_(min=x0_low, max=x0_high)
                if optimize_stimulus:
                    u_var.clamp_(min=-max_stimulus_amplitude, max=max_stimulus_amplitude)
        with torch.no_grad():
            tr_a_chk = _rollout_with_additive_stimulus(candidate_a, x0_var, pa, u_var, dt)
            tr_b_chk = _rollout_with_additive_stimulus(candidate_b, x0_var, pb, u_var, dt)
            if torch.isfinite(tr_a_chk).all() and torch.isfinite(tr_b_chk).all():
                in_bounds = bool(
                    (tr_a_chk >= h_min).all()
                    and (tr_a_chk <= h_max).all()
                    and (tr_b_chk >= h_min).all()
                    and (tr_b_chk <= h_max).all()
                )
                cur_div = float(torch.mean((tr_a_chk - tr_b_chk) ** 2).item())
                if in_bounds and cur_div > best_div:
                    best_div = cur_div
                    best_x0 = x0_var.detach().clone()
                    best_u = u_var.detach().clone()

    with torch.no_grad():
        final_traj_a = _rollout_with_additive_stimulus(candidate_a, best_x0, pa, best_u, dt)
        final_traj_b = _rollout_with_additive_stimulus(candidate_b, best_x0, pb, best_u, dt)

    winner = "undecided"
    err_a = 0.0
    err_b = 0.0
    log_ratio = 0.0

    if oracle_brick is not None:
        po = (
            oracle_params.detach().to(torch.float64)
            if oracle_params is not None
            else oracle_brick.default_params(batch_size=1, dtype=torch.float64, device=best_x0.device)
        )
        if po.ndim == 1:
            po = po.unsqueeze(0)
        with torch.no_grad():
            oracle_traj = _rollout_with_additive_stimulus(oracle_brick, best_x0, po, best_u, dt)
            err_a = float(torch.mean((final_traj_a - oracle_traj) ** 2).item())
            err_b = float(torch.mean((final_traj_b - oracle_traj) ** 2).item())
            log_ratio = math.log((err_b + 1e-15) / (err_a + 1e-15))
            winner = "candidate_a" if err_a <= err_b else "candidate_b"
    elif observed_trajectory is not None:
        obs_t = observed_trajectory.detach().to(torch.float64)
        if obs_t.ndim == 3:
            obs_t = obs_t[0]
        n_eval = min(obs_t.shape[0] - 1, num_steps)
        if n_eval >= 1:
            z_eval = torch.zeros((n_eval, state_dim), dtype=torch.float64, device=best_x0.device)
            x0_eval = x0_base.detach().clone()
            k_obs = min(int(obs_t.shape[-1]), state_dim)
            x0_eval[:k_obs] = obs_t[0, :k_obs]
            with torch.no_grad():
                eval_a = _rollout_with_additive_stimulus(candidate_a, x0_eval, pa, z_eval, dt)
                eval_b = _rollout_with_additive_stimulus(candidate_b, x0_eval, pb, z_eval, dt)
                target_obs = obs_t[: n_eval + 1, :k_obs]
                err_a = float(torch.mean((eval_a[:, :k_obs] - target_obs) ** 2).item())
                err_b = float(torch.mean((eval_b[:, :k_obs] - target_obs) ** 2).item())
                log_ratio = math.log((err_b + 1e-15) / (err_a + 1e-15))
                winner = "candidate_a" if err_a <= err_b else "candidate_b"

    amp_ratio = best_div / max(init_div_mse, 1e-15)
    b_name_eff = (
        f"{candidate_b.name}_passive_baseline_impostor"
        if candidate_b.name == candidate_a.name
        else candidate_b.name
    )
    selected_id = (
        candidate_a.name
        if winner == "candidate_a"
        else (b_name_eff if winner == "candidate_b" else "undecided")
    )
    runner_up_id = b_name_eff if winner == "candidate_a" else candidate_a.name
    runner_err = err_b if winner == "candidate_a" else err_a
    runner_r2 = round(max(1.0 / (1.0 + 10.0 * float(runner_err)), 0.15), 5)
    emit_trace_node(
        layer="layer6_agents",
        operation="design_disambiguating_experiment",
        target_name=f"{candidate_a.name}_vs_{b_name_eff}",
        local_error_metric=min(err_a, err_b),
        decision_margin=max(abs(log_ratio), 1e-4),
        selected_branch_id=selected_id,
        selected_expression=f"winner={winner} ({selected_id}), amp_ratio={amp_ratio:.2f}x",
        alternatives=(
            AlternativeBranchRecord(
                branch_id=runner_up_id,
                expression_or_summary=f"impostor={runner_up_id}",
                objective_score=round(float(runner_err), 6),
                local_r2=runner_r2,
            ),
        ),
        metrics={
            "initial_divergence_mse": round(float(init_div_mse), 6),
            "maximized_divergence_mse": round(float(best_div), 6),
            "divergence_amplification_ratio": round(float(amp_ratio), 4),
            "winning_hypothesis": winner,
            "discrimination_log_ratio": round(float(log_ratio), 4),
        },
    )
    return DisambiguatingExperimentReport(
        optimal_initial_state=best_x0.unsqueeze(0),
        optimal_control_stimulus=best_u,
        initial_divergence_mse=init_div_mse,
        maximized_divergence_mse=best_div,
        divergence_amplification_ratio=amp_ratio,
        trajectory_a=final_traj_a,
        trajectory_b=final_traj_b,
        winning_hypothesis=winner,
        error_mse_a=err_a,
        error_mse_b=err_b,
        discrimination_log_ratio=log_ratio,
    )


def verify_multi_horizon_stability(
    brick: LawBrick,
    params: torch.Tensor,
    initial_state: torch.Tensor,
    dt: float = 0.02,
    base_steps: int = 25,
) -> MultiHorizonStabilityReport:
    """
    Evaluate multi-horizon (1x T, 2x T, 4x T) rollout viability and homeostatic safety margins.
    """
    x0 = initial_state.detach().to(torch.float64)
    if x0.ndim == 1:
        x0 = x0.unsqueeze(0)
    elif x0.ndim == 3:
        x0 = x0[0, 0:1]
    else:
        x0 = x0[0:1]

    p = params.detach().to(torch.float64)
    if p.ndim == 1:
        p = p.unsqueeze(0)
    p = p[:1].expand(x0.shape[0], -1)

    steps_1x = max(int(base_steps), 8)
    steps_2x = steps_1x * 2
    steps_4x = min(steps_1x * 4, 160)

    with torch.no_grad():
        roll_4x = rollout_trajectories(
            brick,
            x0,
            p,
            dt=dt,
            num_steps=steps_4x,
            integrator="auto",
        )

    states_4x = roll_4x.states[0]
    finite_all = bool(torch.isfinite(states_4x).all().item())
    if not finite_all:
        return MultiHorizonStabilityReport(
            base_steps=steps_1x,
            horizon_1x_viable_fraction=0.0,
            horizon_2x_viable_fraction=0.0,
            horizon_4x_viable_fraction=0.0,
            min_safety_margin_4x=-1.0,
            late_horizon_energy_growth=float("inf"),
            asymptotic_contraction_ratio=float("inf"),
            all_horizons_finite=False,
            passed_multi_horizon=False,
        )

    step_mask = roll_4x.step_viable_mask[0].to(torch.float64)
    idx_1x = min(steps_1x + 1, step_mask.numel())
    idx_2x = min(steps_2x + 1, step_mask.numel())
    v_1x = float(torch.mean(step_mask[:idx_1x]).item())
    v_2x = float(torch.mean(step_mask[:idx_2x]).item())
    v_4x = float(torch.mean(step_mask).item())

    min_margin = 1.0
    for i, sp in enumerate(brick.state_specs):
        col = states_4x[:, i]
        c_min = float(torch.min(col).item())
        c_max = float(torch.max(col).item())
        h_min, h_max = float(sp.homeostatic_min), float(sp.homeostatic_max)
        span = max(h_max - h_min, 1e-12)
        margin_i = min(c_min - h_min, h_max - c_max) / span
        min_margin = min(min_margin, margin_i)

    energies = roll_4x.energies[0]
    e_1x = float(energies[min(steps_1x, energies.numel() - 1)].item())
    e_4x = float(energies[-1].item())
    e_scale = max(abs(float(energies[0].item())), abs(e_1x), 1.0)
    late_e_growth = (e_4x - e_1x) / e_scale

    flows = roll_4x.flows[0]
    early_rms = float(torch.sqrt(torch.mean(flows[:idx_1x] ** 2)).item())
    late_rms = float(torch.sqrt(torch.mean(flows[idx_2x:] ** 2)).item()) if idx_2x < flows.shape[0] else early_rms
    contraction_ratio = late_rms / max(early_rms, 1e-9)

    passed = bool(
        finite_all
        and v_1x >= 0.85
        and v_2x >= 0.80
        and v_4x >= 0.75
        and min_margin >= -0.05
    )

    return MultiHorizonStabilityReport(
        base_steps=steps_1x,
        horizon_1x_viable_fraction=round(v_1x, 4),
        horizon_2x_viable_fraction=round(v_2x, 4),
        horizon_4x_viable_fraction=round(v_4x, 4),
        min_safety_margin_4x=round(float(min_margin), 6),
        late_horizon_energy_growth=round(float(late_e_growth), 6),
        asymptotic_contraction_ratio=round(float(contraction_ratio), 6),
        all_horizons_finite=finite_all,
        passed_multi_horizon=passed,
    )
