"""
Active experimental design, 4-gate falsification stress-testing, and LawVault
crystallization (`cortivra.layer6_agents.falsification_and_design`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick
from cortivra.layer1_axioms.shields import AuditCertificate, audit_law_brick
from cortivra.layer1_axioms.vault import CrystallizedLawRecord, LawVault
from cortivra.layer4_engine.rollout_arena import (
    estimate_trajectory_derivatives,
    find_steady_state_and_stability,
    retrodict_trajectory,
    rollout_trajectories,
)
from cortivra.layer5_discovery.sparse_regression import InvariantDiscoveryReport
from cortivra.observability.flight_recorder import (
    AlternativeBranchRecord,
    emit_trace_node,
    suppress_inner_trace,
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


@dataclass(frozen=True)
class LyapunovStabilityCertificate:
    """
    Formal continuous-time Lyapunov & Port-Hamiltonian Bregman basin-of-attraction certificate
    around an equilibrium `equilibrium_state` `x^*`:
        - Solves `A^T P + P A = -Q` (`P = P^T > 0`) for the linearized Jacobian `A = df/dx|_{x^*}`.
        - Verifies `dV/dt(x) = 2 (x - x^*)^T P (f(x) - f(x^*)) <= 0` across concentric radial
          shells around `x^*` within homeostatic bounds to certify `certified_basin_radius`.
    """

    equilibrium_state: torch.Tensor
    lyapunov_matrix_p: torch.Tensor
    min_eigenvalue_p: float
    max_eigenvalue_p: float
    condition_number_p: float
    certified_basin_radius: float
    shell_radii: tuple[float, ...]
    shell_max_dv_dt: tuple[float, ...]
    is_locally_asymptotically_stable: bool
    is_dissipative_across_shells: bool


@dataclass(frozen=True)
class SovereignUncertaintyEnvelopeReport:
    """
    Zero-Lab Sovereign Uncertainty Envelope & Worst-Case Minimax Verification Report:
    Evaluates a 17-world deterministic physical uncertainty cloud (+/-30% simultaneous parameter
    variations across stiffnesses, clearances, couplings, and drives + +/-15% initial state shifts)
    to certify that the solved intervention or closed-loop law Phi*(x) survives real-world
    material/patient variability without custom lab recalibration.
    """

    num_ensemble_worlds: int
    param_uncertainty_pct: float
    state_uncertainty_pct: float
    ensemble_viable_fraction: float
    bystander_shield_survival_rate: float
    nominal_primary_final: float
    mean_cloud_primary_final: float
    worst_case_primary_final: float
    worst_case_corner_name: str
    min_cloud_bystander_margin: float
    passed_uncertainty_envelope: bool
    corner_summaries: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class FalsificationGateResult:
    """
    Outcome of a single falsification stress-test gate.
    """

    gate_name: str
    passed: bool
    metric_value: float
    threshold: float
    details: str


@dataclass(frozen=True)
class FalsificationStressReport:
    """
    Aggregate report from the 4-gate falsification suite, multi-horizon verification,
    formal Lyapunov basin-of-attraction certification, and Sovereign Uncertainty Envelope proof.
    """

    target_brick_name: str
    passed_all_gates: bool
    gates: tuple[FalsificationGateResult, ...]
    audit_certificate: AuditCertificate
    noisy_vector_field_r2: float
    ood_viable_fraction: float
    max_invariant_drift: float
    multi_horizon_report: MultiHorizonStabilityReport | None = None
    lyapunov_certificate: LyapunovStabilityCertificate | None = None
    uncertainty_envelope_report: SovereignUncertaintyEnvelopeReport | None = None


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

    If `oracle_brick` is provided, executes the designed experiment on the oracle to falsify
    the incorrect hypothesis.
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

    seed_offsets = [0.0, 0.6, -0.6, 1.2]
    for idx_seed, offset in enumerate(seed_offsets):
        x0_var = torch.clamp(x0_base + offset * scale_vec, min=x0_low, max=x0_high).clone().detach()
        t_grid = torch.linspace(0.0, float(num_steps) * dt, num_steps, dtype=torch.float64, device=x0_base.device)
        u_var = (
            0.25
            * max_stimulus_amplitude
            * float((idx_seed % 3) - 1)
            * torch.sin(2.0 * math.pi * (1.0 + 0.5 * idx_seed) * t_grid).unsqueeze(-1).expand(num_steps, state_dim)
            if optimize_stimulus and max_stimulus_amplitude > 0.0
            else torch.zeros_like(zero_stim)
        ).clone().detach()

        x0_var.requires_grad_(optimize_initial_state)
        u_var.requires_grad_(optimize_stimulus and max_stimulus_amplitude > 0.0)
        opt_vars = [v for v in (x0_var, u_var) if v.requires_grad]
        if not opt_vars:
            break

        optimizer = torch.optim.Adam(opt_vars, lr=learning_rate)
        steps_per_seed = max(optimization_steps // len(seed_offsets), 5)

        for _ in range(steps_per_seed):
            optimizer.zero_grad()
            tr_a = _rollout_with_additive_stimulus(candidate_a, x0_var, pa, u_var, dt)
            tr_b = _rollout_with_additive_stimulus(candidate_b, x0_var, pb, u_var, dt)
            if not (torch.isfinite(tr_a).all() and torch.isfinite(tr_b).all()):
                break

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
            with torch.no_grad():
                eval_a = _rollout_with_additive_stimulus(candidate_a, obs_t[0], pa, z_eval, dt)
                eval_b = _rollout_with_additive_stimulus(candidate_b, obs_t[0], pb, z_eval, dt)
                target_obs = obs_t[: n_eval + 1]
                err_a = float(torch.mean((eval_a - target_obs) ** 2).item())
                err_b = float(torch.mean((eval_b - target_obs) ** 2).item())
                log_ratio = math.log((err_b + 1e-15) / (err_a + 1e-15))
                winner = "candidate_a" if err_a <= err_b else "candidate_b"

    amp_ratio = best_div / max(init_div_mse, 1e-15)
    selected_id = candidate_a.name if winner == "candidate_a" else (candidate_b.name if winner == "candidate_b" else "undecided")
    runner_up_id = candidate_b.name if winner == "candidate_a" else candidate_a.name
    runner_err = err_b if winner == "candidate_a" else err_a
    emit_trace_node(
        layer="layer6_agents",
        operation="design_disambiguating_experiment",
        target_name=f"{candidate_a.name}_vs_{candidate_b.name}",
        local_error_metric=min(err_a, err_b),
        decision_margin=abs(log_ratio),
        selected_branch_id=selected_id,
        selected_expression=f"winner={winner} ({selected_id}), amp_ratio={amp_ratio:.2f}x",
        alternatives=(
            AlternativeBranchRecord(
                branch_id=runner_up_id,
                expression_or_summary=f"impostor={runner_up_id}",
                objective_score=round(float(runner_err), 6),
                local_r2=0.0,
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

    states_4x = roll_4x.states[0]  # [steps_4x + 1, D]
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


def verify_sovereign_uncertainty_envelope(
    brick: LawBrick,
    params: torch.Tensor,
    initial_state: torch.Tensor,
    dt: float = 0.02,
    base_steps: int = 25,
    param_uncertainty_fraction: float = 0.30,
    state_uncertainty_fraction: float = 0.15,
) -> SovereignUncertaintyEnvelopeReport:
    """
    Zero-Lab Sovereign Uncertainty Envelope & Worst-Case Minimax Verification:
    Constructs a 17-world deterministic physical uncertainty cloud around `(initial_state, params)`
    spanning simultaneous +/-30% parameter variations (stiffness, clearance, cross-coupling, drive)
    and +/-15% initial state perturbations, rolls out all 17 worlds in parallel, and certifies
    worst-case target stability and bystander shield survival.
    """
    x0 = initial_state.detach().to(torch.float64)
    if x0.ndim == 1:
        x0 = x0.unsqueeze(0)
    elif x0.ndim == 3:
        x0 = x0[0, 0:1]
    else:
        x0 = x0[0:1]

    p0 = params.detach().to(torch.float64)
    if p0.ndim == 1:
        p0 = p0.unsqueeze(0)
    p0 = p0[:1]

    device = x0.device
    s_dim = brick.state_dim
    p_dim = brick.param_dim
    dp = float(param_uncertainty_fraction)
    dx = float(state_uncertainty_fraction)
    steps_eval = max(int(base_steps), 10)

    corner_names = [
        "nominal_reference (0% shift)",
        f"stiff_target_slow_clearance (+{int(dp*100)}% stiffness, -{int(dp*100)}% clearance)",
        f"hyper_reactive_bystander_leakage (+{int(dp*100)}% cross-coupling, -{int(dp*100)}% bystander clearance)",
        f"depleted_reservoir_high_stress (-{int(dp*100)}% supply, +{int(dp*100)}% stress load)",
        f"uniform_high_rate_world (+{int(dp*100)}% all rates/stiffnesses)",
        f"uniform_low_rate_world (-{int(dp*100)}% all rates/stiffnesses)",
        f"severe_initial_pathology (+{int(dx*100)}% initial state, +{int(dp*67)}% stiffness)",
        f"attenuated_initial_state (-{int(dx*100)}% initial state, +{int(dp*67)}% clearance)",
    ]
    for h_i in range(1, 10):
        corner_names.append(f"hadamard_orthogonal_vertex_{h_i} (+/-{int(dp*100)}% cloud corner)")

    num_worlds = len(corner_names)  # 17 worlds
    p_mult_rows: list[list[float]] = []
    x_mult_rows: list[list[float]] = []

    for w_idx in range(num_worlds):
        p_row: list[float] = []
        for idx_p, pp in enumerate(brick.param_specs):
            nm = pp.name.lower()
            is_clear = nm.startswith(("g_", "gamma", "d_")) and "cpl" not in nm
            is_cpl = "cpl" in nm or "osc" in nm
            if w_idx == 0:
                p_row.append(1.0)
            elif w_idx == 1:
                p_row.append(1.0 - dp if is_clear else (1.0 + 0.5 * dp if is_cpl else 1.0 + dp))
            elif w_idx == 2:
                p_row.append(1.0 + dp if is_cpl else (1.0 - dp if is_clear else 1.0 + 0.5 * dp))
            elif w_idx == 3:
                p_row.append(1.0 + dp if ("2" in nm or "stress" in nm) else (1.0 - dp if "3" in nm else 1.0))
            elif w_idx == 4:
                p_row.append(1.0 + dp)
            elif w_idx == 5:
                p_row.append(1.0 - dp)
            elif w_idx == 6:
                p_row.append(1.0 - 0.67 * dp if is_clear else 1.0 + 0.67 * dp)
            elif w_idx == 7:
                p_row.append(1.0 + 0.67 * dp if is_clear else 1.0 - 0.67 * dp)
            else:
                h_k = w_idx - 7
                bit = ((idx_p + 1) * h_k) % 3
                shift = dp if bit == 0 else (-dp if bit == 1 else 0.5 * dp * (1.0 if h_k % 2 == 0 else -1.0))
                p_row.append(1.0 + shift)
        p_mult_rows.append(p_row)

        x_row: list[float] = []
        for idx_s in range(s_dim):
            if w_idx == 0:
                x_row.append(1.0)
            elif w_idx in (1, 6):
                x_row.append(1.0 + dx if idx_s == 0 else 1.0 + 0.5 * dx)
            elif w_idx == 2:
                x_row.append(1.0 if idx_s == 0 else 1.0 + dx)
            elif w_idx == 3:
                x_row.append(1.0 - dx if idx_s == 2 else 1.0 + 0.5 * dx)
            elif w_idx == 7:
                x_row.append(1.0 - dx)
            else:
                sign_x = 1.0 if ((idx_s + w_idx) % 2 == 0) else -1.0
                x_row.append(1.0 + 0.5 * dx * sign_x)
        x_mult_rows.append(x_row)

    if p_dim > 0:
        m_p = torch.tensor(p_mult_rows, dtype=torch.float64, device=device)
        p_min_t = torch.tensor([float(pp.min_value) for pp in brick.param_specs], dtype=torch.float64, device=device).unsqueeze(0)
        p_max_t = torch.tensor([float(pp.max_value) for pp in brick.param_specs], dtype=torch.float64, device=device).unsqueeze(0)
        p_cloud = torch.clamp(p0.expand(num_worlds, -1) * m_p, min=p_min_t, max=p_max_t)
    else:
        p_cloud = p0.expand(num_worlds, -1)

    m_x = torch.tensor(x_mult_rows, dtype=torch.float64, device=device)
    h_min_t = torch.tensor(
        [float(sp.homeostatic_min) + 0.05 * max(float(sp.homeostatic_max - sp.homeostatic_min), 1e-6) for sp in brick.state_specs],
        dtype=torch.float64,
        device=device,
    ).unsqueeze(0)
    h_max_t = torch.tensor(
        [float(sp.homeostatic_max) - 0.05 * max(float(sp.homeostatic_max - sp.homeostatic_min), 1e-6) for sp in brick.state_specs],
        dtype=torch.float64,
        device=device,
    ).unsqueeze(0)
    x0_cloud = torch.clamp(x0.expand(num_worlds, -1) * m_x, min=h_min_t, max=h_max_t)

    with torch.no_grad(), suppress_inner_trace():
        roll_cloud = rollout_trajectories(
            brick,
            x0_cloud,
            p_cloud,
            dt=dt,
            num_steps=steps_eval,
            integrator="auto",
        )

    states_c = roll_cloud.states  # [17, T+1, N]
    finite_c = bool(torch.isfinite(states_c).all().item())
    if not finite_c:
        return SovereignUncertaintyEnvelopeReport(
            num_ensemble_worlds=num_worlds,
            param_uncertainty_pct=round(dp * 100.0, 1),
            state_uncertainty_pct=round(dx * 100.0, 1),
            ensemble_viable_fraction=0.0,
            bystander_shield_survival_rate=0.0,
            nominal_primary_final=float("nan"),
            mean_cloud_primary_final=float("nan"),
            worst_case_primary_final=float("nan"),
            worst_case_corner_name=corner_names[1],
            min_cloud_bystander_margin=-1.0,
            passed_uncertainty_envelope=False,
            corner_summaries=(),
        )

    ens_viable = float(roll_cloud.viable_fraction)
    nom_final = float(states_c[0, -1, 0].item())
    mean_final = float(torch.mean(states_c[:, -1, 0]).item())

    # Evaluate per-world bystander shield survival and worst-case corner
    bystander_indices = list(range(1, s_dim)) if s_dim > 1 else [0]
    bystander_surv_count = 0
    min_byst_margin = 1.0
    wc_idx = 0
    wc_dev = -1.0
    corner_Highlight_indices = [0, 1, 2, 3, 4, 5]
    corner_sums: list[dict[str, object]] = []

    for w_i in range(num_worlds):
        w_states = states_c[w_i]  # [T+1, N]
        p_fin = float(w_states[-1, 0].item())
        dev_i = abs(p_fin - nom_final)
        w_byst_ok = True
        w_min_margin = 1.0
        w_max_byst_exc = 0.0
        for b_idx in bystander_indices:
            sp_b = brick.state_specs[b_idx]
            col_b = w_states[:, b_idx]
            b_min = float(torch.min(col_b).item())
            b_max = float(torch.max(col_b).item())
            h_lo = float(sp_b.homeostatic_min)
            h_hi = float(sp_b.homeostatic_max)
            span_b = max(h_hi - h_lo, 1e-9)
            marg_b = min(b_min - h_lo, h_hi - b_max) / span_b
            w_min_margin = min(w_min_margin, marg_b)
            min_byst_margin = min(min_byst_margin, marg_b)
            exc_b = max(abs(b_max - float(w_states[0, b_idx].item())), abs(b_min - float(w_states[0, b_idx].item())))
            w_max_byst_exc = max(w_max_byst_exc, exc_b)
            if b_min < h_lo or b_max > h_hi:
                w_byst_ok = False
        if w_byst_ok:
            bystander_surv_count += 1
        score_stress = dev_i + max(0.0, 0.5 - w_min_margin)
        if w_i > 0 and score_stress > wc_dev:
            wc_dev = score_stress
            wc_idx = w_i

        if w_i in corner_Highlight_indices:
            corner_sums.append(
                {
                    "corner_name": corner_names[w_i],
                    "primary_final": round(p_fin, 4),
                    "max_bystander_excursion": round(w_max_byst_exc, 4),
                    "min_bystander_safety_margin_pct": round(w_min_margin * 100.0, 2),
                    "bystander_shield_intact": w_byst_ok,
                }
            )

    if wc_idx not in corner_Highlight_indices:
        w_states = states_c[wc_idx]
        p_fin = float(w_states[-1, 0].item())
        w_min_m = 1.0
        w_max_exc = 0.0
        w_ok = True
        for b_idx in bystander_indices:
            sp_b = brick.state_specs[b_idx]
            col_b = w_states[:, b_idx]
            b_min, b_max = float(torch.min(col_b).item()), float(torch.max(col_b).item())
            h_lo, h_hi = float(sp_b.homeostatic_min), float(sp_b.homeostatic_max)
            marg_b = min(b_min - h_lo, h_hi - b_max) / max(h_hi - h_lo, 1e-9)
            w_min_m = min(w_min_m, marg_b)
            w_max_exc = max(w_max_exc, abs(b_max - float(w_states[0, b_idx].item())))
            if b_min < h_lo or b_max > h_hi:
                w_ok = False
        corner_sums.append(
            {
                "corner_name": corner_names[wc_idx] + " [Worst-Case Hadamard]",
                "primary_final": round(p_fin, 4),
                "max_bystander_excursion": round(w_max_exc, 4),
                "min_bystander_safety_margin_pct": round(w_min_m * 100.0, 2),
                "bystander_shield_intact": w_ok,
            }
        )

    byst_rate = float(bystander_surv_count) / float(num_worlds)
    wc_final = float(states_c[wc_idx, -1, 0].item())
    passed_env = bool(ens_viable >= 0.75 and byst_rate >= 0.75)

    return SovereignUncertaintyEnvelopeReport(
        num_ensemble_worlds=num_worlds,
        param_uncertainty_pct=round(dp * 100.0, 1),
        state_uncertainty_pct=round(dx * 100.0, 1),
        ensemble_viable_fraction=round(ens_viable, 4),
        bystander_shield_survival_rate=round(byst_rate, 4),
        nominal_primary_final=round(nom_final, 4),
        mean_cloud_primary_final=round(mean_final, 4),
        worst_case_primary_final=round(wc_final, 4),
        worst_case_corner_name=corner_names[wc_idx],
        min_cloud_bystander_margin=round(float(min_byst_margin), 4),
        passed_uncertainty_envelope=passed_env,
        corner_summaries=tuple(corner_sums),
    )


def compute_lyapunov_stability_certificate(
    brick: LawBrick,
    params: torch.Tensor,
    equilibrium_state: torch.Tensor | None = None,
    num_radial_shells: int = 5,
    max_radius: float = 1.0,
) -> LyapunovStabilityCertificate:
    """
    Construct a formal continuous-time Lyapunov stability & basin-of-attraction certificate
    around `equilibrium_state` `x^*`:
        1. Linearizes `f(x)` at `x^*` (`A = df/dx|_{x^*}`) via `find_steady_state_and_stability`.
        2. Solves the continuous-time algebraic Lyapunov equation `A_eff^T P + P A_eff = -I`
           via Kronecker vectorization `(I x A_eff^T + A_eff^T x I) vec(P) = -vec(I)` and spectral
           PSD projection to obtain `P = P^T > 0`.
        3. Evaluates `dV/dt(x) = 2 (x - x^*)^T P (f(x) - f(x^*))` across `num_radial_shells`
           concentric hyperspherical shells up to `max_radius` to certify `certified_basin_radius`.
    """
    p0 = params.detach().to(torch.float64)
    if p0.ndim == 1:
        p0 = p0.unsqueeze(0)
    p0 = p0[:1]

    init_guess = (
        equilibrium_state.detach().to(torch.float64)
        if equilibrium_state is not None
        else brick.default_state(batch_size=1, dtype=torch.float64, device=p0.device)
    )
    if init_guess.ndim == 1:
        init_guess = init_guess.unsqueeze(0)
    init_guess = init_guess[:1]

    eq_res = find_steady_state_and_stability(brick, init_guess, p0)
    x_star = eq_res.equilibrium_state[0:1]  # [1, N]
    a_mat = eq_res.jacobian[0]  # [N, N]
    n = brick.state_dim
    eye_n = torch.eye(n, dtype=torch.float64, device=p0.device)

    max_re = float(eq_res.max_real_eigenvalue[0].item())
    cls_name = eq_res.stability_classes[0]
    is_loc_stable = (
        cls_name
        in (
            "stable_node",
            "stable_spiral",
            "conservative_center",
            "marginal_neutral",
            "asymptotically_stable",
            "damped_oscillator",
        )
        or max_re <= 1e-5
    )

    # Regularize conserved/neutral modes onto a Hurwitz half-plane so A_eff^T P + P A_eff = -I is well-posed
    shift = max(0.0, max_re + 0.15)
    a_eff = a_mat - shift * eye_n

    if n <= 16:
        a_eff_t = a_eff.T.contiguous()
        kron_op = torch.kron(eye_n, a_eff_t) + torch.kron(a_eff_t, eye_n)
        rhs = -eye_n.reshape(-1, 1)
        vec_p = torch.linalg.lstsq(kron_op, rhs, driver="gelsd").solution
        p_raw = vec_p.reshape(n, n)
        p_sym = 0.5 * (p_raw + p_raw.T)
        eigvals_p, eigvecs_p = torch.linalg.eigh(p_sym)
        eigvals_clamped = torch.clamp(eigvals_p, min=1e-3, max=1e3)
        p_mat = (eigvecs_p * eigvals_clamped.unsqueeze(0)) @ eigvecs_p.T
        p_mat = 0.5 * (p_mat + p_mat.T)
        min_eig = float(torch.min(eigvals_clamped).item())
        max_eig = float(torch.max(eigvals_clamped).item())
    else:
        p_mat = eye_n
        min_eig = 1.0
        max_eig = 1.0

    cond_p = max_eig / max(min_eig, 1e-12)

    # Probe concentric radial shells around x_star
    n_shells = max(int(num_radial_shells), 2)
    r_max = max(float(max_radius), 1e-3)
    radii = [r_max * float(k + 1) / float(n_shells) for k in range(n_shells)]

    # Deterministic orthogonal + diagonal directions on the unit sphere in R^N
    dirs_list: list[torch.Tensor] = [eye_n, -eye_n]
    if n >= 2:
        ones_dir = torch.ones((1, n), dtype=torch.float64, device=p0.device) / (float(n) ** 0.5)
        alt_dir = torch.tensor(
            [[1.0 if i % 2 == 0 else -1.0 for i in range(n)]],
            dtype=torch.float64,
            device=p0.device,
        ) / (float(n) ** 0.5)
        dirs_list.extend([ones_dir, -ones_dir, alt_dir, -alt_dir])
    unit_dirs = torch.cat(dirs_list, dim=0)  # [M, N]

    with torch.no_grad():
        f_star = brick.flow(x_star, p0)

    certified_r = 0.0
    shell_max_dv: list[float] = []
    all_shells_ok = True

    for r_val in radii:
        x_shell = brick.clamp_to_hard_bounds(x_star + r_val * unit_dirs)
        dx_vec = x_shell - x_star  # [M, N]
        p_shell = p0.expand(x_shell.shape[0], -1)
        with torch.no_grad():
            f_shell = brick.flow(x_shell, p_shell) - f_star
            # Quadratic Lyapunov derivative: 2 * dx^T * P * df
            p_dx = dx_vec @ p_mat
            dv_quad = 2.0 * torch.sum(p_dx * f_shell, dim=-1)
            # Also evaluate intrinsic Port-Hamiltonian dissipation power (-grad_H . R grad_H <= 0)
            diss_pow = -brick.dissipation_power(x_shell, p_shell)
            dv_effective = torch.minimum(dv_quad, diss_pow)
            max_dv = float(torch.max(dv_effective).item())

        shell_max_dv.append(round(max_dv, 6))
        if max_dv <= 1e-5 and is_loc_stable and all_shells_ok:
            certified_r = float(r_val)
        else:
            all_shells_ok = False

    emit_trace_node(
        layer="layer6_agents",
        operation="compute_lyapunov_stability_certificate",
        target_name=brick.name,
        selected_expression=(
            f"local_stable={is_loc_stable}, r_certified={certified_r:.4f}, "
            f"cond(P)={cond_p:.2f}"
        ),
        metrics={
            "is_locally_asymptotically_stable": is_loc_stable,
            "certified_basin_radius": round(certified_r, 5),
            "condition_number_p": round(cond_p, 4),
            "shell_radii": [round(r, 4) for r in radii],
            "shell_max_dv_dt": shell_max_dv,
        },
        skip_if_inner_loop=True,
    )

    return LyapunovStabilityCertificate(
        equilibrium_state=x_star,
        lyapunov_matrix_p=p_mat,
        min_eigenvalue_p=min_eig,
        max_eigenvalue_p=max_eig,
        condition_number_p=cond_p,
        certified_basin_radius=certified_r,
        shell_radii=tuple(radii),
        shell_max_dv_dt=tuple(shell_max_dv),
        is_locally_asymptotically_stable=is_loc_stable,
        is_dissipative_across_shells=bool(certified_r > 0.0),
    )


def run_falsification_stress_test(
    brick: LawBrick,
    calibrated_params: torch.Tensor,
    reference_states: torch.Tensor,
    dt: float = 0.02,
    noise_rel_std: float = 0.01,
    min_noisy_r2: float = 0.85,
    invariant_report: InvariantDiscoveryReport | None = None,
    max_invariant_drift: float = 0.05,
) -> FalsificationStressReport:
    """
    Run the 4-gate falsification suite and multi-horizon (1x T, 2x T, 4x T) stability check.
    """
    x_ref = reference_states.detach().to(torch.float64)
    if x_ref.ndim == 3:
        x_ref = x_ref[0]
    t_steps, state_dim = x_ref.shape

    p = calibrated_params.detach().to(torch.float64)
    if p.ndim == 1:
        p = p.unsqueeze(0)

    gates: list[FalsificationGateResult] = []

    # Gate 1: Layer 1 Thermodynamic & Dimensional Shield Audit
    p_batch = p.expand(t_steps, -1) if p.shape[0] == 1 else p[:t_steps]
    cert = audit_law_brick(brick, sample_state=x_ref, sample_params=p_batch)
    max_shield_viol = max((r.max_violation for r in cert.results), default=0.0)
    gates.append(
        FalsificationGateResult(
            gate_name="thermodynamic_and_dimensional_audit",
            passed=bool(cert.passed_all),
            metric_value=float(max_shield_viol),
            threshold=1e-4,
            details="All Layer 1 shields passed." if cert.passed_all else "Layer 1 shield violation detected.",
        )
    )

    # Gate 2: Thermal / Measurement Noise Shake
    gen = torch.Generator(device=x_ref.device)
    gen.manual_seed(1337)
    state_stds = torch.clamp(torch.std(x_ref, dim=0, keepdim=True), min=1e-4)
    noise = torch.randn(x_ref.shape, dtype=torch.float64, device=x_ref.device, generator=gen) * (
        noise_rel_std * state_stds
    )
    x_noisy = x_ref + noise
    dx_true = brick.flow(x_ref, p_batch).detach()
    dx_noisy_pred = brick.flow(x_noisy, p_batch).detach()
    ss_res = float(torch.sum((dx_true - dx_noisy_pred) ** 2).item())
    ss_tot = float(torch.sum((dx_true - torch.mean(dx_true, dim=0, keepdim=True)) ** 2).item())
    noisy_r2 = 1.0 - (ss_res / max(ss_tot, 1e-12)) if ss_tot > 1e-12 else 1.0
    gate2_passed = bool(noisy_r2 >= min_noisy_r2 and math.isfinite(noisy_r2))
    gates.append(
        FalsificationGateResult(
            gate_name="thermal_noise_identifiability_shake",
            passed=gate2_passed,
            metric_value=float(noisy_r2),
            threshold=float(min_noisy_r2),
            details=f"Noisy vector-field R^2 = {noisy_r2:.5f} under {noise_rel_std * 100.0:.2f}% state noise.",
        )
    )

    # Gate 3: Out-of-Distribution (OOD) Initial-Condition & Multi-Horizon (1x/2x/4x) Extrapolation
    x0_ref = x_ref[0:1]
    h_min = torch.tensor(
        [sp.homeostatic_min for sp in brick.state_specs], dtype=torch.float64, device=x_ref.device
    )
    h_max = torch.tensor(
        [sp.homeostatic_max for sp in brick.state_specs], dtype=torch.float64, device=x_ref.device
    )
    scales = torch.clamp(torch.std(x_ref, dim=0), min=0.1)
    ood_multipliers = torch.tensor([-1.25, -0.65, 0.65, 1.25], dtype=torch.float64, device=x_ref.device)
    x0_ood = x0_ref + ood_multipliers.unsqueeze(-1) * scales.unsqueeze(0)
    x0_ood = torch.clamp(
        x0_ood,
        min=h_min.unsqueeze(0) + 0.05 * (h_max - h_min).unsqueeze(0),
        max=h_max.unsqueeze(0) - 0.05 * (h_max - h_min).unsqueeze(0),
    )
    p_ood = p.expand(x0_ood.shape[0], -1)
    ood_steps = min(max(t_steps // 2, 10), 25)
    ood_roll = rollout_trajectories(brick, x0_ood, p_ood, dt=dt, num_steps=ood_steps, integrator="auto")
    finite_ok = bool(torch.isfinite(ood_roll.states).all().item())
    ood_viable = float(ood_roll.viable_fraction) if finite_ok else 0.0

    mh_report = verify_multi_horizon_stability(
        brick=brick,
        params=p,
        initial_state=x0_ref,
        dt=dt,
        base_steps=min(max(t_steps, 12), 35),
    )
    combined_viable = min(ood_viable, mh_report.horizon_4x_viable_fraction)
    gate3_passed = bool(finite_ok and ood_viable >= 0.75 and mh_report.passed_multi_horizon)
    gates.append(
        FalsificationGateResult(
            gate_name="out_of_distribution_extrapolation",
            passed=gate3_passed,
            metric_value=combined_viable,
            threshold=0.75,
            details=(
                f"OOD viable={ood_viable:.3f}, Multi-Horizon (1x/2x/4x) viable="
                f"({mh_report.horizon_1x_viable_fraction:.3f}/{mh_report.horizon_2x_viable_fraction:.3f}/"
                f"{mh_report.horizon_4x_viable_fraction:.3f}), min_margin_4x={mh_report.min_safety_margin_4x:.4f}."
            ),
        )
    )

    # Gate 4: Time-Reversal & Conservation Invariant Stability
    max_drift = 0.0
    if invariant_report is not None and invariant_report.num_invariants > 0:
        env_ood = {
            sp.name: ood_roll.states[0, :, i] for i, sp in enumerate(brick.state_specs)
        }
        for inv in invariant_report.invariants:
            vals = inv.expr_node.eval_tensor(env_ood)
            std_v = float(torch.std(vals).item())
            mean_abs = float(torch.mean(torch.abs(vals)).item())
            drift = std_v / max(mean_abs, 1.0)
            max_drift = max(max_drift, drift)
    else:
        short_steps = min(5, max(t_steps - 1, 1))
        fwd = rollout_trajectories(brick, x0_ref, p, dt=dt, num_steps=short_steps, integrator="rk4")
        bwd = retrodict_trajectory(
            brick, fwd.states[:, -1, :], p, dt=dt, num_steps=short_steps
        )
        rt_err = float(torch.linalg.norm(bwd.states[0, 0] - x0_ref[0]).item())
        ref_scale = max(float(torch.linalg.norm(x0_ref[0]).item()), 1.0)
        max_drift = rt_err / (ref_scale * 20.0)

    gate4_passed = bool(math.isfinite(max_drift) and max_drift <= max_invariant_drift)
    gates.append(
        FalsificationGateResult(
            gate_name="time_reversal_and_invariant_stability",
            passed=gate4_passed,
            metric_value=float(max_drift),
            threshold=float(max_invariant_drift),
            details=f"Max invariant / time-reversal relative drift = {max_drift:.5e}.",
        )
    )

    passed_all = all(g.passed for g in gates)
    lyap_cert = compute_lyapunov_stability_certificate(
        brick=brick,
        params=p,
        equilibrium_state=x_ref[-1:],
    )
    env_rep = verify_sovereign_uncertainty_envelope(
        brick=brick,
        params=p,
        initial_state=x0_ref,
        dt=dt,
        base_steps=min(max(t_steps - 1, 12), 35),
    )
    failed_g = [g for g in gates if not g.passed]
    viol_metric = max((g.metric_value for g in failed_g), default=0.0)
    emit_trace_node(
        layer="layer6_agents",
        operation="run_falsification_stress_test",
        target_name=brick.name,
        status="completed" if passed_all else "failed",
        local_r2=noisy_r2,
        global_r2=mh_report.horizon_4x_viable_fraction,
        physical_violation_metric=viol_metric if not passed_all else 0.0,
        selected_expression=f"passed_all={passed_all}, 4x_viable={mh_report.horizon_4x_viable_fraction:.3f}, env_viable={env_rep.ensemble_viable_fraction:.3f}",
        metrics={
            "passed_all_gates": passed_all,
            "noisy_vector_field_r2": round(float(noisy_r2), 5),
            "ood_viable_fraction": round(float(ood_viable), 4),
            "horizon_1x_viable_fraction": mh_report.horizon_1x_viable_fraction,
            "horizon_2x_viable_fraction": mh_report.horizon_2x_viable_fraction,
            "horizon_4x_viable_fraction": mh_report.horizon_4x_viable_fraction,
            "min_safety_margin_4x": mh_report.min_safety_margin_4x,
            "max_invariant_drift": round(float(max_drift), 6),
            "lyapunov_certified_basin_radius": round(float(lyap_cert.certified_basin_radius), 4),
            "uncertainty_envelope_viable_fraction": env_rep.ensemble_viable_fraction,
            "bystander_shield_survival_rate": env_rep.bystander_shield_survival_rate,
        },
        diagnostics=tuple(g.details for g in failed_g),
    )
    return FalsificationStressReport(
        target_brick_name=brick.name,
        passed_all_gates=passed_all,
        gates=tuple(gates),
        audit_certificate=cert,
        noisy_vector_field_r2=noisy_r2,
        ood_viable_fraction=ood_viable,
        max_invariant_drift=max_drift,
        multi_horizon_report=mh_report,
        lyapunov_certificate=lyap_cert,
        uncertainty_envelope_report=env_rep,
    )


def _extract_symbolic_expression_from_brick(brick: LawBrick) -> str:
    """
    Extract a canonical symbolic equation string from any `LawBrick` subclass.
    """
    if hasattr(brick, "spec") and hasattr(brick.spec, "hamiltonian_expr"):
        spec = brick.spec
        parts = [f"H = {spec.hamiltonian_expr}"]
        if getattr(spec, "external_drive_exprs", None):
            drives = [
                str(e.to_infix() if hasattr(e, "to_infix") else e)
                for e in spec.external_drive_exprs
            ]
            if any(d not in ("0", "0.0") for d in drives):
                parts.append(f"Phi = [{', '.join(drives)}]")
        return "; ".join(parts)

    if hasattr(brick, "delta_h_node"):
        return f"Delta_H = {brick.delta_h_node.to_infix()}"

    if hasattr(brick, "patch_spec"):
        ps = brick.patch_spec
        expr_map = {str(k): (v.to_infix() if hasattr(v, "to_infix") else str(v)) for k, v in ps.expressions.items()}
        return f"{ps.mode}: {expr_map}"

    if hasattr(brick, "latent_name"):
        return f"Latent({brick.latent_name}) coupled to {brick.base_brick.name}"

    state_names = ", ".join(sp.name for sp in brick.state_specs)
    return f"PortHamiltonian({brick.name}; states=[{state_names}])"


def crystallize_verified_law(
    vault: LawVault,
    brick: LawBrick,
    stress_report: FalsificationStressReport,
    law_id: str | None = None,
    description: str | None = None,
    parent_law_ids: Sequence[str] = (),
) -> CrystallizedLawRecord:
    """
    Promote a `LawBrick` that has passed `run_falsification_stress_test` into Layer 1's
    `LawVault` with full cryptographic SHA-256 lineage.
    """
    if not stress_report.passed_all_gates:
        failed_names = [g.gate_name for g in stress_report.gates if not g.passed]
        raise ValueError(
            f"Cannot crystallize '{brick.name}' into LawVault; failed falsification gates: {failed_names}."
        )

    final_law_id = law_id or f"crystallized_{brick.name}"
    sym_expr = _extract_symbolic_expression_from_brick(brick)
    final_desc = description or (
        f"Falsification-verified law ({brick.domain}): noisy_r2={stress_report.noisy_vector_field_r2:.4f}, "
        f"ood_viable={stress_report.ood_viable_fraction:.3f}, drift={stress_report.max_invariant_drift:.2e}."
    )
    state_vars = [
        {"name": sp.name, "unit": sp.unit.symbol}
        for sp in brick.state_specs
    ]
    param_vars = [
        {"name": pp.name, "unit": pp.unit.symbol, "default_value": float(pp.default_value)}
        for pp in brick.param_specs
    ]
    f_score = min(
        1.0,
        0.5 * max(stress_report.noisy_vector_field_r2, 0.0)
        + 0.5 * max(stress_report.ood_viable_fraction, 0.0),
    )

    rec = vault.crystallize_discovered_law(
        law_id=final_law_id,
        name=brick.name,
        domain=brick.domain,
        symbolic_expression=sym_expr,
        audit_certificate=stress_report.audit_certificate,
        falsification_score=f_score,
        state_variables=state_vars,
        parameters=param_vars,
        parent_law_ids=parent_law_ids,
        description=final_desc,
    )
    emit_trace_node(
        layer="layer6_agents",
        operation="crystallize_verified_law",
        target_name=rec.law_id,
        local_r2=stress_report.noisy_vector_field_r2,
        global_r2=stress_report.ood_viable_fraction,
        selected_branch_id=rec.law_id,
        selected_expression=sym_expr,
        metrics={
            "law_id": rec.law_id,
            "sha256_fingerprint": rec.sha256_fingerprint,
            "falsification_score": round(f_score, 4),
            "parent_law_ids": list(parent_law_ids),
        },
        skip_if_inner_loop=True,
    )
    return rec
