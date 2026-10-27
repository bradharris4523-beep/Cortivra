"""
Multi-stage parameter elimination funnel and hypothesis portfolio ranker
(`cortivra.layer4_engine.elimination_funnel`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick
from cortivra.layer1_axioms.shields import AuditCertificate, audit_law_brick
from cortivra.layer4_engine.rollout_arena import (
    RolloutResult,
    estimate_trajectory_derivatives,
    rollout_trajectories,
)


@dataclass(frozen=True)
class EliminationFunnelReport:
    """Result from the 4-stage parameter calibration funnel."""

    best_params: torch.Tensor
    best_rollout: RolloutResult
    trajectory_mse: float
    trajectory_r2: float
    vector_field_mse: float
    vector_field_r2: float
    stage_candidate_counts: tuple[int, int, int, int]
    audit_certificate: AuditCertificate


@dataclass(frozen=True)
class HypothesisRankingEntry:
    """Ranking entry for a candidate `LawBrick` in a multi-hypothesis portfolio comparison."""

    brick_name: str
    domain: str
    funnel_report: EliminationFunnelReport
    tunable_param_count: int
    symbolic_complexity: int
    bic_score: float
    passed_thermodynamic_audit: bool


def _infer_param_sampling_bounds(
    brick: LawBrick,
    custom_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> list[tuple[float, float, bool]]:
    """
    Determine finite `(low, high, use_log_scale)` sampling bounds for each parameter in `brick`.
    """
    bounds: list[tuple[float, float, bool]] = []
    for pp in brick.param_specs:
        if not pp.tunable:
            d = float(pp.default_value)
            bounds.append((d, d, False))
            continue

        if custom_bounds and pp.name in custom_bounds:
            lo, hi = float(custom_bounds[pp.name][0]), float(custom_bounds[pp.name][1])
            if hi < lo:
                lo, hi = hi, lo
        else:
            d = float(pp.default_value)
            abs_d = max(abs(d), 1.0)
            if math.isfinite(pp.min_value) and math.isfinite(pp.max_value) and (pp.max_value - pp.min_value) <= 1e4:
                lo, hi = float(pp.min_value), float(pp.max_value)
            elif pp.min_value >= 0.0:
                lo = max(float(pp.min_value), 0.1 * abs_d if d > 0 else 0.0)
                hi = min(float(pp.max_value), 12.0 * abs_d if d > 0 else 12.0)
            else:
                lo = max(float(pp.min_value), d - 6.0 * abs_d)
                hi = min(float(pp.max_value), d + 6.0 * abs_d)
            if hi <= lo:
                hi = lo + 1.0

        use_log = (lo > 1e-9) and ((hi / lo) >= 50.0)
        bounds.append((lo, hi, use_log))
    return bounds


def sample_parameter_universes(
    brick: LawBrick,
    num_samples: int = 256,
    seed: int = 42,
    custom_bounds: Mapping[str, tuple[float, float]] | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Generate `num_samples` space-filling parameter vectors `[num_samples, ParamDim]`
    using a quasi-random Sobol sequence across all tunable parameters of `brick`.
    Sample `#0` is always `brick.default_params(batch_size=1)`.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
    default_p = brick.default_params(batch_size=1, dtype=dtype, device=device)
    if brick.param_dim == 0 or num_samples == 1:
        return default_p.expand(num_samples, -1).clone()

    bounds = _infer_param_sampling_bounds(brick, custom_bounds=custom_bounds)
    tunable_indices = [idx for idx, pp in enumerate(brick.param_specs) if pp.tunable]
    if not tunable_indices:
        return default_p.expand(num_samples, -1).clone()

    sobol = torch.quasirandom.SobolEngine(dimension=len(tunable_indices), scramble=True, seed=seed)
    u = sobol.draw(num_samples).to(dtype=dtype, device=device)  # [num_samples, len(tunable)]

    out = default_p.expand(num_samples, -1).clone()
    for col_i, param_idx in enumerate(tunable_indices):
        lo, hi, use_log = bounds[param_idx]
        u_col = u[:, col_i]
        if use_log:
            log_lo = math.log(lo)
            log_hi = math.log(hi)
            vals = torch.exp(log_lo + u_col * (log_hi - log_lo))
        else:
            vals = lo + u_col * (hi - lo)
        out[:, param_idx] = vals

    # Ensure index 0 is the default parameter vector projected into custom_bounds
    out[0] = _project_tunable_params(brick, default_p, default_p, custom_bounds=custom_bounds)[0]
    return _project_tunable_params(brick, out, default_p, custom_bounds=custom_bounds)


def _project_tunable_params(
    brick: LawBrick,
    params: torch.Tensor,
    default_params_single: torch.Tensor,
    custom_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Clamp tunable parameters to `[min_value, max_value]` (and `custom_bounds`) and lock non-tunables."""
    clamped = brick.clamp_params(params)
    cols: list[torch.Tensor] = []
    for idx, pp in enumerate(brick.param_specs):
        if pp.tunable:
            col = clamped[:, idx]
            if custom_bounds and pp.name in custom_bounds:
                lo, hi = float(custom_bounds[pp.name][0]), float(custom_bounds[pp.name][1])
                if hi < lo:
                    lo, hi = hi, lo
                col = torch.clamp(col, min=lo, max=hi)
            cols.append(col)
        else:
            cols.append(default_params_single[0, idx].expand(clamped.shape[0]))
    return torch.stack(cols, dim=-1) if cols else clamped


def _compute_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = float(torch.sum((y_true - y_pred) ** 2).item())
    mean_true = torch.mean(y_true, dim=tuple(range(y_true.ndim - 1)), keepdim=True)
    ss_tot = float(torch.sum((y_true - mean_true) ** 2).item())
    if ss_tot <= 1e-18:
        return 1.0 if ss_res <= 1e-15 else 0.0
    return max(0.0, 1.0 - (ss_res / ss_tot))


def _levenberg_marquardt_vector_field_polish(
    brick: LawBrick,
    theta_init: torch.Tensor,
    flat_x: torch.Tensor,
    flat_dx: torch.Tensor,
    default_p: torch.Tensor,
    custom_bounds: Mapping[str, tuple[float, float]] | None = None,
    max_iters: int = 8,
) -> torch.Tensor:
    """
    Second-order Levenberg-Marquardt parameter polish (`(J_theta^T J_theta + lambda I) d_theta = -J_theta^T r`)
    on the instantaneous vector field `f(x, theta) - dx/dt`.
    """
    if brick.param_dim == 0 or not any(pp.tunable for pp in brick.param_specs):
        return theta_init

    curr_p = _project_tunable_params(brick, theta_init[:1].detach().clone(), default_p, custom_bounds=custom_bounds)
    n_pts = flat_x.shape[0]
    stride = max(int(n_pts // 24), 1)
    sub_x = flat_x[::stride]
    sub_dx = flat_dx[::stride]
    m_sub = sub_x.shape[0]
    p_dim = brick.param_dim
    eye_p = torch.eye(p_dim, dtype=torch.float64, device=flat_x.device)
    lm_lam = 1e-4

    with torch.no_grad():
        curr_mse = float(torch.mean((brick.flow(flat_x, curr_p.expand(n_pts, -1)) - flat_dx) ** 2).item())

    for _ in range(max_iters):
        if curr_mse <= 1e-22:
            break
        p_req = curr_p.clone().requires_grad_(True)
        res_mat = brick.flow(sub_x, p_req.expand(m_sub, -1)) - sub_dx  # [m_sub, StateDim]
        res_vec = res_mat.reshape(-1)
        n_res = res_vec.numel()

        jac_rows: list[torch.Tensor] = []
        for r_i in range(n_res):
            g_r = torch.autograd.grad(
                res_vec[r_i],
                p_req,
                retain_graph=(r_i < n_res - 1),
                allow_unused=True,
            )[0]
            jac_rows.append(g_r.squeeze(0) if g_r is not None else torch.zeros((p_dim,), dtype=torch.float64, device=flat_x.device))
        j_mat = torch.stack(jac_rows, dim=0)  # [n_res, p_dim]

        lhs = (j_mat.T @ j_mat) + lm_lam * eye_p
        rhs = -(j_mat.T @ res_vec.detach().unsqueeze(-1))
        delta_p = torch.linalg.lstsq(lhs, rhs, driver="gelsd").solution.squeeze(-1).unsqueeze(0)

        accepted = False
        for alpha in (1.0, 0.5, 0.2, 0.05):
            cand_p = _project_tunable_params(brick, curr_p + alpha * delta_p, default_p, custom_bounds=custom_bounds)
            with torch.no_grad():
                cand_mse = float(
                    torch.mean((brick.flow(flat_x, cand_p.expand(n_pts, -1)) - flat_dx) ** 2).item()
                )
            if math.isfinite(cand_mse) and cand_mse < curr_mse:
                curr_p = cand_p
                curr_mse = cand_mse
                lm_lam = max(lm_lam * 0.3, 1e-10)
                accepted = True
                break
        if not accepted:
            lm_lam = min(lm_lam * 5.0, 1e3)

    return curr_p


def run_elimination_funnel(
    brick: LawBrick,
    observed_states: torch.Tensor,
    dt: float,
    observed_dx_dt: torch.Tensor | None = None,
    observed_state_indices: Sequence[int] | None = None,
    initial_full_state: torch.Tensor | None = None,
    num_initial_samples: int = 256,
    num_collocation_survivors: int = 32,
    num_gradient_starts: int = 6,
    gradient_steps: int = 80,
    learning_rate: float = 0.08,
    integrator: str = "rk4",
    custom_bounds: Mapping[str, tuple[float, float]] | None = None,
    seed: int = 42,
) -> EliminationFunnelReport:
    """
    Execute the 4-Stage Parallel Universe Elimination Funnel to identify the optimal physical
    parameters `theta*` of `brick` for an observed trajectory (or multi-experiment ensemble).

    Supports:
        - Single-trajectory (`[T, D]`) or multi-experiment dose-response ensembles (`[E, T, D]`).
        - Full-state observations (`D == StateDim`) or partial-state observations
          (`observed_state_indices` specifying measured state dimensions).
        - Windowed multiple shooting & curriculum horizon continuation to prevent oscillatory
          phase-wrapping local minima.
        - Second-order Levenberg-Marquardt parameter polish for machine-precision convergence.
    """
    if observed_states.ndim == 2:
        obs_traj = observed_states.unsqueeze(0).to(torch.float64)
    elif observed_states.ndim == 3 and observed_states.shape[0] >= 1:
        obs_traj = observed_states.to(torch.float64)
    else:
        raise ValueError(
            f"Expected observed_states of shape [T, D] or [E, T, D], got {tuple(observed_states.shape)}."
        )

    n_exp = int(obs_traj.shape[0])
    t_steps = int(obs_traj.shape[1])
    if t_steps < 2:
        raise ValueError("observed_states must contain at least 2 time steps.")

    obs_idx = (
        tuple(int(i) for i in observed_state_indices)
        if observed_state_indices is not None
        else tuple(range(brick.state_dim))
    )
    full_state_observed = len(obs_idx) == brick.state_dim and obs_idx == tuple(range(brick.state_dim))

    if initial_full_state is not None:
        if initial_full_state.ndim == 1:
            x0_ensemble = initial_full_state.unsqueeze(0).expand(n_exp, -1).to(torch.float64)
        else:
            x0_ensemble = initial_full_state.reshape(-1, brick.state_dim).to(torch.float64)
            if x0_ensemble.shape[0] == 1 and n_exp > 1:
                x0_ensemble = x0_ensemble.expand(n_exp, -1)
    elif full_state_observed:
        x0_ensemble = obs_traj[:, 0, :]
    else:
        x0_ensemble = brick.default_state(batch_size=n_exp, dtype=torch.float64, device=obs_traj.device).clone()
        for col_i, s_idx in enumerate(obs_idx):
            x0_ensemble[:, s_idx] = obs_traj[:, 0, col_i]

    x0_single = x0_ensemble[:1]

    if observed_dx_dt is not None:
        dx_dt_obs = (
            observed_dx_dt.unsqueeze(0).to(torch.float64)
            if observed_dx_dt.ndim == 2
            else observed_dx_dt.to(torch.float64)
        )
    elif full_state_observed:
        dx_dt_obs = estimate_trajectory_derivatives(obs_traj, dt=dt)
    else:
        dx_dt_obs = None

    device = obs_traj.device
    default_p = brick.default_params(batch_size=1, dtype=torch.float64, device=device)

    # Stage 1: Quasi-random Sobol parameter (and optional hidden initial state) sampling
    n_stage1 = max(int(num_initial_samples), 1)
    candidates_s1 = sample_parameter_universes(
        brick,
        num_samples=n_stage1,
        seed=seed,
        custom_bounds=custom_bounds,
        dtype=torch.float64,
        device=device,
    )

    hidden_idx = tuple(i for i in range(brick.state_dim) if i not in obs_idx)
    optimize_hidden_x0 = (not full_state_observed) and (initial_full_state is None) and (len(hidden_idx) > 0)
    x0_s1 = x0_single.expand(n_stage1, -1).clone()
    if optimize_hidden_x0 and n_stage1 > 1:
        sobol_x0 = torch.quasirandom.SobolEngine(
            dimension=len(hidden_idx), scramble=True, seed=seed + 101
        )
        u_x0 = sobol_x0.draw(n_stage1).to(dtype=torch.float64, device=device)
        for hid_pos, s_i in enumerate(hidden_idx):
            sp = brick.state_specs[s_i]
            d_val = float(sp.default_value)
            span = max(abs(d_val), 1.0)
            lo_x = max(float(sp.homeostatic_min), d_val - 3.0 * span)
            hi_x = min(float(sp.homeostatic_max), d_val + 3.0 * span)
            x0_s1[:, s_i] = lo_x + u_x0[:, hid_pos] * (hi_x - lo_x)
        x0_s1[0] = x0_single[0]

    # Stage 2: Vector-field collocation sieve (or short-horizon partial-state sieve)
    n_stage2 = min(max(int(num_collocation_survivors), 1), n_stage1)
    if full_state_observed and dx_dt_obs is not None:
        all_pts_x = obs_traj.reshape(n_exp * t_steps, brick.state_dim)
        all_pts_dx = dx_dt_obs.reshape(n_exp * t_steps, brick.state_dim)
        total_pts = all_pts_x.shape[0]
        sample_t_indices = torch.linspace(0, total_pts - 1, min(total_pts, 24), dtype=torch.long, device=device)
        pts_x = all_pts_x[sample_t_indices, :]
        pts_dx = all_pts_dx[sample_t_indices, :]
        m_pts = pts_x.shape[0]

        vf_losses = torch.zeros((n_stage1,), dtype=torch.float64, device=device)
        for m_i in range(m_pts):
            x_rep = pts_x[m_i : m_i + 1].expand(n_stage1, -1)
            dx_rep = pts_dx[m_i : m_i + 1].expand(n_stage1, -1)
            pred_f = brick.flow(x_rep, candidates_s1)
            err = torch.mean((pred_f - dx_rep) ** 2, dim=-1)
            err = torch.where(torch.isfinite(err), err, torch.full_like(err, 1e18))
            vf_losses = vf_losses + err / float(m_pts)

        top_s2_idx = torch.argsort(vf_losses)[:n_stage2]
        candidates_s2 = candidates_s1[top_s2_idx]
        x0_s2 = x0_s1[top_s2_idx]
    else:
        short_steps = min(t_steps - 1, 8)
        short_roll = rollout_trajectories(
            brick, x0_s1, candidates_s1, dt=dt, num_steps=short_steps, integrator=integrator,
            record_flows_and_energies=False,
        )
        pred_sub = short_roll.states[:, :, obs_idx]
        target_sub = obs_traj[:1, : short_steps + 1, :].expand(n_stage1, -1, -1)
        short_mse = torch.mean((pred_sub - target_sub) ** 2, dim=(1, 2))
        viable_pen = torch.where(short_roll.all_time_viable_mask, 0.0, 1e12)
        scores = torch.where(torch.isfinite(short_mse), short_mse + viable_pen, torch.full_like(short_mse, 1e18))
        top_s2_idx = torch.argsort(scores)[:n_stage2]
        candidates_s2 = candidates_s1[top_s2_idx]
        x0_s2 = x0_s1[top_s2_idx]

    # Stage 3: Short-horizon viability filter and batched multi-start optimization
    n_stage3 = min(max(int(num_gradient_starts), 1), n_stage2)
    mid_steps = min(t_steps - 1, 15)
    mid_roll = rollout_trajectories(
        brick, x0_s2, candidates_s2, dt=dt, num_steps=mid_steps, integrator=integrator,
        record_flows_and_energies=False,
    )
    pred_mid = mid_roll.states[:, :, obs_idx]
    target_mid = obs_traj[:1, : mid_steps + 1, :].expand(n_stage2, -1, -1)
    mid_mse = torch.mean((pred_mid - target_mid) ** 2, dim=(1, 2))
    viable_pen_s2 = torch.where(mid_roll.all_time_viable_mask, 0.0, 1e12)
    mid_scores = torch.where(torch.isfinite(mid_mse), mid_mse + viable_pen_s2, torch.full_like(mid_mse, 1e18))
    top_s3_idx = torch.argsort(mid_scores)[:n_stage3]
    candidates_s3 = candidates_s2[top_s3_idx].clone()

    any_tunable = any(pp.tunable for pp in brick.param_specs)
    x0_candidates_s3 = x0_s2[top_s3_idx].clone()

    if (any_tunable or optimize_hidden_x0) and gradient_steps > 0:
        theta_opt = candidates_s3.clone().detach().requires_grad_(any_tunable and brick.param_dim > 0)
        opt_vars: list[torch.Tensor] = []
        if any_tunable and brick.param_dim > 0:
            opt_vars.append(theta_opt)

        if optimize_hidden_x0:
            hidden_init = torch.stack(
                [x0_candidates_s3[:, h_i] for h_i in hidden_idx], dim=-1
            ).detach().clone().requires_grad_(True)
            opt_vars.append(hidden_init)
        else:
            hidden_init = None

        optimizer = torch.optim.Adam(opt_vars, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(gradient_steps, 1), eta_min=learning_rate * 0.05
        )

        def _assemble_batch_x0(h_vec: torch.Tensor | None) -> torch.Tensor:
            if h_vec is None:
                return x0_single.expand(n_stage3, -1)
            cols: list[torch.Tensor] = []
            obs_pos = 0
            hid_pos = 0
            for s_i in range(brick.state_dim):
                if s_i in obs_idx:
                    cols.append(obs_traj[0, 0, obs_pos].expand(n_stage3))
                    obs_pos += 1
                else:
                    sp = brick.state_specs[s_i]
                    cols.append(
                        torch.clamp(h_vec[:, hid_pos], min=sp.homeostatic_min, max=sp.homeostatic_max)
                    )
                    hid_pos += 1
            return torch.stack(cols, dim=-1)

        if full_state_observed and dx_dt_obs is not None:
            flat_x = obs_traj.reshape(n_exp * t_steps, brick.state_dim)
            flat_dx = dx_dt_obs.reshape(n_exp * t_steps, brick.state_dim)
            n_total_pts = flat_x.shape[0]
            x_exp = flat_x.unsqueeze(0).expand(n_stage3, -1, -1).reshape(n_stage3 * n_total_pts, brick.state_dim)
            dx_exp = flat_dx.unsqueeze(0).expand(n_stage3, -1, -1).reshape(n_stage3 * n_total_pts, brick.state_dim)

            colloc_steps = gradient_steps if observed_dx_dt is not None else max(int(gradient_steps * 0.65), 1)
            polish_steps = gradient_steps - colloc_steps

            for _ in range(colloc_steps):
                optimizer.zero_grad()
                p_proj = _project_tunable_params(brick, theta_opt, default_p, custom_bounds=custom_bounds)
                p_exp = p_proj.unsqueeze(1).expand(-1, n_total_pts, -1).reshape(n_stage3 * n_total_pts, brick.param_dim)
                f_pred = brick.flow(x_exp, p_exp)
                loss = torch.mean((f_pred - dx_exp) ** 2)
                loss.backward()
                if theta_opt.grad is not None:
                    torch.nn.utils.clip_grad_norm_([theta_opt], max_norm=100.0)
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    theta_opt.copy_(_project_tunable_params(brick, theta_opt, default_p, custom_bounds=custom_bounds))

            if polish_steps > 0:
                # Windowed Multiple Shooting across short segments to eliminate phase-wrapping local minima
                win_len = min(t_steps - 1, 6)
                max_start = max(t_steps - win_len - 1, 0)
                n_wins = min(max_start + 1, 4)
                win_starts = torch.linspace(0, max_start, n_wins, dtype=torch.long, device=device)
                # Extract window initial states [n_wins, StateDim] and targets [n_wins, win_len + 1, StateDim]
                win_x0 = obs_traj[0, win_starts, :]
                win_targets = torch.stack([obs_traj[0, int(s.item()) : int(s.item()) + win_len + 1, :] for s in win_starts], dim=0)
                win_x0_batch = win_x0.unsqueeze(0).expand(n_stage3, -1, -1).reshape(n_stage3 * n_wins, brick.state_dim)
                win_target_batch = win_targets.unsqueeze(0).expand(n_stage3, -1, -1, -1).reshape(
                    n_stage3 * n_wins, win_len + 1, brick.state_dim
                )

                for _ in range(polish_steps):
                    optimizer.zero_grad()
                    p_proj = _project_tunable_params(brick, theta_opt, default_p, custom_bounds=custom_bounds)
                    p_win = p_proj.unsqueeze(1).expand(-1, n_wins, -1).reshape(n_stage3 * n_wins, brick.param_dim)
                    roll = rollout_trajectories(
                        brick, win_x0_batch, p_win, dt=dt, num_steps=win_len, integrator=integrator,
                        record_flows_and_energies=False,
                    )
                    per_u = torch.mean((roll.states - win_target_batch) ** 2, dim=(1, 2))
                    valid_u = torch.isfinite(per_u) & (per_u < 1e12)
                    if not bool(valid_u.any().item()):
                        break
                    loss = torch.mean(torch.where(valid_u, per_u, torch.zeros_like(per_u)))
                    loss.backward()
                    for v in opt_vars:
                        if v.grad is not None:
                            torch.nan_to_num_(v.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    torch.nn.utils.clip_grad_norm_(opt_vars, max_norm=100.0)
                    optimizer.step()
                    scheduler.step()
                    with torch.no_grad():
                        theta_opt.copy_(_project_tunable_params(brick, theta_opt, default_p, custom_bounds=custom_bounds))
        else:
            # 2-Stage Curriculum Horizon Continuation for partial-state trajectory shooting
            short_h = min(t_steps - 1, max(t_steps // 3, 8))
            stage_a_steps = int(gradient_steps * 0.4) if short_h < (t_steps - 1) else 0
            stage_b_steps = gradient_steps - stage_a_steps

            for h_steps, n_iters in ((short_h, stage_a_steps), (t_steps - 1, stage_b_steps)):
                if n_iters <= 0:
                    continue
                target_sub_h = obs_traj[:1, : h_steps + 1, :].expand(n_stage3, -1, -1)
                for _ in range(n_iters):
                    optimizer.zero_grad()
                    p_proj = _project_tunable_params(brick, theta_opt, default_p, custom_bounds=custom_bounds)
                    x0_s3 = _assemble_batch_x0(hidden_init)
                    roll = rollout_trajectories(
                        brick, x0_s3, p_proj, dt=dt, num_steps=h_steps, integrator=integrator,
                        record_flows_and_energies=False,
                    )
                    pred_obs = roll.states[:, :, obs_idx]
                    per_u = torch.mean((pred_obs - target_sub_h) ** 2, dim=(1, 2))
                    valid_u = torch.isfinite(per_u) & (per_u < 1e12)
                    if not bool(valid_u.any().item()):
                        break
                    loss = torch.mean(torch.where(valid_u, per_u, torch.zeros_like(per_u)))
                    loss.backward()
                    for v in opt_vars:
                        if v.grad is not None:
                            torch.nan_to_num_(v.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    torch.nn.utils.clip_grad_norm_(opt_vars, max_norm=100.0)
                    optimizer.step()
                    scheduler.step()
                    with torch.no_grad():
                        theta_opt.copy_(_project_tunable_params(brick, theta_opt, default_p, custom_bounds=custom_bounds))
                        if hidden_init is not None:
                            for hid_pos, s_i in enumerate(hidden_idx):
                                sp = brick.state_specs[s_i]
                                hidden_init[:, hid_pos].clamp_(min=sp.homeostatic_min, max=sp.homeostatic_max)

        candidates_s3 = _project_tunable_params(brick, theta_opt.detach(), default_p, custom_bounds=custom_bounds)
        x0_candidates_s3 = _assemble_batch_x0(hidden_init.detach() if hidden_init is not None else None)

        # Second-order Levenberg-Marquardt polish on the top candidate when vector-field is observed
        if full_state_observed and dx_dt_obs is not None:
            flat_x_all = obs_traj.reshape(n_exp * t_steps, brick.state_dim)
            flat_dx_all = dx_dt_obs.reshape(n_exp * t_steps, brick.state_dim)
            vf_s3 = torch.zeros((n_stage3,), dtype=torch.float64, device=device)
            for k_i in range(n_stage3):
                p_k = candidates_s3[k_i : k_i + 1].expand(flat_x_all.shape[0], -1)
                vf_s3[k_i] = torch.mean((brick.flow(flat_x_all, p_k) - flat_dx_all) ** 2)
            vf_s3 = torch.where(torch.isfinite(vf_s3), vf_s3, torch.full_like(vf_s3, 1e18))
            top_lm_idx = int(torch.argmin(vf_s3).item())
            polished_top = _levenberg_marquardt_vector_field_polish(
                brick=brick,
                theta_init=candidates_s3[top_lm_idx : top_lm_idx + 1],
                flat_x=flat_x_all,
                flat_dx=flat_dx_all,
                default_p=default_p,
                custom_bounds=custom_bounds,
                max_iters=8,
            )
            candidates_s3 = candidates_s3.clone()
            candidates_s3[top_lm_idx] = polished_top[0]

    # Stage 4: Full-horizon rollout verification and Layer 1 thermodynamic audit
    full_rollouts = rollout_trajectories(
        brick, x0_candidates_s3, candidates_s3, dt=dt, num_steps=t_steps - 1, integrator=integrator,
        record_flows_and_energies=True,
    )
    pred_full_obs = full_rollouts.states[:, :, obs_idx]
    target_full_obs = obs_traj[:1].expand(n_stage3, -1, -1)
    traj_mses = torch.mean((pred_full_obs - target_full_obs) ** 2, dim=(1, 2))
    viable_pen_final = torch.where(full_rollouts.all_time_viable_mask, 0.0, 1e12)

    if full_state_observed and dx_dt_obs is not None:
        flat_x_all = obs_traj.reshape(n_exp * t_steps, brick.state_dim)
        flat_dx_all = dx_dt_obs.reshape(n_exp * t_steps, brick.state_dim)
        vf_mses = torch.zeros((n_stage3,), dtype=torch.float64, device=device)
        for k_i in range(n_stage3):
            p_k = candidates_s3[k_i : k_i + 1].expand(flat_x_all.shape[0], -1)
            f_k = brick.flow(flat_x_all, p_k)
            vf_mses[k_i] = torch.mean((f_k - flat_dx_all) ** 2)
        total_scores = vf_mses + 0.1 * traj_mses + viable_pen_final
    else:
        total_scores = traj_mses + viable_pen_final

    total_scores = torch.where(torch.isfinite(total_scores), total_scores, torch.full_like(total_scores, 1e18))
    best_idx = int(torch.argmin(total_scores).item())
    best_params = candidates_s3[best_idx : best_idx + 1].clone()
    best_x0 = (
        x0_ensemble
        if (full_state_observed or initial_full_state is not None)
        else x0_candidates_s3[best_idx : best_idx + 1].expand(n_exp, -1).clone()
    )
    best_roll = rollout_trajectories(
        brick, best_x0, best_params.expand(n_exp, -1), dt=dt, num_steps=t_steps - 1, integrator=integrator,
        record_flows_and_energies=True,
    )

    best_pred_obs = best_roll.states[:, :, obs_idx]
    t_mse = float(torch.mean((best_pred_obs - obs_traj) ** 2).item())
    t_r2 = _compute_r2(obs_traj.reshape(-1, len(obs_idx)), best_pred_obs.reshape(-1, len(obs_idx)))

    if full_state_observed and dx_dt_obs is not None:
        flat_x_all = obs_traj.reshape(n_exp * t_steps, brick.state_dim)
        flat_dx_all = dx_dt_obs.reshape(n_exp * t_steps, brick.state_dim)
        p_rep = best_params.expand(flat_x_all.shape[0], -1)
        f_best = brick.flow(flat_x_all, p_rep)
        v_mse = float(torch.mean((f_best - flat_dx_all) ** 2).item())
        v_r2 = _compute_r2(flat_dx_all, f_best)
        audit_states = flat_x_all
        audit_params = p_rep
    else:
        v_mse = t_mse
        v_r2 = t_r2
        audit_states = best_roll.states.reshape(n_exp * t_steps, brick.state_dim)
        audit_params = best_params.expand(n_exp * t_steps, -1)

    cert = audit_law_brick(brick, sample_state=audit_states, sample_params=audit_params)

    return EliminationFunnelReport(
        best_params=best_params,
        best_rollout=best_roll,
        trajectory_mse=t_mse,
        trajectory_r2=t_r2,
        vector_field_mse=v_mse,
        vector_field_r2=v_r2,
        stage_candidate_counts=(n_stage1, n_stage2, n_stage3, 1),
        audit_certificate=cert,
    )


def rank_hypothesis_portfolio(
    bricks: Sequence[LawBrick],
    observed_states: torch.Tensor,
    dt: float,
    observed_dx_dt: torch.Tensor | None = None,
    num_initial_samples: int = 128,
    gradient_steps: int = 60,
    integrator: str = "rk4",
) -> list[HypothesisRankingEntry]:
    """
    Evaluate a portfolio of competing `LawBrick` hypotheses on the same observed trajectory
    and rank them from lowest (best) to highest Bayesian Information Criterion (`BIC`) score,
    prioritizing models that pass Layer 1's thermodynamic audit.
    """
    if not bricks:
        raise ValueError("`bricks` portfolio cannot be empty.")

    n_obs = max(int(observed_states.numel()), 1)
    entries: list[HypothesisRankingEntry] = []

    for brick in bricks:
        rep = run_elimination_funnel(
            brick=brick,
            observed_states=observed_states,
            dt=dt,
            observed_dx_dt=observed_dx_dt,
            num_initial_samples=num_initial_samples,
            gradient_steps=gradient_steps,
            integrator=integrator,
        )
        k_params = sum(1 for pp in brick.param_specs if pp.tunable)
        sym_comp = (
            int(brick.total_symbolic_complexity())
            if hasattr(brick, "total_symbolic_complexity")
            else (
                int(brick.total_patch_complexity())
                if hasattr(brick, "total_patch_complexity")
                else k_params * 2
            )
        )
        mse_val = max(min(rep.vector_field_mse, rep.trajectory_mse), 1e-15)
        bic = float(n_obs * math.log(mse_val) + (k_params + 0.25 * sym_comp) * math.log(max(n_obs, 2)))
        if not rep.audit_certificate.passed_all:
            bic += 1e6

        entries.append(
            HypothesisRankingEntry(
                brick_name=brick.name,
                domain=brick.domain,
                funnel_report=rep,
                tunable_param_count=k_params,
                symbolic_complexity=sym_comp,
                bic_score=bic,
                passed_thermodynamic_audit=rep.audit_certificate.passed_all,
            )
        )

    entries.sort(key=lambda e: (not e.passed_thermodynamic_audit, e.bic_score))
    return entries
