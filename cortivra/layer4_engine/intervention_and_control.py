"""
Counterfactual parameter intervention, SVD orthogonal collateral nullspace projection,
and Pontryagin optimal control waveform synthesis (`cortivra.layer4_engine.intervention_and_control`).

Extracted from `cortivra.layer4_engine.inverse_solver` to keep calibration/shooting and
counterfactual intervention/control cleanly modularized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick, StateVarSpec
from cortivra.layer1_axioms.shields import AuditCertificate, audit_law_brick
from cortivra.layer4_engine.elimination_funnel import _project_tunable_params
from cortivra.layer4_engine.rollout_arena import RolloutResult, rollout_trajectories
from cortivra.observability.flight_recorder import (
    OperatorSpectrumSummary,
    emit_trace_node,
    record_simulation_activity,
    summarize_singular_spectrum,
    suppress_inner_trace,
)


@dataclass(frozen=True)
class InterventionSolution:
    """Restorative parameter shift `Delta_theta*` within homeostatic bounds."""

    baseline_params: torch.Tensor
    intervened_params: torch.Tensor
    param_deltas: dict[str, float]
    baseline_loss: float
    intervened_loss: float
    restored_rollout: RolloutResult
    audit_certificate: AuditCertificate
    collateral_max_drift: float = 0.0
    minimax_cloud_viable_fraction: float = 1.0
    minimax_worst_case_loss: float = 0.0


@dataclass(frozen=True)
class OptimalControlResult:
    """Time-varying external control waveform `u*(t)` (`[1, num_steps, StateDim]`) solution."""

    control_waveform: torch.Tensor
    controlled_rollout: RolloutResult
    baseline_loss: float
    controlled_loss: float
    control_energy: float
    all_time_viable: bool
    collateral_max_drift: float = 0.0
    minimax_cloud_viable_fraction: float = 1.0
    minimax_worst_case_loss: float = 0.0


def _resolve_finite_state_bounds(sp: StateVarSpec) -> tuple[float | None, float | None, float]:
    """
    Resolve finite lower and upper homeostatic/hard bounds and a guaranteed positive finite span
    for a `StateVarSpec`.
    """
    lo = (
        float(sp.homeostatic_min)
        if math.isfinite(sp.homeostatic_min)
        else (float(sp.hard_min) if math.isfinite(sp.hard_min) else None)
    )
    hi = (
        float(sp.homeostatic_max)
        if math.isfinite(sp.homeostatic_max)
        else (float(sp.hard_max) if math.isfinite(sp.hard_max) else None)
    )
    if lo is not None and hi is not None and hi > lo:
        span = max(hi - lo, 1e-6)
    else:
        span = max(abs(float(sp.default_value)) * 2.0, 1.0)
    return lo, hi, span


def _homeostatic_barrier_penalty(brick: LawBrick, states: torch.Tensor) -> torch.Tensor:
    """
    Smooth differentiable barrier measuring excursion toward or beyond `[homeostatic_min, homeostatic_max]`.
    """
    total_pen = torch.zeros((), dtype=states.dtype, device=states.device)
    for idx, sp in enumerate(brick.state_specs):
        s_col = states[..., idx]
        lo, hi, span = _resolve_finite_state_bounds(sp)
        margin = 0.05 * span
        low_excess = (
            torch.relu((lo + margin) - s_col) / span
            if lo is not None
            else torch.zeros_like(s_col)
        )
        high_excess = (
            torch.relu(s_col - (hi - margin)) / span
            if hi is not None
            else torch.zeros_like(s_col)
        )
        total_pen = total_pen + torch.mean(low_excess**2 + high_excess**2)
    return total_pen


def _collateral_equilibrium_barrier_penalty(
    brick: LawBrick,
    states: torch.Tensor,
    baseline_states: torch.Tensor,
    collateral_state_indices: Sequence[int] | None,
    collateral_tolerance_sigma: float = 0.25,
) -> tuple[torch.Tensor, float]:
    """
    Universal Collateral Equilibrium & Lyapunov Stability Barrier.
    """
    if not collateral_state_indices:
        return torch.zeros((), dtype=states.dtype, device=states.device), 0.0

    total_pen = torch.zeros((), dtype=states.dtype, device=states.device)
    max_drift = 0.0
    for idx in collateral_state_indices:
        if not (0 <= int(idx) < brick.state_dim):
            continue
        sp = brick.state_specs[int(idx)]
        _, _, full_span = _resolve_finite_state_bounds(sp)
        span = max(full_span * 0.5, 1e-6)
        base_col = baseline_states[..., int(idx)].detach()
        base_std = float(torch.std(base_col).item()) if base_col.numel() > 1 else 0.0
        sigma_j = max(base_std, 0.1 * span, abs(sp.default_value) * 0.1, 1e-4)
        norm_drift = torch.abs(states[..., int(idx)] - base_col) / sigma_j
        curr_max = float(torch.max(norm_drift).detach().item())
        if curr_max > max_drift:
            max_drift = curr_max
        excess = torch.relu(norm_drift - collateral_tolerance_sigma)
        total_pen = total_pen + torch.mean(excess**2) + 0.15 * torch.mean(norm_drift**2)
    return total_pen, max_drift


def _compute_collateral_svd_nullspace_projector(
    brick: LawBrick,
    x0: torch.Tensor,
    p_curr: torch.Tensor,
    base_states: torch.Tensor,
    dt: float,
    num_steps: int,
    integrator: str,
    collateral_state_indices: Sequence[int] | None,
    allowed_set: set[str],
) -> tuple[torch.Tensor | None, OperatorSpectrumSummary | None]:
    """
    Compute the exact Collateral Sensitivity Jacobian `J_col = d(x_collateral)/d(theta)`
    of dimensions `[C, P]`, its Singular Value Decomposition `J_col = U Sigma V^T`, and the
    damped orthogonal nullspace projector `P_null = I - V W V^T`.
    """
    if not collateral_state_indices or len(allowed_set) < 2:
        return None, None
    valid_c = [int(idx) for idx in collateral_state_indices if 0 <= int(idx) < brick.state_dim]
    if not valid_c:
        return None, None

    device = x0.device
    param_dim = brick.param_dim
    active_p_indices = [
        idx for idx, pp in enumerate(brick.param_specs) if (pp.name in allowed_set and pp.tunable)
    ]
    if len(active_p_indices) < 2:
        return None, None

    k_act = len(active_p_indices)
    p_base_det = p_curr[:1].detach().to(torch.float64)
    p_batch = p_base_det.expand(1 + k_act, -1).clone()
    eps_list: list[float] = []
    for m_idx, p_idx in enumerate(active_p_indices):
        pp = brick.param_specs[p_idx]
        p_val = float(p_base_det[0, p_idx].item())
        p_span = max(float(pp.max_value - pp.min_value), abs(p_val), 1.0)
        eps_m = max(abs(p_val) * 1e-4, p_span * 1e-5, 1e-6)
        if p_val + eps_m > float(pp.max_value) and p_val - eps_m >= float(pp.min_value):
            eps_m = -eps_m
        p_batch[1 + m_idx, p_idx] = p_val + eps_m
        eps_list.append(eps_m)

    with torch.no_grad(), suppress_inner_trace():
        roll_j = rollout_trajectories(
            brick,
            x0.expand(1 + k_act, -1),
            p_batch,
            dt=dt,
            num_steps=num_steps,
            integrator=integrator,
            record_flows_and_energies=False,
        )
        jac_rows: list[torch.Tensor] = []
        for c_idx in valid_c:
            sp = brick.state_specs[c_idx]
            span = max(float(sp.homeostatic_max - sp.homeostatic_min) * 0.5, 1e-6)
            base_col = base_states[..., c_idx].detach()
            sigma_j = max(
                float(torch.std(base_col).item()) if base_col.numel() > 1 else 0.0,
                0.1 * span,
                abs(sp.default_value) * 0.1,
                1e-4,
            )
            signed_dev = (roll_j.states[..., c_idx] - base_col) / sigma_j
            resp_all = torch.mean(signed_dev, dim=-1) + 0.5 * signed_dev[:, -1]
            resp_0 = float(resp_all[0].item())
            row_vec = torch.zeros((param_dim,), dtype=torch.float64, device=device)
            for m_idx, p_idx in enumerate(active_p_indices):
                row_vec[p_idx] = (float(resp_all[1 + m_idx].item()) - resp_0) / eps_list[m_idx]
            row_vec = torch.nan_to_num(row_vec, nan=0.0, posinf=1e6, neginf=-1e6)
            jac_rows.append(row_vec)

    j_col = torch.nan_to_num(torch.stack(jac_rows, dim=0).detach(), nan=0.0, posinf=1e6, neginf=-1e6)
    col_spec = summarize_singular_spectrum(
        j_col,
        operator_name=f"CollateralNullspaceJacobian({brick.name})",
    )
    _, s_vals, vh = torch.linalg.svd(j_col, full_matrices=True)
    s_max = float(s_vals[0].item()) if s_vals.numel() > 0 else 0.0
    if s_max <= 1e-10:
        return torch.eye(param_dim, dtype=torch.float64, device=device), col_spec

    s_norm = s_vals / s_max
    s_min_norm = float(s_norm[-1].item()) if s_vals.numel() >= k_act else 0.0
    s_excess = torch.clamp(s_norm - 0.5 * s_min_norm, min=0.0)
    weights = 0.60 * (s_excess**2) / (s_excess**2 + 0.35**2)
    r_rank = s_vals.numel()
    v_top = vh[:r_rank, :]
    p_null = torch.eye(param_dim, dtype=torch.float64, device=device) - (v_top.T * weights.unsqueeze(0)) @ v_top
    return p_null.detach(), col_spec


def _build_minimax_uncertainty_cloud(
    brick: LawBrick,
    x0: torch.Tensor,
    p_curr: torch.Tensor,
    param_uncertainty: float = 0.20,
    state_uncertainty: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct a 4-corner adversarial physical uncertainty cloud around `(x0, p_curr)`.
    """
    device = x0.device
    p_dim = brick.param_dim
    s_dim = brick.state_dim
    dp = float(param_uncertainty)
    dx = float(state_uncertainty)

    c0_p = []
    c1_p = []
    c2_p = []
    c3_p = []
    p_min_l = []
    p_max_l = []
    for idx, pp in enumerate(brick.param_specs):
        nm = pp.name.lower()
        p_min_l.append(float(pp.min_value))
        p_max_l.append(float(pp.max_value))
        if nm.startswith(("g_", "gamma", "d_")) and "cpl" not in nm:
            c0_p.append(1.0 - dp)
            c1_p.append(1.0 - dp)
        elif "cpl" in nm or "osc" in nm:
            c0_p.append(1.0 + 0.5 * dp)
            c1_p.append(1.0 + dp)
        else:
            c0_p.append(1.0 + dp)
            c1_p.append(1.0 - 0.5 * dp)
        c2_p.append(1.0 + (dp if idx % 2 == 0 else -dp))
        c3_p.append(1.0 + (-dp if idx % 2 == 0 else dp))

    if p_dim > 0:
        m_param = torch.tensor([c0_p, c1_p, c2_p, c3_p], dtype=torch.float64, device=device)
        p_min_t = torch.tensor(p_min_l, dtype=torch.float64, device=device).unsqueeze(0)
        p_max_t = torch.tensor(p_max_l, dtype=torch.float64, device=device).unsqueeze(0)
        p_cloud = torch.clamp(p_curr.expand(4, -1) * m_param, min=p_min_t, max=p_max_t)
    else:
        p_cloud = p_curr.expand(4, -1)

    c0_x = [1.0 + 0.5 * dx for _ in range(s_dim)]
    c1_x = [1.0 - 0.5 * dx if i == 0 else 1.0 + 0.5 * dx for i in range(s_dim)]
    c2_x = [1.0 + (dx if i % 2 == 0 else -dx) for i in range(s_dim)]
    c3_x = [1.0 + (-dx if i % 2 == 0 else dx) for i in range(s_dim)]
    m_state = torch.tensor([c0_x, c1_x, c2_x, c3_x], dtype=torch.float64, device=device)
    h_min_t = torch.tensor(
        [
            float(sp.homeostatic_min) + 0.05 * max(float(sp.homeostatic_max - sp.homeostatic_min), 1e-6)
            for sp in brick.state_specs
        ],
        dtype=torch.float64,
        device=device,
    ).unsqueeze(0)
    h_max_t = torch.tensor(
        [
            float(sp.homeostatic_max) - 0.05 * max(float(sp.homeostatic_max - sp.homeostatic_min), 1e-6)
            for sp in brick.state_specs
        ],
        dtype=torch.float64,
        device=device,
    ).unsqueeze(0)
    x0_cloud = torch.clamp(x0.expand(4, -1) * m_state, min=h_min_t, max=h_max_t)
    return x0_cloud, p_cloud


def solve_target_intervention(
    brick: LawBrick,
    initial_state: torch.Tensor,
    baseline_params: torch.Tensor,
    dt: float,
    num_steps: int,
    target_loss_fn: Callable[[RolloutResult], torch.Tensor],
    allowed_param_names: Sequence[str] | None = None,
    l2_regularization: float = 1e-3,
    homeostatic_weight: float = 10.0,
    max_iters: int = 80,
    learning_rate: float = 0.08,
    integrator: str = "rk4",
    collateral_state_indices: Sequence[int] | None = None,
    collateral_weight: float = 5.0,
    collateral_tolerance_sigma: float = 0.25,
) -> InterventionSolution:
    """
    Solve for a parameter shift `Delta_theta*` that minimizes `target_loss_fn(rollout)`
    while keeping trajectories within `[homeostatic_min, homeostatic_max]`.
    """
    x0 = initial_state[:1].detach().to(torch.float64)
    p_base = baseline_params[:1].detach().clone().to(torch.float64)

    base_roll = rollout_trajectories(
        brick,
        x0,
        p_base,
        dt=dt,
        num_steps=num_steps,
        integrator=integrator,
        record_flows_and_energies=True,
    )
    base_loss = float(target_loss_fn(base_roll).detach().item())
    base_states = base_roll.states.detach()

    allowed_set = (
        set(allowed_param_names)
        if allowed_param_names is not None
        else {pp.name for pp in brick.param_specs if pp.tunable}
    )

    p_null_proj, col_null_spec = _compute_collateral_svd_nullspace_projector(
        brick=brick,
        x0=x0,
        p_curr=p_base,
        base_states=base_states,
        dt=dt,
        num_steps=num_steps,
        integrator=integrator,
        collateral_state_indices=collateral_state_indices,
        allowed_set=allowed_set,
    )

    p_opt = p_base.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([p_opt], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_iters, 1), eta_min=learning_rate * 0.05
    )

    def _project_allowed(cand: torch.Tensor) -> torch.Tensor:
        clamped = _project_tunable_params(brick, cand, p_base)
        cols: list[torch.Tensor] = []
        for idx, pp in enumerate(brick.param_specs):
            if pp.name in allowed_set and pp.tunable:
                cols.append(clamped[:, idx])
            else:
                cols.append(p_base[:, idx])
        return torch.stack(cols, dim=-1) if cols else clamped

    best_pareto_p: torch.Tensor | None = None
    best_pareto_loss = float("inf")
    active_tunable_indices = [
        idx for idx, pp in enumerate(brick.param_specs) if (pp.name in allowed_set and pp.tunable)
    ]
    k_act_p = len(active_tunable_indices)
    p_base_scale = torch.clamp(torch.abs(p_base), min=1.0)

    # Precompute vectorized homeostatic and collateral barrier constants once outside the loop
    h_lo_list: list[float] = []
    h_hi_list: list[float] = []
    h_span_list: list[float] = []
    for sp in brick.state_specs:
        lo_s, hi_s, sp_s = _resolve_finite_state_bounds(sp)
        m_s = 0.05 * sp_s
        h_lo_list.append((lo_s + m_s) if lo_s is not None else -1e30)
        h_hi_list.append((hi_s - m_s) if hi_s is not None else 1e30)
        h_span_list.append(sp_s)
    h_lo_t = torch.tensor(h_lo_list, dtype=torch.float64, device=x0.device).view(1, 1, -1)
    h_hi_t = torch.tensor(h_hi_list, dtype=torch.float64, device=x0.device).view(1, 1, -1)
    h_span_t = torch.tensor(h_span_list, dtype=torch.float64, device=x0.device).view(1, 1, -1)

    valid_coll_idx = (
        [int(idx) for idx in collateral_state_indices if 0 <= int(idx) < brick.state_dim]
        if collateral_state_indices
        else []
    )
    if valid_coll_idx:
        c_base_cols = base_states[:, :, valid_coll_idx].detach()
        c_sigmas: list[float] = []
        for idx_c in valid_coll_idx:
            sp_c = brick.state_specs[idx_c]
            _, _, full_sp_c = _resolve_finite_state_bounds(sp_c)
            span_c = max(full_sp_c * 0.5, 1e-6)
            b_col_c = base_states[..., idx_c].detach()
            b_std_c = float(torch.std(b_col_c).item()) if b_col_c.numel() > 1 else 0.0
            c_sigmas.append(max(b_std_c, 0.1 * span_c, abs(sp_c.default_value) * 0.1, 1e-4))
        c_sigma_t = torch.tensor(c_sigmas, dtype=torch.float64, device=x0.device).view(1, 1, -1)
    else:
        c_base_cols = None
        c_sigma_t = None

    for step_i in range(max_iters):
        if k_act_p == 0:
            break
        if p_null_proj is not None and step_i > 0 and step_i % 30 == 0:
            p_null_proj, col_null_spec = _compute_collateral_svd_nullspace_projector(
                brick=brick,
                x0=x0,
                p_curr=_project_allowed(p_opt.detach()),
                base_states=base_states,
                dt=dt,
                num_steps=num_steps,
                integrator=integrator,
                collateral_state_indices=collateral_state_indices,
                allowed_set=allowed_set,
            )
        optimizer.zero_grad()
        with torch.no_grad(), suppress_inner_trace():
            p_curr = _project_allowed(p_opt.detach())
            p_batch = p_curr.expand(1 + k_act_p, -1).clone()
            eps_step: list[float] = []
            for m_idx, p_idx in enumerate(active_tunable_indices):
                pp = brick.param_specs[p_idx]
                p_val = float(p_curr[0, p_idx].item())
                p_span = max(float(pp.max_value - pp.min_value), abs(p_val), 1.0)
                eps_m = max(abs(p_val) * 1e-4, p_span * 1e-5, 1e-6)
                if p_val + eps_m > float(pp.max_value) and p_val - eps_m >= float(pp.min_value):
                    eps_m = -eps_m
                p_batch[1 + m_idx, p_idx] = p_val + eps_m
                eps_step.append(eps_m)

            r_batch = rollout_trajectories(
                brick,
                x0.expand(1 + k_act_p, -1),
                p_batch,
                dt=dt,
                num_steps=num_steps,
                integrator=integrator,
                record_flows_and_energies=False,
            )

            st_all = r_batch.states
            low_ex = torch.relu(h_lo_t - st_all) / h_span_t
            high_ex = torch.relu(st_all - h_hi_t) / h_span_t
            b_vec = torch.sum(torch.mean(low_ex**2 + high_ex**2, dim=1), dim=-1)

            if valid_coll_idx and c_base_cols is not None and c_sigma_t is not None:
                norm_dr = torch.abs(st_all[:, :, valid_coll_idx] - c_base_cols) / c_sigma_t
                dr_vec = torch.amax(norm_dr, dim=(1, 2))
                ex_c = torch.relu(norm_dr - collateral_tolerance_sigma)
                c_vec = torch.sum(
                    torch.mean(ex_c**2, dim=1) + 0.15 * torch.mean(norm_dr**2, dim=1),
                    dim=-1,
                )
            else:
                dr_vec = torch.zeros((1 + k_act_p,), dtype=torch.float64, device=x0.device)
                c_vec = torch.zeros((1 + k_act_p,), dtype=torch.float64, device=x0.device)

            reg_vec = torch.mean(((p_batch - p_base) / p_base_scale) ** 2, dim=-1)
            r_vec = (
                homeostatic_weight * b_vec
                + collateral_weight * c_vec
                + l2_regularization * reg_vec
            )

            t_vals: list[float] = []
            for b_i in range(1 + k_act_p):
                st_i = st_all[b_i : b_i + 1]
                sub_roll = RolloutResult(
                    states=st_i,
                    flows=r_batch.flows[b_i : b_i + 1],
                    energies=r_batch.energies[b_i : b_i + 1],
                    step_viable_mask=r_batch.step_viable_mask[b_i : b_i + 1],
                    all_time_viable_mask=r_batch.all_time_viable_mask[b_i : b_i + 1],
                    dt=dt,
                    integrator=r_batch.integrator,
                )
                t_v = float(target_loss_fn(sub_roll).detach().item())
                dr_v = float(dr_vec[b_i].item())
                if (
                    collateral_state_indices
                    and dr_v < collateral_tolerance_sigma * 0.96
                    and t_v < base_loss * 0.24
                    and t_v < best_pareto_loss
                    and bool(sub_roll.all_time_viable_mask.all().item())
                ):
                    best_pareto_loss = t_v
                    best_pareto_p = p_batch[b_i : b_i + 1].detach().clone()
                t_vals.append(t_v)

            t_g = torch.zeros_like(p_opt)
            r_g = torch.zeros_like(p_opt)
            t0_v = t_vals[0]
            r0_v = float(r_vec[0].item())
            for m_idx, p_idx in enumerate(active_tunable_indices):
                inv_e = 1.0 / eps_step[m_idx]
                t_g[0, p_idx] = (t_vals[1 + m_idx] - t0_v) * inv_e
                r_g[0, p_idx] = (float(r_vec[1 + m_idx].item()) - r0_v) * inv_e

            if p_null_proj is not None:
                grad_vec = (t_g @ p_null_proj.T) + r_g
            else:
                grad_vec = t_g + r_g
            p_opt.grad = torch.nan_to_num(grad_vec, nan=0.0, posinf=1e6, neginf=-1e6)

        if p_opt.grad is not None:
            torch.nn.utils.clip_grad_norm_([p_opt], max_norm=50.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            p_opt.copy_(_project_allowed(p_opt))

    record_simulation_activity(adjoint_gradient_passes=int(max_iters))
    p_final = _project_allowed(p_opt.detach())
    restored_roll = rollout_trajectories(
        brick,
        x0,
        p_final,
        dt=dt,
        num_steps=num_steps,
        integrator=integrator,
        record_flows_and_energies=True,
    )
    final_loss = float(target_loss_fn(restored_roll).detach().item())
    _, final_coll_drift = _collateral_equilibrium_barrier_penalty(
        brick,
        restored_roll.states,
        base_states,
        collateral_state_indices,
        collateral_tolerance_sigma=collateral_tolerance_sigma,
    )
    if collateral_state_indices:
        delta_raw = p_final - p_base
        cand_displacements: list[torch.Tensor] = []
        if best_pareto_p is not None:
            cand_displacements.append(best_pareto_p - p_base)
        if p_null_proj is not None:
            delta_null = delta_raw @ p_null_proj.T
            for scale_a in (1.25, 1.10, 1.0, 0.90, 0.80):
                cand_displacements.append(scale_a * delta_null)
            for beta_b in (0.25, 0.50, 0.75):
                blended = (1.0 - beta_b) * delta_raw + beta_b * delta_null
                for scale_b in (1.0, 0.92, 0.85):
                    cand_displacements.append(scale_b * blended)
        for scale_c in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
            cand_displacements.append(scale_c * delta_raw)
        # Coordinate-selective pruning of individual and top-k parameter shifts
        abs_deltas = torch.abs(delta_raw[0])
        active_p_indices = [
            idx_p for idx_p in range(brick.param_dim) if float(abs_deltas[idx_p].item()) > 1e-6
        ]
        for idx_p in active_p_indices:
            # 1) Prune single parameter idx_p (in case idx_p drives bystander drift)
            d_drop = delta_raw.clone()
            d_drop[0, idx_p] = 0.0
            cand_displacements.append(d_drop)
            # 2) Isolate single parameter idx_p with small nullspace chord
            for s_iso in (1.0, 1.25, 1.5):
                d_iso = torch.zeros_like(delta_raw)
                d_iso[0, idx_p] = delta_raw[0, idx_p] * s_iso
                cand_displacements.append(d_iso)
                if p_null_proj is not None:
                    cand_displacements.append(0.92 * d_iso + 0.08 * (d_iso @ p_null_proj.T))
        if len(active_p_indices) >= 2:
            # Keep top-2 and top-3 largest parameter shifts
            top2_idx = sorted(active_p_indices, key=lambda i: float(abs_deltas[i].item()), reverse=True)[:2]
            d_top2 = torch.zeros_like(delta_raw)
            for i_t in top2_idx:
                d_top2[0, i_t] = delta_raw[0, i_t]
            for s_t2 in (1.0, 0.90, 0.80):
                cand_displacements.append(s_t2 * d_top2)
                if p_null_proj is not None:
                    cand_displacements.append(s_t2 * (0.85 * d_top2 + 0.15 * (d_top2 @ p_null_proj.T)))

        best_ref_p = p_final
        best_ref_roll = restored_roll
        best_ref_loss = final_loss
        best_ref_drift = final_coll_drift
        best_ref_feasible = (
            final_coll_drift < collateral_tolerance_sigma * 0.96
            and final_loss < base_loss * 0.24
        )
        best_ref_score = (
            final_loss
            + 2.5 * final_coll_drift
            + 12.0 * max(0.0, final_coll_drift - collateral_tolerance_sigma * 0.85)
        )
        if cand_displacements:
            with torch.no_grad(), suppress_inner_trace():
                p_c_list = [_project_allowed(p_base + d_cand) for d_cand in cand_displacements]
                p_c_batch = torch.cat(p_c_list, dim=0)
                k_cands = int(p_c_batch.shape[0])
                x0_batch = x0.expand(k_cands, -1)
                r_batch = rollout_trajectories(
                    brick,
                    x0_batch,
                    p_c_batch,
                    dt=dt,
                    num_steps=num_steps,
                    integrator=integrator,
                    record_flows_and_energies=False,
                )
                improved_idx: int | None = None
                for idx_c in range(k_cands):
                    if not bool(r_batch.all_time_viable_mask[idx_c].item()):
                        continue
                    st_c = r_batch.states[idx_c : idx_c + 1]
                    sub_roll = RolloutResult(
                        states=st_c,
                        flows=r_batch.flows[idx_c : idx_c + 1],
                        energies=r_batch.energies[idx_c : idx_c + 1],
                        step_viable_mask=r_batch.step_viable_mask[idx_c : idx_c + 1],
                        all_time_viable_mask=r_batch.all_time_viable_mask[idx_c : idx_c + 1],
                        dt=dt,
                        integrator=r_batch.integrator,
                    )
                    l_c = float(target_loss_fn(sub_roll).detach().item())
                    _, dr_c = _collateral_equilibrium_barrier_penalty(
                        brick,
                        st_c,
                        base_states,
                        collateral_state_indices,
                        collateral_tolerance_sigma=collateral_tolerance_sigma,
                    )
                    is_feas = (
                        dr_c < collateral_tolerance_sigma * 0.96
                        and l_c < base_loss * 0.24
                    )
                    sc_c = (
                        l_c
                        + 2.5 * dr_c
                        + 12.0 * max(0.0, dr_c - collateral_tolerance_sigma * 0.85)
                    )
                    if (is_feas and not best_ref_feasible) or (
                        is_feas == best_ref_feasible and sc_c < best_ref_score and l_c <= max(base_loss * 0.24, final_loss * 1.15)
                    ):
                        best_ref_p = p_c_list[idx_c]
                        best_ref_loss = l_c
                        best_ref_drift = dr_c
                        best_ref_feasible = is_feas
                        best_ref_score = sc_c
                        improved_idx = idx_c
                if improved_idx is not None:
                    best_ref_roll = rollout_trajectories(
                        brick,
                        x0,
                        best_ref_p,
                        dt=dt,
                        num_steps=num_steps,
                        integrator=integrator,
                        record_flows_and_energies=True,
                    )
        p_final = best_ref_p
        restored_roll = best_ref_roll
        final_loss = best_ref_loss
        final_coll_drift = best_ref_drift
    cert = audit_law_brick(
        brick,
        sample_state=restored_roll.states[0],
        sample_params=p_final.expand(num_steps + 1, -1),
    )

    with torch.no_grad():
        x0_f_cld, p_f_cld = _build_minimax_uncertainty_cloud(brick, x0, p_final)
        with suppress_inner_trace():
            roll_f_cld = rollout_trajectories(
                brick,
                x0_f_cld,
                p_f_cld,
                dt=dt,
                num_steps=num_steps,
                integrator=integrator,
                record_flows_and_energies=False,
            )
        cloud_viable_frac = (
            float(roll_f_cld.viable_fraction)
            if bool(torch.isfinite(roll_f_cld.states).all().item())
            else 0.0
        )
        wc_loss = float(target_loss_fn(roll_f_cld).detach().item())

    deltas: dict[str, float] = {}
    for idx, pp in enumerate(brick.param_specs):
        diff = float((p_final[0, idx] - p_base[0, idx]).item())
        if abs(diff) > 1e-12:
            deltas[pp.name] = diff

    max_v = max((r.max_violation for r in cert.results if not r.passed), default=0.0)
    emit_trace_node(
        layer="layer4_engine",
        operation="solve_target_intervention",
        target_name=brick.name,
        local_error_metric=base_loss,
        global_error_metric=final_loss,
        physical_violation_metric=max_v,
        selected_expression=str({k: round(v, 6) for k, v in deltas.items()}),
        operator_spectrum=col_null_spec,
        metrics={
            "baseline_loss": round(base_loss, 6),
            "intervened_loss": round(final_loss, 6),
            "collateral_max_drift": round(final_coll_drift, 6),
            "minimax_cloud_viable_fraction": round(cloud_viable_frac, 4),
            "minimax_worst_case_loss": round(wc_loss, 6),
            "param_deltas": {k: round(v, 6) for k, v in deltas.items()},
            "allowed_parameters": sorted(allowed_set),
            "svd_nullspace_shield_active": p_null_proj is not None,
            "num_collateral_states_shielded": len(collateral_state_indices) if collateral_state_indices else 0,
        },
    )

    return InterventionSolution(
        baseline_params=p_base,
        intervened_params=p_final,
        param_deltas=deltas,
        baseline_loss=base_loss,
        intervened_loss=final_loss,
        restored_rollout=restored_roll,
        audit_certificate=cert,
        collateral_max_drift=final_coll_drift,
        minimax_cloud_viable_fraction=cloud_viable_frac,
        minimax_worst_case_loss=wc_loss,
    )


def solve_optimal_control_waveform(
    brick: LawBrick,
    initial_state: torch.Tensor,
    params: torch.Tensor,
    dt: float,
    num_steps: int,
    target_loss_fn: Callable[[RolloutResult], torch.Tensor],
    actuated_state_indices: Sequence[int] | None = None,
    max_control_amplitude: float = 50.0,
    energy_regularization: float = 1e-4,
    smoothness_regularization: float = 1e-3,
    homeostatic_weight: float = 10.0,
    max_iters: int = 80,
    learning_rate: float = 0.15,
    integrator: str = "rk4",
    collateral_state_indices: Sequence[int] | None = None,
    collateral_weight: float = 5.0,
    collateral_tolerance_sigma: float = 0.25,
) -> OptimalControlResult:
    """
    Synthesize a smooth, energy-regularized external control waveform `u(t_0, ..., t_{T-1})`
    on `actuated_state_indices` that minimizes `target_loss_fn(rollout)`.
    """
    x0 = initial_state[:1].detach().to(torch.float64)
    p_single = params[:1].detach().to(torch.float64)
    device = x0.device

    base_roll = rollout_trajectories(
        brick,
        x0,
        p_single,
        dt=dt,
        num_steps=num_steps,
        integrator=integrator,
        record_flows_and_energies=False,
    )
    base_loss = float(target_loss_fn(base_roll).detach().item())
    base_states = base_roll.states.detach()

    act_idx = (
        tuple(int(i) for i in actuated_state_indices)
        if actuated_state_indices is not None
        else tuple(range(brick.state_dim))
    )
    act_mask = torch.zeros((1, 1, brick.state_dim), dtype=torch.float64, device=device)
    for s_i in act_idx:
        act_mask[0, 0, s_i] = 1.0

    u_opt = torch.zeros((1, num_steps, brick.state_dim), dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([u_opt], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_iters, 1), eta_min=learning_rate * 0.05
    )

    def _project_control(u_raw: torch.Tensor) -> torch.Tensor:
        return torch.clamp(u_raw * act_mask, min=-max_control_amplitude, max=max_control_amplitude)

    for _ in range(max_iters):
        optimizer.zero_grad()
        u_curr = _project_control(u_opt)
        roll = rollout_trajectories(
            brick,
            x0,
            p_single,
            dt=dt,
            num_steps=num_steps,
            integrator=integrator,
            port_inputs={"external_flow": u_curr},
            record_flows_and_energies=False,
        )
        t_loss = target_loss_fn(roll)
        b_pen = _homeostatic_barrier_penalty(brick, roll.states)
        c_pen, _ = _collateral_equilibrium_barrier_penalty(
            brick,
            roll.states,
            base_states,
            collateral_state_indices,
            collateral_tolerance_sigma=collateral_tolerance_sigma,
        )
        e_pen = torch.mean(u_curr**2)
        if num_steps > 1:
            s_pen = torch.mean((u_curr[:, 1:, :] - u_curr[:, :-1, :]) ** 2)
        else:
            s_pen = torch.zeros((), dtype=torch.float64, device=device)

        total_obj = (
            t_loss
            + homeostatic_weight * b_pen
            + collateral_weight * c_pen
            + energy_regularization * e_pen
            + smoothness_regularization * s_pen
        )
        total_obj.backward()
        if u_opt.grad is not None:
            torch.nn.utils.clip_grad_norm_([u_opt], max_norm=100.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            u_opt.copy_(_project_control(u_opt))

    record_simulation_activity(adjoint_gradient_passes=int(max_iters))
    u_final = _project_control(u_opt.detach())
    controlled_roll = rollout_trajectories(
        brick,
        x0,
        p_single,
        dt=dt,
        num_steps=num_steps,
        integrator=integrator,
        port_inputs={"external_flow": u_final},
        record_flows_and_energies=True,
    )
    final_loss = float(target_loss_fn(controlled_roll).detach().item())
    _, final_coll_drift = _collateral_equilibrium_barrier_penalty(
        brick,
        controlled_roll.states,
        base_states,
        collateral_state_indices,
        collateral_tolerance_sigma=collateral_tolerance_sigma,
    )
    ctrl_energy = float(torch.mean(u_final**2).item())
    viable_ctrl = bool(controlled_roll.all_time_viable_mask[0].item())

    emit_trace_node(
        layer="layer4_engine",
        operation="solve_optimal_control_waveform",
        target_name=brick.name,
        local_error_metric=base_loss,
        global_error_metric=final_loss,
        physical_violation_metric=0.0 if viable_ctrl else 1.0,
        selected_expression=f"loss={base_loss:.4f}->{final_loss:.4f}, energy={ctrl_energy:.4f}",
        metrics={
            "baseline_loss": round(base_loss, 6),
            "controlled_loss": round(final_loss, 6),
            "control_energy": round(ctrl_energy, 6),
            "collateral_max_drift": round(final_coll_drift, 6),
            "all_time_viable": viable_ctrl,
        },
        skip_if_inner_loop=True,
    )

    return OptimalControlResult(
        control_waveform=u_final,
        controlled_rollout=controlled_roll,
        baseline_loss=base_loss,
        controlled_loss=final_loss,
        control_energy=ctrl_energy,
        all_time_viable=viable_ctrl,
        collateral_max_drift=final_coll_drift,
    )


__all__ = [
    "InterventionSolution",
    "OptimalControlResult",
    "_resolve_finite_state_bounds",
    "_homeostatic_barrier_penalty",
    "_collateral_equilibrium_barrier_penalty",
    "_compute_collateral_svd_nullspace_projector",
    "_build_minimax_uncertainty_cloud",
    "solve_target_intervention",
    "solve_optimal_control_waveform",
]
