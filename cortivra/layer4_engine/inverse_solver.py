"""
Inverse parameter calibration, residual isolation, two-point boundary shooting,
Fisher information / parameter leverage analysis, and counterfactual target intervention
(`cortivra.layer4_engine.inverse_solver`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick
from cortivra.layer1_axioms.shields import (
    AuditCertificate,
    LatentResidualAnalysis,
    analyze_unexplained_residual,
    audit_law_brick,
)
from cortivra.layer4_engine.elimination_funnel import (
    EliminationFunnelReport,
    _project_tunable_params,
    run_elimination_funnel,
)
from cortivra.layer4_engine.rollout_arena import (
    RolloutResult,
    estimate_trajectory_derivatives,
    rollout_trajectories,
)
from cortivra.observability.flight_recorder import (
    OperatorSpectrumSummary,
    emit_trace_node,
    summarize_singular_spectrum,
)


@dataclass(frozen=True)
class CalibrationAndResidualReport:
    """Inverse parameter calibration result and Layer 1 unmodeled residual analysis."""

    best_params: torch.Tensor
    calibrated_param_map: dict[str, float]
    funnel_report: EliminationFunnelReport
    residual_analysis: LatentResidualAnalysis

    @property
    def latent_mechanism_required(self) -> bool:
        """True if known laws cannot explain the trajectory even after parameter calibration."""
        return self.residual_analysis.latent_term_justified


@dataclass(frozen=True)
class BoundaryShootingResult:
    """Two-point boundary-value shooting solution bridging `t_0` to `t_T`."""

    solved_initial_state: torch.Tensor
    rollout: RolloutResult
    terminal_error_norm: float
    converged: bool
    all_time_viable: bool


@dataclass(frozen=True)
class SensitivityAndFisherReport:
    """Parameter susceptibility, Fisher Information Matrix `F = J_theta^T J_theta`, and leverage ranking."""

    param_names: tuple[str, ...]
    objective_gradients: torch.Tensor
    log_sensitivities: torch.Tensor
    homeostatic_risk_scores: torch.Tensor
    restorative_leverage_scores: torch.Tensor
    ranked_param_names: tuple[str, ...]
    fisher_matrix: torch.Tensor
    fisher_eigenvalues: torch.Tensor
    stiffest_eigenvector: torch.Tensor


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


@dataclass(frozen=True)
class OptimalControlResult:
    """Time-varying external control waveform `u*(t)` (`[1, num_steps, StateDim]`) solution."""

    control_waveform: torch.Tensor
    controlled_rollout: RolloutResult
    baseline_loss: float
    controlled_loss: float
    control_energy: float
    all_time_viable: bool


def calibrate_and_audit_residual(
    brick: LawBrick,
    observed_states: torch.Tensor,
    dt: float,
    observed_dx_dt: torch.Tensor | None = None,
    observed_state_indices: Sequence[int] | None = None,
    initial_full_state: torch.Tensor | None = None,
    noise_floor_std: float = 1e-3,
    min_snr_threshold: float = 3.0,
    min_coherence_r2: float = 0.50,
    num_initial_samples: int = 256,
    num_collocation_survivors: int = 32,
    num_gradient_starts: int = 6,
    gradient_steps: int = 90,
    learning_rate: float = 0.08,
    integrator: str = "rk4",
    custom_bounds: Mapping[str, tuple[float, float]] | None = None,
    seed: int = 42,
) -> CalibrationAndResidualReport:
    """
    Calibrate tunable parameters of `brick` via `run_elimination_funnel` and evaluate
    `analyze_unexplained_residual(..., parameter_fit_exhausted=True)` on the calibrated residuals.
    """
    funnel = run_elimination_funnel(
        brick=brick,
        observed_states=observed_states,
        dt=dt,
        observed_dx_dt=observed_dx_dt,
        observed_state_indices=observed_state_indices,
        initial_full_state=initial_full_state,
        num_initial_samples=num_initial_samples,
        num_collocation_survivors=num_collocation_survivors,
        num_gradient_starts=num_gradient_starts,
        gradient_steps=gradient_steps,
        learning_rate=learning_rate,
        integrator=integrator,
        custom_bounds=custom_bounds,
        seed=seed,
    )

    obs_3d = observed_states.unsqueeze(0).to(torch.float64) if observed_states.ndim == 2 else observed_states.to(torch.float64)
    n_exp, t_steps, d_obs = obs_3d.shape
    total_pts = n_exp * t_steps

    full_observed = (
        observed_state_indices is None
        or len(observed_state_indices) == brick.state_dim
    )
    if full_observed:
        eval_states = obs_3d.reshape(total_pts, brick.state_dim)
        if observed_dx_dt is not None:
            dx_3d = (
                observed_dx_dt.unsqueeze(0).to(torch.float64)
                if observed_dx_dt.ndim == 2
                else observed_dx_dt.to(torch.float64)
            )
            eval_dx_dt = dx_3d.reshape(total_pts, brick.state_dim)
        else:
            eval_dx_dt = estimate_trajectory_derivatives(obs_3d, dt=dt).reshape(total_pts, brick.state_dim)
    else:
        roll_states = funnel.best_rollout.states.detach()
        eval_states = roll_states.reshape(total_pts, brick.state_dim)
        eval_dx_dt = estimate_trajectory_derivatives(roll_states, dt=dt).reshape(total_pts, brick.state_dim)
        if observed_state_indices is not None:
            obs_deriv = estimate_trajectory_derivatives(obs_3d, dt=dt).reshape(total_pts, d_obs)
            eval_dx_dt = eval_dx_dt.clone()
            for col_i, s_idx in enumerate(observed_state_indices):
                eval_dx_dt[:, s_idx] = obs_deriv[:, col_i]

    params_expanded = funnel.best_params.expand(total_pts, -1)
    residual_report = analyze_unexplained_residual(
        brick=brick,
        state=eval_states,
        params=params_expanded,
        observed_dx_dt=eval_dx_dt,
        noise_floor_std=noise_floor_std,
        parameter_fit_exhausted=True,
        min_snr_threshold=min_snr_threshold,
        min_coherence_r2=min_coherence_r2,
    )

    param_map = {
        pp.name: float(funnel.best_params[0, idx].item())
        for idx, pp in enumerate(brick.param_specs)
    }

    return CalibrationAndResidualReport(
        best_params=funnel.best_params,
        calibrated_param_map=param_map,
        funnel_report=funnel,
        residual_analysis=residual_report,
    )


def solve_boundary_value_shooting(
    brick: LawBrick,
    known_initial_indices: Sequence[int],
    known_initial_values: torch.Tensor,
    target_final_indices: Sequence[int],
    target_final_values: torch.Tensor,
    params: torch.Tensor,
    dt: float,
    num_steps: int,
    integrator: str = "rk4",
    max_iters: int = 80,
    learning_rate: float = 0.1,
    tol: float = 1e-3,
) -> BoundaryShootingResult:
    """
    Solve for unknown initial state components at `t_0` so that the trajectory starting
    with `x[known_initial_indices](t_0) = known_initial_values` arrives at
    `x[target_final_indices](t_T) = target_final_values` after `num_steps` steps of size `dt`.
    """
    known_idx = tuple(int(i) for i in known_initial_indices)
    target_idx = tuple(int(i) for i in target_final_indices)
    free_idx = tuple(i for i in range(brick.state_dim) if i not in known_idx)
    if not free_idx:
        raise ValueError("At least one initial state component must be free to shoot.")

    p_single = params[:1].to(torch.float64)
    device = p_single.device
    known_vals = known_initial_values.reshape(-1).to(dtype=torch.float64, device=device)
    target_vals = target_final_values.reshape(-1).to(dtype=torch.float64, device=device)

    base_x0 = brick.default_state(batch_size=1, dtype=torch.float64, device=device)
    for col_i, s_i in enumerate(known_idx):
        base_x0[0, s_i] = known_vals[col_i]

    free_guess = torch.tensor(
        [float(base_x0[0, s_i].item()) for s_i in free_idx],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )

    def _assemble_x0(free_vec: torch.Tensor) -> torch.Tensor:
        cols: list[torch.Tensor] = []
        free_pos = 0
        known_pos = 0
        for s_i in range(brick.state_dim):
            if s_i in known_idx:
                cols.append(known_vals[known_pos])
                known_pos += 1
            else:
                cols.append(free_vec[free_pos])
                free_pos += 1
        return torch.stack(cols, dim=-1).unsqueeze(0)

    optimizer = torch.optim.Adam([free_guess], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)

    # Phase 1: Coarse Adam warm-start (up to 15 steps) to enter basin of attraction
    warm_steps = min(max_iters, 15)
    for _ in range(warm_steps):
        optimizer.zero_grad()
        x0_curr = _assemble_x0(free_guess)
        roll = rollout_trajectories(
            brick, x0_curr, p_single, dt=dt, num_steps=num_steps, integrator=integrator,
            record_flows_and_energies=False,
        )
        x_end = roll.states[0, -1, target_idx]
        err_sq = torch.sum((x_end - target_vals) ** 2)
        if float(torch.sqrt(err_sq).detach().item()) <= tol:
            break
        err_sq.backward()
        if free_guess.grad is not None:
            torch.nn.utils.clip_grad_norm_([free_guess], max_norm=50.0)
        optimizer.step()
        scheduler.step(float(err_sq.detach().item()))

    # Phase 2: Exact Autograd Gauss-Newton / Levenberg-Marquardt boundary shooting refinement
    free_curr = free_guess.detach().clone()
    n_free = free_curr.numel()
    n_targ = len(target_idx)
    eye_free = torch.eye(n_free, dtype=torch.float64, device=device)
    lm_lambda = 1e-4
    shoot_spec: OperatorSpectrumSummary | None = None

    for _ in range(max(max_iters - warm_steps, 10)):
        f_req = free_curr.clone().requires_grad_(True)
        x0_curr = _assemble_x0(f_req)
        roll = rollout_trajectories(
            brick, x0_curr, p_single, dt=dt, num_steps=num_steps, integrator=integrator,
            record_flows_and_energies=False,
        )
        res_vec = roll.states[0, -1, target_idx] - target_vals  # [n_targ]
        curr_norm = float(torch.linalg.norm(res_vec).detach().item())
        if curr_norm <= tol:
            free_curr = f_req.detach()
            break

        jac_rows: list[torch.Tensor] = []
        for k_i in range(n_targ):
            g_k = torch.autograd.grad(
                res_vec[k_i],
                f_req,
                retain_graph=(k_i < n_targ - 1),
            )[0]
            jac_rows.append(g_k)
        j_shoot = torch.stack(jac_rows, dim=0)  # [n_targ, n_free]
        shoot_spec = summarize_singular_spectrum(
            j_shoot.detach(),
            operator_name=f"BoundaryShootingJacobian({brick.name})",
        )

        lhs = (j_shoot.T @ j_shoot) + lm_lambda * eye_free
        rhs = -(j_shoot.T @ res_vec.detach().unsqueeze(-1))
        delta_free = torch.linalg.lstsq(lhs, rhs, driver="gelsd").solution.squeeze(-1)

        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.1):
            cand_free = free_curr + alpha * delta_free
            with torch.no_grad():
                x0_cand = brick.clamp_to_hard_bounds(_assemble_x0(cand_free))
                roll_cand = rollout_trajectories(
                    brick, x0_cand, p_single, dt=dt, num_steps=num_steps, integrator=integrator,
                    record_flows_and_energies=False,
                )
                cand_norm = float(
                    torch.linalg.norm(roll_cand.states[0, -1, target_idx] - target_vals).item()
                )
            if cand_norm < curr_norm:
                free_curr = cand_free
                lm_lambda = max(lm_lambda * 0.3, 1e-9)
                accepted = True
                break
        if not accepted:
            lm_lambda = min(lm_lambda * 5.0, 1e3)

    with torch.no_grad():
        x0_solved = brick.clamp_to_hard_bounds(_assemble_x0(free_curr))
    final_roll = rollout_trajectories(
        brick, x0_solved, p_single, dt=dt, num_steps=num_steps, integrator=integrator,
        record_flows_and_energies=True,
    )
    term_err = float(torch.linalg.norm(final_roll.states[0, -1, target_idx] - target_vals).item())
    conv_ok = bool(term_err <= max(tol * 5.0, 1e-2))
    viable_ok = bool(final_roll.all_time_viable_mask[0].item())

    emit_trace_node(
        layer="layer4_engine",
        operation="solve_boundary_value_shooting",
        target_name=brick.name,
        status="completed" if conv_ok else "failed",
        global_error_metric=term_err,
        physical_violation_metric=0.0 if viable_ok else 1.0,
        selected_expression=f"x(0)={[round(float(v), 5) for v in x0_solved[0].tolist()]}",
        operator_spectrum=shoot_spec,
        metrics={
            "terminal_error_norm": round(term_err, 8),
            "converged": conv_ok,
            "all_time_viable": viable_ok,
        },
        skip_if_inner_loop=True,
    )

    return BoundaryShootingResult(
        solved_initial_state=x0_solved,
        rollout=final_roll,
        terminal_error_norm=term_err,
        converged=conv_ok,
        all_time_viable=viable_ok,
    )


def _homeostatic_barrier_penalty(brick: LawBrick, states: torch.Tensor) -> torch.Tensor:
    """
    Smooth differentiable barrier measuring excursion toward or beyond `[homeostatic_min, homeostatic_max]`.
    Returns zero inside the central 80% safe zone and grows quadratically near/past boundaries.
    """
    total_pen = torch.zeros((), dtype=states.dtype, device=states.device)
    for idx, sp in enumerate(brick.state_specs):
        s_col = states[..., idx]
        span = max(float(sp.homeostatic_max - sp.homeostatic_min), 1e-6)
        margin = 0.05 * span
        low_excess = torch.relu((sp.homeostatic_min + margin) - s_col) / span
        high_excess = torch.relu(s_col - (sp.homeostatic_max - margin)) / span
        total_pen = total_pen + torch.mean(low_excess**2 + high_excess**2)
    return total_pen


def compute_parameter_leverage_and_fisher(
    brick: LawBrick,
    initial_state: torch.Tensor,
    params: torch.Tensor,
    dt: float,
    num_steps: int,
    objective_fn: Callable[[RolloutResult], torch.Tensor],
    integrator: str = "rk4",
) -> SensitivityAndFisherReport:
    """
    Compute the exact parameter susceptibility gradient `dJ/d(theta)`, logarithmic sensitivity,
    homeostatic boundary risk, Restorative Leverage ranking, and the `[ParamDim, ParamDim]`
    Fisher Information Matrix `F = J_theta^T J_theta` with its stiff/sloppy eigen-spectrum.
    """
    x0 = initial_state[:1].detach().to(torch.float64)
    p_req = params[:1].detach().clone().to(torch.float64).requires_grad_(True)
    param_dim = brick.param_dim
    param_names = tuple(pp.name for pp in brick.param_specs)

    roll = rollout_trajectories(
        brick, x0, p_req, dt=dt, num_steps=num_steps, integrator=integrator,
        record_flows_and_energies=True,
    )
    obj_val = objective_fn(roll)
    obj_grad = torch.autograd.grad(obj_val, p_req, retain_graph=True)[0].squeeze(0)

    barrier_val = _homeostatic_barrier_penalty(brick, roll.states)
    barrier_grad = torch.autograd.grad(barrier_val, p_req, retain_graph=True, allow_unused=True)[0]
    if barrier_grad is None:
        risk_scores = torch.zeros_like(obj_grad)
    else:
        risk_scores = torch.abs(barrier_grad.squeeze(0))

    # Compute Fisher Information Matrix F = J_theta^T J_theta across trajectory states
    flat_states = roll.states.reshape(-1)
    # Subsample up to 24 state coordinates along trajectory for fast exact Jacobian construction
    stride = max(int(flat_states.numel() // 24), 1)
    sampled_outputs = flat_states[::stride]
    jac_rows: list[torch.Tensor] = []
    for k_i in range(sampled_outputs.numel()):
        g_k = torch.autograd.grad(
            sampled_outputs[k_i],
            p_req,
            retain_graph=(k_i < sampled_outputs.numel() - 1),
            allow_unused=True,
        )[0]
        jac_rows.append(g_k.squeeze(0) if g_k is not None else torch.zeros_like(obj_grad))

    j_mat = torch.stack(jac_rows, dim=0)  # [M, ParamDim]
    fisher = (j_mat.T @ j_mat) / max(float(j_mat.shape[0]), 1.0)
    fisher_sym = 0.5 * (fisher + fisher.T)
    eigvals, eigvecs = torch.linalg.eigh(fisher_sym)
    # Sort descending (stiffest -> sloppiest)
    desc_idx = torch.argsort(eigvals, descending=True)
    eigvals_desc = torch.clamp(eigvals[desc_idx], min=0.0)
    stiffest_vec = eigvecs[:, desc_idx[0]] if param_dim > 0 else torch.zeros((0,), dtype=torch.float64)

    p_abs = torch.clamp(torch.abs(p_req.detach().squeeze(0)), min=1.0)
    log_sens = p_abs * torch.abs(obj_grad.detach())

    tunable_mask = torch.tensor(
        [1.0 if pp.tunable else 0.0 for pp in brick.param_specs],
        dtype=torch.float64,
        device=x0.device,
    )
    leverage = (log_sens * tunable_mask) / (1.0 + 10.0 * risk_scores.detach())
    rank_order = torch.argsort(leverage, descending=True).tolist()
    ranked_names = tuple(param_names[idx] for idx in rank_order)

    f_spec = summarize_singular_spectrum(
        eigvals_desc.detach(),
        operator_name=f"FisherMatrix({brick.name})",
        matrix_shape=(param_dim, param_dim),
    )
    emit_trace_node(
        layer="layer4_engine",
        operation="compute_parameter_leverage_and_fisher",
        target_name=brick.name,
        selected_expression=", ".join(ranked_names),
        operator_spectrum=f_spec,
        metrics={
            "ranked_param_names": list(ranked_names),
            "fisher_eigenvalues": [round(float(v), 6) for v in eigvals_desc.tolist()],
            "restorative_leverage_scores": {
                param_names[i]: round(float(leverage[i].item()), 6) for i in range(param_dim)
            },
        },
    )

    return SensitivityAndFisherReport(
        param_names=param_names,
        objective_gradients=obj_grad.detach(),
        log_sensitivities=log_sens,
        homeostatic_risk_scores=risk_scores.detach(),
        restorative_leverage_scores=leverage,
        ranked_param_names=ranked_names,
        fisher_matrix=fisher_sym.detach(),
        fisher_eigenvalues=eigvals_desc.detach(),
        stiffest_eigenvector=stiffest_vec.detach(),
    )


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
) -> InterventionSolution:
    """
    Solve for a parameter shift `Delta_theta*` that minimizes `target_loss_fn(rollout)`
    while keeping trajectories within `[homeostatic_min, homeostatic_max]` and passing `audit_law_brick`.
    """
    x0 = initial_state[:1].detach().to(torch.float64)
    p_base = baseline_params[:1].detach().clone().to(torch.float64)

    base_roll = rollout_trajectories(
        brick, x0, p_base, dt=dt, num_steps=num_steps, integrator=integrator,
        record_flows_and_energies=True,
    )
    base_loss = float(target_loss_fn(base_roll).detach().item())

    allowed_set = (
        set(allowed_param_names)
        if allowed_param_names is not None
        else {pp.name for pp in brick.param_specs if pp.tunable}
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

    for _ in range(max_iters):
        optimizer.zero_grad()
        p_curr = _project_allowed(p_opt)
        roll = rollout_trajectories(
            brick, x0, p_curr, dt=dt, num_steps=num_steps, integrator=integrator,
            record_flows_and_energies=False,
        )
        t_loss = target_loss_fn(roll)
        b_pen = _homeostatic_barrier_penalty(brick, roll.states)
        reg_pen = torch.mean(((p_curr - p_base) / torch.clamp(torch.abs(p_base), min=1.0)) ** 2)
        total_obj = t_loss + homeostatic_weight * b_pen + l2_regularization * reg_pen
        total_obj.backward()
        if p_opt.grad is not None:
            torch.nn.utils.clip_grad_norm_([p_opt], max_norm=50.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            p_opt.copy_(_project_allowed(p_opt))

    p_final = _project_allowed(p_opt.detach())
    restored_roll = rollout_trajectories(
        brick, x0, p_final, dt=dt, num_steps=num_steps, integrator=integrator,
        record_flows_and_energies=True,
    )
    final_loss = float(target_loss_fn(restored_roll).detach().item())
    cert = audit_law_brick(
        brick,
        sample_state=restored_roll.states[0],
        sample_params=p_final.expand(num_steps + 1, -1),
    )

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
        metrics={
            "baseline_loss": round(base_loss, 6),
            "intervened_loss": round(final_loss, 6),
            "param_deltas": {k: round(v, 6) for k, v in deltas.items()},
            "allowed_parameters": sorted(allowed_set),
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
) -> OptimalControlResult:
    """
    Synthesize a smooth, energy-regularized external control waveform `u(t_0, ..., t_{T-1})`
    on `actuated_state_indices` that minimizes `target_loss_fn(rollout)` within homeostatic bounds.
    """
    x0 = initial_state[:1].detach().to(torch.float64)
    p_single = params[:1].detach().to(torch.float64)
    device = x0.device

    base_roll = rollout_trajectories(
        brick, x0, p_single, dt=dt, num_steps=num_steps, integrator=integrator,
        record_flows_and_energies=False,
    )
    base_loss = float(target_loss_fn(base_roll).detach().item())

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
        e_pen = torch.mean(u_curr**2)
        if num_steps > 1:
            s_pen = torch.mean((u_curr[:, 1:, :] - u_curr[:, :-1, :]) ** 2)
        else:
            s_pen = torch.zeros((), dtype=torch.float64, device=device)

        total_obj = (
            t_loss
            + homeostatic_weight * b_pen
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
    )
