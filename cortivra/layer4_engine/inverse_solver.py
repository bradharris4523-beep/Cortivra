"""
Inverse parameter calibration, residual isolation, two-point boundary shooting,
Fisher information / parameter leverage analysis, and counterfactual target intervention
(`cortivra.layer4_engine.inverse_solver`).

Counterfactual target intervention (`solve_target_intervention`), SVD collateral nullspace
projection, and Pontryagin optimal control (`solve_optimal_control_waveform`) are implemented
in `cortivra.layer4_engine.intervention_and_control` and re-exported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick
from cortivra.layer1_axioms.shields import (
    LatentResidualAnalysis,
    analyze_unexplained_residual,
)
from cortivra.layer4_engine.elimination_funnel import (
    EliminationFunnelReport,
    run_elimination_funnel,
)
from cortivra.layer4_engine.intervention_and_control import (
    InterventionSolution,
    OptimalControlResult,
    _build_minimax_uncertainty_cloud,
    _collateral_equilibrium_barrier_penalty,
    _compute_collateral_svd_nullspace_projector,
    _homeostatic_barrier_penalty,
    _resolve_finite_state_bounds,
    solve_optimal_control_waveform,
    solve_target_intervention,
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
    suppress_inner_trace,
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

    obs_3d = (
        observed_states.unsqueeze(0).to(torch.float64)
        if observed_states.ndim == 2
        else observed_states.to(torch.float64)
    )
    n_exp, t_steps, d_obs = obs_3d.shape
    total_pts = n_exp * t_steps

    full_observed = observed_state_indices is None or len(observed_state_indices) == brick.state_dim
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

    warm_steps = min(max_iters, 15)
    for _ in range(warm_steps):
        optimizer.zero_grad()
        x0_curr = _assemble_x0(free_guess)
        roll = rollout_trajectories(
            brick,
            x0_curr,
            p_single,
            dt=dt,
            num_steps=num_steps,
            integrator=integrator,
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
            brick,
            x0_curr,
            p_single,
            dt=dt,
            num_steps=num_steps,
            integrator=integrator,
            record_flows_and_energies=False,
        )
        res_vec = roll.states[0, -1, target_idx] - target_vals
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
        j_shoot = torch.stack(jac_rows, dim=0)
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
                    brick,
                    x0_cand,
                    p_single,
                    dt=dt,
                    num_steps=num_steps,
                    integrator=integrator,
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
        brick,
        x0_solved,
        p_single,
        dt=dt,
        num_steps=num_steps,
        integrator=integrator,
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
    p_base = params[:1].detach().clone().to(torch.float64)
    param_dim = brick.param_dim
    param_names = tuple(pp.name for pp in brick.param_specs)
    device = x0.device

    if param_dim == 0:
        empty_v = torch.zeros((0,), dtype=torch.float64, device=device)
        empty_m = torch.zeros((0, 0), dtype=torch.float64, device=device)
        f_spec = summarize_singular_spectrum(
            empty_v,
            operator_name=f"FisherMatrix({brick.name})",
            matrix_shape=(0, 0),
        )
        return SensitivityAndFisherReport(
            param_names=param_names,
            objective_gradients=empty_v,
            log_sensitivities=empty_v,
            homeostatic_risk_scores=empty_v,
            restorative_leverage_scores=empty_v,
            ranked_param_names=param_names,
            fisher_matrix=empty_m,
            fisher_eigenvalues=empty_v,
            stiffest_eigenvector=empty_v,
        )

    p_batch = p_base.expand(1 + param_dim, -1).clone()
    eps_list: list[float] = []
    for idx_p, pp in enumerate(brick.param_specs):
        p_val = float(p_base[0, idx_p].item())
        p_span = max(float(pp.max_value - pp.min_value), abs(p_val), 1.0)
        eps_p = max(abs(p_val) * 1e-4, p_span * 1e-5, 1e-6)
        if p_val + eps_p > float(pp.max_value) and p_val - eps_p >= float(pp.min_value):
            eps_p = -eps_p
        p_batch[1 + idx_p, idx_p] = p_val + eps_p
        eps_list.append(eps_p)

    with torch.no_grad(), suppress_inner_trace():
        r_batch = rollout_trajectories(
            brick,
            x0.expand(1 + param_dim, -1),
            p_batch,
            dt=dt,
            num_steps=num_steps,
            integrator=integrator,
            record_flows_and_energies=False,
        )
        st_all = r_batch.states
        obj_vals: list[float] = []
        bar_vals: list[float] = []
        for b_i in range(1 + param_dim):
            st_i = st_all[b_i : b_i + 1]
            sub_r = RolloutResult(
                states=st_i,
                flows=r_batch.flows[b_i : b_i + 1],
                energies=r_batch.energies[b_i : b_i + 1],
                step_viable_mask=r_batch.step_viable_mask[b_i : b_i + 1],
                all_time_viable_mask=r_batch.all_time_viable_mask[b_i : b_i + 1],
                dt=dt,
                integrator=r_batch.integrator,
            )
            obj_vals.append(float(objective_fn(sub_r).detach().item()))
            bar_vals.append(float(_homeostatic_barrier_penalty(brick, st_i).detach().item()))

        obj_grad = torch.zeros((param_dim,), dtype=torch.float64, device=device)
        risk_scores = torch.zeros((param_dim,), dtype=torch.float64, device=device)
        obj_0 = obj_vals[0]
        bar_0 = bar_vals[0]
        for idx_p in range(param_dim):
            inv_e = 1.0 / eps_list[idx_p]
            obj_grad[idx_p] = (obj_vals[1 + idx_p] - obj_0) * inv_e
            risk_scores[idx_p] = abs((bar_vals[1 + idx_p] - bar_0) * inv_e)

        obj_grad = torch.nan_to_num(obj_grad, nan=0.0, posinf=1e6, neginf=-1e6)
        risk_scores = torch.nan_to_num(risk_scores, nan=0.0, posinf=1e6, neginf=0.0)

        flat_0 = st_all[0].reshape(-1)
        stride = max(int(flat_0.numel() // 24), 1)
        base_out = flat_0[::stride]
        jac_cols: list[torch.Tensor] = []
        for idx_p in range(param_dim):
            pert_out = st_all[1 + idx_p].reshape(-1)[::stride]
            col_p = (pert_out - base_out) / eps_list[idx_p]
            jac_cols.append(torch.nan_to_num(col_p, nan=0.0, posinf=1e6, neginf=-1e6))

        j_mat = torch.stack(jac_cols, dim=-1)
        fisher = (j_mat.T @ j_mat) / max(float(j_mat.shape[0]), 1.0)
        fisher_sym = 0.5 * (fisher + fisher.T)
        eigvals, eigvecs = torch.linalg.eigh(fisher_sym)
        desc_idx = torch.argsort(eigvals, descending=True)
        eigvals_desc = torch.clamp(eigvals[desc_idx], min=0.0)
        stiffest_vec = eigvecs[:, desc_idx[0]] if param_dim > 0 else torch.zeros((0,), dtype=torch.float64)

        p_abs = torch.clamp(torch.abs(p_base.squeeze(0)), min=1.0)
        log_sens = p_abs * torch.abs(obj_grad)

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


__all__ = [
    "CalibrationAndResidualReport",
    "BoundaryShootingResult",
    "SensitivityAndFisherReport",
    "InterventionSolution",
    "OptimalControlResult",
    "calibrate_and_audit_residual",
    "solve_boundary_value_shooting",
    "_resolve_finite_state_bounds",
    "_homeostatic_barrier_penalty",
    "_collateral_equilibrium_barrier_penalty",
    "compute_parameter_leverage_and_fisher",
    "_compute_collateral_svd_nullspace_projector",
    "_build_minimax_uncertainty_cloud",
    "solve_target_intervention",
    "solve_optimal_control_waveform",
]
