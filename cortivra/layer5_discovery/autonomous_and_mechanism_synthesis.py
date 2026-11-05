"""
Autonomous zero-prior coupled ODE discovery, Helmholtz-Hodge vector-field
decomposition, and inverse symbolic feedback mechanism synthesis
(`cortivra.layer5_discovery.autonomous_and_mechanism_synthesis`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick, ParamSpec, StateVarSpec
from cortivra.layer1_axioms.shields import AuditCertificate, audit_law_brick
from cortivra.layer1_axioms.units import DIMENSIONLESS, Unit, s
from cortivra.layer3_compiler.ast_nodes import ConstNode, ExprNode
from cortivra.layer3_compiler.symbolic_compiler import (
    CompiledLawBrick,
    SymbolicSystemSpec,
    compile_symbolic_system,
)
from cortivra.layer4_engine.elimination_funnel import run_elimination_funnel
from cortivra.layer4_engine.inverse_solver import (
    OptimalControlResult,
    solve_optimal_control_waveform,
    solve_target_intervention,
)
from cortivra.layer4_engine.rollout_arena import (
    estimate_trajectory_derivatives,
    estimate_trajectory_noise_std,
    rollout_trajectories,
    savitzky_golay_denoise_and_differentiate,
)
from cortivra.layer5_discovery.basis_library import generate_symbolic_basis_library
from cortivra.layer5_discovery.latent_discovery import (
    ArchetypeFingerprint,
    classify_mathematical_archetype,
    compile_symbolic_candidate,
)
from cortivra.layer5_discovery.sparse_regression import (
    InvariantDiscoveryReport,
    SymbolicCandidate,
    discover_conservation_invariants,
    discover_fractional_power_and_log_candidates,
    discover_rational_pade_candidates,
    solve_sparse_pareto_front,
)
from cortivra.observability.flight_recorder import emit_trace_node


@dataclass(frozen=True)
class HelmholtzHodgeDecomposition:
    """
    Continuous-time Helmholtz-Hodge & Metriplectic Jacobian decomposition of a discovered
    vector field `F(x)` along an observed trajectory `x(t)` (`[N, D]`):
        `J_F(x) = dF/dx = J_skew(x) - R_sym(x)`
    """

    mean_skew_jacobian: tuple[tuple[float, ...], ...]
    mean_dissipative_symmetric_matrix: tuple[tuple[float, ...], ...]
    conservative_energy_fraction: float
    dissipative_contraction_fraction: float
    mean_divergence_rate: float
    is_volume_contracting: bool
    dominant_regime: str


def decompose_vector_field_helmholtz_hodge(
    brick: LawBrick,
    states: torch.Tensor,
    params: torch.Tensor,
) -> HelmholtzHodgeDecomposition:
    """
    Compute the trajectory-averaged Helmholtz-Hodge Jacobian decomposition of `brick.flow(x, p)`
    into skew-symmetric conservative circulation `J_skew` and symmetric dissipative contraction `R_sym`.
    """
    x_eval = states.detach().to(torch.float64)
    if x_eval.ndim == 3:
        x_eval = x_eval.reshape(-1, x_eval.shape[-1])
    if x_eval.shape[0] > 32:
        idx_sub = torch.linspace(0, x_eval.shape[0] - 1, 32, dtype=torch.long, device=x_eval.device)
        x_eval = x_eval[idx_sub]

    p_eval = params.detach().to(torch.float64)
    if p_eval.ndim == 1:
        p_eval = p_eval.unsqueeze(0)
    p_row = p_eval[:1]

    n_pts, d = x_eval.shape
    eps = 1e-6
    j_accum = torch.zeros((d, d), dtype=torch.float64, device=x_eval.device)

    with torch.no_grad():
        p_batch = p_row.expand(n_pts, -1)
        for j in range(d):
            dx_vec = torch.zeros_like(x_eval)
            step_j = torch.clamp(eps * torch.abs(x_eval[:, j]), min=eps)
            dx_vec[:, j] = step_j
            f_plus = brick.flow(x_eval + dx_vec, p_batch)
            f_minus = brick.flow(x_eval - dx_vec, p_batch)
            df_dxj = (f_plus - f_minus) / (2.0 * step_j.unsqueeze(-1))
            j_accum[:, j] = torch.mean(df_dxj, dim=0)

        j_skew = 0.5 * (j_accum - j_accum.T)
        r_sym = -0.5 * (j_accum + j_accum.T)
        skew_norm_sq = float(torch.sum(j_skew * j_skew).item())
        sym_norm_sq = float(torch.sum(r_sym * r_sym).item())
        tot_sq = max(skew_norm_sq + sym_norm_sq, 1e-18)
        cons_frac = skew_norm_sq / tot_sq
        diss_frac = sym_norm_sq / tot_sq
        mean_div = float(torch.trace(j_accum).item())

    if cons_frac >= 0.65:
        regime = "conservative_symplectic_dominant"
    elif diss_frac >= 0.65 and mean_div <= 1e-5:
        regime = "dissipative_gradient_contraction"
    else:
        regime = "metriplectic_mixed_conservative_dissipative"

    return HelmholtzHodgeDecomposition(
        mean_skew_jacobian=tuple(
            tuple(round(float(j_skew[r, c].item()), 6) for c in range(d)) for r in range(d)
        ),
        mean_dissipative_symmetric_matrix=tuple(
            tuple(round(float(r_sym[r, c].item()), 6) for c in range(d)) for r in range(d)
        ),
        conservative_energy_fraction=round(cons_frac, 6),
        dissipative_contraction_fraction=round(diss_frac, 6),
        mean_divergence_rate=round(mean_div, 6),
        is_volume_contracting=bool(mean_div < -1e-6),
        dominant_regime=regime,
    )


@dataclass(frozen=True)
class AutonomousDiscoveryReport:
    """
    Result of discovering a complete coupled multi-variable differential equation system
    from raw trajectory data (`discover_autonomous_symbolic_system`) with zero prior `base_brick`.
    """

    compiled_brick: CompiledLawBrick
    calibrated_params: torch.Tensor
    per_state_expressions: dict[str, str]
    invariant_report: InvariantDiscoveryReport
    trajectory_r2: float
    vector_field_r2: float
    archetype: ArchetypeFingerprint
    audit_certificate: AuditCertificate
    helmholtz_hodge_decomposition: HelmholtzHodgeDecomposition | None = None


def discover_autonomous_symbolic_system(
    name: str,
    domain: str,
    state_specs: Sequence[StateVarSpec],
    observed_states: torch.Tensor,
    dt: float,
    observed_dx_dt: torch.Tensor | None = None,
    time_unit: Unit = s,
    energy_unit: Unit = DIMENSIONLESS,
    max_polynomial_degree: int = 3,
    max_terms_per_state: int = 2,
    include_rational_primitives: bool = True,
    include_trigonometric: bool = True,
    include_fractional_and_log: bool = True,
) -> AutonomousDiscoveryReport:
    """
    Discover an entire coupled multi-variable system of differential equations directly from
    raw trajectory data `observed_states` (`[T, D]` or `[E, T, D]`) without requiring any
    prior baseline model (`base_brick`), compile it into a standalone `CompiledLawBrick`,
    extract its conservation invariants, and polish its parameters via Layer 4.
    """
    x_in = observed_states.detach().to(torch.float64)
    state_dim = len(state_specs)
    if x_in.ndim == 3:
        num_exps, t_steps, _ = x_in.shape
        x_flat = x_in.reshape(num_exps * t_steps, state_dim)
    else:
        num_exps = 1
        t_steps, _ = x_in.shape
        x_flat = x_in

    var_bump_auto = 0.0
    x_eval_states = x_in
    if observed_dx_dt is not None:
        dx_in = observed_dx_dt.detach().to(torch.float64)
        dx_flat = dx_in.reshape(num_exps * t_steps, state_dim) if dx_in.ndim == 3 else dx_in
    else:
        noise_std = estimate_trajectory_noise_std(x_in)
        sig_std = max(float(torch.std(x_in).item()), 1e-12)
        is_noisy = t_steps >= 9 and noise_std > 0.008 * sig_std and noise_std > 25.0 * (dt**3) * sig_std
        if is_noisy:
            var_bump_auto = 0.75
            x_smooth, dx_in = savitzky_golay_denoise_and_differentiate(x_in, dt=dt)
            x_eval_states = x_smooth
            x_flat = x_smooth.reshape(num_exps * t_steps, state_dim) if x_smooth.ndim == 3 else x_smooth
            dx_flat = dx_in.reshape(num_exps * t_steps, state_dim) if dx_in.ndim == 3 else dx_in
        elif x_in.ndim == 3:
            dx_list = [estimate_trajectory_derivatives(x_in[e], dt=dt) for e in range(num_exps)]
            dx_in = torch.stack(dx_list, dim=0)
            dx_flat = dx_in.reshape(num_exps * t_steps, state_dim)
        else:
            dx_in = estimate_trajectory_derivatives(x_in, dt=dt)
            dx_flat = dx_in

    blank_spec = SymbolicSystemSpec(
        name=f"{name}_blank_carrier",
        domain=domain,
        state_specs=state_specs,
        param_specs=(),
        time_unit=time_unit,
        energy_unit=energy_unit,
        hamiltonian_expr="0.0",
    )
    blank_brick = compile_symbolic_system(blank_spec)
    empty_params = torch.zeros((1, 0), dtype=torch.float64, device=x_flat.device)

    inv_report = discover_conservation_invariants(
        brick=blank_brick,
        states=x_flat,
        flows=dx_flat,
        max_degree=max_polynomial_degree + 1,
        include_cross_terms=True,
        include_trigonometric=include_trigonometric,
        include_logarithmic=True,
    )

    discovered_drive_exprs: list[ExprNode] = []
    all_param_specs: list[ParamSpec] = []
    seen_p_names: set[str] = set()
    per_state_str: dict[str, str] = {}
    last_cand: SymbolicCandidate | None = None

    for idx in range(state_dim):
        col_vals = x_eval_states[..., idx]
        col_std = float(torch.std(col_vals).item())
        col_noise = estimate_trajectory_noise_std(x_in[..., idx : idx + 1]) if t_steps >= 6 else 0.0
        if col_std <= 1e-7 or (t_steps >= 9 and col_noise > 1e-12 and (col_std / col_noise) < 2.2):
            zero_node = ConstNode(0.0, state_specs[idx].unit / time_unit)
            discovered_drive_exprs.append(zero_node)
            per_state_str[state_specs[idx].name] = "0"
            continue

        r_single = torch.zeros_like(dx_flat)
        r_single[:, idx] = dx_flat[:, idx]

        lib_i = generate_symbolic_basis_library(
            brick=blank_brick,
            states=x_flat,
            base_params=empty_params,
            thermodynamic_slot="active_driven",
            affected_state_indices=[idx],
            residual_flow=r_single,
            max_polynomial_degree=max_polynomial_degree,
            include_rational_primitives=include_rational_primitives,
            include_trigonometric=include_trigonometric,
        )
        extra_i: list[SymbolicCandidate] = []
        if include_rational_primitives:
            extra_i = discover_rational_pade_candidates(
                brick=blank_brick,
                states=x_flat,
                target_residual=r_single,
                affected_state_indices=[idx],
            )
        if include_fractional_and_log:
            extra_i.extend(
                discover_fractional_power_and_log_candidates(
                    brick=blank_brick,
                    states=x_flat,
                    target_residual=r_single,
                    affected_state_indices=[idx],
                )
            )
        pareto_i = solve_sparse_pareto_front(
            library=lib_i,
            target_residual=r_single,
            dt=dt,
            max_active_terms=max_terms_per_state,
            extra_candidates=extra_i,
            num_experiments=num_exps,
            states=x_flat,
            variational_bump_weight=var_bump_auto,
        )
        best_i = pareto_i.best_candidate
        if best_i is not None and len(best_i.active_term_ids) >= 3:
            r2_tol = 0.015 if var_bump_auto > 0.0 else 0.0015
            min_r2_floor = 0.96 if var_bump_auto > 0.0 else 0.9990
            simpler_cands = [
                c_s
                for c_s in pareto_i.all_candidates
                if len(c_s.active_term_ids) < len(best_i.active_term_ids)
                and c_s.total_complexity <= best_i.total_complexity
                and (
                    c_s.vector_field_r2 >= 0.9992
                    or (
                        c_s.vector_field_r2 >= min_r2_floor
                        and c_s.vector_field_r2 >= best_i.vector_field_r2 - r2_tol
                    )
                )
            ]
            if simpler_cands:
                simpler_cands.sort(key=lambda c_s: (len(c_s.active_term_ids), c_s.bic_score))
                best_i = simpler_cands[0]

        if (
            best_i is not None
            and idx in best_i.patch_expressions
            and best_i.vector_field_r2 >= 0.55
            and float(torch.std(dx_flat[:, idx]).item()) > 1e-7
        ):
            node_i = best_i.patch_expressions[idx]
            discovered_drive_exprs.append(node_i)
            per_state_str[state_specs[idx].name] = node_i.to_infix()
            for ps in best_i.param_specs:
                if ps.name not in seen_p_names:
                    all_param_specs.append(ps)
                    seen_p_names.add(ps.name)
            last_cand = best_i
        else:
            zero_node = ConstNode(0.0, state_specs[idx].unit / time_unit)
            discovered_drive_exprs.append(zero_node)
            per_state_str[state_specs[idx].name] = "0"

    synthesized_spec = SymbolicSystemSpec(
        name=name,
        domain=domain,
        state_specs=state_specs,
        param_specs=all_param_specs,
        time_unit=time_unit,
        energy_unit=energy_unit,
        hamiltonian_expr="0.0",
        external_drive_exprs=discovered_drive_exprs,
    )
    compiled_brick = compile_symbolic_system(synthesized_spec)

    funnel = run_elimination_funnel(
        brick=compiled_brick,
        observed_states=x_eval_states,
        dt=dt,
        observed_dx_dt=dx_in,
        num_initial_samples=48,
        num_collocation_survivors=12,
        num_gradient_starts=4,
        gradient_steps=40,
    )
    is_null_law = all(v == "0" for v in per_state_str.values())
    vf_r2 = 0.0 if is_null_law else funnel.vector_field_r2
    tr_r2 = 0.0 if is_null_law else funnel.trajectory_r2
    arch = classify_mathematical_archetype(compiled_brick, candidate=last_cand, is_latent_augmented=False)
    hh_decomp = decompose_vector_field_helmholtz_hodge(
        brick=compiled_brick,
        states=x_flat,
        params=funnel.best_params,
    )
    emit_trace_node(
        layer="layer5_discovery",
        operation="discover_autonomous_symbolic_system",
        target_name=compiled_brick.name,
        local_error_metric=funnel.vector_field_mse,
        local_r2=vf_r2,
        global_error_metric=funnel.trajectory_mse,
        global_r2=tr_r2,
        selected_branch_id=compiled_brick.name,
        selected_expression="; ".join(f"d({k})/dt = {v}" for k, v in per_state_str.items()),
        metrics={
            "per_state_expressions": per_state_str,
            "num_invariants": inv_report.num_invariants,
            "archetype": arch.archetype_name,
            "helmholtz_hodge_regime": hh_decomp.dominant_regime,
            "conservative_energy_fraction": hh_decomp.conservative_energy_fraction,
            "dissipative_contraction_fraction": hh_decomp.dissipative_contraction_fraction,
        },
        skip_if_inner_loop=True,
    )

    return AutonomousDiscoveryReport(
        compiled_brick=compiled_brick,
        calibrated_params=funnel.best_params,
        per_state_expressions=per_state_str,
        invariant_report=inv_report,
        trajectory_r2=tr_r2,
        vector_field_r2=vf_r2,
        archetype=arch,
        audit_certificate=funnel.audit_certificate,
        helmholtz_hodge_decomposition=hh_decomp,
    )


@dataclass(frozen=True)
class SymbolicMechanismInventionReport:
    """
    End-to-end report from Inverse Symbolic Mechanism Invention (`synthesize_symbolic_feedback_mechanism`).
    """

    synthesized_brick: LawBrick
    calibrated_params: torch.Tensor
    invented_candidate: SymbolicCandidate
    invented_expression: str
    optimal_control_report: OptimalControlResult
    baseline_target_loss: float
    autonomous_closed_loop_loss: float
    loss_reduction_ratio: float
    autonomous_trajectory: torch.Tensor
    viable_across_trajectory: bool
    max_collateral_drift: float
    audit_certificate: AuditCertificate


def synthesize_symbolic_feedback_mechanism(
    base_brick: LawBrick,
    initial_state: torch.Tensor,
    base_params: torch.Tensor,
    dt: float,
    num_steps: int,
    target_loss_fn: Callable[[torch.Tensor], torch.Tensor],
    actuated_state_indices: Sequence[int] | None = None,
    collateral_state_indices: Sequence[int] | None = None,
    max_collateral_drift: float | None = None,
    max_control_amplitude: float = 5.0,
    control_optimization_steps: int = 24,
    polish_optimization_steps: int = 24,
) -> SymbolicMechanismInventionReport:
    """
    Invent a closed-form symbolic feedback mechanism `u_catalyst(x, theta_new)` when existing
    parameters `base_params` cannot steer `base_brick` to the target state.
    """
    x0 = initial_state.detach().to(torch.float64)
    if x0.ndim == 1:
        x0 = x0.unsqueeze(0)
    x0 = x0[:1]

    p_base = base_params.detach().to(torch.float64)
    if p_base.ndim == 1:
        p_base = p_base.unsqueeze(0)
    p_base = p_base[:1]

    act_indices = (
        tuple(int(i) for i in actuated_state_indices)
        if actuated_state_indices is not None
        else tuple(range(base_brick.state_dim))
    )
    coll_tol = float(max_collateral_drift) if max_collateral_drift is not None else 0.25

    def _roll_loss_fn(roll_or_traj: object) -> torch.Tensor:
        if hasattr(roll_or_traj, "states"):
            return target_loss_fn(getattr(roll_or_traj, "states"))
        assert isinstance(roll_or_traj, torch.Tensor)
        return target_loss_fn(roll_or_traj)

    ctrl_res = solve_optimal_control_waveform(
        brick=base_brick,
        initial_state=x0,
        params=p_base,
        dt=dt,
        num_steps=num_steps,
        target_loss_fn=_roll_loss_fn,
        actuated_state_indices=act_indices,
        collateral_state_indices=collateral_state_indices,
        collateral_tolerance_sigma=coll_tol,
        max_control_amplitude=max_control_amplitude,
        max_iters=control_optimization_steps,
    )

    x_star = ctrl_res.controlled_rollout.states[0, :-1].detach()
    u_star = ctrl_res.control_waveform[0].detach()

    lib = generate_symbolic_basis_library(
        brick=base_brick,
        states=x_star,
        base_params=p_base,
        thermodynamic_slot="active_driven",
        affected_state_indices=act_indices,
        residual_flow=u_star,
        max_polynomial_degree=2,
        include_rational_primitives=True,
        include_trigonometric=False,
    )
    # Orthogonal Hilbert-space projection Pi_{Phi(x*)} u*(t) onto the Markovian state-measurable
    # feedback subspace so non-Markovian open-loop boundary transients u_perp(t) do not corrupt
    # sparse state-feedback symbol identification.
    u_feedback_target = u_star
    if len(lib.terms) >= 2:
        try:
            with torch.no_grad():
                u_proj = u_star.clone()
                for act_i in act_indices:
                    u_col = u_star[:, act_i]
                    if float(torch.var(u_col).item()) <= 1e-12:
                        continue
                    cols_i = [
                        t_obj.flow_column[:, act_i]
                        for t_obj in lib.terms
                        if t_obj.target_key == act_i
                        and float(torch.linalg.norm(t_obj.flow_column[:, act_i]).item()) > 1e-12
                    ]
                    if len(cols_i) >= 2:
                        phi_i = torch.stack(cols_i, dim=-1)
                        # Orthogonal basis projection in state-feedback subspace preserving DC equilibrium intercept
                        u_mean = torch.mean(u_col)
                        u_centered = u_col - u_mean
                        phi_mean = torch.mean(phi_i, dim=0, keepdim=True)
                        phi_centered = phi_i - phi_mean
                        norms = torch.clamp(torch.linalg.norm(phi_centered, dim=0), min=1e-12)
                        corrs = torch.abs(phi_centered.T @ u_centered) / norms
                        top_k = min(3, phi_i.shape[1])
                        top_idx = torch.topk(corrs, k=top_k).indices
                        ones_col = torch.ones((u_col.shape[0], 1), dtype=torch.float64, device=u_star.device)
                        phi_sub = torch.cat([ones_col, phi_i[:, top_idx]], dim=-1)
                        gram = phi_sub.T @ phi_sub + 1e-8 * torch.eye(
                            phi_sub.shape[1], dtype=torch.float64, device=u_star.device
                        )
                        w_sub = torch.linalg.solve(gram, phi_sub.T @ u_col)
                        u_hat = phi_sub @ w_sub
                        if float(torch.var(u_hat).item()) > 1e-10:
                            u_proj[:, act_i] = u_hat
                u_feedback_target = u_proj
        except Exception:
            u_feedback_target = u_star

    pade_cands = discover_rational_pade_candidates(
        brick=base_brick,
        states=x_star,
        target_residual=u_feedback_target,
        affected_state_indices=act_indices,
    )
    pareto = solve_sparse_pareto_front(
        library=lib,
        target_residual=u_feedback_target,
        dt=dt,
        max_active_terms=2,
        extra_candidates=pade_cands,
    )
    best_cand = pareto.best_candidate
    if best_cand is None:
        raise RuntimeError(
            f"synthesize_symbolic_feedback_mechanism could not find a dimensionally valid candidate for '{base_brick.name}'."
        )

    n_base_p = base_brick.param_dim
    candidate_pool: list[SymbolicCandidate] = []
    seen_c_ids: set[str] = set()
    for c_item in [best_cand, *pareto.all_candidates]:
        if c_item.candidate_id not in seen_c_ids:
            candidate_pool.append(c_item)
            seen_c_ids.add(c_item.candidate_id)

    patched_brick: LawBrick | None = None
    init_full_params: torch.Tensor | None = None
    best_cl_key: tuple[int, int, float] | None = None
    with torch.no_grad():
        for c_item in candidate_pool[:8]:
            try:
                c_brick = compile_symbolic_candidate(
                    base_brick=base_brick,
                    candidate=c_item,
                    name_prefix="synthesized_catalyst",
                )
                c_params = c_brick.default_params(
                    batch_size=1, dtype=torch.float64, device=p_base.device
                )
                if n_base_p > 0:
                    c_params[:, :n_base_p] = p_base[:, :n_base_p]
                c_roll = rollout_trajectories(
                    c_brick, x0, c_params, dt=dt, num_steps=num_steps
                )
                c_finite = bool(torch.isfinite(c_roll.states).all().item())
                c_loss = (
                    float(_roll_loss_fn(c_roll).item())
                    if c_finite
                    else float("inf")
                )
                if not math.isfinite(c_loss):
                    c_finite = False
                    c_loss = float("inf")
                c_viable = bool(c_roll.all_time_viable_mask.all().item()) if c_finite else False
                c_key = (0 if c_finite else 1, 0 if c_viable else 1, c_loss)
                if best_cl_key is None or c_key < best_cl_key:
                    best_cl_key = c_key
                    best_cand = c_item
                    patched_brick = c_brick
                    init_full_params = c_params
            except Exception:
                continue

    if patched_brick is None or init_full_params is None:
        patched_brick = compile_symbolic_candidate(
            base_brick=base_brick,
            candidate=best_cand,
            name_prefix="synthesized_catalyst",
        )
        init_full_params = patched_brick.default_params(
            batch_size=1, dtype=torch.float64, device=p_base.device
        )
        if n_base_p > 0:
            init_full_params[:, :n_base_p] = p_base[:, :n_base_p]

    new_param_names = [pp.name for pp in patched_brick.param_specs[n_base_p:]]

    if new_param_names:
        interv_sol = solve_target_intervention(
            brick=patched_brick,
            initial_state=x0,
            baseline_params=init_full_params,
            dt=dt,
            num_steps=num_steps,
            target_loss_fn=_roll_loss_fn,
            allowed_param_names=new_param_names,
            collateral_state_indices=collateral_state_indices,
            collateral_tolerance_sigma=coll_tol,
            max_iters=polish_optimization_steps,
            learning_rate=0.14,
            l2_regularization=1e-5,
        )
        polished_params = interv_sol.intervened_params
        auto_loss = interv_sol.intervened_loss
        auto_traj = interv_sol.restored_rollout.states
        viable_ok = bool(interv_sol.restored_rollout.all_time_viable_mask.all().item())
        col_drift = interv_sol.collateral_max_drift
    else:
        polished_params = init_full_params
        roll = rollout_trajectories(patched_brick, x0, polished_params, dt=dt, num_steps=num_steps)
        auto_traj = roll.states
        with torch.no_grad():
            auto_loss = float(_roll_loss_fn(roll).item())
        viable_ok = bool(roll.all_time_viable_mask.all().item())
        col_drift = 0.0

    with torch.no_grad():
        roll_init = rollout_trajectories(patched_brick, x0, init_full_params, dt=dt, num_steps=num_steps)
        init_cand_finite = bool(torch.isfinite(roll_init.states).all().item())
        init_cand_loss = float(_roll_loss_fn(roll_init).item()) if init_cand_finite else float("inf")
        init_cand_viable = bool(roll_init.all_time_viable_mask.all().item()) if init_cand_finite else False
        auto_finite = math.isfinite(auto_loss) and bool(torch.isfinite(auto_traj).all().item())
        if init_cand_finite and (
            not auto_finite
            or (init_cand_viable and (not viable_ok or init_cand_loss < auto_loss))
        ):
            polished_params = init_full_params
            auto_loss = init_cand_loss
            auto_traj = roll_init.states
            viable_ok = init_cand_viable

    base_loss = ctrl_res.baseline_loss
    reduction_ratio = base_loss / max(auto_loss, 1e-12)
    traj_2d = auto_traj[0] if auto_traj.ndim == 3 else auto_traj
    p_batch = polished_params.expand(traj_2d.shape[0], -1)
    cert = audit_law_brick(patched_brick, sample_state=traj_2d, sample_params=p_batch)

    expr_str = "; ".join(
        f"u_{base_brick.state_specs[idx].name}(x) = {node.to_infix()}"
        for idx, node in sorted(best_cand.patch_expressions.items())
    )

    emit_trace_node(
        layer="layer5_discovery",
        operation="synthesize_symbolic_feedback_mechanism",
        target_name=patched_brick.name,
        local_r2=best_cand.vector_field_r2,
        global_error_metric=auto_loss,
        decision_margin=base_loss - auto_loss,
        selected_branch_id=best_cand.candidate_id,
        selected_expression=expr_str,
        metrics={
            "baseline_target_loss": round(base_loss, 6),
            "optimal_control_loss": round(ctrl_res.controlled_loss, 6),
            "autonomous_closed_loop_loss": round(auto_loss, 6),
            "loss_reduction_ratio": round(reduction_ratio, 4),
            "invented_expression": expr_str,
            "passed_audit": cert.passed_all,
        },
        skip_if_inner_loop=True,
    )

    return SymbolicMechanismInventionReport(
        synthesized_brick=patched_brick,
        calibrated_params=polished_params,
        invented_candidate=best_cand,
        invented_expression=expr_str,
        optimal_control_report=ctrl_res,
        baseline_target_loss=base_loss,
        autonomous_closed_loop_loss=auto_loss,
        loss_reduction_ratio=reduction_ratio,
        autonomous_trajectory=auto_traj,
        viable_across_trajectory=viable_ok,
        max_collateral_drift=col_drift,
        audit_certificate=cert,
    )
