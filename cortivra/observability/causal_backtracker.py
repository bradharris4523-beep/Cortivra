"""
Universal causal contradiction backtracker, stage-transition fracture detector, and
symbolic/parameter force attribution (`cortivra.observability.causal_backtracker`).
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from cortivra.observability.flight_recorder import (
    AlternativeBranchRecord,
    CausalTraceDAG,
    CausalTraceNode,
    suppress_inner_trace,
)

if TYPE_CHECKING:
    from cortivra.layer1_axioms.hamiltonian import LawBrick
    from cortivra.layer3_compiler.ast_nodes import ExprNode


@dataclass(frozen=True)
class TermForceAttribution:
    """
    Exact force, power, trajectory-integrated adjoint sensitivity, and symbolic sub-term
    attribution at the state `x(t*)` and step `t*` where a physical impossibility, blowup,
    or trajectory divergence occurred.
    """

    contradiction_step_index: int
    contradiction_state_index: int
    contradiction_state_name: str
    state_vector_at_contradiction: dict[str, float]
    total_flow_at_contradiction: float
    conservative_flow_share: float
    dissipative_flow_share: float
    external_drive_flow_share: float
    dissipative_power_at_contradiction: float
    local_jacobian_max_real_eigenvalue: float
    dominant_term_or_parameter: str
    dominant_contribution_fraction: float
    term_contributions: dict[str, float]
    parameter_sensitivities: dict[str, float]
    cumulative_trajectory_sensitivities: dict[str, float] = field(default_factory=dict)
    first_thermodynamic_violation_step: int | None = None
    max_energy_power_residual: float = 0.0


@dataclass(frozen=True)
class PipelineFractureEdge:
    """
    Edge `parent_node -> child_node` in the `CausalTraceDAG` where information-theoretic
    monotonicity or physical consistency fractured (e.g., high local vector-field R^2
    followed by global rollout divergence or shield violation).
    """

    parent_node_id: str
    parent_operation: str
    child_node_id: str
    child_operation: str
    upstream_r2: float | None
    downstream_r2: float | None
    r2_drop: float
    error_amplification_ratio: float
    physical_violation_delta: float
    fracture_summary: str


@dataclass(frozen=True)
class DecisionVulnerabilityNode:
    """
    Upstream decision node in the `CausalTraceDAG` whose ill-conditioned operator spectrum
    or narrow posterior margin allowed an unphysical or suboptimal branch to be selected.
    """

    node_id: str
    layer: str
    operation: str
    target_name: str
    condition_number: float
    effective_rank: int
    decision_margin: float | None
    selected_branch_id: str | None
    selected_expression: str | None
    recommended_alternative: AlternativeBranchRecord | None
    vulnerability_score: float
    causal_explanation: str


@dataclass(frozen=True)
class CausalContradictionReport:
    """
    Universal root-cause diagnosis produced by backtracking from any physical or predictive
    contradiction through the `CausalTraceDAG`.
    """

    contradiction_detected: bool
    contradiction_type: str
    failing_node_id: str | None
    failing_operation: str | None
    force_attribution: TermForceAttribution | None
    pipeline_fractures: tuple[PipelineFractureEdge, ...] = field(default_factory=tuple)
    root_cause_nodes: tuple[DecisionVulnerabilityNode, ...] = field(default_factory=tuple)
    primary_root_cause: DecisionVulnerabilityNode | None = None
    causal_chain_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decompose_additive_ast_nodes(
    node: ExprNode,
    scale_sign: float = 1.0,
) -> list[tuple[str, ExprNode, float]]:
    """
    Recursively decompose a Layer 3 `ExprNode` into additive `(infix_label, sub_node, signed_scale)`
    branches so each additive physical mechanism can be evaluated exactly on GPU/CPU tensors.
    """
    from cortivra.layer3_compiler.ast_nodes import BinaryOpNode, UnaryOpNode

    if isinstance(node, BinaryOpNode) and node.op == "+":
        return _decompose_additive_ast_nodes(node.left, scale_sign) + _decompose_additive_ast_nodes(
            node.right, scale_sign
        )
    if isinstance(node, BinaryOpNode) and node.op == "-":
        return _decompose_additive_ast_nodes(node.left, scale_sign) + _decompose_additive_ast_nodes(
            node.right, -scale_sign
        )
    if isinstance(node, UnaryOpNode) and node.op == "-":
        return _decompose_additive_ast_nodes(node.operand, -scale_sign)

    raw_label = node.to_infix()
    label = f"-({raw_label})" if scale_sign < 0.0 else raw_label
    return [(label, node, scale_sign)]


def _evaluate_native_layer3_ast_terms_at_state(
    brick: LawBrick,
    x_crit: torch.Tensor,
    p_single: torch.Tensor,
    state_idx: int,
) -> dict[str, float]:
    """
    Evaluate every additive Layer 3 `ExprNode` sub-expression's force contribution
    to `dx[state_idx]/dt` at `(x_crit, p_single)`.
    """
    contribs: dict[str, float] = {}
    env: dict[str, torch.Tensor] = {}
    for idx, sp in enumerate(brick.state_specs):
        if idx < x_crit.shape[1]:
            env[sp.name] = x_crit[:, idx]
    for idx, pp in enumerate(brick.param_specs):
        if idx < p_single.shape[1]:
            env[pp.name] = p_single[:, idx]

    with torch.no_grad():
        eff = brick.effort(x_crit, p_single)[0]

        if hasattr(brick, "_drive_asts") and state_idx < len(getattr(brick, "_drive_asts")):
            drv_node, drv_scale = getattr(brick, "_drive_asts")[state_idx]
            for lbl, sub_node, s_sign in _decompose_additive_ast_nodes(drv_node, drv_scale):
                val = float((sub_node.eval_tensor(env)[0] * s_sign).item())
                if math.isfinite(val) and abs(val) > 1e-12:
                    contribs[lbl] = round(contribs.get(lbl, 0.0) + val, 6)

        if hasattr(brick, "_diss_asts") and state_idx < len(getattr(brick, "_diss_asts")):
            diss_node, diss_scale = getattr(brick, "_diss_asts")[state_idx]
            e_i = float(eff[state_idx].item())
            for lbl, sub_node, s_sign in _decompose_additive_ast_nodes(diss_node, diss_scale):
                r_val = float((sub_node.eval_tensor(env)[0] * s_sign).item())
                flow_val = -r_val * e_i
                if math.isfinite(flow_val) and abs(flow_val) > 1e-12:
                    key_lbl = f"-({lbl}) * e_{brick.state_specs[state_idx].name}"
                    contribs[key_lbl] = round(contribs.get(key_lbl, 0.0) + flow_val, 6)

        if hasattr(brick, "_skew_asts") and hasattr(brick, "_effort_asts"):
            skew_map = getattr(brick, "_skew_asts")
            for (i_s, j_s), (j_node, j_scale) in skew_map.items():
                if i_s == state_idx or j_s == state_idx:
                    other_idx = j_s if i_s == state_idx else i_s
                    orient = 1.0 if i_s == state_idx else -1.0
                    j_val = float((j_node.eval_tensor(env)[0] * j_scale * orient).item())
                    e_other_ast = getattr(brick, "_effort_asts")[other_idx]
                    h_scale = float(getattr(brick, "_h_scale", 1.0))
                    if e_other_ast is not None:
                        for lbl, sub_e, s_sign in _decompose_additive_ast_nodes(e_other_ast, h_scale * j_val):
                            val = float((sub_e.eval_tensor(env)[0] * s_sign).item())
                            if math.isfinite(val) and abs(val) > 1e-12:
                                contribs[lbl] = round(contribs.get(lbl, 0.0) + val, 6)

        if hasattr(brick, "_compiled_patches"):
            base_b = getattr(brick, "base_brick", None)
            base_p_dim = int(getattr(brick, "base_param_dim", brick.param_dim))
            if base_b is not None:
                base_terms = _evaluate_native_layer3_ast_terms_at_state(
                    base_b, x_crit[:, : base_b.state_dim], p_single[:, :base_p_dim], state_idx
                )
                for k_b, v_b in base_terms.items():
                    contribs[k_b] = v_b

            t_mode = str(getattr(brick, "thermodynamic_mode", "active_driven"))
            patches = getattr(brick, "_compiled_patches")
            for key, (p_node, p_scale) in patches.items():
                if t_mode == "active_driven" and key == state_idx:
                    for lbl, sub_n, s_sign in _decompose_additive_ast_nodes(p_node, p_scale):
                        val = float((sub_n.eval_tensor(env)[0] * s_sign).item())
                        if math.isfinite(val) and abs(val) > 1e-12:
                            contribs[lbl] = round(contribs.get(lbl, 0.0) + val, 6)
                elif t_mode == "passive_dissipative" and key == state_idx:
                    e_i = float(eff[state_idx].item())
                    for lbl, sub_n, s_sign in _decompose_additive_ast_nodes(p_node, p_scale):
                        r_val = float((sub_n.eval_tensor(env)[0] * s_sign).item())
                        flow_val = -r_val * e_i
                        if math.isfinite(flow_val) and abs(flow_val) > 1e-12:
                            contribs[f"-({lbl}) * e_{brick.state_specs[state_idx].name}"] = round(flow_val, 6)
                elif t_mode == "conservative_coupling" and isinstance(key, tuple):
                    i_c, j_c = key
                    if i_c == state_idx or j_c == state_idx:
                        other_c = j_c if i_c == state_idx else i_c
                        orient = 1.0 if i_c == state_idx else -1.0
                        e_other = float(eff[other_c].item())
                        for lbl, sub_n, s_sign in _decompose_additive_ast_nodes(p_node, p_scale * orient * e_other):
                            val = float((sub_n.eval_tensor(env)[0] * s_sign).item())
                            if math.isfinite(val) and abs(val) > 1e-12:
                                contribs[f"({lbl}) * e_{brick.state_specs[other_c].name}"] = round(val, 6)

        if hasattr(brick, "delta_hamiltonian_ast") and hasattr(brick, "_delta_effort_asts"):
            base_b = getattr(brick, "base_brick", None)
            base_p_dim = int(getattr(brick, "base_param_dim", brick.param_dim))
            if base_b is not None:
                base_terms = _evaluate_native_layer3_ast_terms_at_state(
                    base_b, x_crit, p_single[:, :base_p_dim], state_idx
                )
                for k_b, v_b in base_terms.items():
                    contribs[k_b] = v_b
            j_mat = brick.structure_matrix(x_crit, p_single)[0]
            r_mat = brick.dissipation_matrix(x_crit, p_single)[0]
            jr_row = j_mat[state_idx] - r_mat[state_idx]
            h_scale = float(getattr(brick, "_h_scale", 1.0))
            delta_effs = getattr(brick, "_delta_effort_asts")
            for k_col, de_node in enumerate(delta_effs):
                w_jr = float(jr_row[k_col].item())
                if abs(w_jr) <= 1e-12:
                    continue
                for lbl, sub_n, s_sign in _decompose_additive_ast_nodes(de_node, h_scale * w_jr):
                    val = float((sub_n.eval_tensor(env)[0] * s_sign).item())
                    if math.isfinite(val) and abs(val) > 1e-12:
                        contribs[lbl] = round(contribs.get(lbl, 0.0) + val, 6)

        if hasattr(brick, "coupling_ast") and hasattr(brick, "latent_drive_ast"):
            t_idx = int(getattr(brick, "target_state_index", 0))
            b_dim = int(getattr(brick, "base_state_dim", brick.state_dim - 1))
            if state_idx == t_idx:
                c_node = getattr(brick, "coupling_ast")
                val = float(c_node.eval_tensor(env)[0].item())
                if math.isfinite(val) and abs(val) > 1e-12:
                    contribs[c_node.to_infix()] = round(val, 6)
            elif state_idx == b_dim:
                d_node = getattr(brick, "latent_drive_ast")
                for lbl, sub_n, s_sign in _decompose_additive_ast_nodes(d_node, 1.0):
                    val = float((sub_n.eval_tensor(env)[0] * s_sign).item())
                    if math.isfinite(val) and abs(val) > 1e-12:
                        contribs[lbl] = round(val, 6)

    return contribs


def attribute_state_space_contradiction(
    brick: LawBrick,
    states: torch.Tensor,
    params: torch.Tensor | None = None,
    reference_states: torch.Tensor | None = None,
    dt: float = 0.02,
) -> TermForceAttribution:
    """
    Pinpoint the exact time step `t*` and state dimension `d*` where `states` first
    exhibits a physical violation (non-finite blowup, homeostatic boundary breach,
    or maximum divergence from `reference_states`), and decompose:
        - Instantaneous conservative, dissipative, and external drive force shares at `x(t*)`.
        - Local state Jacobian eigenvalue `max_i Re(lambda_i(dF/dx))` at `x(t*)`.
        - Exact Layer 3 symbolic `ExprNode` sub-term force contributions at `x(t*)`.
        - Instantaneous parameter sensitivities `|theta_j * dF_d(x(t*)) / d_theta_j|`.
        - Cumulative trajectory-integrated adjoint sensitivities `|theta_j * dx_d(t*) / d_theta_j|`
          rolled out from `t=0 -> t*`.
        - Trajectory-wide 1st/2nd Law thermodynamic power residual and earliest violation step.
    """
    x_seq = states.detach().to(torch.float64)
    if x_seq.ndim == 3:
        x_seq = x_seq[0]
    elif x_seq.ndim == 1:
        x_seq = x_seq.unsqueeze(0)

    p_row = (
        params.detach().to(torch.float64)
        if params is not None
        else brick.default_params(batch_size=1, dtype=torch.float64, device=x_seq.device)
    )
    if p_row.ndim == 1:
        p_row = p_row.unsqueeze(0)
    p_single = p_row[:1]

    t_steps, state_dim = x_seq.shape
    h_min = torch.tensor(
        [sp.homeostatic_min for sp in brick.state_specs], dtype=torch.float64, device=x_seq.device
    )
    h_max = torch.tensor(
        [sp.homeostatic_max for sp in brick.state_specs], dtype=torch.float64, device=x_seq.device
    )

    # Locate critical step t*
    finite_mask = torch.isfinite(x_seq).all(dim=-1)
    if not bool(finite_mask.all().item()):
        non_finite_indices = torch.nonzero(~finite_mask, as_tuple=False).flatten()
        first_bad = int(non_finite_indices[0].item())
        crit_t = max(first_bad - 1, 0)
    else:
        breach_low = torch.relu(h_min.unsqueeze(0) - x_seq)
        breach_high = torch.relu(x_seq - h_max.unsqueeze(0))
        breach_total = breach_low + breach_high
        if bool((breach_total > 0.0).any().item()):
            breach_steps = torch.nonzero(breach_total.sum(dim=-1) > 0.0, as_tuple=False).flatten()
            crit_t = int(breach_steps[0].item())
        elif reference_states is not None:
            ref_seq = reference_states.detach().to(torch.float64)
            if ref_seq.ndim == 3:
                ref_seq = ref_seq[0]
            n_cmp = min(t_steps, ref_seq.shape[0])
            d_cmp = min(state_dim, ref_seq.shape[1])
            diff_norm = torch.sum((x_seq[:n_cmp, :d_cmp] - ref_seq[:n_cmp, :d_cmp]) ** 2, dim=-1)
            crit_t = int(torch.argmax(diff_norm).item())
        else:
            crit_t = t_steps - 1

    x_crit = x_seq[crit_t : crit_t + 1].clone()
    x_crit = torch.nan_to_num(x_crit, nan=0.0, posinf=1e4, neginf=-1e4)

    with torch.no_grad():
        f_total = brick.flow(x_crit, p_single)[0]
        f_cons = brick.conservative_flow(x_crit, p_single)[0]
        f_diss = brick.dissipative_flow(x_crit, p_single)[0]
        f_ext = f_total - (f_cons - f_diss)
        eff = brick.effort(x_crit, p_single)[0]
        diss_power = float(torch.sum(eff * f_diss).item())

        x_clean_all = torch.nan_to_num(x_seq, nan=0.0, posinf=1e4, neginf=-1e4)
        mod_mask = torch.max(torch.abs(x_clean_all), dim=-1).values <= 1e4
        x_mod = x_clean_all[mod_mask] if bool(mod_mask.any().item()) else x_clean_all[:1]
        p_mod = p_single.expand(x_mod.shape[0], -1)
        diss_all = brick.dissipation_power(x_mod, p_mod)
        neg_diss_steps = torch.nonzero(diss_all < -1e-4, as_tuple=False).flatten()
        first_thermo_step: int | None = (
            int(neg_diss_steps[0].item()) if neg_diss_steps.numel() > 0 else None
        )
        e_all = brick.effort(x_mod, p_mod)
        f_all = brick.flow(x_mod, p_mod)
        f_cons_all = brick.conservative_flow(x_mod, p_mod)
        f_diss_all = brick.dissipative_flow(x_mod, p_mod)
        f_ext_all = f_all - (f_cons_all - f_diss_all)
        dh_dt_all = torch.sum(e_all * f_all, dim=-1)
        p_ext_all = torch.sum(e_all * f_ext_all, dim=-1)
        power_scale = torch.clamp(torch.abs(dh_dt_all) + torch.abs(p_ext_all) + torch.abs(diss_all), min=1.0)
        power_res_all = torch.abs(dh_dt_all - (p_ext_all - diss_all)) / power_scale
        max_power_res = float(torch.max(power_res_all).item()) if power_res_all.numel() > 0 else 0.0

    # Identify affected state dimension d*
    if reference_states is not None:
        ref_seq = reference_states.detach().to(torch.float64)
        if ref_seq.ndim == 3:
            ref_seq = ref_seq[0]
        c_idx = min(crit_t, ref_seq.shape[0] - 1)
        d_cmp = min(state_dim, ref_seq.shape[1])
        crit_d = int(torch.argmax(torch.abs(x_crit[0, :d_cmp] - ref_seq[c_idx, :d_cmp])).item())
    else:
        span_vec = torch.clamp(h_max - h_min, min=1e-9)
        rel_exc = torch.maximum(
            (h_min - x_crit[0]) / span_vec,
            (x_crit[0] - h_max) / span_vec,
        )
        if bool((rel_exc > 0.0).any().item()):
            crit_d = int(torch.argmax(rel_exc).item())
        else:
            crit_d = int(torch.argmax(torch.abs(f_total)).item())

    state_name = brick.state_specs[crit_d].name
    tot_f_d = float(f_total[crit_d].item())
    cons_f_d = float(f_cons[crit_d].item())
    diss_f_d = float(-f_diss[crit_d].item())
    ext_f_d = float(f_ext[crit_d].item())
    abs_sum = abs(cons_f_d) + abs(diss_f_d) + abs(ext_f_d)
    denom_f = max(abs_sum, 1e-12)

    # Local state Jacobian dF/dx at x(t*)
    max_real_eig = 0.0
    try:
        with torch.enable_grad():
            x_req = x_crit.clone().requires_grad_(True)
            f_req = brick.flow(x_req, p_single)[0]
            jac_rows: list[torch.Tensor] = []
            for i_s in range(state_dim):
                if f_req[i_s].requires_grad:
                    g_row = torch.autograd.grad(
                        f_req[i_s],
                        x_req,
                        retain_graph=(i_s < state_dim - 1),
                        allow_unused=True,
                    )[0]
                    jac_rows.append(
                        g_row[0]
                        if g_row is not None
                        else torch.zeros(state_dim, dtype=torch.float64, device=x_seq.device)
                    )
                else:
                    jac_rows.append(torch.zeros(state_dim, dtype=torch.float64, device=x_seq.device))
            jac_mat = torch.stack(jac_rows, dim=0)
            if torch.isfinite(jac_mat).all():
                eig_vals = torch.linalg.eigvals(jac_mat)
                max_real_eig = float(torch.max(eig_vals.real).item())
    except Exception:
        max_real_eig = 0.0

    # Instantaneous parameter sensitivities |theta_j * dF_d / d_theta_j|
    param_sens: dict[str, float] = {}
    if brick.param_dim > 0:
        try:
            with torch.enable_grad():
                p_req = p_single.clone().requires_grad_(True)
                f_p = brick.flow(x_crit, p_req)[0, crit_d]
                if f_p.requires_grad:
                    g_p = torch.autograd.grad(f_p, p_req, allow_unused=True)[0]
                    if g_p is not None:
                        for j_p, pp in enumerate(brick.param_specs):
                            p_val = float(p_single[0, j_p].item())
                            grad_val = float(g_p[0, j_p].item())
                            scaled_contrib = (
                                abs(p_val * grad_val) if abs(p_val) > 1e-12 else abs(grad_val)
                            )
                            param_sens[pp.name] = round(scaled_contrib, 6)
        except Exception:
            pass

    # Cumulative trajectory-integrated adjoint sensitivities |theta_j * dx_d(t*) / d_theta_j|
    cum_traj_sens: dict[str, float] = {}
    if brick.param_dim > 0 and crit_t >= 1:
        try:
            n_adj_steps = min(crit_t, 25)
            start_idx = max(crit_t - n_adj_steps, 0)
            x_start = x_seq[start_idx : start_idx + 1].clone()
            with suppress_inner_trace(), torch.enable_grad():
                p_adj = p_single.clone().requires_grad_(True)
                curr_x = x_start
                for _ in range(crit_t - start_idx):
                    curr_x = brick.step_rk4(curr_x, p_adj, dt=dt)
                target_out = curr_x[0, crit_d]
                if target_out.requires_grad:
                    g_traj = torch.autograd.grad(target_out, p_adj, allow_unused=True)[0]
                    if g_traj is not None:
                        for j_p, pp in enumerate(brick.param_specs):
                            p_val = float(p_single[0, j_p].item())
                            gv = float(g_traj[0, j_p].item())
                            if math.isfinite(gv):
                                cum_traj_sens[pp.name] = round(
                                    abs(p_val * gv) if abs(p_val) > 1e-12 else abs(gv), 6
                                )
        except Exception:
            pass

    term_contribs = _evaluate_native_layer3_ast_terms_at_state(brick, x_crit, p_single, crit_d)

    if not term_contribs:
        if abs(cons_f_d) > 1e-12:
            term_contribs["conservative_flow"] = round(cons_f_d, 6)
        if abs(diss_f_d) > 1e-12:
            term_contribs["dissipative_flow"] = round(diss_f_d, 6)
        if abs(ext_f_d) > 1e-12:
            term_contribs["external_drive_flow"] = round(ext_f_d, 6)

    dom_name = "none"
    dom_frac = 0.0
    if term_contribs:
        total_abs_t = sum(abs(v) for v in term_contribs.values())
        best_t = max(term_contribs.items(), key=lambda kv: abs(kv[1]))
        dom_name = best_t[0]
        dom_frac = abs(best_t[1]) / max(total_abs_t, 1e-12)
    elif param_sens:
        total_abs_p = sum(abs(v) for v in param_sens.values())
        best_p = max(param_sens.items(), key=lambda kv: abs(kv[1]))
        dom_name = best_p[0]
        dom_frac = abs(best_p[1]) / max(total_abs_p, 1e-12)

    state_dict = {
        sp.name: round(float(x_crit[0, i].item()), 6) for i, sp in enumerate(brick.state_specs)
    }

    return TermForceAttribution(
        contradiction_step_index=crit_t,
        contradiction_state_index=crit_d,
        contradiction_state_name=state_name,
        state_vector_at_contradiction=state_dict,
        total_flow_at_contradiction=round(tot_f_d, 6),
        conservative_flow_share=round(abs(cons_f_d) / denom_f, 4),
        dissipative_flow_share=round(abs(diss_f_d) / denom_f, 4),
        external_drive_flow_share=round(abs(ext_f_d) / denom_f, 4),
        dissipative_power_at_contradiction=round(diss_power, 6),
        local_jacobian_max_real_eigenvalue=round(max_real_eig, 6),
        dominant_term_or_parameter=dom_name,
        dominant_contribution_fraction=round(dom_frac, 4),
        term_contributions=term_contribs,
        parameter_sensitivities=param_sens,
        cumulative_trajectory_sensitivities=cum_traj_sens,
        first_thermodynamic_violation_step=first_thermo_step,
        max_energy_power_residual=round(max_power_res, 8),
    )


def _score_node_vulnerability(
    node: CausalTraceNode,
    dominant_symbol: str | None = None,
) -> DecisionVulnerabilityNode | None:
    """
    Evaluate the mathematical vulnerability of an upstream decision node based on:
        1. Operator spectral conditioning (`log10(condition_number)` and rank deficiency).
        2. Posterior decision margin (`decision_margin` between the selected branch and runner-up).
        3. Local-to-global error divergence (`local_r2` vs `global_r2`).
        4. Symbolic overlap with `dominant_symbol` (the term/parameter responsible for downstream failure).
    """
    cond = node.operator_spectrum.condition_number if node.operator_spectrum is not None else 1.0
    eff_rank = node.operator_spectrum.effective_rank if node.operator_spectrum is not None else 1
    mat_cols = node.operator_spectrum.matrix_shape[1] if node.operator_spectrum is not None else 1
    rank_deficit = max(mat_cols - eff_rank, 0)

    cond_score = min(math.log10(max(cond, 1.0)) / 10.0, 1.0)
    rank_score = min(float(rank_deficit) * 0.25, 0.75)

    margin_score = 0.0
    if node.decision_margin is not None:
        # Small BIC/objective margin (< 2.0) between distinct branches indicates high epistemic ambiguity
        margin_score = 1.0 / (1.0 + max(float(node.decision_margin), 0.0) * 0.5)

    local_global_gap = 0.0
    if node.local_r2 is not None and node.global_r2 is not None:
        local_global_gap = max(0.0, float(node.local_r2) - float(node.global_r2))

    symbol_match_bonus = 0.0
    if dominant_symbol and dominant_symbol != "none":
        expr_text = f"{node.selected_expression or ''} {node.target_name} {node.metrics}"
        tokens = [
            tok
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", dominant_symbol)
            if len(tok) >= 1
        ]
        if dominant_symbol in expr_text or any(tok in expr_text for tok in tokens):
            symbol_match_bonus = 0.45

    vuln = (
        0.30 * cond_score
        + 0.20 * rank_score
        + 0.35 * margin_score
        + 0.40 * local_global_gap
        + symbol_match_bonus
    )
    if (
        vuln < 0.08
        and node.status == "completed"
        and node.physical_violation_metric <= 1e-4
        and not node.alternatives
    ):
        return None

    rec_alt: AlternativeBranchRecord | None = node.alternatives[0] if node.alternatives else None
    reasons: list[str] = []
    if cond >= 1e3:
        reasons.append(f"ill-conditioned operator (kappa={cond:.3e}, rank={eff_rank}/{mat_cols})")
    if node.decision_margin is not None and node.decision_margin < 4.0:
        alt_txt = f" over '{rec_alt.expression_or_summary}'" if rec_alt else ""
        reasons.append(f"narrow decision margin (delta={node.decision_margin:.4f}{alt_txt})")
    if local_global_gap >= 0.05:
        reasons.append(
            f"local-to-global fidelity drop (local_r2={node.local_r2:.4f} -> global_r2={node.global_r2:.4f})"
        )
    if symbol_match_bonus > 0.0:
        reasons.append(f"originated dominant term/parameter '{dominant_symbol}'")
    if node.physical_violation_metric > 1e-4:
        reasons.append(f"physical violation={node.physical_violation_metric:.3e}")

    explanation = (
        f"Node '{node.node_id}' ({node.layer}.{node.operation} on '{node.target_name}'): "
        + (", ".join(reasons) if reasons else "evaluated candidate decision branch.")
    )

    return DecisionVulnerabilityNode(
        node_id=node.node_id,
        layer=node.layer,
        operation=node.operation,
        target_name=node.target_name,
        condition_number=round(cond, 4),
        effective_rank=eff_rank,
        decision_margin=node.decision_margin,
        selected_branch_id=node.selected_branch_id,
        selected_expression=node.selected_expression,
        recommended_alternative=rec_alt,
        vulnerability_score=round(vuln, 4),
        causal_explanation=explanation,
    )


def analyze_trace_and_backtrack_contradictions(
    dag: CausalTraceDAG,
    brick: LawBrick | None = None,
    states: torch.Tensor | None = None,
    params: torch.Tensor | None = None,
    reference_states: torch.Tensor | None = None,
    r2_contradiction_threshold: float = 0.90,
) -> CausalContradictionReport:
    """
    Inspect `dag` (and optional active `brick` / `states`) for any physical impossibility,
    shield violation, rollout blowup, or local-to-global stage contradiction, and trace
    backward up the causal graph to identify the exact originating decision node and
    alternative branch.
    """
    fractures: list[PipelineFractureEdge] = []
    failing_node: CausalTraceNode | None = None
    contradiction_type = "none"

    for node in dag.nodes:
        # Check intra-node local vs global fracture (e.g. collocation R^2 vs trajectory rollout R^2)
        if node.local_r2 is not None and node.global_r2 is not None:
            drop_self = float(node.local_r2) - float(node.global_r2)
            if drop_self >= 0.08:
                amp = (
                    float(node.global_error_metric) / max(float(node.local_error_metric or 1e-12), 1e-12)
                    if node.global_error_metric is not None
                    else (1.0 + drop_self * 10.0)
                )
                fractures.append(
                    PipelineFractureEdge(
                        parent_node_id=node.node_id,
                        parent_operation=f"{node.operation}[local_collocation]",
                        child_node_id=node.node_id,
                        child_operation=f"{node.operation}[global_rollout]",
                        upstream_r2=node.local_r2,
                        downstream_r2=node.global_r2,
                        r2_drop=round(drop_self, 6),
                        error_amplification_ratio=round(amp, 4),
                        physical_violation_delta=node.physical_violation_metric,
                        fracture_summary=(
                            f"Intra-stage fracture in '{node.node_id}': local vector-field R^2={node.local_r2:.4f} "
                            f"degraded to global trajectory R^2={node.global_r2:.4f} (drop={drop_self:.4f})."
                        ),
                    )
                )

        # Check parent -> child inter-node fractures
        p_r2 = node.global_r2 if node.global_r2 is not None else node.local_r2
        for p_id in node.parent_ids:
            parent = dag.get_node(p_id)
            if parent is None:
                continue
            up_r2 = parent.global_r2 if parent.global_r2 is not None else parent.local_r2
            if up_r2 is not None and p_r2 is not None:
                r2_drop = float(up_r2) - float(p_r2)
                if r2_drop >= 0.08 or (
                    node.physical_violation_metric - parent.physical_violation_metric > 1e-3
                ):
                    up_err = parent.global_error_metric or parent.local_error_metric or 1e-9
                    dn_err = node.global_error_metric or node.local_error_metric or 0.0
                    amp_ratio = float(dn_err) / max(float(up_err), 1e-12)
                    v_delta = node.physical_violation_metric - parent.physical_violation_metric
                    fractures.append(
                        PipelineFractureEdge(
                            parent_node_id=parent.node_id,
                            parent_operation=parent.operation,
                            child_node_id=node.node_id,
                            child_operation=node.operation,
                            upstream_r2=up_r2,
                            downstream_r2=p_r2,
                            r2_drop=round(r2_drop, 6),
                            error_amplification_ratio=round(amp_ratio, 4),
                            physical_violation_delta=round(v_delta, 6),
                            fracture_summary=(
                                f"Stage transition '{parent.operation}' -> '{node.operation}' fractured: "
                                f"R^2 dropped {up_r2:.4f} -> {p_r2:.4f} (violation_delta={v_delta:.3e})."
                            ),
                        )
                    )

        # Detect terminal or intermediate failure nodes
        node_r2 = node.global_r2 if node.global_r2 is not None else node.local_r2
        if node.status == "failed":
            failing_node = node
            contradiction_type = "compile_or_runtime_exception"
        elif node.physical_violation_metric > 1e-4:
            failing_node = node
            contradiction_type = "physical_axiom_or_viability_violation"
        elif node_r2 is not None and node_r2 < r2_contradiction_threshold:
            failing_node = node
            contradiction_type = "predictive_divergence_or_model_mismatch"

    force_attr: TermForceAttribution | None = None
    if brick is not None and states is not None:
        force_attr = attribute_state_space_contradiction(
            brick=brick,
            states=states,
            params=params,
            reference_states=reference_states,
        )
        if contradiction_type == "none":
            if not bool(torch.isfinite(states).all().item()):
                contradiction_type = "non_finite_state_blowup"
            elif force_attr.local_jacobian_max_real_eigenvalue > 5.0 and len(fractures) > 0:
                contradiction_type = "unstable_positive_lyapunov_eigenvalue"

    if failing_node is None and fractures:
        last_frac = fractures[-1]
        failing_node = dag.get_node(last_frac.child_node_id)
        if contradiction_type == "none":
            contradiction_type = "stage_transition_monotonicity_fracture"

    has_contradiction = bool(
        failing_node is not None or len(fractures) > 0 or contradiction_type != "none"
    )
    dom_sym = force_attr.dominant_term_or_parameter if force_attr is not None else None

    # Walk backward from failing_node (or across all decision nodes if general audit)
    candidate_ancestors: list[CausalTraceNode] = []
    if failing_node is not None:
        candidate_ancestors = [failing_node, *dag.ancestors_of(failing_node.node_id)]
    else:
        candidate_ancestors = list(reversed(dag.nodes))

    vuln_nodes: list[DecisionVulnerabilityNode] = []
    seen_ids: set[str] = set()
    for anc in candidate_ancestors:
        if anc.node_id in seen_ids:
            continue
        seen_ids.add(anc.node_id)
        v_node = _score_node_vulnerability(anc, dominant_symbol=dom_sym)
        if v_node is not None:
            vuln_nodes.append(v_node)

    vuln_nodes.sort(key=lambda v: v.vulnerability_score, reverse=True)
    primary_rc = vuln_nodes[0] if vuln_nodes else None

    if not has_contradiction:
        summary = (
            f"All {len(dag.nodes)} causal DAG nodes satisfied physical admissibility and "
            f"stage-to-stage monotonicity contracts with zero contradictions "
            f"(merkle_root={dag.merkle_root_hash[:12]}...)."
        )
    else:
        parts: list[str] = [f"Detected '{contradiction_type}'"]
        if failing_node is not None:
            parts[0] += f" at node '{failing_node.node_id}' ({failing_node.layer}.{failing_node.operation})."
        else:
            parts[0] += "."
        if force_attr is not None:
            parts.append(
                f"At step t*={force_attr.contradiction_step_index} on state '{force_attr.contradiction_state_name}' "
                f"(x={force_attr.state_vector_at_contradiction}), dominant force contributor is "
                f"'{force_attr.dominant_term_or_parameter}' ({force_attr.dominant_contribution_fraction * 100.0:.1f}% share, "
                f"max_real_eig={force_attr.local_jacobian_max_real_eigenvalue:+.4f})."
            )
        if fractures:
            parts.append(fractures[0].fracture_summary)
        if primary_rc is not None:
            alt_str = (
                f" Runner-up alternative at that node was '{primary_rc.recommended_alternative.expression_or_summary}' "
                f"(score={primary_rc.recommended_alternative.objective_score:.4f}, R^2={primary_rc.recommended_alternative.local_r2:.4f})."
                if primary_rc.recommended_alternative is not None
                else ""
            )
            parts.append(f"Backtracked root cause -> {primary_rc.causal_explanation}{alt_str}")
        summary = " ".join(parts)

    return CausalContradictionReport(
        contradiction_detected=has_contradiction,
        contradiction_type=contradiction_type,
        failing_node_id=failing_node.node_id if failing_node is not None else None,
        failing_operation=failing_node.operation if failing_node is not None else None,
        force_attribution=force_attr,
        pipeline_fractures=tuple(fractures),
        root_cause_nodes=tuple(vuln_nodes),
        primary_root_cause=primary_rc,
        causal_chain_summary=summary,
    )
