"""
Weak-integral sparse symbolic regression, Pareto-front selection, and Sanathanan-Koerner
rational Padé discovery (`cortivra.layer5_discovery.sparse_regression`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick, ParamSpec
from cortivra.layer1_axioms.units import DIMENSIONLESS
from cortivra.layer3_compiler.ast_nodes import (
    BinaryOpNode,
    ConstNode,
    ExprNode,
    PrimitiveCallNode,
    SymbolNode,
    UnaryOpNode,
)
from cortivra.layer5_discovery.basis_library import (
    CandidateBasisTerm,
    SymbolicBasisLibrary,
    estimate_characteristic_scales,
)


@dataclass(frozen=True)
class SymbolicCandidate:
    """
    Single symbolic law candidate on the complexity-accuracy Pareto front.

    Attributes:
        candidate_id: Identifier for the candidate.
        thermodynamic_slot: One of `"passive_dissipative"`, `"hamiltonian_potential"`,
                            `"conservative_coupling"`, or `"active_driven"`.
        patch_expressions: Mapping from state index `i`, pair `(i, j)`, or `"hamiltonian"`
                           to symbolic `ExprNode`.
        param_specs: Tuple of `ParamSpec`s (both linear amplitudes and embedded nonlinear scales)
                     required by `patch_expressions`, initialized to the regression-fitted values.
        active_term_ids: Tuple of `term_id` strings included in this candidate.
        total_complexity: Total AST structural node count across all `patch_expressions`.
        num_parameters: Number of tunable parameters in `param_specs`.
        vector_field_mse: Mean squared error against the target residual vector field.
        vector_field_r2: Coefficient of determination `R^2` against the target residual vector field.
        bic_score: Bayesian Information Criterion (`N * ln(MSE) + k_eff * ln(N)`).
        is_pareto_optimal: True if no other candidate has both lower complexity and lower MSE.
    """

    candidate_id: str
    thermodynamic_slot: str
    patch_expressions: dict[int | tuple[int, int] | str, ExprNode]
    param_specs: tuple[ParamSpec, ...]
    active_term_ids: tuple[str, ...]
    total_complexity: int
    num_parameters: int
    vector_field_mse: float
    vector_field_r2: float
    bic_score: float
    is_pareto_optimal: bool = True


@dataclass(frozen=True)
class ParetoDiscoveryReport:
    """
    Result of sparse symbolic regression and rational Padé discovery on a residual vector field.

    Attributes:
        pareto_front: Tuple of Pareto-optimal `SymbolicCandidate`s ordered by `bic_score`.
        all_candidates: Tuple of all evaluated `SymbolicCandidate`s ordered by `bic_score`.
        best_candidate: Top-ranked `SymbolicCandidate` by `bic_score`.
    """

    pareto_front: tuple[SymbolicCandidate, ...]
    all_candidates: tuple[SymbolicCandidate, ...]
    best_candidate: SymbolicCandidate | None


def build_weak_integral_system(
    flow_columns: torch.Tensor,
    target_residual: torch.Tensor,
    dt: float = 0.01,
    window_steps: int = 4,
    integral_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct a combined pointwise and weak-integral linear system `(A, b)` from
    `flow_columns` (`[T, StateDim, NumTerms]`) and `target_residual` (`[T, StateDim]`).

    Integrating over sliding windows `[t_k, t_{k+w}]` via the trapezoidal rule filters out
    high-frequency derivative estimation noise while preserving exact linear coefficients.
    """
    t_steps, state_dim, num_terms = flow_columns.shape
    a_pt = flow_columns.reshape(t_steps * state_dim, num_terms)
    b_pt = target_residual.reshape(t_steps * state_dim)

    if t_steps < window_steps + 2 or integral_weight <= 0.0 or window_steps < 2:
        return a_pt, b_pt

    num_windows = t_steps - window_steps
    int_a_rows: list[torch.Tensor] = []
    int_b_rows: list[torch.Tensor] = []

    # Trapezoidal weights [0.5, 1.0, ..., 1.0, 0.5] normalized by window length
    trap_w = torch.ones((window_steps + 1,), dtype=flow_columns.dtype, device=flow_columns.device)
    trap_w[0] = 0.5
    trap_w[-1] = 0.5
    trap_w = trap_w / float(window_steps)

    for start in range(num_windows):
        end = start + window_steps + 1
        win_cols = torch.einsum("w,wsm->sm", trap_w, flow_columns[start:end])
        win_targ = torch.einsum("w,ws->s", trap_w, target_residual[start:end])
        int_a_rows.append(win_cols)
        int_b_rows.append(win_targ)

    a_int = torch.cat(int_a_rows, dim=0) * integral_weight
    b_int = torch.cat(int_b_rows, dim=0) * integral_weight

    return torch.cat([a_pt, a_int], dim=0), torch.cat([b_pt, b_int], dim=0)


def _solve_subspace_coefficients(
    a_sub: torch.Tensor,
    b_vec: torch.Tensor,
    non_negative_mask: torch.Tensor,
    ridge_lambda: float = 1e-10,
) -> torch.Tensor:
    """
    Solve `min_c ||a_sub @ c - b_vec||_2^2 + ridge_lambda * ||c||_2^2` subject to
    `c_j >= 0` for indices where `non_negative_mask[j]` is True.
    """
    k = a_sub.shape[1]
    if k == 0:
        return torch.zeros((0,), dtype=a_sub.dtype, device=a_sub.device)

    ata = a_sub.T @ a_sub + ridge_lambda * torch.eye(k, dtype=a_sub.dtype, device=a_sub.device)
    atb = a_sub.T @ b_vec
    try:
        c = torch.linalg.solve(ata, atb)
    except RuntimeError:
        c = torch.linalg.lstsq(a_sub, b_vec.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)

    if not bool(non_negative_mask.any().item()):
        return c

    # Active-set / projected Newton-PGD refinement for non-negative constrained coordinates
    if bool((c[non_negative_mask] >= 0.0).all().item()):
        return c

    c = torch.where(non_negative_mask, torch.clamp(c, min=0.0), c)
    eig_max = float(torch.linalg.matrix_norm(ata, ord=2).item())
    step_lr = 1.0 / max(eig_max, 1e-8)
    for _ in range(120):
        grad = ata @ c - atb
        c_next = c - step_lr * grad
        c_next = torch.where(non_negative_mask, torch.clamp(c_next, min=0.0), c_next)
        if float(torch.max(torch.abs(c_next - c)).item()) < 1e-11:
            c = c_next
            break
        c = c_next
    return c


def _assemble_candidate_from_support(
    candidate_id: str,
    library: SymbolicBasisLibrary,
    support_indices: Sequence[int],
    coeffs: torch.Tensor,
    flow_columns_pt: torch.Tensor,
    target_residual_pt: torch.Tensor,
) -> SymbolicCandidate | None:
    """Build a `SymbolicCandidate` from active basis indices and fitted physical coefficients."""
    active_pairs = [
        (idx, float(coeffs[pos].item()))
        for pos, idx in enumerate(support_indices)
        if abs(float(coeffs[pos].item())) > 1e-8
    ]
    if not active_pairs:
        return None

    t_steps, state_dim, _ = flow_columns_pt.shape
    n_obs = max(t_steps * state_dim, 1)

    pred_flow = torch.zeros_like(target_residual_pt)
    grouped_exprs: dict[int | tuple[int, int] | str, list[ExprNode]] = {}
    param_specs_list: list[ParamSpec] = []
    seen_param_names: set[str] = set()
    active_ids: list[str] = []

    for basis_idx, coeff_val in active_pairs:
        term: CandidateBasisTerm = library.terms[basis_idx]
        if term.require_non_negative_coeff and coeff_val <= 1e-9:
            continue

        pred_flow = pred_flow + coeff_val * term.flow_column
        active_ids.append(term.term_id)

        # Create ParamSpec initialized to the regression-fitted coefficient
        base_ps = term.param_spec
        min_v = 0.0 if term.require_non_negative_coeff else min(base_ps.min_value, coeff_val * 10.0, -abs(coeff_val) * 10.0 - 10.0)
        max_v = max(base_ps.max_value, abs(coeff_val) * 10.0 + 10.0)
        updated_ps = ParamSpec(
            name=base_ps.name,
            unit=base_ps.unit,
            default_value=coeff_val,
            min_value=min_v,
            max_value=max_v,
            tunable=True,
        )
        if updated_ps.name not in seen_param_names:
            param_specs_list.append(updated_ps)
            seen_param_names.add(updated_ps.name)

        for extra_ps in term.extra_param_specs:
            if extra_ps.name not in seen_param_names:
                param_specs_list.append(extra_ps)
                seen_param_names.add(extra_ps.name)

        grouped_exprs.setdefault(term.target_key, []).append(term.expr_node)

    if not grouped_exprs:
        return None

    patch_expressions: dict[int | tuple[int, int] | str, ExprNode] = {}
    total_complexity = 0
    for key, expr_list in grouped_exprs.items():
        combined = expr_list[0]
        for nxt in expr_list[1:]:
            combined = BinaryOpNode("+", combined, nxt)
        simplified = combined.simplify()
        patch_expressions[key] = simplified
        total_complexity += simplified.tree_complexity()

    res_err = target_residual_pt - pred_flow
    mse = float(torch.mean(res_err * res_err).item())
    y_var = float(torch.sum((target_residual_pt - torch.mean(target_residual_pt)) ** 2).item()) / float(n_obs)
    r2 = 1.0 - (mse / max(y_var, 1e-18))

    num_params = len(param_specs_list)
    # Effective model complexity includes parameter count and structural AST complexity
    k_eff = float(num_params) + 0.25 * float(total_complexity)
    bic = float(n_obs) * math.log(max(mse, 1e-24)) + k_eff * math.log(max(n_obs, 2))

    return SymbolicCandidate(
        candidate_id=candidate_id,
        thermodynamic_slot=library.thermodynamic_slot,
        patch_expressions=patch_expressions,
        param_specs=tuple(param_specs_list),
        active_term_ids=tuple(active_ids),
        total_complexity=total_complexity,
        num_parameters=num_params,
        vector_field_mse=mse,
        vector_field_r2=r2,
        bic_score=bic,
        is_pareto_optimal=True,
    )


def discover_rational_pade_candidates(
    brick: LawBrick,
    states: torch.Tensor,
    target_residual: torch.Tensor,
    affected_state_indices: Sequence[int] | None = None,
    num_sk_iterations: int = 8,
) -> list[SymbolicCandidate]:
    """
    Discover saturating / rational Padé laws of the form:
        `Phi_i(x) = (v_max * x_j^n) / (K_d^n + x_j^n)`  or  `v_max * K_i^n / (K_i^n + x_j^n)`
    using Sanathanan-Koerner Iteratively Reweighted Least Squares (IRLS) to directly estimate
    both numerator amplitude `v_max` and denominator scale `K_d^n` without nonlinear local minima.
    """
    x = states.detach().to(torch.float64)
    r = target_residual.detach().to(torch.float64)
    t_steps, state_dim = x.shape
    n_obs = max(t_steps * state_dim, 1)
    y_var = float(torch.sum((r - torch.mean(r)) ** 2).item()) / float(n_obs)

    target_indices = (
        list(affected_state_indices)
        if affected_state_indices
        else list(range(state_dim))
    )
    scales = estimate_characteristic_scales(brick, x, residual_flow=r)
    discovered: list[SymbolicCandidate] = []

    for idx in target_indices:
        r_col = r[:, idx]
        if float(torch.linalg.norm(r_col).item()) <= 1e-9:
            continue
        sp_target = brick.state_specs[idx]
        flow_unit = sp_target.unit / brick.time_unit

        for j, sp_state in enumerate(brick.state_specs):
            xj_abs = torch.clamp(torch.abs(x[:, j]), min=1e-12)
            sym_j = SymbolNode(sp_state.name, sp_state.unit, role="state")

            for n_exp in (1, 2, 3, 4):
                z = xj_abs ** float(n_exp)  # x_j^n
                # Model 1 (Activating Hill / Michaelis-Menten): r = (a * z) / (1 + b * z)
                # Rearranged: a * z - b * (r * z) = r, where b = 1 / K_d^n > 0
                weights = torch.ones((t_steps,), dtype=torch.float64, device=x.device)
                best_a = 0.0
                best_b = 1.0 / (scales[sp_state.name] ** float(n_exp))

                for _ in range(num_sk_iterations):
                    w_col = weights.unsqueeze(-1)
                    design = torch.stack([z, -r_col * z], dim=-1) * w_col
                    rhs = r_col * weights
                    sol = torch.linalg.lstsq(design, rhs.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)
                    a_est = float(sol[0].item())
                    b_est = float(sol[1].item())
                    if b_est <= 1e-9:
                        b_est = 1.0 / max(scales[sp_state.name] ** float(n_exp), 1e-6)
                    denom = torch.clamp(1.0 + b_est * z, min=1e-6)
                    weights = 1.0 / denom
                    best_a, best_b = a_est, b_est

                kd_fit = (1.0 / max(best_b, 1e-12)) ** (1.0 / float(n_exp))
                vmax_fit = best_a / max(best_b, 1e-12)

                if abs(vmax_fit) > 1e-8 and math.isfinite(kd_fit) and kd_fit > 1e-7:
                    pred_col = vmax_fit * (z / (kd_fit ** float(n_exp) + z))
                    pred_full = torch.zeros_like(r)
                    pred_full[:, idx] = pred_col
                    mse = float(torch.mean((r - pred_full) ** 2).item())
                    r2 = 1.0 - (mse / max(y_var, 1e-18))

                    if r2 > 0.50:
                        vmax_spec = ParamSpec(
                            name=f"vmax_sk_{idx}_{sp_state.name}_n{n_exp}",
                            unit=flow_unit,
                            default_value=vmax_fit,
                            min_value=min(-abs(vmax_fit) * 10.0 - 10.0, vmax_fit * 10.0),
                            max_value=max(abs(vmax_fit) * 10.0 + 10.0, 10.0),
                        )
                        kd_spec = ParamSpec(
                            name=f"kd_sk_{idx}_{sp_state.name}_n{n_exp}",
                            unit=sp_state.unit,
                            default_value=kd_fit,
                            min_value=max(kd_fit * 0.05, 1e-6),
                            max_value=max(kd_fit * 20.0, 10.0),
                        )
                        vmax_sym = SymbolNode(vmax_spec.name, flow_unit, role="param")
                        kd_sym = SymbolNode(kd_spec.name, sp_state.unit, role="param")
                        expr_node = BinaryOpNode(
                            "*",
                            vmax_sym,
                            PrimitiveCallNode(
                                "hill",
                                (sym_j, kd_sym, ConstNode(float(n_exp), DIMENSIONLESS)),
                            ),
                        )
                        comp = expr_node.tree_complexity()
                        k_eff = 2.0 + 0.25 * float(comp)
                        bic = float(n_obs) * math.log(max(mse, 1e-24)) + k_eff * math.log(max(n_obs, 2))
                        discovered.append(
                            SymbolicCandidate(
                                candidate_id=f"pade_hill_{idx}_{sp_state.name}_n{n_exp}",
                                thermodynamic_slot="active_driven",
                                patch_expressions={idx: expr_node},
                                param_specs=(vmax_spec, kd_spec),
                                active_term_ids=(f"pade_hill_{idx}_{sp_state.name}_n{n_exp}",),
                                total_complexity=comp,
                                num_parameters=2,
                                vector_field_mse=mse,
                                vector_field_r2=r2,
                                bic_score=bic,
                                is_pareto_optimal=True,
                            )
                        )

                # Model 2 (Inhibitory Hill saturation): r = a / (1 + b * z) = a * K_i^n / (K_i^n + z)
                # Rearranged: a * 1 - b * (r * z) = r, where V_max = a, b = 1 / K_i^n > 0
                weights_inh = torch.ones((t_steps,), dtype=torch.float64, device=x.device)
                best_a_inh = 0.0
                best_b_inh = 1.0 / (scales[sp_state.name] ** float(n_exp))
                ones_col = torch.ones_like(z)

                for _ in range(num_sk_iterations):
                    w_col = weights_inh.unsqueeze(-1)
                    design_inh = torch.stack([ones_col, -r_col * z], dim=-1) * w_col
                    rhs_inh = r_col * weights_inh
                    sol_inh = torch.linalg.lstsq(design_inh, rhs_inh.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)
                    a_inh = float(sol_inh[0].item())
                    b_inh = float(sol_inh[1].item())
                    if b_inh <= 1e-9:
                        b_inh = 1.0 / max(scales[sp_state.name] ** float(n_exp), 1e-6)
                    denom_inh = torch.clamp(1.0 + b_inh * z, min=1e-6)
                    weights_inh = 1.0 / denom_inh
                    best_a_inh, best_b_inh = a_inh, b_inh

                ki_fit = (1.0 / max(best_b_inh, 1e-12)) ** (1.0 / float(n_exp))
                vmax_inh_fit = best_a_inh

                if abs(vmax_inh_fit) > 1e-8 and math.isfinite(ki_fit) and ki_fit > 1e-7:
                    ki_n = ki_fit ** float(n_exp)
                    pred_inh_col = vmax_inh_fit * (ki_n / (ki_n + z))
                    pred_inh_full = torch.zeros_like(r)
                    pred_inh_full[:, idx] = pred_inh_col
                    mse_inh = float(torch.mean((r - pred_inh_full) ** 2).item())
                    r2_inh = 1.0 - (mse_inh / max(y_var, 1e-18))

                    if r2_inh > 0.50:
                        vmax_inh_spec = ParamSpec(
                            name=f"vmax_inh_sk_{idx}_{sp_state.name}_n{n_exp}",
                            unit=flow_unit,
                            default_value=vmax_inh_fit,
                            min_value=min(-abs(vmax_inh_fit) * 10.0 - 10.0, vmax_inh_fit * 10.0),
                            max_value=max(abs(vmax_inh_fit) * 10.0 + 10.0, 10.0),
                        )
                        ki_spec = ParamSpec(
                            name=f"ki_sk_{idx}_{sp_state.name}_n{n_exp}",
                            unit=sp_state.unit,
                            default_value=ki_fit,
                            min_value=max(ki_fit * 0.05, 1e-6),
                            max_value=max(ki_fit * 20.0, 10.0),
                        )
                        vmax_inh_sym = SymbolNode(vmax_inh_spec.name, flow_unit, role="param")
                        ki_sym = SymbolNode(ki_spec.name, sp_state.unit, role="param")
                        inh_node = BinaryOpNode(
                            "*",
                            vmax_inh_sym,
                            PrimitiveCallNode(
                                "hill_inhib",
                                (sym_j, ki_sym, ConstNode(float(n_exp), DIMENSIONLESS)),
                            ),
                        )
                        comp_inh = inh_node.tree_complexity()
                        k_eff_inh = 2.0 + 0.25 * float(comp_inh)
                        bic_inh = float(n_obs) * math.log(max(mse_inh, 1e-24)) + k_eff_inh * math.log(max(n_obs, 2))
                        discovered.append(
                            SymbolicCandidate(
                                candidate_id=f"pade_hill_inhib_{idx}_{sp_state.name}_n{n_exp}",
                                thermodynamic_slot="active_driven",
                                patch_expressions={idx: inh_node},
                                param_specs=(vmax_inh_spec, ki_spec),
                                active_term_ids=(f"pade_hill_inhib_{idx}_{sp_state.name}_n{n_exp}",),
                                total_complexity=comp_inh,
                                num_parameters=2,
                                vector_field_mse=mse_inh,
                                vector_field_r2=r2_inh,
                                bic_score=bic_inh,
                                is_pareto_optimal=True,
                            )
                        )

            # Model 3 (General Padé [1/2] Substrate-Inhibition / Bell-Shaped Quotient):
            # r = (a1 * x) / (1 + b1 * x + b2 * x^2) -> a1*x - b1*(r*x) - b2*(r*x^2) = r
            x1 = xj_abs
            x2 = xj_abs * xj_abs
            w_sub = torch.ones((t_steps,), dtype=torch.float64, device=x.device)
            a1_best, b1_best, b2_best = 0.0, 1.0 / scales[sp_state.name], 1.0 / (scales[sp_state.name] ** 2)
            for _ in range(num_sk_iterations):
                wc = w_sub.unsqueeze(-1)
                des_sub = torch.stack([x1, -r_col * x1, -r_col * x2], dim=-1) * wc
                rhs_sub = r_col * w_sub
                sol_sub = torch.linalg.lstsq(des_sub, rhs_sub.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)
                a1_e = float(sol_sub[0].item())
                b1_e = max(float(sol_sub[1].item()), 0.0)
                b2_e = max(float(sol_sub[2].item()), 0.0)
                if b1_e + b2_e <= 1e-9:
                    b1_e = 1.0 / max(scales[sp_state.name], 1e-6)
                denom_sub = torch.clamp(1.0 + b1_e * x1 + b2_e * x2, min=1e-6)
                w_sub = 1.0 / denom_sub
                a1_best, b1_best, b2_best = a1_e, b1_e, b2_e

            if abs(a1_best) > 1e-8 and b2_best > 1e-7:
                pred_sub_col = (a1_best * x1) / (1.0 + b1_best * x1 + b2_best * x2)
                pred_sub_full = torch.zeros_like(r)
                pred_sub_full[:, idx] = pred_sub_col
                mse_sub = float(torch.mean((r - pred_sub_full) ** 2).item())
                r2_sub = 1.0 - (mse_sub / max(y_var, 1e-18))

                if r2_sub > 0.60:
                    a1_u = flow_unit / sp_state.unit
                    b1_u = DIMENSIONLESS / sp_state.unit
                    b2_u = DIMENSIONLESS / (sp_state.unit ** 2)
                    p_a1 = ParamSpec(
                        name=f"a1_pade_{idx}_{sp_state.name}",
                        unit=a1_u,
                        default_value=a1_best,
                        min_value=-abs(a1_best) * 10.0 - 10.0,
                        max_value=abs(a1_best) * 10.0 + 10.0,
                    )
                    p_b1 = ParamSpec(
                        name=f"b1_pade_{idx}_{sp_state.name}",
                        unit=b1_u,
                        default_value=b1_best,
                        min_value=0.0,
                        max_value=max(b1_best * 20.0 + 10.0, 10.0),
                    )
                    p_b2 = ParamSpec(
                        name=f"b2_pade_{idx}_{sp_state.name}",
                        unit=b2_u,
                        default_value=b2_best,
                        min_value=1e-8,
                        max_value=max(b2_best * 20.0 + 10.0, 10.0),
                    )
                    num_node = BinaryOpNode("*", SymbolNode(p_a1.name, a1_u, role="param"), sym_j)
                    den_node = BinaryOpNode(
                        "+",
                        BinaryOpNode(
                            "+",
                            ConstNode(1.0, DIMENSIONLESS),
                            BinaryOpNode("*", SymbolNode(p_b1.name, b1_u, role="param"), sym_j),
                        ),
                        BinaryOpNode(
                            "*",
                            SymbolNode(p_b2.name, b2_u, role="param"),
                            BinaryOpNode("**", sym_j, ConstNode(2.0, DIMENSIONLESS)),
                        ),
                    )
                    rat_node = BinaryOpNode("/", num_node, den_node)
                    comp_rat = rat_node.tree_complexity()
                    k_eff_rat = 3.0 + 0.25 * float(comp_rat)
                    bic_rat = float(n_obs) * math.log(max(mse_sub, 1e-24)) + k_eff_rat * math.log(max(n_obs, 2))
                    discovered.append(
                        SymbolicCandidate(
                            candidate_id=f"pade_rat12_{idx}_{sp_state.name}",
                            thermodynamic_slot="active_driven",
                            patch_expressions={idx: rat_node},
                            param_specs=(p_a1, p_b1, p_b2),
                            active_term_ids=(f"pade_rat12_{idx}_{sp_state.name}",),
                            total_complexity=comp_rat,
                            num_parameters=3,
                            vector_field_mse=mse_sub,
                            vector_field_r2=r2_sub,
                            bic_score=bic_rat,
                            is_pareto_optimal=True,
                        )
                    )

    return discovered


def solve_sparse_pareto_front(
    library: SymbolicBasisLibrary,
    target_residual: torch.Tensor,
    dt: float = 0.01,
    max_active_terms: int = 4,
    ridge_lambda: float = 1e-6,
    window_steps: int = 4,
    integral_weight: float = 0.5,
    extra_candidates: Sequence[SymbolicCandidate] = (),
) -> ParetoDiscoveryReport:
    """
    Discover the Pareto front of sparse symbolic equations from `library` that explain
    `target_residual` (`[T, StateDim]`).

    Combines:
        1. Dual pointwise + weak integral collocation system construction.
        2. Unit-column-normalized Orthogonal Matching Pursuit (OMP) for cardinalities `1..max_active_terms`.
        3. Unit-column-normalized Sequential Thresholded Ridge Regression (STRidge) across a geometric
           grid of sparsity thresholds.
        4. Exact non-negative subspace refinement (`c_m >= 0`) when required by the 2nd Law.
        5. Pareto dominance filtering across `(total_complexity, vector_field_mse)`.
    """
    r_pt = target_residual.detach().to(torch.float64)
    num_terms = len(library.terms)
    if num_terms == 0 and not extra_candidates:
        raise ValueError("SymbolicBasisLibrary contains 0 valid candidate terms.")

    candidates_by_signature: dict[tuple[str, ...], SymbolicCandidate] = {}
    for ext in extra_candidates:
        candidates_by_signature[ext.active_term_ids] = ext

    if num_terms > 0:
        flow_cols_pt = torch.stack([t.flow_column.to(torch.float64) for t in library.terms], dim=-1)
        a_mat, b_vec = build_weak_integral_system(
            flow_cols_pt,
            r_pt,
            dt=dt,
            window_steps=window_steps,
            integral_weight=integral_weight,
        )

        # Normalize columns to unit L2 norm for scale-invariant threshold & greedy selection
        col_norms = torch.clamp(torch.linalg.norm(a_mat, dim=0), min=1e-12)
        a_norm = a_mat / col_norms.unsqueeze(0)
        non_neg_all = torch.tensor(
            [t.require_non_negative_coeff for t in library.terms],
            dtype=torch.bool,
            device=r_pt.device,
        )

        discovered_supports: set[tuple[int, ...]] = set()
        k_limit = min(max_active_terms, num_terms)

        # Method A: Lookahead Beam-OMP + vectorized 2x2 Gram pair screening
        residual_omp = b_vec.clone()
        active_omp: list[int] = []
        for k_step in range(1, k_limit + 1):
            correlations = a_norm.T @ residual_omp  # [M]
            # For non-negative terms, negative correlation with residual cannot be selected alone
            eff_scores = torch.where(
                non_neg_all,
                torch.clamp(correlations, min=0.0),
                torch.abs(correlations),
            )
            for idx_used in active_omp:
                eff_scores[idx_used] = -1.0
            best_next = int(torch.argmax(eff_scores).item())
            if float(eff_scores[best_next].item()) <= 1e-12:
                break
            active_omp.append(best_next)
            supp_tuple = tuple(sorted(active_omp))
            discovered_supports.add(supp_tuple)

            sub_t = list(supp_tuple)
            c_sub = _solve_subspace_coefficients(
                a_mat[:, sub_t], b_vec, non_neg_all[sub_t], ridge_lambda=1e-10
            )
            residual_omp = b_vec - a_mat[:, sub_t] @ c_sub

        # Evaluate all single-term models
        for s_idx in range(num_terms):
            discovered_supports.add((int(s_idx),))

        # Vectorized all-pairs 2x2 Gram subspace projection to capture competing positive/negative
        # production-degradation pairs (e.g. alpha*x - beta*x*y) where individual 1D correlation is weak
        if k_limit >= 2 and num_terms >= 2:
            gram = a_norm.T @ a_norm  # [M, M]
            r_corr = a_norm.T @ b_vec  # [M]
            denom = torch.clamp(1.0 - gram * gram, min=1e-7)
            c_i_mat = (r_corr.unsqueeze(1) - gram * r_corr.unsqueeze(0)) / denom
            c_j_mat = (r_corr.unsqueeze(0) - gram * r_corr.unsqueeze(1)) / denom
            # Respect non-negativity constraints in 2x2 screening
            valid_i = (~non_neg_all).unsqueeze(1) | (c_i_mat >= 0.0)
            valid_j = (~non_neg_all).unsqueeze(0) | (c_j_mat >= 0.0)
            proj_sq = torch.where(
                valid_i & valid_j,
                c_i_mat * r_corr.unsqueeze(1) + c_j_mat * r_corr.unsqueeze(0),
                torch.zeros_like(gram),
            )
            proj_sq = torch.triu(proj_sq, diagonal=1)
            flat_scores = proj_sq.reshape(-1)
            num_top_pairs = min(24, int(torch.sum(flat_scores > 0.0).item()))
            if num_top_pairs > 0:
                top_pair_flat = torch.argsort(flat_scores, descending=True)[:num_top_pairs].tolist()
                for flat_idx in top_pair_flat:
                    i_p = int(flat_idx // num_terms)
                    j_p = int(flat_idx % num_terms)
                    if i_p < j_p:
                        pair_supp = (i_p, j_p)
                        discovered_supports.add(pair_supp)
                        # If k_limit >= 3, extend top pairs greedily up to k_limit
                        if k_limit >= 3:
                            beam_list = [i_p, j_p]
                            for _ in range(3, k_limit + 1):
                                c_b = _solve_subspace_coefficients(
                                    a_mat[:, beam_list], b_vec, non_neg_all[beam_list], ridge_lambda=1e-10
                                )
                                res_b = b_vec - a_mat[:, beam_list] @ c_b
                                corr_b = torch.abs(a_norm.T @ res_b)
                                for u_i in beam_list:
                                    corr_b[u_i] = -1.0
                                nxt_b = int(torch.argmax(corr_b).item())
                                if float(corr_b[nxt_b].item()) <= 1e-12:
                                    break
                                beam_list.append(nxt_b)
                                discovered_supports.add(tuple(sorted(beam_list)))

        # Method B: Sequential Thresholded Ridge Regression (STRidge) with backward elimination
        c_init_norm = _solve_sparse_ridge(a_norm, b_vec, non_neg_all, ridge_lambda=ridge_lambda)
        max_c = float(torch.max(torch.abs(c_init_norm)).item()) if c_init_norm.numel() > 0 else 1.0
        thresholds = [max_c * frac for frac in (0.01, 0.03, 0.07, 0.15, 0.25, 0.40, 0.60, 0.80)]

        for thresh in thresholds:
            active_mask = torch.abs(c_init_norm) >= thresh
            for _ in range(num_terms + 4):
                if not bool(active_mask.any().item()):
                    break
                act_idx = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
                sub_c = _solve_sparse_ridge(
                    a_norm[:, act_idx], b_vec, non_neg_all[act_idx], ridge_lambda=ridge_lambda
                )
                if len(act_idx) > k_limit:
                    # Iteratively prune the single weakest normalized coefficient
                    drop_count = max(1, (len(act_idx) - k_limit + 1) // 2)
                    keep_k = len(act_idx) - drop_count
                    top_k_local = torch.argsort(torch.abs(sub_c), descending=True)[:keep_k].tolist()
                    act_idx = sorted(act_idx[i] for i in top_k_local)
                    new_mask = torch.zeros_like(active_mask)
                    new_mask[act_idx] = True
                    active_mask = new_mask
                    if len(act_idx) <= k_limit:
                        discovered_supports.add(tuple(act_idx))
                    continue

                keep_local = torch.abs(sub_c) >= thresh
                if bool(keep_local.all().item()):
                    break
                new_mask = torch.zeros_like(active_mask)
                for local_i, global_i in enumerate(act_idx):
                    if bool(keep_local[local_i].item()):
                        new_mask[global_i] = True
                active_mask = new_mask

            if bool(active_mask.any().item()):
                supp = tuple(sorted(int(i) for i in torch.nonzero(active_mask, as_tuple=False).flatten().tolist()))
                if 1 <= len(supp) <= k_limit:
                    discovered_supports.add(supp)

        # Map each target_key to its linear/constant basis index (if any) so biased linear
        # base parameters (e.g. over-estimated linear damping when fitting cubic damping) can
        # be de-biased during nonlinear basis evaluation
        const_idx_by_key: dict[int | tuple[int, int] | str, int] = {}
        if library.thermodynamic_slot == "passive_dissipative":
            for idx_t, t_obj in enumerate(library.terms):
                if t_obj.term_id.endswith("_const"):
                    const_idx_by_key[t_obj.target_key] = idx_t

        # Unpenalized physical-scale subspace polish on every discovered support
        for idx_s, supp in enumerate(sorted(discovered_supports)):
            sub_list = list(supp)
            # Check if a nonlinear passive_dissipative term can be de-biased against its linear effort column
            debias_indices: list[int] = []
            if library.thermodynamic_slot == "passive_dissipative":
                for s_i in sub_list:
                    t_key = library.terms[s_i].target_key
                    c_idx = const_idx_by_key.get(t_key)
                    if c_idx is not None and c_idx not in sub_list and c_idx not in debias_indices:
                        debias_indices.append(c_idx)

            if debias_indices:
                aug_list = sub_list + debias_indices
                aug_non_neg = non_neg_all[aug_list].clone()
                # Allow the linear baseline shift to be unconstrained so it removes base-parameter bias
                aug_non_neg[len(sub_list) :] = False
                c_aug = _solve_subspace_coefficients(
                    a_mat[:, aug_list], b_vec, aug_non_neg, ridge_lambda=1e-12
                )
                c_phys = c_aug[: len(sub_list)]
                r_debiased = r_pt.clone()
                for pos_d, d_idx in enumerate(debias_indices):
                    c_d = float(c_aug[len(sub_list) + pos_d].item())
                    r_debiased = r_debiased - c_d * flow_cols_pt[:, :, d_idx]
            else:
                c_phys = _solve_subspace_coefficients(
                    a_mat[:, sub_list], b_vec, non_neg_all[sub_list], ridge_lambda=1e-12
                )
                r_debiased = r_pt

            cand = _assemble_candidate_from_support(
                candidate_id=f"{library.thermodynamic_slot}_cand_{idx_s}",
                library=library,
                support_indices=sub_list,
                coeffs=c_phys,
                flow_columns_pt=flow_cols_pt,
                target_residual_pt=r_debiased,
            )
            if cand is not None:
                prev = candidates_by_signature.get(cand.active_term_ids)
                if prev is None or cand.vector_field_mse < prev.vector_field_mse:
                    candidates_by_signature[cand.active_term_ids] = cand

    all_cands = sorted(candidates_by_signature.values(), key=lambda c: c.bic_score)
    if not all_cands:
        # Fallback to best single basis term if all supports were pruned
        best_idx = 0
        c_phys = _solve_subspace_coefficients(
            a_mat[:, [best_idx]], b_vec, non_neg_all[[best_idx]], ridge_lambda=1e-12
        )
        fallback = _assemble_candidate_from_support(
            candidate_id=f"{library.thermodynamic_slot}_fallback",
            library=library,
            support_indices=[best_idx],
            coeffs=c_phys,
            flow_columns_pt=flow_cols_pt,
            target_residual_pt=r_pt,
        )
        if fallback is not None:
            all_cands = [fallback]

    # Mark Pareto-optimal candidates (non-dominated in (total_complexity, vector_field_mse))
    final_all: list[SymbolicCandidate] = []
    pareto_list: list[SymbolicCandidate] = []
    for cand in all_cands:
        dominated = False
        for other in all_cands:
            if other.candidate_id == cand.candidate_id:
                continue
            if (
                other.total_complexity <= cand.total_complexity
                and other.vector_field_mse <= cand.vector_field_mse * (1.0 - 1e-5)
            ) or (
                other.total_complexity < cand.total_complexity
                and other.vector_field_mse <= cand.vector_field_mse * 1.02
            ):
                dominated = True
                break
        updated = SymbolicCandidate(
            candidate_id=cand.candidate_id,
            thermodynamic_slot=cand.thermodynamic_slot,
            patch_expressions=cand.patch_expressions,
            param_specs=cand.param_specs,
            active_term_ids=cand.active_term_ids,
            total_complexity=cand.total_complexity,
            num_parameters=cand.num_parameters,
            vector_field_mse=cand.vector_field_mse,
            vector_field_r2=cand.vector_field_r2,
            bic_score=cand.bic_score,
            is_pareto_optimal=not dominated,
        )
        final_all.append(updated)
        if not dominated:
            pareto_list.append(updated)

    pareto_list.sort(key=lambda c: c.bic_score)
    final_all.sort(key=lambda c: c.bic_score)

    return ParetoDiscoveryReport(
        pareto_front=tuple(pareto_list),
        all_candidates=tuple(final_all),
        best_candidate=pareto_list[0] if pareto_list else (final_all[0] if final_all else None),
    )


def _solve_sparse_ridge(
    a_norm: torch.Tensor,
    b_vec: torch.Tensor,
    non_neg_mask: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    return _solve_subspace_coefficients(a_norm, b_vec, non_neg_mask, ridge_lambda=ridge_lambda)


@dataclass(frozen=True)
class ConservationInvariant:
    """
    Symbolic first integral / conserved quantity `I(x) = sum_m w_m * psi_m(x) = const`
    discovered in the nullspace of the Lie derivative operator `grad(I(x)) . dx/dt == 0`.

    Attributes:
        invariant_id: Identifier for the discovered conserved quantity.
        expr_node: Symbolic `ExprNode` for `I(x)`.
        infix_expression: Human-readable infix equation string for `I(x)`.
        term_weights: Mapping from basis term string to physical coefficient `w_m`.
        relative_drift: Ratio `Std(I(x(t))) / sum_m(|w_m| * Std(psi_m(x(t))))`.
        lie_derivative_rms: RMS of `dI/dt = grad(I(x)) . dx/dt` along the trajectory.
    """

    invariant_id: str
    expr_node: ExprNode
    infix_expression: str
    term_weights: dict[str, float]
    relative_drift: float
    lie_derivative_rms: float


@dataclass(frozen=True)
class InvariantDiscoveryReport:
    """
    Report of all independent algebraic conservation laws `I_k(x) = const` discovered
    along an observed state trajectory.

    Attributes:
        invariants: Tuple of independent `ConservationInvariant`s ordered by `relative_drift`.
        num_invariants: Number of independent conserved quantities discovered.
        singular_values: Tuple of normalized Lie-derivative singular values.
    """

    invariants: tuple[ConservationInvariant, ...]
    num_invariants: int
    singular_values: tuple[float, ...]


def _rref_sparsify_nullspace(basis_vectors: torch.Tensor, pivot_tol: float = 1e-4) -> torch.Tensor:
    """
    Transform a `[K, M]` nullspace basis matrix into Reduced Row Echelon Form (RREF)
    so each row is a maximally sparse, canonical conservation law with leading coefficient 1.
    """
    mat = basis_vectors.clone().to(torch.float64)
    k_rows, m_cols = mat.shape
    pivot_row = 0

    for col in range(m_cols):
        if pivot_row >= k_rows:
            break
        sub_abs = torch.abs(mat[pivot_row:, col])
        max_rel = int(torch.argmax(sub_abs).item())
        max_val = float(sub_abs[max_rel].item())
        if max_val <= pivot_tol:
            continue

        swap_r = pivot_row + max_rel
        if swap_r != pivot_row:
            tmp = mat[pivot_row].clone()
            mat[pivot_row] = mat[swap_r]
            mat[swap_r] = tmp

        mat[pivot_row] = mat[pivot_row] / mat[pivot_row, col]
        for r in range(k_rows):
            if r != pivot_row and abs(float(mat[r, col].item())) > 1e-12:
                mat[r] = mat[r] - mat[r, col] * mat[pivot_row]
        pivot_row += 1

    mat[torch.abs(mat) < 1e-6] = 0.0
    return mat


def discover_conservation_invariants(
    brick: LawBrick,
    states: torch.Tensor,
    flows: torch.Tensor | None = None,
    params: torch.Tensor | None = None,
    max_degree: int = 4,
    nullspace_tol: float = 1e-3,
    include_cross_terms: bool = False,
    include_trigonometric: bool = False,
    include_logarithmic: bool = False,
) -> InvariantDiscoveryReport:
    """
    Discover exact symbolic first integrals / conserved quantities `I(x) = const` along
    `states` (`[T, D]` or `[E, T, D]`) by constructing the symbolic Lie derivative matrix
    `L_{t, m} = grad_x(psi_m(x(t))) . dx/dt(t)` and extracting sparse nullspace vectors
    via SVD + Gauss-Jordan RREF canonicalization.

    Supports polynomial powers, bilinear cross-products `x_i * x_j` (for angular momentum
    `q_x * p_y - q_y * p_x` and coupled oscillators), trigonometric terms `cos(x_i), sin(x_i)`
    (for nonlinear pendulums), and logarithmic terms `log(x_i)` (for Lotka-Volterra invariants).
    """
    x_in = states.detach().to(torch.float64)
    if x_in.ndim == 3:
        x = x_in.reshape(-1, x_in.shape[-1])
    else:
        x = x_in

    total_pts, state_dim = x.shape
    if flows is not None:
        dx_in = flows.detach().to(torch.float64)
        dx = dx_in.reshape(-1, state_dim) if dx_in.ndim == 3 else dx_in
    else:
        p = (
            params.detach().to(torch.float64)
            if params is not None
            else brick.default_params(batch_size=1, dtype=torch.float64, device=x.device)
        )
        if p.ndim == 1:
            p = p.unsqueeze(0)
        p_batch = p.expand(total_pts, -1) if p.shape[0] == 1 else p
        dx = brick.flow(x, p_batch).detach().to(torch.float64)

    state_symbols = [SymbolNode(sp.name, sp.unit, role="state") for sp in brick.state_specs]
    env = {sp.name: x[:, i] for i, sp in enumerate(brick.state_specs)}

    def _build_subspace_nodes(
        sub_max_deg: int,
        sub_cross: bool,
        sub_trig: bool,
        sub_log: bool,
    ) -> list[tuple[str, ExprNode]]:
        nodes: list[tuple[str, ExprNode]] = []
        for i, sym_i in enumerate(state_symbols):
            for deg in range(1, sub_max_deg + 1):
                if deg == 1:
                    node: ExprNode = sym_i
                    label = sym_i.name
                else:
                    node = BinaryOpNode(
                        "*",
                        ConstNode(1.0 / float(deg), DIMENSIONLESS),
                        BinaryOpNode("**", sym_i, ConstNode(float(deg), DIMENSIONLESS)),
                    )
                    label = f"(1/{deg})*{sym_i.name}^{deg}"
                nodes.append((label, node))

            arg_dimless: ExprNode = (
                sym_i
                if sym_i.unit.dimension.is_dimensionless
                else BinaryOpNode("*", ConstNode(1.0, DIMENSIONLESS / sym_i.unit), sym_i)
            )
            if sub_trig:
                nodes.append((f"cos({sym_i.name})", UnaryOpNode("cos", arg_dimless)))
                nodes.append((f"sin({sym_i.name})", UnaryOpNode("sin", arg_dimless)))

            if sub_log and bool((x[:, i] > 1e-4).all().item()):
                nodes.append((f"log({sym_i.name})", UnaryOpNode("log", arg_dimless)))

        if sub_cross and sub_max_deg >= 2:
            for i in range(state_dim):
                for j in range(i + 1, state_dim):
                    sym_i, sym_j = state_symbols[i], state_symbols[j]
                    nodes.append(
                        (f"{sym_i.name}*{sym_j.name}", BinaryOpNode("*", sym_i, sym_j))
                    )
        return nodes

    # Structured parsimony hierarchy so mixed polynomial + transcendental bases do not form
    # spurious 1D Taylor approximations that obscure sparse first integrals
    subspace_configs: list[tuple[int, bool, bool, bool]] = [
        (1, False, False, False),
    ]
    if max_degree >= 2 or include_cross_terms:
        subspace_configs.append((max_degree, include_cross_terms, False, False))
    if include_trigonometric:
        subspace_configs.append((min(max_degree, 2), False, True, False))
    if include_logarithmic:
        subspace_configs.append((1, False, False, True))
    full_cfg = (max_degree, include_cross_terms, include_trigonometric, include_logarithmic)
    if full_cfg not in subspace_configs:
        subspace_configs.append(full_cfg)

    invariants: list[ConservationInvariant] = []
    kept_term_signatures: list[set[str]] = []
    last_s_norm: tuple[float, ...] = ()

    for sub_deg, sub_cross, sub_trig, sub_log in subspace_configs:
        candidate_nodes = _build_subspace_nodes(sub_deg, sub_cross, sub_trig, sub_log)
        kept_labels: list[str] = []
        kept_nodes: list[ExprNode] = []
        val_cols: list[torch.Tensor] = []
        lie_cols: list[torch.Tensor] = []
        grad_mats: list[torch.Tensor] = []
        grad_norms: list[float] = []
        std_scales: list[float] = []

        for label, node in candidate_nodes:
            vals = node.eval_tensor(env)
            std_v = float(torch.std(vals).item())
            if std_v <= 1e-6 or not torch.isfinite(vals).all():
                continue

            grad_cols = [node.diff(sp.name).simplify().eval_tensor(env) for sp in brick.state_specs]
            grad_mat = torch.stack(grad_cols, dim=-1)  # [N, D]
            g_norm = float(torch.linalg.norm(grad_mat).item())
            if g_norm <= 1e-9:
                continue
            lie_deriv = torch.sum(grad_mat * dx, dim=-1)  # [N]

            kept_labels.append(label)
            kept_nodes.append(node)
            val_cols.append(vals)
            lie_cols.append(lie_deriv / std_v)
            grad_mats.append(grad_mat)
            grad_norms.append(g_norm)
            std_scales.append(std_v)

        if not lie_cols:
            continue

        l_mat = torch.stack(lie_cols, dim=-1)  # [N, M]
        val_mat = torch.stack(val_cols, dim=-1)  # [N, M]
        grad_tensor = torch.stack(grad_mats, dim=-1)  # [N, D, M]
        grad_norm_vec = torch.tensor(grad_norms, dtype=torch.float64, device=x.device)
        std_vec = torch.tensor(std_scales, dtype=torch.float64, device=x.device)

        _, s_vals, vh = torch.linalg.svd(l_mat, full_matrices=False)
        s_norm = s_vals / math.sqrt(max(total_pts, 1))
        last_s_norm = tuple(float(v) for v in s_norm.tolist())
        null_mask = s_norm <= nullspace_tol
        null_count = int(torch.sum(null_mask).item())
        if null_count == 0:
            continue

        null_vh_norm = vh[null_mask, :]  # [K, M]
        null_phys = null_vh_norm / std_vec.unsqueeze(0)
        sparse_phys = _rref_sparsify_nullspace(null_phys)

        for k_idx in range(sparse_phys.shape[0]):
            w_row = sparse_phys[k_idx]
            active_idx = torch.nonzero(torch.abs(w_row) > 1e-5, as_tuple=False).flatten().tolist()
            if not active_idx:
                continue

            lead_val = float(w_row[active_idx[0]].item())
            w_row = w_row / lead_val

            inv_vals = val_mat @ w_row
            inv_std = float(torch.std(inv_vals).item())
            weighted_term_std = float(torch.sum(torch.abs(w_row) * std_vec).item())
            rel_drift = inv_std / max(weighted_term_std, 1e-12)
            if rel_drift > nullspace_tol * 5.0:
                continue

            # Reject spurious 1D Taylor identities where grad_x(I(x)) vanishes identically
            inv_grad = torch.einsum("ndm,m->nd", grad_tensor, w_row)
            inv_grad_norm = float(torch.linalg.norm(inv_grad).item())
            weighted_grad_norm = float(torch.sum(torch.abs(w_row) * grad_norm_vec).item())
            grad_non_degeneracy = inv_grad_norm / max(weighted_grad_norm, 1e-12)
            if grad_non_degeneracy < 0.05:
                continue

            lie_phys = (l_mat * std_vec.unsqueeze(0)) @ w_row
            lie_rms = float(torch.sqrt(torch.mean(lie_phys * lie_phys)).item())

            term_weights: dict[str, float] = {}
            expr_parts: list[ExprNode] = []
            target_inv_unit = kept_nodes[active_idx[0]].infer_unit()
            for m_i in active_idx:
                c_val = float(w_row[m_i].item())
                if abs(c_val) <= 1e-5:
                    continue
                term_weights[kept_labels[m_i]] = c_val
                node_u = kept_nodes[m_i].infer_unit()
                coeff_u = target_inv_unit / node_u
                term_node = (
                    kept_nodes[m_i]
                    if (
                        abs(c_val - 1.0) < 1e-6
                        and coeff_u.dimension.is_dimensionless
                        and abs(coeff_u.scale - 1.0) < 1e-9
                    )
                    else BinaryOpNode("*", ConstNode(c_val, coeff_u), kept_nodes[m_i])
                )
                expr_parts.append(term_node)

            if not expr_parts:
                continue

            sig_set = set(term_weights.keys())
            if any(sig_set == prev_s or prev_s.issubset(sig_set) for prev_s in kept_term_signatures):
                continue

            combined_expr = expr_parts[0]
            for nxt in expr_parts[1:]:
                combined_expr = BinaryOpNode("+", combined_expr, nxt)
            simplified = combined_expr.simplify()

            kept_term_signatures.append(sig_set)
            invariants.append(
                ConservationInvariant(
                    invariant_id=f"invariant_{len(invariants)}",
                    expr_node=simplified,
                    infix_expression=simplified.to_infix(),
                    term_weights=term_weights,
                    relative_drift=rel_drift,
                    lie_derivative_rms=lie_rms,
                )
            )

    invariants.sort(key=lambda inv: (len(inv.term_weights), inv.relative_drift))
    return InvariantDiscoveryReport(
        invariants=tuple(invariants),
        num_invariants=len(invariants),
        singular_values=last_s_norm,
    )
