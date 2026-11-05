"""
Sanathanan-Koerner rational Padé, cooperative Hill/Michaelis-Menten, and continuous
fractional power-law / logarithmic symbolic discovery (`cortivra.layer5_discovery.rational_pade_discovery`).
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
from cortivra.layer5_discovery.basis_library import estimate_characteristic_scales
from cortivra.observability.flight_recorder import (
    AlternativeBranchRecord,
    emit_trace_node,
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
            best_univariate_mse = float("inf")

            for n_exp in (1, 2, 3, 4):
                z = xj_abs ** float(n_exp)  # x_j^n
                # Model 1 (Activating Hill / Michaelis-Menten): r = (a * z) / (1 + b * z)
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
                    best_univariate_mse = min(best_univariate_mse, mse)
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

            # Model 3 (General Padé [1/2] Substrate-Inhibition / Bell-Shaped Quotient)
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

            # Model 4 (Multivariate Competitive / Allosteric Padé [1/(1,1)] Quotient):
            #   f(x_j, x_k) = (a_1 * x_j) / (1 + b_1 * x_j + b_2 * x_k) with b_1, b_2 >= 0
            if state_dim >= 2:
                for k, sp_mod in enumerate(brick.state_specs):
                    if k == j:
                        continue
                    xk_abs = torch.abs(x[:, k])
                    if float(torch.max(xk_abs).item() - torch.min(xk_abs).item()) <= 1e-6:
                        continue
                    sym_k = SymbolNode(sp_mod.name, sp_mod.unit, role="state")
                    w_comp = torch.ones((t_steps,), dtype=torch.float64, device=x.device)
                    ac_best = 0.0
                    bc1_best = 1.0 / max(scales[sp_state.name], 1e-6)
                    bc2_best = 1.0 / max(scales[sp_mod.name], 1e-6)
                    for _ in range(num_sk_iterations):
                        wc_k = w_comp.unsqueeze(-1)
                        des_comp = torch.stack([x1, -r_col * x1, -r_col * xk_abs], dim=-1) * wc_k
                        rhs_comp = r_col * w_comp
                        sol_comp = torch.linalg.lstsq(
                            des_comp, rhs_comp.unsqueeze(-1), driver="gelsd"
                        ).solution.squeeze(-1)
                        ac_e = float(sol_comp[0].item())
                        bc1_e = max(float(sol_comp[1].item()), 0.0)
                        bc2_e = max(float(sol_comp[2].item()), 0.0)
                        if bc1_e + bc2_e <= 1e-9:
                            bc1_e = 1.0 / max(scales[sp_state.name], 1e-6)
                        denom_comp = torch.clamp(1.0 + bc1_e * x1 + bc2_e * xk_abs, min=1e-6)
                        w_comp = 1.0 / denom_comp
                        ac_best, bc1_best, bc2_best = ac_e, bc1_e, bc2_e

                    if abs(ac_best) > 1e-8 and bc1_best > 1e-7 and bc2_best > 1e-5:
                        pred_comp_col = (ac_best * x1) / (1.0 + bc1_best * x1 + bc2_best * xk_abs)
                        pred_comp_full = torch.zeros_like(r)
                        pred_comp_full[:, idx] = pred_comp_col
                        mse_comp = float(torch.mean((r - pred_comp_full) ** 2).item())
                        r2_comp = 1.0 - (mse_comp / max(y_var, 1e-18))
                        # Require genuine improvement over univariate rational models
                        if r2_comp > 0.75 and mse_comp < 0.70 * best_univariate_mse:
                            ac_u = flow_unit / sp_state.unit
                            bc1_u = DIMENSIONLESS / sp_state.unit
                            bc2_u = DIMENSIONLESS / sp_mod.unit
                            p_ac = ParamSpec(
                                name=f"a1_comp_{idx}_{sp_state.name}_{sp_mod.name}",
                                unit=ac_u,
                                default_value=ac_best,
                                min_value=-abs(ac_best) * 10.0 - 10.0,
                                max_value=abs(ac_best) * 10.0 + 10.0,
                            )
                            p_bc1 = ParamSpec(
                                name=f"b1_comp_{idx}_{sp_state.name}_{sp_mod.name}",
                                unit=bc1_u,
                                default_value=bc1_best,
                                min_value=0.0,
                                max_value=max(bc1_best * 20.0 + 10.0, 10.0),
                            )
                            p_bc2 = ParamSpec(
                                name=f"b2_comp_{idx}_{sp_state.name}_{sp_mod.name}",
                                unit=bc2_u,
                                default_value=bc2_best,
                                min_value=1e-8,
                                max_value=max(bc2_best * 20.0 + 10.0, 10.0),
                            )
                            num_c_node = BinaryOpNode("*", SymbolNode(p_ac.name, ac_u, role="param"), sym_j)
                            den_c_node = BinaryOpNode(
                                "+",
                                BinaryOpNode(
                                    "+",
                                    ConstNode(1.0, DIMENSIONLESS),
                                    BinaryOpNode("*", SymbolNode(p_bc1.name, bc1_u, role="param"), sym_j),
                                ),
                                BinaryOpNode("*", SymbolNode(p_bc2.name, bc2_u, role="param"), sym_k),
                            )
                            comp_node = BinaryOpNode("/", num_c_node, den_c_node)
                            comp_c = comp_node.tree_complexity()
                            k_eff_c = 3.5 + 0.30 * float(comp_c)
                            bic_c = float(n_obs) * math.log(max(mse_comp, 1e-24)) + k_eff_c * math.log(max(n_obs, 2))
                            discovered.append(
                                SymbolicCandidate(
                                    candidate_id=f"pade_comp_{idx}_{sp_state.name}_{sp_mod.name}",
                                    thermodynamic_slot="active_driven",
                                    patch_expressions={idx: comp_node},
                                    param_specs=(p_ac, p_bc1, p_bc2),
                                    active_term_ids=(f"pade_comp_{idx}_{sp_state.name}_{sp_mod.name}",),
                                    total_complexity=comp_c,
                                    num_parameters=3,
                                    vector_field_mse=mse_comp,
                                    vector_field_r2=r2_comp,
                                    bic_score=bic_c,
                                    is_pareto_optimal=True,
                                )
                            )

    if discovered:
        discovered.sort(key=lambda c: c.bic_score)
        best_pade = discovered[0]
        pade_margin = (
            float(discovered[1].bic_score - best_pade.bic_score)
            if len(discovered) >= 2
            else None
        )
        pade_expr = "; ".join(f"{k}: {v.to_infix()}" for k, v in best_pade.patch_expressions.items())
        pade_alts = [
            AlternativeBranchRecord(
                branch_id=c.candidate_id,
                expression_or_summary="; ".join(f"{k}: {v.to_infix()}" for k, v in c.patch_expressions.items()),
                objective_score=round(float(c.bic_score), 4),
                local_r2=round(float(c.vector_field_r2), 6),
                complexity=c.total_complexity,
            )
            for c in discovered[1:4]
        ]
        emit_trace_node(
            layer="layer5_discovery",
            operation="discover_rational_pade_candidates",
            target_name=brick.name,
            local_error_metric=best_pade.vector_field_mse,
            local_r2=best_pade.vector_field_r2 if best_pade.vector_field_r2 >= 0.90 else None,
            decision_margin=pade_margin,
            selected_branch_id=best_pade.candidate_id,
            selected_expression=pade_expr,
            alternatives=pade_alts,
            metrics={
                "num_pade_candidates": len(discovered),
                "best_pade_r2": round(float(best_pade.vector_field_r2), 6),
            },
            skip_if_inner_loop=True,
        )
    return discovered


def discover_fractional_power_and_log_candidates(
    brick: LawBrick,
    states: torch.Tensor,
    target_residual: torch.Tensor,
    affected_state_indices: Sequence[int] | None = None,
) -> list[SymbolicCandidate]:
    """
    Discover continuous non-integer fractional power-law mechanisms (`c * x_j^alpha` and
    allometric `c_1 * x_j^alpha + c_2 * x_j` with `alpha not in Z`) and logarithmic
    creep/relaxation mechanisms (`c * log(1 + x_j)`) via Variable Projection (VarPro)
    and Golden-Section Profile-Likelihood optimization.
    """
    x = states.detach().to(torch.float64)
    r = target_residual.detach().to(torch.float64)
    t_steps, state_dim = x.shape
    if t_steps < 8:
        return []
    n_obs = max(t_steps * state_dim, 1)
    y_var = float(torch.sum((r - torch.mean(r)) ** 2).item()) / float(n_obs)
    if y_var <= 1e-18:
        return []

    target_indices = (
        list(affected_state_indices)
        if affected_state_indices
        else list(range(state_dim))
    )
    discovered: list[SymbolicCandidate] = []
    golden_phi = 0.5 * (math.sqrt(5.0) - 1.0)

    for idx in target_indices:
        r_col = r[:, idx]
        if float(torch.linalg.norm(r_col).item()) <= 1e-9:
            continue
        sp_target = brick.state_specs[idx]
        flow_unit = sp_target.unit / brick.time_unit

        for j, sp_state in enumerate(brick.state_specs):
            xj_raw = x[:, j]
            if float(torch.min(xj_raw).item()) < -1e-6:
                continue
            xj = torch.clamp(xj_raw, min=1e-12)
            if float(torch.max(xj).item() - torch.min(xj).item()) <= 1e-4:
                continue
            sym_j = SymbolNode(sp_state.name, sp_state.unit, role="state")

            best_1term_int_mse = float("inf")
            for deg_int in (1, 2, 3):
                z_int = xj ** float(deg_int)
                c_int = float(torch.dot(z_int, r_col).item()) / max(float(torch.dot(z_int, z_int).item()), 1e-18)
                p_int = torch.zeros_like(r)
                p_int[:, idx] = c_int * z_int
                m_int = float(torch.mean((r - p_int) ** 2).item())
                best_1term_int_mse = min(best_1term_int_mse, m_int)

            des_int3 = torch.stack([xj, xj ** 2, xj ** 3], dim=-1)
            sol_int3 = torch.linalg.lstsq(des_int3, r_col.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)
            p_int3 = torch.zeros_like(r)
            p_int3[:, idx] = des_int3 @ sol_int3
            int_poly3_mse = float(torch.mean((r - p_int3) ** 2).item())

            def _eval_1term(alpha_val: float) -> tuple[float, float]:
                z_a = xj ** alpha_val
                zz = float(torch.dot(z_a, z_a).item())
                if zz <= 1e-18:
                    return float("inf"), 0.0
                c_a = float(torch.dot(z_a, r_col).item()) / zz
                err_col = r_col - c_a * z_a
                mse_a = float(torch.sum(err_col * err_col).item()) / float(n_obs)
                return mse_a, c_a

            coarse_alphas = [0.15 + 0.08 * k for k in range(46)]
            best_a1 = 1.0
            best_mse1 = float("inf")
            for a_try in coarse_alphas:
                m_try, _ = _eval_1term(a_try)
                if m_try < best_mse1:
                    best_mse1, best_a1 = m_try, a_try

            lo_a = max(0.10, best_a1 - 0.12)
            hi_a = min(3.90, best_a1 + 0.12)
            x1_g = hi_a - golden_phi * (hi_a - lo_a)
            x2_g = lo_a + golden_phi * (hi_a - lo_a)
            f1_g, _ = _eval_1term(x1_g)
            f2_g, _ = _eval_1term(x2_g)
            for _ in range(20):
                if f1_g < f2_g:
                    hi_a = x2_g
                    x2_g = x1_g
                    f2_g = f1_g
                    x1_g = hi_a - golden_phi * (hi_a - lo_a)
                    f1_g, _ = _eval_1term(x1_g)
                else:
                    lo_a = x1_g
                    x1_g = x2_g
                    f1_g = f2_g
                    x2_g = lo_a + golden_phi * (hi_a - lo_a)
                    f2_g, _ = _eval_1term(x2_g)

            alpha_star = round(0.5 * (lo_a + hi_a), 4)
            for canonical_rat in (0.25, 0.3333, 0.5, 0.6667, 0.75, 1.25, 1.3333, 1.5, 1.6667, 1.75, 2.25, 2.5):
                if abs(alpha_star - canonical_rat) <= 0.0035:
                    alpha_star = canonical_rat
                    break
            mse_star, c_star = _eval_1term(alpha_star)
            r2_star = 1.0 - (mse_star / max(y_var, 1e-18))

            if (
                abs(alpha_star - round(alpha_star)) >= 0.06
                and abs(c_star) > 1e-7
                and r2_star >= 0.995
                and mse_star < 0.12 * best_1term_int_mse
            ):
                c_unit = flow_unit / (sp_state.unit ** alpha_star)
                a_tag = str(alpha_star).replace(".", "p")
                p_c = ParamSpec(
                    name=f"c_fpow_{idx}_{sp_state.name}_{a_tag}",
                    unit=c_unit,
                    default_value=c_star,
                    min_value=min(-abs(c_star) * 10.0 - 10.0, c_star * 10.0),
                    max_value=max(abs(c_star) * 10.0 + 10.0, 10.0),
                )
                fpow_node = BinaryOpNode(
                    "*",
                    SymbolNode(p_c.name, c_unit, role="param"),
                    BinaryOpNode("**", sym_j, ConstNode(float(alpha_star), DIMENSIONLESS)),
                )
                comp_f = fpow_node.tree_complexity()
                k_eff_f = 1.5 + 0.25 * float(comp_f)
                bic_f = float(n_obs) * math.log(max(mse_star, 1e-24)) + k_eff_f * math.log(max(n_obs, 2))
                discovered.append(
                    SymbolicCandidate(
                        candidate_id=f"frac_pow_{idx}_{sp_state.name}_{a_tag}",
                        thermodynamic_slot="active_driven",
                        patch_expressions={idx: fpow_node},
                        param_specs=(p_c,),
                        active_term_ids=(f"frac_pow_{idx}_{sp_state.name}_{a_tag}",),
                        total_complexity=comp_f,
                        num_parameters=1,
                        vector_field_mse=mse_star,
                        vector_field_r2=r2_star,
                        bic_score=bic_f,
                        is_pareto_optimal=True,
                    )
                )

            def _eval_2term(alpha_val: float) -> tuple[float, float, float]:
                if abs(alpha_val - 1.0) < 0.06:
                    return float("inf"), 0.0, 0.0
                z_a = xj ** alpha_val
                des_2 = torch.stack([z_a, xj], dim=-1)
                sol_2 = torch.linalg.lstsq(des_2, r_col.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)
                c1_v = float(sol_2[0].item())
                c2_v = float(sol_2[1].item())
                err_2 = r_col - (c1_v * z_a + c2_v * xj)
                mse_2 = float(torch.sum(err_2 * err_2).item()) / float(n_obs)
                return mse_2, c1_v, c2_v

            best_a2 = 0.75
            best_mse2 = float("inf")
            for a_try in coarse_alphas:
                if abs(a_try - round(a_try)) < 0.06:
                    continue
                m2_try, _, _ = _eval_2term(a_try)
                if m2_try < best_mse2:
                    best_mse2, best_a2 = m2_try, a_try

            lo2 = max(0.15, best_a2 - 0.10)
            hi2 = min(3.60, best_a2 + 0.10)
            g1 = hi2 - golden_phi * (hi2 - lo2)
            g2 = lo2 + golden_phi * (hi2 - lo2)
            v1, _, _ = _eval_2term(g1)
            v2, _, _ = _eval_2term(g2)
            for _ in range(18):
                if v1 < v2:
                    hi2 = g2
                    g2 = g1
                    v2 = v1
                    g1 = hi2 - golden_phi * (hi2 - lo2)
                    v1, _, _ = _eval_2term(g1)
                else:
                    lo2 = g1
                    g1 = g2
                    v1 = v2
                    g2 = lo2 + golden_phi * (hi2 - lo2)
                    v2, _, _ = _eval_2term(g2)

            alpha2_star = round(0.5 * (lo2 + hi2), 4)
            for canonical_rat in (0.25, 0.3333, 0.5, 0.6667, 0.75, 1.25, 1.3333, 1.5, 1.6667, 1.75, 2.5):
                if abs(alpha2_star - canonical_rat) <= 0.004:
                    alpha2_star = canonical_rat
                    break
            mse2_star, c1_star, c2_star = _eval_2term(alpha2_star)
            r2_2star = 1.0 - (mse2_star / max(y_var, 1e-18))

            if (
                abs(alpha2_star - round(alpha2_star)) >= 0.06
                and abs(c1_star) > 1e-6
                and abs(c2_star) > 1e-6
                and r2_2star >= 0.9985
                and mse2_star < 0.15 * int_poly3_mse
            ):
                u_c1 = flow_unit / (sp_state.unit ** alpha2_star)
                u_c2 = flow_unit / sp_state.unit
                a2_tag = str(alpha2_star).replace(".", "p")
                p_c1 = ParamSpec(
                    name=f"c1_allom_{idx}_{sp_state.name}_{a2_tag}",
                    unit=u_c1,
                    default_value=c1_star,
                    min_value=min(-abs(c1_star) * 10.0 - 10.0, c1_star * 10.0),
                    max_value=max(abs(c1_star) * 10.0 + 10.0, 10.0),
                )
                p_c2 = ParamSpec(
                    name=f"c2_allom_{idx}_{sp_state.name}_lin",
                    unit=u_c2,
                    default_value=c2_star,
                    min_value=min(-abs(c2_star) * 10.0 - 10.0, c2_star * 10.0),
                    max_value=max(abs(c2_star) * 10.0 + 10.0, 10.0),
                )
                term_pow = BinaryOpNode(
                    "*",
                    SymbolNode(p_c1.name, u_c1, role="param"),
                    BinaryOpNode("**", sym_j, ConstNode(float(alpha2_star), DIMENSIONLESS)),
                )
                term_lin = BinaryOpNode("*", SymbolNode(p_c2.name, u_c2, role="param"), sym_j)
                allom_node = BinaryOpNode("+", term_pow, term_lin).simplify()
                comp_a2 = allom_node.tree_complexity()
                k_eff_a2 = 2.5 + 0.25 * float(comp_a2)
                bic_a2 = float(n_obs) * math.log(max(mse2_star, 1e-24)) + k_eff_a2 * math.log(max(n_obs, 2))
                discovered.append(
                    SymbolicCandidate(
                        candidate_id=f"allometric_{idx}_{sp_state.name}_{a2_tag}",
                        thermodynamic_slot="active_driven",
                        patch_expressions={idx: allom_node},
                        param_specs=(p_c1, p_c2),
                        active_term_ids=(f"allometric_{idx}_{sp_state.name}_{a2_tag}",),
                        total_complexity=comp_a2,
                        num_parameters=2,
                        vector_field_mse=mse2_star,
                        vector_field_r2=r2_2star,
                        bic_score=bic_a2,
                        is_pareto_optimal=True,
                    )
                )

            if sp_state.unit.dimension.is_dimensionless:
                z_log = torch.log(1.0 + xj)
                zz_l = float(torch.dot(z_log, z_log).item())
                if zz_l > 1e-16:
                    c_log = float(torch.dot(z_log, r_col).item()) / zz_l
                    err_l = r_col - c_log * z_log
                    mse_l = float(torch.sum(err_l * err_l).item()) / float(n_obs)
                    r2_l = 1.0 - (mse_l / max(y_var, 1e-18))
                    if abs(c_log) > 1e-7 and r2_l >= 0.9985 and mse_l < 0.15 * best_1term_int_mse:
                        p_clog = ParamSpec(
                            name=f"c_log_{idx}_{sp_state.name}",
                            unit=flow_unit,
                            default_value=c_log,
                            min_value=min(-abs(c_log) * 10.0 - 10.0, c_log * 10.0),
                            max_value=max(abs(c_log) * 10.0 + 10.0, 10.0),
                        )
                        log_node = BinaryOpNode(
                            "*",
                            SymbolNode(p_clog.name, flow_unit, role="param"),
                            UnaryOpNode("log", BinaryOpNode("+", ConstNode(1.0, DIMENSIONLESS), sym_j)),
                        )
                        comp_l = log_node.tree_complexity()
                        k_eff_l = 1.5 + 0.25 * float(comp_l)
                        bic_l = float(n_obs) * math.log(max(mse_l, 1e-24)) + k_eff_l * math.log(max(n_obs, 2))
                        discovered.append(
                            SymbolicCandidate(
                                candidate_id=f"log_creep_{idx}_{sp_state.name}",
                                thermodynamic_slot="active_driven",
                                patch_expressions={idx: log_node},
                                param_specs=(p_clog,),
                                active_term_ids=(f"log_creep_{idx}_{sp_state.name}",),
                                total_complexity=comp_l,
                                num_parameters=1,
                                vector_field_mse=mse_l,
                                vector_field_r2=r2_l,
                                bic_score=bic_l,
                                is_pareto_optimal=True,
                            )
                        )

    return discovered
