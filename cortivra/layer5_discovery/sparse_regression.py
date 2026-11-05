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
from cortivra.layer5_discovery.rational_pade_discovery import (
    SymbolicCandidate,
    discover_fractional_power_and_log_candidates,
    discover_rational_pade_candidates,
)
from cortivra.observability.flight_recorder import (
    AlternativeBranchRecord,
    OperatorSpectrumSummary,
    emit_trace_node,
    summarize_singular_spectrum,
)


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


def build_variational_bump_integral_system(
    flow_columns: torch.Tensor,
    target_residual: torch.Tensor,
    dt: float = 0.01,
    window_steps: int = 8,
    bump_degree: int = 4,
    num_experiments: int = 1,
    states: torch.Tensor | None = None,
    base_flow: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct a pure Weak-Form Variational Integral linear system `(G_weak, b_weak)` using
    compactly supported `C^p` bump test functions `phi(tau) = (1 - tau^2)^p` on `tau in [-1, 1]`.

    When `states` (`[E*T, StateDim]`) is provided, uses exact integration by parts:
        `integral phi(t) * (dx/dt - F_base(x)) dt = -integral (dphi/dt) * x(t) dt - integral phi(t) * F_base(x) dt`
    so the noisy state trajectory `x(t)` is NEVER differentiated!
    Respects `num_experiments` boundaries so windows never cross between distinct experiments.
    """
    total_pts, state_dim, num_terms = flow_columns.shape
    n_exp = max(int(num_experiments), 1)
    t_per_exp = total_pts // n_exp
    w = min(max(int(window_steps), 4), max(t_per_exp - 1, 1))
    if t_per_exp < w + 1 or w < 2:
        return (
            flow_columns.reshape(total_pts * state_dim, num_terms),
            target_residual.reshape(total_pts * state_dim),
        )

    p_deg = max(int(bump_degree), 2)
    tau = torch.linspace(-1.0, 1.0, w + 1, dtype=flow_columns.dtype, device=flow_columns.device)
    one_minus_tau2 = torch.clamp(1.0 - tau * tau, min=0.0)
    phi = one_minus_tau2 ** float(p_deg)
    half_len = 0.5 * float(w) * dt
    dphi_dt = (-2.0 * float(p_deg) * tau * (one_minus_tau2 ** float(p_deg - 1))) / max(half_len, 1e-12)

    trap_w = torch.ones((w + 1,), dtype=flow_columns.dtype, device=flow_columns.device)
    trap_w[0] = 0.5
    trap_w[-1] = 0.5
    norm_factor = max(float(torch.sum(trap_w * phi).item()), 1e-12)
    quad_phi = (trap_w * phi) / norm_factor
    quad_dphi = (trap_w * dphi_dt) / norm_factor

    g_rows: list[torch.Tensor] = []
    b_rows: list[torch.Tensor] = []
    stride = max(w // 4, 1)

    for exp_i in range(n_exp):
        base_offset = exp_i * t_per_exp
        for local_start in range(0, t_per_exp - w, stride):
            start = base_offset + local_start
            end = start + w + 1
            win_cols = torch.einsum("w,wsm->sm", quad_phi, flow_columns[start:end])
            if states is not None:
                x_win = states[start:end].to(flow_columns.dtype)
                rhs_ibp = -torch.einsum("w,ws->s", quad_dphi, x_win)
                if base_flow is not None:
                    f_win = base_flow[start:end].to(flow_columns.dtype)
                    rhs_ibp = rhs_ibp - torch.einsum("w,ws->s", quad_phi, f_win)
                win_targ = rhs_ibp
            else:
                win_targ = torch.einsum("w,ws->s", quad_phi, target_residual[start:end])
            g_rows.append(win_cols)
            b_rows.append(win_targ)

    if not g_rows:
        return (
            flow_columns.reshape(total_pts * state_dim, num_terms),
            target_residual.reshape(total_pts * state_dim),
        )
    return torch.cat(g_rows, dim=0), torch.cat(b_rows, dim=0)


def build_weak_integral_system(
    flow_columns: torch.Tensor,
    target_residual: torch.Tensor,
    dt: float = 0.01,
    window_steps: int = 4,
    integral_weight: float = 0.5,
    num_experiments: int = 1,
    states: torch.Tensor | None = None,
    base_flow: torch.Tensor | None = None,
    variational_bump_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct a combined pointwise, trapezoidal weak-integral, and variational bump-test
    linear system `(A, b)` from `flow_columns` (`[E*T, StateDim, NumTerms]`) and
    `target_residual` (`[E*T, StateDim]`).

    Respects `num_experiments` boundaries so sliding integral windows never cross between
    distinct trajectory experiments `[e * T, (e + 1) * T)`.
    """
    total_pts, state_dim, num_terms = flow_columns.shape
    a_pt = flow_columns.reshape(total_pts * state_dim, num_terms)
    b_pt = target_residual.reshape(total_pts * state_dim)

    n_exp = max(int(num_experiments), 1)
    t_per_exp = total_pts // n_exp

    if t_per_exp < window_steps + 2 or integral_weight <= 0.0 or window_steps < 2:
        return a_pt, b_pt

    int_a_rows: list[torch.Tensor] = []
    int_b_rows: list[torch.Tensor] = []

    trap_w = torch.ones((window_steps + 1,), dtype=flow_columns.dtype, device=flow_columns.device)
    trap_w[0] = 0.5
    trap_w[-1] = 0.5
    trap_w = trap_w / float(window_steps)

    for exp_i in range(n_exp):
        base_offset = exp_i * t_per_exp
        for local_start in range(t_per_exp - window_steps):
            start = base_offset + local_start
            end = start + window_steps + 1
            win_cols = torch.einsum("w,wsm->sm", trap_w, flow_columns[start:end])
            win_targ = torch.einsum("w,ws->s", trap_w, target_residual[start:end])
            int_a_rows.append(win_cols)
            int_b_rows.append(win_targ)

    a_int = torch.cat(int_a_rows, dim=0) * integral_weight
    b_int = torch.cat(int_b_rows, dim=0) * integral_weight
    a_out = torch.cat([a_pt, a_int], dim=0)
    b_out = torch.cat([b_pt, b_int], dim=0)

    if variational_bump_weight > 0.0 and t_per_exp >= 7:
        g_var, b_var = build_variational_bump_integral_system(
            flow_columns=flow_columns,
            target_residual=target_residual,
            dt=dt,
            window_steps=min(max(window_steps * 2, 6), t_per_exp - 1),
            bump_degree=4,
            num_experiments=n_exp,
            states=states,
            base_flow=base_flow,
        )
        a_out = torch.cat([a_out, g_var * variational_bump_weight], dim=0)
        b_out = torch.cat([b_out, b_var * variational_bump_weight], dim=0)

    return a_out, b_out


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


def solve_sparse_pareto_front(
    library: SymbolicBasisLibrary,
    target_residual: torch.Tensor,
    dt: float = 0.01,
    max_active_terms: int = 4,
    ridge_lambda: float = 1e-6,
    window_steps: int = 4,
    integral_weight: float = 0.5,
    extra_candidates: Sequence[SymbolicCandidate] = (),
    num_experiments: int = 1,
    states: torch.Tensor | None = None,
    base_flow: torch.Tensor | None = None,
    variational_bump_weight: float = 0.0,
) -> ParetoDiscoveryReport:
    """
    Discover the Pareto front of sparse symbolic equations from `library` that explain
    `target_residual` (`[E*T, StateDim]`).

    Combines:
        1. Dual pointwise + weak integral + C^4 variational bump collocation system construction
           with multi-experiment boundary protection (`num_experiments`).
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

    design_spectrum: OperatorSpectrumSummary | None = None
    if num_terms > 0:
        flow_cols_pt = torch.stack([t.flow_column.to(torch.float64) for t in library.terms], dim=-1)
        a_mat, b_vec = build_weak_integral_system(
            flow_cols_pt,
            r_pt,
            dt=dt,
            window_steps=window_steps,
            integral_weight=integral_weight,
            num_experiments=num_experiments,
            states=states,
            base_flow=base_flow,
            variational_bump_weight=variational_bump_weight,
        )

        col_norms = torch.clamp(torch.linalg.norm(a_mat, dim=0), min=1e-12)
        a_norm = a_mat / col_norms.unsqueeze(0)
        design_spectrum = summarize_singular_spectrum(
            a_norm,
            operator_name=f"WeakIntegralDesign({library.thermodynamic_slot})",
        )
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

        for s_idx in range(num_terms):
            discovered_supports.add((int(s_idx),))

        if k_limit >= 2 and num_terms >= 2:
            gram = a_norm.T @ a_norm  # [M, M]
            r_corr = a_norm.T @ b_vec  # [M]
            denom = torch.clamp(1.0 - gram * gram, min=1e-7)
            c_i_mat = (r_corr.unsqueeze(1) - gram * r_corr.unsqueeze(0)) / denom
            c_j_mat = (r_corr.unsqueeze(0) - gram * r_corr.unsqueeze(1)) / denom
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

        # Method B: Sequential Thresholded Ridge Regression (STRidge)
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

        const_idx_by_key: dict[int | tuple[int, int] | str, int] = {}
        if library.thermodynamic_slot == "passive_dissipative":
            for idx_t, t_obj in enumerate(library.terms):
                if t_obj.term_id.endswith("_const"):
                    const_idx_by_key[t_obj.target_key] = idx_t

        for idx_s, supp in enumerate(sorted(discovered_supports)):
            sub_list = list(supp)
            debias_indices: list[int] = []
            if library.thermodynamic_slot == "passive_dissipative":
                for s_i in sub_list:
                    t_key = library.terms[s_i].target_key
                    c_idx = const_idx_by_key.get(t_key)
                    if c_idx is not None and c_idx not in sub_list and c_idx not in debias_indices:
                        debias_indices.append(c_idx)

            if debias_indices and variational_bump_weight <= 0.0:
                aug_list = sub_list + debias_indices
                aug_non_neg = non_neg_all[aug_list].clone()
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
    best_cand = pareto_list[0] if pareto_list else (final_all[0] if final_all else None)

    if best_cand is not None:
        bic_margin = (
            float(final_all[1].bic_score - best_cand.bic_score)
            if len(final_all) >= 2
            else None
        )
        best_expr_str = "; ".join(
            f"{k}: {v.to_infix()}" for k, v in best_cand.patch_expressions.items()
        )
        alts = [
            AlternativeBranchRecord(
                branch_id=c.candidate_id,
                expression_or_summary="; ".join(f"{k}: {v.to_infix()}" for k, v in c.patch_expressions.items()),
                objective_score=round(float(c.bic_score), 4),
                local_r2=round(float(c.vector_field_r2), 6),
                complexity=c.total_complexity,
            )
            for c in final_all[1:5]
        ]
        emit_trace_node(
            layer="layer5_discovery",
            operation="solve_sparse_pareto_front",
            target_name=library.thermodynamic_slot,
            local_error_metric=best_cand.vector_field_mse,
            local_r2=best_cand.vector_field_r2,
            decision_margin=bic_margin,
            selected_branch_id=best_cand.candidate_id,
            selected_expression=best_expr_str,
            operator_spectrum=design_spectrum,
            alternatives=alts,
            metrics={
                "num_pareto_candidates": len(pareto_list),
                "num_total_candidates": len(final_all),
                "winning_bic": round(float(best_cand.bic_score), 4),
            },
        )

    return ParetoDiscoveryReport(
        pareto_front=tuple(pareto_list),
        all_candidates=tuple(final_all),
        best_candidate=best_cand,
    )


def _solve_sparse_ridge(
    a_norm: torch.Tensor,
    b_vec: torch.Tensor,
    non_neg_mask: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    return _solve_subspace_coefficients(a_norm, b_vec, non_neg_mask, ridge_lambda=ridge_lambda)


from cortivra.layer5_discovery.invariant_discovery import (
    ConservationInvariant,
    InvariantDiscoveryReport,
    _rref_sparsify_nullspace,
    discover_conservation_invariants,
)
