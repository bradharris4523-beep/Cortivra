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
from cortivra.observability.flight_recorder import (
    AlternativeBranchRecord,
    OperatorSpectrumSummary,
    emit_trace_node,
    summarize_singular_spectrum,
)



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
        x_lie = x
        dx_lie = dx
    else:
        p = (
            params.detach().to(torch.float64)
            if params is not None
            else brick.default_params(batch_size=1, dtype=torch.float64, device=x.device)
        )
        if p.ndim == 1:
            p = p.unsqueeze(0)
        p_single = p[:1]
        p_batch = p_single.expand(total_pts, -1) if p.shape[0] == 1 else p
        dx = brick.flow(x, p_batch).detach().to(torch.float64)

        # Augment Lie-derivative operator evaluation with orthogonal phase-space perturbations
        # around the trajectory so single-curve parameterizations on dissipative attractors
        # are distinguished from true phase-space first integrals (grad I(x) . f(x) == 0).
        if state_dim >= 2 and total_pts >= 4 and float(torch.linalg.norm(dx).item()) > 1e-9:
            anchor_indices = [0, total_pts // 3, (2 * total_pts) // 3, total_pts - 1]
            anchors = x[anchor_indices]  # [4, D]
            coord_stds = torch.clamp(torch.std(x, dim=0), min=0.08)
            probe_rows: list[torch.Tensor] = []
            for d_i in range(state_dim):
                for sign in (-0.35, 0.35):
                    pert = anchors.clone()
                    pert[:, d_i] = pert[:, d_i] + sign * coord_stds[d_i]
                    if bool((x[:, d_i] > 1e-3).all().item()):
                        min_pos = float(torch.min(x[:, d_i]).item()) * 0.25
                        pert[:, d_i] = torch.clamp(pert[:, d_i], min=max(min_pos, 1e-3))
                    probe_rows.append(brick.clamp_to_hard_bounds(pert))
            x_probe = torch.cat(probe_rows, dim=0)
            x_lie = torch.cat([x, x_probe], dim=0)
            p_lie = p_single.expand(x_lie.shape[0], -1)
            dx_lie = brick.flow(x_lie, p_lie).detach().to(torch.float64)
        else:
            x_lie = x
            dx_lie = dx

    lie_pts = int(x_lie.shape[0])
    state_symbols = [SymbolNode(sp.name, sp.unit, role="state") for sp in brick.state_specs]
    env = {sp.name: x[:, i] for i, sp in enumerate(brick.state_specs)}
    env_lie = {sp.name: x_lie[:, i] for i, sp in enumerate(brick.state_specs)}

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

            grad_cols = [node.diff(sp.name).simplify().eval_tensor(env_lie) for sp in brick.state_specs]
            grad_mat = torch.stack(grad_cols, dim=-1)  # [N_lie, D]
            g_norm = float(torch.linalg.norm(grad_mat).item())
            if g_norm <= 1e-9 or not torch.isfinite(grad_mat).all():
                continue
            lie_deriv = torch.sum(grad_mat * dx_lie, dim=-1)  # [N_lie]

            kept_labels.append(label)
            kept_nodes.append(node)
            val_cols.append(vals)
            lie_cols.append(lie_deriv / std_v)
            grad_mats.append(grad_mat)
            grad_norms.append(g_norm)
            std_scales.append(std_v)

        if not lie_cols:
            continue

        l_mat = torch.stack(lie_cols, dim=-1)  # [N_lie, M]
        val_mat = torch.stack(val_cols, dim=-1)  # [N, M]
        grad_tensor = torch.stack(grad_mats, dim=-1)  # [N_lie, D, M]
        grad_norm_vec = torch.tensor(grad_norms, dtype=torch.float64, device=x.device)
        std_vec = torch.tensor(std_scales, dtype=torch.float64, device=x.device)

        _, s_vals, vh = torch.linalg.svd(l_mat, full_matrices=False)
        s_norm = s_vals / math.sqrt(max(lie_pts, 1))
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
    if last_s_norm:
        lie_spec = summarize_singular_spectrum(
            torch.tensor(last_s_norm, dtype=torch.float64),
            operator_name=f"LieDerivative({brick.name})",
            null_abs_tol=nullspace_tol,
        )
    else:
        lie_spec = None

    emit_trace_node(
        layer="layer5_discovery",
        operation="discover_conservation_invariants",
        target_name=brick.name,
        selected_expression=", ".join(inv.infix_expression for inv in invariants) or "none",
        operator_spectrum=lie_spec,
        metrics={
            "num_invariants": len(invariants),
            "invariants": [inv.infix_expression for inv in invariants],
        },
    )
    return InvariantDiscoveryReport(
        invariants=tuple(invariants),
        num_invariants=len(invariants),
        singular_values=last_s_norm,
    )
