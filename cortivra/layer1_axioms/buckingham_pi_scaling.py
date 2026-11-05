"""
Buckingham-Pi dimensionless group discovery and canonical non-dimensional scaling
(`cortivra.layer1_axioms.buckingham_pi_scaling`).

Extracted from `cortivra.layer1_axioms.units` to keep core SI dimensional algebra and
Buckingham-Pi matrix nullspace analysis cleanly separated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from cortivra.layer1_axioms.units import (
    DIMENSIONLESS,
    Dimension,
    Quantity,
    Unit,
    _clean_exp,
)
from cortivra.observability.flight_recorder import (
    emit_trace_node,
    summarize_singular_spectrum,
)


@dataclass(frozen=True)
class DimensionlessGroup:
    """Dimensionless Pi group `prod_i (var_i ^ exponent_i)` in the nullspace of the SI dimension matrix."""

    variable_names: tuple[str, ...]
    exponents: tuple[float, ...]

    def format_expression(self) -> str:
        num_parts: list[str] = []
        den_parts: list[str] = []
        for name, exp in zip(self.variable_names, self.exponents):
            if abs(exp) < 1e-6:
                continue
            abs_exp = abs(exp)
            exp_str = (
                ""
                if abs(abs_exp - 1.0) < 1e-6
                else (
                    f"^{int(round(abs_exp))}"
                    if abs(abs_exp - round(abs_exp)) < 1e-5
                    else f"^{abs_exp:.2g}"
                )
            )
            term = f"{name}{exp_str}"
            if exp > 0:
                num_parts.append(term)
            else:
                den_parts.append(term)
        num = " * ".join(num_parts) if num_parts else "1"
        if not den_parts:
            return num
        den = " * ".join(den_parts)
        return f"({num}) / ({den})"


def discover_dimensionless_groups(
    variables: Sequence[tuple[str, Unit | Dimension | Quantity]],
    tol: float = 1e-9,
) -> list[DimensionlessGroup]:
    """
    Compute the basis of dimensionless groups for `variables` via the right nullspace
    of the 7 x k SI dimensional exponent matrix `D` (`D * alpha = 0`).
    """
    if not variables:
        return []

    names = tuple(v[0] for v in variables)
    if len(set(names)) != len(names):
        raise ValueError("Variable names passed to discover_dimensionless_groups must be unique.")

    dims: list[Dimension] = []
    for _, obj in variables:
        if isinstance(obj, Dimension):
            dims.append(obj)
        elif isinstance(obj, (Unit, Quantity)):
            dims.append(obj.dimension)
        else:
            raise TypeError(f"Expected Unit, Dimension, or Quantity, got {type(obj).__name__}.")

    k = len(dims)
    d_mat = np.column_stack([d.as_vector for d in dims])  # (7, k)

    _, s_vals, vt = np.linalg.svd(d_mat, full_matrices=True)
    rank = int(np.sum(s_vals > tol))
    null_dim = k - rank
    if null_dim <= 0:
        return []

    null_basis = vt[rank:, :].copy()  # (null_dim, k)

    pivot_row = 0
    for col in range(k):
        if pivot_row >= null_dim:
            break
        max_row = pivot_row + int(np.argmax(np.abs(null_basis[pivot_row:, col])))
        if abs(null_basis[max_row, col]) <= 1e-7:
            continue
        null_basis[[pivot_row, max_row]] = null_basis[[max_row, pivot_row]]
        null_basis[pivot_row] = null_basis[pivot_row] / null_basis[pivot_row, col]
        for r in range(null_dim):
            if r != pivot_row and abs(null_basis[r, col]) > 1e-7:
                null_basis[r] = null_basis[r] - null_basis[r, col] * null_basis[pivot_row]
        pivot_row += 1

    groups: list[DimensionlessGroup] = []
    for row in null_basis:
        cleaned = tuple(_clean_exp(float(val)) for val in row)
        residual = np.max(np.abs(d_mat @ np.array(cleaned, dtype=np.float64)))
        if residual <= 1e-5 and any(abs(x) > 1e-6 for x in cleaned):
            groups.append(DimensionlessGroup(variable_names=names, exponents=cleaned))

    d_spec = summarize_singular_spectrum(
        torch.as_tensor(d_mat, dtype=torch.float64),
        operator_name=f"SIDimensionalMatrix(7x{k})",
        null_abs_tol=tol,
    )
    emit_trace_node(
        layer="layer1_axioms",
        operation="discover_dimensionless_groups",
        target_name=",".join(names[:4]) + ("..." if len(names) > 4 else ""),
        selected_expression="; ".join(g.format_expression() for g in groups) if groups else "none",
        operator_spectrum=d_spec,
        metrics={
            "num_variables": k,
            "dimensional_rank": rank,
            "num_pi_groups": len(groups),
            "pi_groups": [g.format_expression() for g in groups],
        },
        skip_if_inner_loop=True,
    )
    return groups


@dataclass(frozen=True)
class CanonicalNonDimensionalScaling:
    """
    Canonical Buckingham-Pi non-dimensionalization and timescale-separation analysis
    for any dynamical system (`compute_canonical_nondimensional_scaling`).
    """

    characteristic_state_scales: dict[str, float]
    characteristic_time_scale_s: float
    fastest_timescale_s: float
    slowest_timescale_s: float
    timescale_separation_epsilon: float
    dimensionless_pi_groups: tuple[DimensionlessGroup, ...]
    evaluated_pi_values: dict[str, float]
    scale_invariance_max_drift: float
    scale_invariance_verified: bool


def compute_canonical_nondimensional_scaling(
    state_specs: Sequence[object],
    param_specs: Sequence[object],
    time_unit: Unit,
    sample_states: torch.Tensor | None = None,
    sample_params: torch.Tensor | None = None,
) -> CanonicalNonDimensionalScaling:
    """
    Project a dynamical system's states and parameters onto canonical dimensionless
    Buckingham-Pi coordinates, extract the stiff timescale-separation ratio
    `epsilon = tau_fast / tau_slow`, and verify exact invariance under arbitrary
    rescalings of the 7 base SI dimensions.
    """
    state_scales: dict[str, float] = {}
    s_flat = (
        sample_states.detach().to(torch.float64)
        if sample_states is not None and sample_states.numel() > 0
        else None
    )
    for idx_s, sp in enumerate(state_specs):
        nm = str(getattr(sp, "name", f"x_{idx_s}"))
        def_v = abs(float(getattr(sp, "default_value", 1.0)))
        h_min = float(getattr(sp, "homeostatic_min", -10.0))
        h_max = float(getattr(sp, "homeostatic_max", 10.0))
        span_ref = 0.5 * max(abs(h_max - h_min), 1e-6)
        emp_ref = 0.0
        if s_flat is not None:
            if s_flat.ndim >= 2 and idx_s < s_flat.shape[-1]:
                col = torch.abs(s_flat[..., idx_s])
                emp_ref = float(torch.max(col).item())
            elif s_flat.ndim == 1 and idx_s < s_flat.numel():
                emp_ref = abs(float(s_flat[idx_s].item()))
        state_scales[nm] = max(emp_ref, def_v, span_ref, 1e-6)

    p_flat = (
        sample_params.detach().to(torch.float64).flatten()
        if sample_params is not None and sample_params.numel() > 0
        else None
    )
    param_vals: dict[str, float] = {}
    timescales: list[float] = []
    variables_for_pi: list[tuple[str, Unit]] = []
    seen_names: set[str] = set()

    for sp in state_specs:
        nm = str(getattr(sp, "name", "x"))
        u_sp = getattr(sp, "unit", DIMENSIONLESS)
        if nm not in seen_names and isinstance(u_sp, Unit):
            variables_for_pi.append((nm, u_sp))
            seen_names.add(nm)

    inv_t_dim = time_unit.dimension ** -1
    inv_t2_dim = time_unit.dimension ** -2

    for idx_p, pp in enumerate(param_specs):
        p_nm = str(getattr(pp, "name", f"p_{idx_p}"))
        u_pp = getattr(pp, "unit", DIMENSIONLESS)
        val_p = (
            float(p_flat[idx_p].item())
            if (p_flat is not None and idx_p < p_flat.numel())
            else float(getattr(pp, "default_value", 1.0))
        )
        mag_p = max(abs(val_p), 1e-6)
        param_vals[p_nm] = mag_p
        if p_nm not in seen_names and isinstance(u_pp, Unit):
            variables_for_pi.append((p_nm, u_pp))
            seen_names.add(p_nm)

        if isinstance(u_pp, Unit):
            if u_pp.dimension == inv_t_dim or p_nm.startswith(("g_", "gamma", "d_", "c_drv_")):
                timescales.append(1.0 / mag_p)
            elif u_pp.dimension == inv_t2_dim or p_nm.startswith(("k_", "omega")):
                timescales.append(1.0 / math.sqrt(mag_p))

    if not timescales:
        timescales = [1.0]
    tau_fast = max(min(timescales), 1e-9)
    tau_slow = max(max(timescales), tau_fast)
    tau_0 = float(np.median(np.array(timescales, dtype=np.float64)))
    eps_sep = min(max(tau_fast / max(tau_slow, 1e-12), 1e-9), 1.0)

    pi_groups = tuple(discover_dimensionless_groups(variables_for_pi)) if len(variables_for_pi) >= 2 else ()
    all_mag_map: dict[str, float] = {**state_scales, **param_vals}
    unit_by_name: dict[str, Unit] = dict(variables_for_pi)

    base_rescalings = np.array([2.3, 3.1, 1.7, 1.9, 2.9, 1.3, 2.1], dtype=np.float64)
    evaluated_pi: dict[str, float] = {}
    max_drift = 0.0

    for grp in pi_groups:
        expr_s = grp.format_expression()
        log_val = 0.0
        log_rescaled = 0.0
        for v_nm, exp_v in zip(grp.variable_names, grp.exponents):
            if abs(exp_v) < 1e-9:
                continue
            v_mag = max(all_mag_map.get(v_nm, 1.0), 1e-12)
            u_obj = unit_by_name.get(v_nm, DIMENSIONLESS)
            dim_vec = u_obj.dimension.as_vector
            scale_factor = float(np.prod(base_rescalings ** dim_vec))
            log_val += exp_v * math.log(v_mag)
            log_rescaled += exp_v * math.log(v_mag * scale_factor)
        log_val_clamped = max(min(log_val, 60.0), -60.0)
        log_resc_clamped = max(min(log_rescaled, 60.0), -60.0)
        pi_v = math.exp(log_val_clamped)
        pi_v_resc = math.exp(log_resc_clamped)
        rel_drift = abs(pi_v - pi_v_resc) / max(abs(pi_v), 1e-12)
        max_drift = max(max_drift, rel_drift)
        evaluated_pi[expr_s] = round(pi_v, 6)

    return CanonicalNonDimensionalScaling(
        characteristic_state_scales=state_scales,
        characteristic_time_scale_s=round(tau_0, 6),
        fastest_timescale_s=round(tau_fast, 6),
        slowest_timescale_s=round(tau_slow, 6),
        timescale_separation_epsilon=round(eps_sep, 6),
        dimensionless_pi_groups=pi_groups,
        evaluated_pi_values=evaluated_pi,
        scale_invariance_max_drift=float(max_drift),
        scale_invariance_verified=bool(max_drift <= 1e-9),
    )


__all__ = [
    "DimensionlessGroup",
    "discover_dimensionless_groups",
    "CanonicalNonDimensionalScaling",
    "compute_canonical_nondimensional_scaling",
]
