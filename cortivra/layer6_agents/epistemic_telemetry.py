"""
Differential epistemic telemetry serializer and cross-domain LawVault isomorphism
search (`cortivra.layer6_agents.epistemic_telemetry`).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
import re
from typing import Any, Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick
from cortivra.layer1_axioms.shields import AuditCertificate, LatentResidualAnalysis
from cortivra.layer1_axioms.units import discover_dimensionless_groups
from cortivra.layer1_axioms.vault import LawVault
from cortivra.layer4_engine.inverse_solver import (
    CalibrationAndResidualReport,
    SensitivityAndFisherReport,
)
from cortivra.layer4_engine.rollout_arena import EquilibriumLandscapeReport
from cortivra.layer5_discovery.latent_discovery import NonMarkovianMemoryReport
from cortivra.layer5_discovery.sparse_regression import InvariantDiscoveryReport
from cortivra.layer6_agents.falsification_and_design import verify_multi_horizon_stability
from cortivra.observability.flight_recorder import suppress_inner_trace


@dataclass(frozen=True)
class StateTelemetrySummary:
    """
    Mathematical telemetry for a single state variable along a trajectory.
    """

    index: int
    name: str
    unit_symbol: str
    homeostatic_bounds: tuple[float, float]
    trajectory_initial: float
    trajectory_final: float
    trajectory_min: float
    trajectory_max: float
    safety_margin_fraction: float
    homeostatic_violated: bool


@dataclass(frozen=True)
class ParamTelemetrySummary:
    """
    Mathematical telemetry for a single model parameter.
    """

    index: int
    name: str
    unit_symbol: str
    current_value: float
    bounds: tuple[float, float]
    tunable: bool
    normalized_leverage: float


@dataclass(frozen=True)
class EquilibriumTelemetrySummary:
    """
    Summary of phase-space steady-state equilibria and local Jacobian stability.
    """

    num_equilibria: int
    is_multistable: bool
    equilibria: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SensitivityTelemetrySummary:
    """
    Summary of Fisher Information Matrix spectrum and causal parameter leverage.
    """

    condition_number: float
    effective_rank: int
    singular_values: tuple[float, ...]
    top_leverage_parameters: tuple[str, ...]
    sloppy_parameters: tuple[str, ...]
    principal_eigen_directions: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class ResidualTelemetrySummary:
    """
    Summary of unexplained vector-field residual and non-Markovian memory diagnostics.
    """

    has_unmodeled_mechanism: bool
    max_snr: float
    state_coherence_r2: float
    thermodynamic_class: str
    net_power_integral: float
    is_non_markovian_hysteresis: bool
    loop_area_ratio: float
    is_oscillatory_residual: bool
    dominant_residual_freq_hz: float
    affected_state_names: tuple[str, ...]


@dataclass(frozen=True)
class VaultAnalogueMatch:
    """
    Cross-domain mathematical isomorphism discovered in the Layer 1 `LawVault`.
    """

    law_id: str
    name: str
    source_domain: str
    tier: str
    symbolic_expression: str
    description: str
    isomorphism_score: float
    shared_dimensionless_rank: int
    matched_operators: tuple[str, ...] = ()


@dataclass(frozen=True)
class MultiHorizonTelemetrySummary:
    """
    Multi-horizon (1x T, 2x T, 4x T) stability and safety-margin telemetry.
    """

    horizon_1x_viable_fraction: float
    horizon_2x_viable_fraction: float
    horizon_4x_viable_fraction: float
    min_safety_margin_4x: float
    late_horizon_energy_growth: float
    asymptotic_contraction_ratio: float
    passed_multi_horizon: bool


@dataclass(frozen=True)
class DifferentialEpistemicBriefing:
    """
    Mathematical telemetry state of a dynamical investigation.
    """

    system_name: str
    domain: str
    state_dim: int
    param_dim: int
    time_unit: str
    energy_unit: str
    mean_phase_space_divergence: float
    volume_behavior: str
    initial_hamiltonian_energy: float
    final_hamiltonian_energy: float
    mean_dissipative_power: float
    thermodynamic_audit_passed: bool
    state_summaries: tuple[StateTelemetrySummary, ...]
    param_summaries: tuple[ParamTelemetrySummary, ...]
    dimensionless_pi_groups: tuple[str, ...]
    equilibrium_summary: EquilibriumTelemetrySummary | None
    sensitivity_summary: SensitivityTelemetrySummary | None
    residual_summary: ResidualTelemetrySummary | None
    conserved_invariants: tuple[str, ...]
    vault_analogues: tuple[VaultAnalogueMatch, ...]
    compiler_or_shield_diagnostics: tuple[str, ...]
    target_state_objective: dict[str, float] | None = None
    multi_horizon_summary: MultiHorizonTelemetrySummary | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _compute_local_divergence(
    brick: LawBrick,
    states: torch.Tensor,
    params: torch.Tensor,
) -> tuple[float, str]:
    x_sub = states[: min(states.shape[0], 12)].detach().to(torch.float64)
    n_pts, d_dim = x_sub.shape
    p_row = params.detach().to(torch.float64)
    if p_row.ndim == 1:
        p_row = p_row.unsqueeze(0)
    p_batch = p_row.expand(n_pts, -1) if p_row.shape[0] == 1 else p_row[:n_pts]

    with torch.enable_grad():
        x_req = x_sub.clone().requires_grad_(True)
        f_val = brick.flow(x_req, p_batch)
        if not f_val.requires_grad:
            return 0.0, "volume_preserving_liouville"
        trace_acc = torch.zeros(n_pts, dtype=torch.float64, device=x_sub.device)
        for i in range(d_dim):
            if not f_val[:, i].requires_grad:
                continue
            grad_i = torch.autograd.grad(
                f_val[:, i].sum(),
                x_req,
                create_graph=False,
                retain_graph=(i < d_dim - 1),
                allow_unused=True,
            )[0]
            if grad_i is not None:
                trace_acc = trace_acc + grad_i[:, i]
        mean_div = float(torch.mean(trace_acc).item())

    if abs(mean_div) <= 1e-5:
        behavior = "volume_preserving_liouville"
    elif mean_div < 0.0:
        behavior = "volume_contracting_attractor"
    else:
        behavior = "volume_expanding_instability"
    return mean_div, behavior


def _extract_symbolic_text_from_brick(brick: LawBrick) -> str:
    parts: list[str] = [brick.name.lower()]
    if hasattr(brick, "spec"):
        spec = getattr(brick, "spec")
        parts.append(str(getattr(spec, "hamiltonian_expr", "")))
        for d_ex in getattr(spec, "dissipation_diag_exprs", ()):
            parts.append(str(d_ex))
        for e_ex in getattr(spec, "external_drive_exprs", ()):
            parts.append(str(e_ex))
    if hasattr(brick, "delta_hamiltonian_ast"):
        parts.append(str(getattr(brick, "delta_hamiltonian_ast").to_infix()))
    if hasattr(brick, "patch_spec"):
        ps = getattr(brick, "patch_spec")
        for expr_v in getattr(ps, "expressions", {}).values():
            parts.append(str(expr_v.to_infix() if hasattr(expr_v, "to_infix") else expr_v))
    if hasattr(brick, "base_brick"):
        parts.append(_extract_symbolic_text_from_brick(getattr(brick, "base_brick")))
    return " ".join(parts).lower()


def _extract_operator_features_from_text(text: str) -> set[str]:
    low = text.lower()
    feats: set[str] = set()
    if any(k in low for k in ("sin(", "cos(", "sin ", "cos ", "pendulum", "phase", "oscillat")):
        feats.add("trigonometric_periodic")
    if any(k in low for k in ("/", "mm_sat", "hill", "michaelis", "monod", "saturat", "pade")):
        feats.add("rational_saturation")
    if any(k in low for k in ("exp(", "bernoulli", "ghk", "nernst", "arrhenius", "boltzmann")):
        feats.add("exponential_barrier")
    if any(k in low for k in ("** 2", "^2", "** 3", "^3", "** 4", "^4", "cubic", "quartic", "duffing", "k_f")):
        feats.add("polynomial_nonlinearity")
    if any(k in low for k in ("laplacian", "grad", "div", "∇", "∂", "diffusion", "advection")):
        feats.add("spatial_continuum_pde")
    if any(k in low for k in ("damping", "gamma", "r_ohm", "clear", "dissipat", "visco", "conductance", "g_")):
        feats.add("passive_dissipation")
    if any(k in low for k in ("hamiltonian", "p ** 2", "p^2", "kepler", "inductance", "capacitance", "symplectic")):
        feats.add("skew_symplectic_oscillator")
    return feats


def _extract_brick_operator_features(brick: LawBrick) -> set[str]:
    feats = _extract_operator_features_from_text(_extract_symbolic_text_from_brick(brick))
    with torch.no_grad():
        x_sample = brick.default_state(batch_size=4, dtype=torch.float64)
        p_sample = brick.default_params(batch_size=4, dtype=torch.float64)
        f_cons = brick.conservative_flow(x_sample, p_sample)
        f_diss = brick.dissipative_flow(x_sample, p_sample)
        if float(torch.linalg.norm(f_cons).item()) > 1e-7:
            feats.add("skew_symplectic_oscillator")
        if float(torch.linalg.norm(f_diss).item()) > 1e-7:
            feats.add("passive_dissipation")
    return feats


def search_vault_isomorphisms(
    brick: LawBrick,
    vault: LawVault | None,
    top_k: int = 5,
) -> list[VaultAnalogueMatch]:
    """
    Query Layer 1's `LawVault` for crystallized laws across any scientific domain whose
    differential operator signature, thermodynamic decomposition, and dimensionless
    Buckingham-Pi rank are mathematically isomorphic to `brick`.
    """
    if vault is None:
        return []

    variables: list[tuple[str, object]] = []
    seen: set[str] = set()
    for sp in brick.state_specs:
        variables.append((sp.name, sp.unit))
        seen.add(sp.name)
    for pp in brick.param_specs:
        if pp.name not in seen:
            variables.append((pp.name, pp.unit))
            seen.add(pp.name)
    variables.append(("t_char", brick.time_unit))

    with suppress_inner_trace():
        pi_groups = discover_dimensionless_groups(variables)  # type: ignore[arg-type]
    target_pi_count = len(pi_groups)
    brick_feats = _extract_brick_operator_features(brick)

    all_records = vault.list_laws(include_discovered=True)
    matches: list[VaultAnalogueMatch] = []
    for rec in all_records:
        if rec.name == brick.name or rec.law_id == f"crystallized_{brick.name}":
            continue
        rec_text = f"{rec.name} {rec.symbolic_expression} {rec.description}"
        rec_feats = _extract_operator_features_from_text(rec_text)
        shared_ops = sorted(brick_feats.intersection(rec_feats))
        union_ops = brick_feats.union(rec_feats)
        jaccard = len(shared_ops) / max(len(union_ops), 1)

        rec_state_dim = len(rec.state_variables)
        rec_param_dim = len(rec.parameters)
        dim_proximity = 1.0 / (1.0 + 0.25 * abs(rec_state_dim - brick.state_dim) + 0.1 * abs(rec_param_dim - brick.param_dim))
        cross_domain_bonus = 0.20 if (rec.domain != brick.domain and len(shared_ops) > 0) else 0.05

        score = 0.45 * jaccard + 0.30 * dim_proximity + cross_domain_bonus + (0.05 if target_pi_count >= 1 else 0.0)
        score = min(max(score, 0.05), 0.99)

        matches.append(
            VaultAnalogueMatch(
                law_id=rec.law_id,
                name=rec.name,
                source_domain=rec.domain,
                tier=str(rec.tier),
                symbolic_expression=rec.symbolic_expression,
                description=rec.description,
                isomorphism_score=round(score, 4),
                shared_dimensionless_rank=target_pi_count,
                matched_operators=tuple(shared_ops),
            )
        )

    matches.sort(
        key=lambda m: (
            len(m.matched_operators) == 0,
            m.source_domain == brick.domain,
            -m.isomorphism_score,
            m.law_id,
        )
    )
    return matches[:top_k]


def format_compiler_or_shield_diagnostic(
    error_or_cert: Exception | AuditCertificate,
    candidate_spec_summary: str | None = None,
) -> str:
    """
    Format a Layer 3 compile-time error (`DimensionalMismatchError`, `SymbolicParseError`)
    or a Layer 1 `AuditCertificate` violation into a precise mathematical diagnostic for
    automated self-repair.
    """
    prefix = f"[Spec: {candidate_spec_summary}] " if candidate_spec_summary else ""
    if isinstance(error_or_cert, Exception):
        err_type = type(error_or_cert).__name__
        return f"{prefix}{err_type}: {error_or_cert}"

    failed_shields = [r for r in error_or_cert.results if not r.passed]
    if not failed_shields:
        return f"{prefix}All Layer 1 thermodynamic and dimensional shields passed."

    parts = [
        f"{r.shield_name} failed (max_violation={r.max_violation:.3e} > tol={r.tolerance:.3e}): {r.message}"
        for r in failed_shields
    ]
    return prefix + " | ".join(parts)


def build_differential_epistemic_briefing(
    brick: LawBrick,
    states: torch.Tensor | None = None,
    params: torch.Tensor | None = None,
    audit_certificate: AuditCertificate | None = None,
    equilibrium_report: EquilibriumLandscapeReport | None = None,
    identifiability_report: SensitivityAndFisherReport | None = None,
    calibration_report: CalibrationAndResidualReport | None = None,
    residual_analysis: LatentResidualAnalysis | None = None,
    memory_report: NonMarkovianMemoryReport | None = None,
    invariant_report: InvariantDiscoveryReport | None = None,
    vault: LawVault | None = None,
    compiler_diagnostics: Sequence[str] = (),
    target_state_objective: dict[str, float] | None = None,
) -> DifferentialEpistemicBriefing:
    """
    Construct a `DifferentialEpistemicBriefing` from `brick` and Layer 1–5 telemetry.
    """
    p_tensor = (
        params.detach().to(torch.float64)
        if params is not None
        else brick.default_params(batch_size=1, dtype=torch.float64)
    )
    if p_tensor.ndim == 1:
        p_tensor = p_tensor.unsqueeze(0)

    if states is not None:
        x_tensor = states.detach().to(torch.float64)
        if x_tensor.ndim == 3:
            x_tensor = x_tensor[0]
    else:
        x_tensor = brick.default_state(batch_size=1, dtype=torch.float64)

    mean_div, vol_behavior = _compute_local_divergence(brick, x_tensor, p_tensor)

    with torch.no_grad():
        n_traj = x_tensor.shape[0]
        p_traj = p_tensor.expand(n_traj, -1) if p_tensor.shape[0] == 1 else p_tensor[:n_traj]
        h_vals = brick.hamiltonian_energy(x_tensor, p_traj)
        h_init = float(h_vals[0].item()) if h_vals.numel() > 0 else 0.0
        h_final = float(h_vals[-1].item()) if h_vals.numel() > 0 else 0.0
        eff = brick.effort(x_tensor, p_traj)
        f_diss = brick.dissipative_flow(x_tensor, p_traj)
        diss_power = float(torch.mean((eff * (-f_diss)).sum(dim=-1)).item())

    leverage_by_idx: dict[int, float] = {j: 0.0 for j in range(brick.param_dim)}
    sens_summary: SensitivityTelemetrySummary | None = None

    if identifiability_report is not None and brick.param_dim > 0:
        lev_vec = identifiability_report.restorative_leverage_scores.detach().to(torch.float64)
        max_lev = float(torch.max(lev_vec).item()) if lev_vec.numel() > 0 else 1.0
        if max_lev <= 1e-12:
            f_mat = identifiability_report.fisher_matrix.detach().to(torch.float64)
            lev_vec = torch.sqrt(torch.clamp(torch.diag(f_mat), min=0.0))
            max_lev = float(torch.max(lev_vec).item()) if lev_vec.numel() > 0 else 1.0

        for j in range(min(brick.param_dim, lev_vec.numel())):
            leverage_by_idx[j] = float(lev_vec[j].item()) / max(max_lev, 1e-12)

        ranked_params = sorted(
            range(brick.param_dim),
            key=lambda idx: leverage_by_idx.get(idx, 0.0),
            reverse=True,
        )
        top_names = tuple(
            brick.param_specs[idx].name
            for idx in ranked_params
            if leverage_by_idx.get(idx, 0.0) >= 0.15 and brick.param_specs[idx].tunable
        )
        sloppy_names = tuple(
            brick.param_specs[idx].name
            for idx in ranked_params
            if leverage_by_idx.get(idx, 0.0) < 0.05 or not brick.param_specs[idx].tunable
        )

        eig_vals = identifiability_report.fisher_eigenvalues.detach().to(torch.float64)
        max_ev = float(torch.max(eig_vals).item()) if eig_vals.numel() > 0 else 1.0
        min_ev = float(torch.min(eig_vals).item()) if eig_vals.numel() > 0 else 1.0
        cond_num = max_ev / max(min_ev, 1e-12)
        eff_rank = int(torch.sum(eig_vals > (max_ev * 1e-4)).item())

        stiff_vec = identifiability_report.stiffest_eigenvector.detach().to(torch.float64)
        dir_map = {
            brick.param_specs[j].name: round(float(stiff_vec[j].item()), 4)
            for j in range(min(brick.param_dim, stiff_vec.numel()))
            if abs(float(stiff_vec[j].item())) >= 0.05
        }

        sens_summary = SensitivityTelemetrySummary(
            condition_number=round(cond_num, 4),
            effective_rank=eff_rank,
            singular_values=tuple(round(float(v), 6) for v in eig_vals.tolist()),
            top_leverage_parameters=top_names,
            sloppy_parameters=sloppy_names,
            principal_eigen_directions=(dir_map,) if dir_map else (),
        )
    elif brick.param_dim > 0:
        x_probe = x_tensor[: min(x_tensor.shape[0], 16)].detach()
        p_probe = p_tensor[0:1].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            f_probe = brick.flow(x_probe, p_probe.expand(x_probe.shape[0], -1))
            f_sq_sum = (f_probe * f_probe).sum()
            g_vec = torch.autograd.grad(f_sq_sum, p_probe, allow_unused=True)[0]
            if g_vec is not None:
                col_norms = [
                    float(torch.abs(g_vec[0, j]).item()) if brick.param_specs[j].tunable else 0.0
                    for j in range(brick.param_dim)
                ]
            else:
                col_norms = [0.0 for _ in range(brick.param_dim)]
        max_cn = max(col_norms) if col_norms else 1.0
        for j, cn in enumerate(col_norms):
            leverage_by_idx[j] = cn / max(max_cn, 1e-12)

    state_summaries: list[StateTelemetrySummary] = []
    for i, sp in enumerate(brick.state_specs):
        col = x_tensor[:, i]
        c_init = float(col[0].item())
        c_final = float(col[-1].item())
        c_min = float(torch.min(col).item())
        c_max = float(torch.max(col).item())
        h_min, h_max = float(sp.homeostatic_min), float(sp.homeostatic_max)
        span = max(h_max - h_min, 1e-12)
        dist_low = c_min - h_min
        dist_high = h_max - c_max
        margin = min(dist_low, dist_high) / span
        violated = bool(c_min < h_min or c_max > h_max)

        state_summaries.append(
            StateTelemetrySummary(
                index=i,
                name=sp.name,
                unit_symbol=sp.unit.symbol,
                homeostatic_bounds=(h_min, h_max),
                trajectory_initial=round(c_init, 6),
                trajectory_final=round(c_final, 6),
                trajectory_min=round(c_min, 6),
                trajectory_max=round(c_max, 6),
                safety_margin_fraction=round(float(margin), 6),
                homeostatic_violated=violated,
            )
        )

    param_summaries: list[ParamTelemetrySummary] = []
    p_row = p_tensor[0]
    for j, pp in enumerate(brick.param_specs):
        param_summaries.append(
            ParamTelemetrySummary(
                index=j,
                name=pp.name,
                unit_symbol=pp.unit.symbol,
                current_value=round(float(p_row[j].item()), 6),
                bounds=(float(pp.min_value), float(pp.max_value)),
                tunable=bool(pp.tunable),
                normalized_leverage=round(leverage_by_idx.get(j, 0.0), 4),
            )
        )

    eq_summary: EquilibriumTelemetrySummary | None = None
    if equilibrium_report is not None:
        eq_items: list[dict[str, Any]] = []
        n_eq = int(equilibrium_report.distinct_equilibria.shape[0])
        for k_eq in range(n_eq):
            pt = equilibrium_report.distinct_equilibria[k_eq]
            coords = {
                brick.state_specs[i].name: round(float(pt[i].item()), 6)
                for i in range(brick.state_dim)
            }
            st_class = equilibrium_report.stability_classes[k_eq]
            max_re = float(equilibrium_report.max_real_eigenvalues[k_eq].item())
            max_im = float(equilibrium_report.max_imag_eigenvalues[k_eq].item())
            is_st = st_class in ("stable_node", "stable_spiral", "conservative_center")
            eq_items.append(
                {
                    "coordinates": coords,
                    "stability_class": st_class,
                    "is_stable": is_st,
                    "max_real_eigenvalue": round(max_re, 6),
                    "max_oscillation_freq_hz": round(max_im / (2.0 * math.pi), 6),
                }
            )
        eq_summary = EquilibriumTelemetrySummary(
            num_equilibria=n_eq,
            is_multistable=(len(equilibrium_report.stable_attractor_indices) >= 2 or n_eq >= 2),
            equilibria=tuple(eq_items),
        )

    res_obj = (
        residual_analysis
        if residual_analysis is not None
        else (calibration_report.residual_analysis if calibration_report is not None else None)
    )
    res_summary: ResidualTelemetrySummary | None = None
    if res_obj is not None:
        is_nm = bool(memory_report.is_non_markovian) if memory_report is not None else False
        loop_ratio = float(memory_report.hysteresis_ratio) if memory_report is not None else 0.0
        is_osc = bool(getattr(res_obj, "is_oscillatory", False))
        dom_freq = float(getattr(res_obj, "dominant_frequency_hz", 0.0))
        aff_names = tuple(
            brick.state_specs[idx].name
            for idx in res_obj.affected_state_indices
            if 0 <= idx < brick.state_dim
        )
        has_unmodeled = bool(
            res_obj.latent_term_justified
            or (calibration_report is not None and calibration_report.latent_mechanism_required)
            or (res_obj.max_snr >= 3.0 and is_nm)
        )
        mean_p = (
            float(res_obj.missing_power.mean().item())
            if hasattr(res_obj, "missing_power") and isinstance(res_obj.missing_power, torch.Tensor)
            else 0.0
        )
        res_summary = ResidualTelemetrySummary(
            has_unmodeled_mechanism=has_unmodeled,
            max_snr=round(float(res_obj.max_snr), 4),
            state_coherence_r2=round(float(res_obj.state_coherence_r2), 4),
            thermodynamic_class=str(res_obj.thermodynamic_class),
            net_power_integral=round(mean_p, 6),
            is_non_markovian_hysteresis=is_nm,
            loop_area_ratio=round(loop_ratio, 4),
            is_oscillatory_residual=is_osc,
            dominant_residual_freq_hz=round(dom_freq, 4),
            affected_state_names=aff_names,
        )

    variables: list[tuple[str, object]] = []
    seen_v: set[str] = set()
    for sp in brick.state_specs:
        variables.append((sp.name, sp.unit))
        seen_v.add(sp.name)
    for pp in brick.param_specs:
        if pp.name not in seen_v:
            variables.append((pp.name, pp.unit))
            seen_v.add(pp.name)
    variables.append(("t_char", brick.time_unit))
    pi_groups = discover_dimensionless_groups(variables)  # type: ignore[arg-type]
    pi_list = [g.format_expression() for g in pi_groups]
    p_map_now = {brick.param_specs[j].name: float(p_row[j].item()) for j in range(brick.param_dim)}
    stiff_keys = [k for k in p_map_now if re.match(r"^k(?:_|\d)", k)]
    clear_keys = [k for k in p_map_now if re.match(r"^(?:g(?:_|\d)|gamma)", k) and "cpl" not in k]
    if stiff_keys and clear_keys:
        k0_n, g0_n = stiff_keys[0], clear_keys[0]
        k0_v = max(abs(p_map_now[k0_n]), 1e-6)
        g0_v = max(abs(p_map_now[g0_n]), 1e-6)
        da_0 = k0_v / g0_v
        zeta_0 = g0_v / (2.0 * math.sqrt(k0_v))
        pi_list.insert(0, f"Pi_Damkohler_0 = {k0_n} / {g0_n} = {da_0:.4f}")
        pi_list.insert(1, f"Pi_DampingRatio_0 = {g0_n} / (2*sqrt({k0_n})) = {zeta_0:.4f}")
        cpl_keys = [k for k in p_map_now if "cpl" in k]
        if cpl_keys:
            bi_0 = abs(p_map_now[cpl_keys[0]]) / g0_v
            pi_list.insert(2, f"Pi_BiotLeakage_0 = {cpl_keys[0]} / {g0_n} = {bi_0:.4f}")
    pi_strs = tuple(pi_list)

    inv_strs: tuple[str, ...] = ()
    if invariant_report is not None:
        inv_strs = tuple(inv.infix_expression for inv in invariant_report.invariants)

    vault_matches = tuple(search_vault_isomorphisms(brick, vault=vault))
    audit_passed = True if audit_certificate is None else bool(audit_certificate.passed_all)

    mh_raw = verify_multi_horizon_stability(
        brick=brick,
        params=p_tensor,
        initial_state=x_tensor[0:1],
        dt=0.02,
        base_steps=16,
    )
    mh_summary = MultiHorizonTelemetrySummary(
        horizon_1x_viable_fraction=mh_raw.horizon_1x_viable_fraction,
        horizon_2x_viable_fraction=mh_raw.horizon_2x_viable_fraction,
        horizon_4x_viable_fraction=mh_raw.horizon_4x_viable_fraction,
        min_safety_margin_4x=mh_raw.min_safety_margin_4x,
        late_horizon_energy_growth=mh_raw.late_horizon_energy_growth,
        asymptotic_contraction_ratio=mh_raw.asymptotic_contraction_ratio,
        passed_multi_horizon=mh_raw.passed_multi_horizon,
    )

    return DifferentialEpistemicBriefing(
        system_name=brick.name,
        domain=brick.domain,
        state_dim=brick.state_dim,
        param_dim=brick.param_dim,
        time_unit=brick.time_unit.symbol,
        energy_unit=brick.energy_unit.symbol,
        mean_phase_space_divergence=round(mean_div, 6),
        volume_behavior=vol_behavior,
        initial_hamiltonian_energy=round(h_init, 6),
        final_hamiltonian_energy=round(h_final, 6),
        mean_dissipative_power=round(diss_power, 6),
        thermodynamic_audit_passed=audit_passed,
        state_summaries=tuple(state_summaries),
        param_summaries=tuple(param_summaries),
        dimensionless_pi_groups=pi_strs,
        equilibrium_summary=eq_summary,
        sensitivity_summary=sens_summary,
        residual_summary=res_summary,
        conserved_invariants=inv_strs,
        vault_analogues=vault_matches,
        compiler_or_shield_diagnostics=tuple(compiler_diagnostics),
        target_state_objective=target_state_objective,
        multi_horizon_summary=mh_summary,
    )
