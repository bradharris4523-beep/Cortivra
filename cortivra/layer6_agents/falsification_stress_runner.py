"""
4-Gate Popperian falsification stress-testing and LawVault SHA-256 crystallization
(`cortivra.layer6_agents.falsification_stress_runner`).

Extracted from `cortivra.layer6_agents.falsification_and_design` to keep Lyapunov/Langevin/Uncertainty
certification and 4-gate stress-testing + LawVault crystallization cleanly modularized.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick
from cortivra.layer1_axioms.shields import audit_law_brick
from cortivra.layer1_axioms.vault import CrystallizedLawRecord, LawVault
from cortivra.layer4_engine.rollout_arena import retrodict_trajectory, rollout_trajectories
from cortivra.layer5_discovery.sparse_regression import InvariantDiscoveryReport
from cortivra.layer6_agents.experiment_design import verify_multi_horizon_stability
from cortivra.layer6_agents.falsification_and_design import (
    FalsificationGateResult,
    FalsificationStressReport,
    compute_lyapunov_stability_certificate,
    verify_fluctuation_dissipation_langevin_and_kramers_escape,
    verify_parametric_uncertainty_envelope,
)
from cortivra.observability.flight_recorder import emit_trace_node


def run_falsification_stress_test(
    brick: LawBrick,
    calibrated_params: torch.Tensor,
    reference_states: torch.Tensor,
    dt: float = 0.02,
    noise_rel_std: float = 0.01,
    min_noisy_r2: float = 0.85,
    invariant_report: InvariantDiscoveryReport | None = None,
    max_invariant_drift: float = 0.05,
) -> FalsificationStressReport:
    """
    Run the 4-gate falsification suite and multi-horizon (1x T, 2x T, 4x T) stability check.
    """
    x_ref = reference_states.detach().to(torch.float64)
    if x_ref.ndim == 3:
        x_ref = x_ref[0]
    t_steps, _ = x_ref.shape

    p = calibrated_params.detach().to(torch.float64)
    if p.ndim == 1:
        p = p.unsqueeze(0)

    gates: list[FalsificationGateResult] = []

    p_batch = p.expand(t_steps, -1) if p.shape[0] == 1 else p[:t_steps]
    cert = audit_law_brick(brick, sample_state=x_ref, sample_params=p_batch)
    max_shield_viol = max((r.max_violation for r in cert.results), default=0.0)
    gates.append(
        FalsificationGateResult(
            gate_name="thermodynamic_and_dimensional_audit",
            passed=bool(cert.passed_all),
            metric_value=float(max_shield_viol),
            threshold=1e-4,
            details="All Layer 1 shields passed." if cert.passed_all else "Layer 1 shield violation detected.",
        )
    )

    gen = torch.Generator(device=x_ref.device)
    gen.manual_seed(1337)
    state_stds = torch.clamp(torch.std(x_ref, dim=0, keepdim=True), min=1e-4)
    noise = torch.randn(x_ref.shape, dtype=torch.float64, device=x_ref.device, generator=gen) * (
        noise_rel_std * state_stds
    )
    x_noisy = x_ref + noise
    dx_true = brick.flow(x_ref, p_batch).detach()
    dx_noisy_pred = brick.flow(x_noisy, p_batch).detach()
    ss_res = float(torch.sum((dx_true - dx_noisy_pred) ** 2).item())
    ss_tot = float(torch.sum((dx_true - torch.mean(dx_true, dim=0, keepdim=True)) ** 2).item())
    is_trivial_null_flow = bool(
        ss_tot <= 1e-12
        and float(torch.linalg.norm(dx_true).item()) <= 1e-12
        and float(torch.std(x_ref).item()) > 1e-4
    )
    noisy_r2 = 0.0 if is_trivial_null_flow else (1.0 - (ss_res / max(ss_tot, 1e-12)) if ss_tot > 1e-12 else 1.0)
    gate2_passed = bool(noisy_r2 >= min_noisy_r2 and math.isfinite(noisy_r2))
    gates.append(
        FalsificationGateResult(
            gate_name="thermal_noise_identifiability_shake",
            passed=gate2_passed,
            metric_value=float(noisy_r2),
            threshold=float(min_noisy_r2),
            details=f"Noisy vector-field R^2 = {noisy_r2:.5f} under {noise_rel_std * 100.0:.2f}% state noise.",
        )
    )

    x0_ref = x_ref[0:1]
    h_min = torch.tensor(
        [sp.homeostatic_min for sp in brick.state_specs], dtype=torch.float64, device=x_ref.device
    )
    h_max = torch.tensor(
        [sp.homeostatic_max for sp in brick.state_specs], dtype=torch.float64, device=x_ref.device
    )
    scales = torch.clamp(torch.std(x_ref, dim=0), min=0.1)
    ood_multipliers = torch.tensor([-1.25, -0.65, 0.65, 1.25], dtype=torch.float64, device=x_ref.device)
    x0_ood = x0_ref + ood_multipliers.unsqueeze(-1) * scales.unsqueeze(0)
    x0_ood = torch.clamp(
        x0_ood,
        min=h_min.unsqueeze(0) + 0.05 * (h_max - h_min).unsqueeze(0),
        max=h_max.unsqueeze(0) - 0.05 * (h_max - h_min).unsqueeze(0),
    )
    p_ood = p.expand(x0_ood.shape[0], -1)
    ood_steps = min(max(t_steps // 2, 10), 25)
    ood_roll = rollout_trajectories(brick, x0_ood, p_ood, dt=dt, num_steps=ood_steps, integrator="auto")
    finite_ok = bool(torch.isfinite(ood_roll.states).all().item())
    ood_viable = float(ood_roll.viable_fraction) if finite_ok else 0.0

    mh_report = verify_multi_horizon_stability(
        brick=brick,
        params=p,
        initial_state=x0_ref,
        dt=dt,
        base_steps=min(max(t_steps, 12), 35),
    )
    combined_viable = min(ood_viable, mh_report.horizon_4x_viable_fraction)
    gate3_passed = bool(finite_ok and ood_viable >= 0.75 and mh_report.passed_multi_horizon)
    gates.append(
        FalsificationGateResult(
            gate_name="out_of_distribution_extrapolation",
            passed=gate3_passed,
            metric_value=combined_viable,
            threshold=0.75,
            details=(
                f"OOD viable={ood_viable:.3f}, Multi-Horizon (1x/2x/4x) viable="
                f"({mh_report.horizon_1x_viable_fraction:.3f}/{mh_report.horizon_2x_viable_fraction:.3f}/"
                f"{mh_report.horizon_4x_viable_fraction:.3f}), min_margin_4x={mh_report.min_safety_margin_4x:.4f}."
            ),
        )
    )

    max_drift = 0.0
    if invariant_report is not None and invariant_report.num_invariants > 0:
        env_ood = {
            sp.name: ood_roll.states[0, :, i] for i, sp in enumerate(brick.state_specs)
        }
        matching_invs = [
            inv
            for inv in invariant_report.invariants
            if inv.expr_node.free_symbols().issubset(set(env_ood.keys()))
        ]
        if matching_invs:
            for inv in matching_invs:
                vals = inv.expr_node.eval_tensor(env_ood)
                std_v = float(torch.std(vals).item())
                mean_abs = float(torch.mean(torch.abs(vals)).item())
                drift = std_v / max(mean_abs, 1.0)
                max_drift = max(max_drift, drift)
        else:
            for inv in invariant_report.invariants:
                max_drift = max(max_drift, float(inv.relative_drift))
    else:
        short_steps = min(5, max(t_steps - 1, 1))
        fwd = rollout_trajectories(brick, x0_ref, p, dt=dt, num_steps=short_steps, integrator="rk4")
        bwd = retrodict_trajectory(
            brick, fwd.states[:, -1, :], p, dt=dt, num_steps=short_steps
        )
        rt_err = float(torch.linalg.norm(bwd.states[0, 0] - x0_ref[0]).item())
        ref_scale = max(float(torch.linalg.norm(x0_ref[0]).item()), 1.0)
        max_drift = rt_err / (ref_scale * 20.0)

    gate4_passed = bool(math.isfinite(max_drift) and max_drift <= max_invariant_drift)
    gates.append(
        FalsificationGateResult(
            gate_name="time_reversal_and_invariant_stability",
            passed=gate4_passed,
            metric_value=float(max_drift),
            threshold=float(max_invariant_drift),
            details=f"Max invariant / time-reversal relative drift = {max_drift:.5e}.",
        )
    )

    passed_all = all(g.passed for g in gates)
    lyap_cert = compute_lyapunov_stability_certificate(
        brick=brick,
        params=p,
        equilibrium_state=x_ref[-1:],
    )
    env_rep = verify_parametric_uncertainty_envelope(
        brick=brick,
        params=p,
        initial_state=x0_ref,
        dt=dt,
        base_steps=min(max(t_steps - 1, 12), 35),
    )
    langevin_rep = verify_fluctuation_dissipation_langevin_and_kramers_escape(
        brick=brick,
        params=p,
        initial_state=x0_ref,
        dt=dt,
        num_steps=min(max(t_steps - 1, 12), 25),
        lyapunov_cert=lyap_cert,
    )
    failed_g = [g for g in gates if not g.passed]
    viol_metric = max((g.metric_value for g in failed_g), default=0.0)
    emit_trace_node(
        layer="layer6_agents",
        operation="run_falsification_stress_test",
        target_name=brick.name,
        status="completed" if passed_all else "failed",
        local_r2=noisy_r2,
        global_r2=mh_report.horizon_4x_viable_fraction,
        physical_violation_metric=viol_metric if not passed_all else 0.0,
        selected_expression=(
            f"passed_all={passed_all}, 4x_viable={mh_report.horizon_4x_viable_fraction:.3f}, "
            f"env_viable={env_rep.ensemble_viable_fraction:.3f}"
        ),
        metrics={
            "passed_all_gates": passed_all,
            "noisy_vector_field_r2": round(float(noisy_r2), 5),
            "ood_viable_fraction": round(float(ood_viable), 4),
            "horizon_1x_viable_fraction": mh_report.horizon_1x_viable_fraction,
            "horizon_2x_viable_fraction": mh_report.horizon_2x_viable_fraction,
            "horizon_4x_viable_fraction": mh_report.horizon_4x_viable_fraction,
            "min_safety_margin_4x": mh_report.min_safety_margin_4x,
            "max_invariant_drift": round(float(max_drift), 6),
            "lyapunov_certified_basin_radius": round(float(lyap_cert.certified_basin_radius), 4),
            "uncertainty_envelope_viable_fraction": env_rep.ensemble_viable_fraction,
            "bystander_shield_survival_rate": env_rep.bystander_shield_survival_rate,
            "process_capability_cpk": env_rep.process_capability_cpk,
            "first_pass_manufacturing_yield_pct": env_rep.first_pass_manufacturing_yield_pct,
            "kramers_barrier_depth_kt": langevin_rep.kramers_barrier_depth_kt,
            "langevin_viable_fraction": langevin_rep.langevin_viable_fraction,
        },
        diagnostics=tuple(g.details for g in failed_g),
    )
    return FalsificationStressReport(
        target_brick_name=brick.name,
        passed_all_gates=passed_all,
        gates=tuple(gates),
        audit_certificate=cert,
        noisy_vector_field_r2=noisy_r2,
        ood_viable_fraction=ood_viable,
        max_invariant_drift=max_drift,
        multi_horizon_report=mh_report,
        lyapunov_certificate=lyap_cert,
        uncertainty_envelope_report=env_rep,
        langevin_kramers_report=langevin_rep,
    )


def _extract_symbolic_expression_from_brick(brick: LawBrick) -> str:
    """
    Extract a canonical symbolic equation string from any `LawBrick` subclass.
    """
    if hasattr(brick, "spec") and hasattr(brick.spec, "hamiltonian_expr"):
        spec = brick.spec
        parts = [f"H = {spec.hamiltonian_expr}"]
        if getattr(spec, "external_drive_exprs", None):
            drives = [
                str(e.to_infix() if hasattr(e, "to_infix") else e)
                for e in spec.external_drive_exprs
            ]
            if any(d not in ("0", "0.0") for d in drives):
                parts.append(f"Phi = [{', '.join(drives)}]")
        return "; ".join(parts)

    if hasattr(brick, "delta_h_node"):
        return f"Delta_H = {brick.delta_h_node.to_infix()}"

    if hasattr(brick, "patch_spec"):
        ps = brick.patch_spec
        expr_map = {str(k): (v.to_infix() if hasattr(v, "to_infix") else str(v)) for k, v in ps.expressions.items()}
        return f"{ps.mode}: {expr_map}"

    if hasattr(brick, "latent_name"):
        return f"Latent({brick.latent_name}) coupled to {brick.base_brick.name}"

    state_names = ", ".join(sp.name for sp in brick.state_specs)
    return f"PortHamiltonian({brick.name}; states=[{state_names}])"


def crystallize_verified_law(
    vault: LawVault,
    brick: LawBrick,
    stress_report: FalsificationStressReport,
    law_id: str | None = None,
    description: str | None = None,
    parent_law_ids: Sequence[str] = (),
) -> CrystallizedLawRecord:
    """
    Promote a `LawBrick` that has passed `run_falsification_stress_test` into Layer 1's
    `LawVault` with full cryptographic SHA-256 lineage.
    """
    if not stress_report.passed_all_gates:
        failed_names = [g.gate_name for g in stress_report.gates if not g.passed]
        raise ValueError(
            f"Cannot crystallize '{brick.name}' into LawVault; failed falsification gates: {failed_names}."
        )

    final_law_id = law_id or f"crystallized_{brick.name}"
    sym_expr = _extract_symbolic_expression_from_brick(brick)
    final_desc = description or (
        f"Falsification-verified law ({brick.domain}): noisy_r2={stress_report.noisy_vector_field_r2:.4f}, "
        f"ood_viable={stress_report.ood_viable_fraction:.3f}, drift={stress_report.max_invariant_drift:.2e}."
    )
    state_vars = [
        {"name": sp.name, "unit": sp.unit.symbol}
        for sp in brick.state_specs
    ]
    param_vars = [
        {"name": pp.name, "unit": pp.unit.symbol, "default_value": float(pp.default_value)}
        for pp in brick.param_specs
    ]
    f_score = min(
        1.0,
        0.5 * max(stress_report.noisy_vector_field_r2, 0.0)
        + 0.5 * max(stress_report.ood_viable_fraction, 0.0),
    )

    rec = vault.crystallize_discovered_law(
        law_id=final_law_id,
        name=brick.name,
        domain=brick.domain,
        symbolic_expression=sym_expr,
        audit_certificate=stress_report.audit_certificate,
        falsification_score=f_score,
        state_variables=state_vars,
        parameters=param_vars,
        parent_law_ids=parent_law_ids,
        description=final_desc,
    )
    emit_trace_node(
        layer="layer6_agents",
        operation="crystallize_verified_law",
        target_name=rec.law_id,
        local_r2=stress_report.noisy_vector_field_r2,
        global_r2=stress_report.ood_viable_fraction,
        selected_branch_id=rec.law_id,
        selected_expression=sym_expr,
        metrics={
            "law_id": rec.law_id,
            "sha256_fingerprint": rec.sha256_fingerprint,
            "falsification_score": round(f_score, 4),
            "parent_law_ids": list(parent_law_ids),
        },
        skip_if_inner_loop=True,
    )
    return rec


__all__ = [
    "run_falsification_stress_test",
    "_extract_symbolic_expression_from_brick",
    "crystallize_verified_law",
]
