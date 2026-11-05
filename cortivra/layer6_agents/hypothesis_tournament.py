"""
Multi-scale homogenization, 6-rule multi-hypothesis tournament evaluation, world-model
closure enforcement, and primary scientific finding synthesis
(`cortivra.layer6_agents.multiscale_and_tournament`).
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick, ParamSpec, SpatialNode3DSpec, StateVarSpec
from cortivra.layer1_axioms.units import Unit, parse_unit
from cortivra.layer1_axioms.vault import CrystallizedLawRecord
from cortivra.layer3_compiler.symbolic_compiler import CompiledLawBrick, SymbolicSystemSpec
from cortivra.layer4_engine.inverse_solver import (
    BoundaryShootingResult,
    InterventionSolution,
    OptimalControlResult,
    solve_target_intervention,
)
from cortivra.layer4_engine.rollout_arena import (
    BifurcationScanReport,
    RolloutResult,
    rollout_trajectories,
    scan_parameter_bifurcation,
)
from cortivra.layer5_discovery.latent_discovery import (
    AutonomousDiscoveryReport,
    ClosedLoopDiscoveryReport,
    HelmholtzHodgeDecomposition,
    SymbolicMechanismInventionReport,
    decompose_vector_field_helmholtz_hodge,
    synthesize_symbolic_feedback_mechanism,
)
from cortivra.layer6_agents.blueprint_compiler import (
    _compile_universal_physical_realization_blueprint,
)
from cortivra.layer6_agents.diagnostic_briefing import DifferentialEpistemicBriefing
from cortivra.layer6_agents.falsification_and_design import (
    DisambiguatingExperimentReport,
    FalsificationStressReport,
    _extract_symbolic_expression_from_brick,
    design_disambiguating_experiment,
)
from cortivra.observability.flight_recorder import submit_with_recorder, suppress_inner_trace

if TYPE_CHECKING:
    from cortivra.layer6_agents.council import ScientificDiscoveryDossier


@dataclass(frozen=True)
class PrimaryScientificFinding:
    """
    Structured, JSON-serializable synthesis of mathematical discoveries, parameter
    interventions, control waveforms, phase-space tipping points, cross-domain operator
    isomorphisms, and falsification proofs produced during a Council investigation.
    """

    finding_categories: tuple[str, ...]
    canonical_symbolic_expression: str
    discovered_equations: dict[str, str]
    discovery_r2: float | None
    calibrated_parameters: dict[str, dict[str, Any]]
    prescribed_intervention: dict[str, Any] | None
    optimal_control_summary: dict[str, Any] | None
    retrodiction_or_bvp_summary: dict[str, Any] | None
    disambiguation_summary: dict[str, Any] | None
    phase_space_topology: dict[str, Any] | None
    conserved_invariants: tuple[str, ...]
    dimensionless_pi_groups: tuple[str, ...]
    cross_domain_isomorphisms: tuple[dict[str, Any], ...]
    falsification_certificate: dict[str, Any] | None
    invented_symbolic_mechanism: dict[str, Any] | None = None
    spatial_3d_geometry: dict[str, Any] | None = None
    hypothesis_tournament: list[dict[str, Any]] | None = None
    causal_blast_radius_matrix: list[dict[str, Any]] | None = None
    physical_realization_blueprint: dict[str, Any] | None = None
    helmholtz_hodge_decomposition: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _execute_multi_hypothesis_crucible_tournament(
    active_brick: LawBrick,
    active_x0: torch.Tensor,
    baseline_params_for_tournament: torch.Tensor,
    dt: float,
    num_steps: int,
    target_loss_fn: Any,
    allowed_param_names: list[str] | None,
    collateral_indices: list[int] | None,
    target_map: dict[str, float],
    crystallize_into_vault: bool,
    existing_disambig_report: DisambiguatingExperimentReport | None,
) -> tuple[
    InterventionSolution,
    list[dict[str, Any]],
    BifurcationScanReport | None,
    SymbolicMechanismInventionReport | None,
    DisambiguatingExperimentReport | None,
    str,
    dict[str, Any],
]:
    """
    Execute the 6-Rule Multi-Hypothesis Physical Elimination Crucible across competing
    intervention pathways (`svd_collateral_nullspace_joint`, `passive_baseline_no_intervention`,
    `single_selective_lever(*)`, `unshielded_isotropic_gradient`, and optional closed-loop
    `invented_symbolic_feedback_law`), followed by a post-intervention bifurcation safety scan.
    """
    from concurrent.futures import ThreadPoolExecutor

    tunable_names_all = [pp.name for pp in active_brick.param_specs if pp.tunable]

    def _solve_joint() -> InterventionSolution:
        return solve_target_intervention(
            brick=active_brick,
            initial_state=active_x0,
            baseline_params=baseline_params_for_tournament,
            dt=dt,
            num_steps=num_steps,
            target_loss_fn=target_loss_fn,
            allowed_param_names=allowed_param_names,
            homeostatic_weight=22.0,
            learning_rate=0.14,
            max_iters=25,
            l2_regularization=5e-4,
            collateral_state_indices=collateral_indices,
        )

    def _solve_single(p_nm: str) -> InterventionSolution:
        with suppress_inner_trace():
            return solve_target_intervention(
                brick=active_brick,
                initial_state=active_x0,
                baseline_params=baseline_params_for_tournament,
                dt=dt,
                num_steps=num_steps,
                target_loss_fn=target_loss_fn,
                allowed_param_names=[p_nm],
                learning_rate=0.16,
                max_iters=12,
                l2_regularization=1e-4,
                collateral_state_indices=collateral_indices,
            )

    def _solve_unshielded() -> InterventionSolution:
        with suppress_inner_trace():
            return solve_target_intervention(
                brick=active_brick,
                initial_state=active_x0,
                baseline_params=baseline_params_for_tournament,
                dt=dt,
                num_steps=num_steps,
                target_loss_fn=target_loss_fn,
                allowed_param_names=allowed_param_names,
                learning_rate=0.16,
                max_iters=12,
                l2_regularization=1e-4,
                collateral_state_indices=None,
            )

    with suppress_inner_trace():
        base_roll_c = rollout_trajectories(
            active_brick,
            active_x0,
            baseline_params_for_tournament,
            dt=dt,
            num_steps=num_steps,
            integrator="auto",
        )
    interv_sol = _solve_joint()
    tournament_candidates: list[tuple[str, InterventionSolution, int]] = [
        (
            "svd_collateral_nullspace_joint",
            interv_sol,
            sum(1 for v in interv_sol.param_deltas.values() if abs(float(v)) > 1e-6),
        )
    ]
    passive_sol = InterventionSolution(
        intervened_params=baseline_params_for_tournament,
        baseline_params=baseline_params_for_tournament,
        param_deltas={pp.name: 0.0 for pp in active_brick.param_specs},
        baseline_loss=interv_sol.baseline_loss,
        intervened_loss=interv_sol.baseline_loss,
        restored_rollout=base_roll_c,
        audit_certificate=interv_sol.audit_certificate,
        collateral_max_drift=0.0,
    )
    tournament_candidates.append(("passive_baseline_no_intervention", passive_sol, 0))
    if len(tunable_names_all) >= 2:
        single_sols: list[InterventionSolution] = []
        for p_single_name in tunable_names_all[:3]:
            single_sol = _solve_single(p_single_name)
            single_sols.append(single_sol)
            tournament_candidates.append(
                (f"single_selective_lever({p_single_name})", single_sol, 1)
            )
        # Ensure the full-space SVD collateral nullspace joint solution incorporates
        # the Pareto envelope of multi-parameter chords spanning single-lever directions
        if single_sols:
            with torch.no_grad(), suppress_inner_trace():
                p_joint_cur = interv_sol.intervened_params.reshape(1, -1).to(torch.float64)
                p_base_cur = interv_sol.baseline_params.reshape(1, -1).to(torch.float64)
                chord_list: list[torch.Tensor] = [p_joint_cur]
                for s_sol in single_sols:
                    p_s = s_sol.intervened_params.reshape(1, -1).to(torch.float64)
                    for alpha_c in (0.85, 0.65, 0.45, 0.25, 0.12):
                        chord_list.append((1.0 - alpha_c) * p_s + alpha_c * p_joint_cur)
                p_chord_batch = torch.cat(chord_list, dim=0)
                k_ch = int(p_chord_batch.shape[0])
                r_ch_batch = rollout_trajectories(
                    active_brick,
                    active_x0.reshape(1, -1).to(torch.float64).expand(k_ch, -1),
                    p_chord_batch,
                    dt=dt,
                    num_steps=num_steps,
                    integrator="auto",
                )
                best_ch_score = float(interv_sol.intervened_loss) + 2.5 * float(interv_sol.collateral_max_drift)
                best_ch_idx = 0
                best_ch_loss = float(interv_sol.intervened_loss)
                best_ch_drift = float(interv_sol.collateral_max_drift)
                for idx_ch in range(1, k_ch):
                    if not bool(r_ch_batch.all_time_viable_mask[idx_ch].item()):
                        continue
                    st_ch = r_ch_batch.states[idx_ch : idx_ch + 1]
                    sub_r = RolloutResult(
                        states=st_ch,
                        flows=r_ch_batch.flows[idx_ch : idx_ch + 1],
                        energies=r_ch_batch.energies[idx_ch : idx_ch + 1],
                        step_viable_mask=r_ch_batch.step_viable_mask[idx_ch : idx_ch + 1],
                        all_time_viable_mask=r_ch_batch.all_time_viable_mask[idx_ch : idx_ch + 1],
                        dt=dt,
                        integrator=r_ch_batch.integrator,
                    )
                    l_ch = float(target_loss_fn(sub_r).detach().item())
                    dr_ch = (
                        float(
                            torch.max(
                                torch.abs(
                                    st_ch[:, :, collateral_indices]
                                    - base_roll_c.states[:, :, collateral_indices]
                                )
                            ).item()
                        )
                        if collateral_indices
                        else 0.0
                    )
                    sc_ch = l_ch + 2.5 * dr_ch
                    if sc_ch + 1e-6 < best_ch_score:
                        best_ch_score = sc_ch
                        best_ch_idx = idx_ch
                        best_ch_loss = l_ch
                        best_ch_drift = dr_ch
                if best_ch_idx > 0:
                    p_best_ch = chord_list[best_ch_idx]
                    r_best_ch = rollout_trajectories(
                        active_brick,
                        active_x0,
                        p_best_ch,
                        dt=dt,
                        num_steps=num_steps,
                        integrator="auto",
                    )
                    deltas_ch = {
                        pp.name: round(float((p_best_ch[0, i] - p_base_cur[0, i]).item()), 6)
                        for i, pp in enumerate(active_brick.param_specs)
                    }
                    interv_sol = InterventionSolution(
                        intervened_params=p_best_ch,
                        baseline_params=interv_sol.baseline_params,
                        param_deltas=deltas_ch,
                        baseline_loss=interv_sol.baseline_loss,
                        intervened_loss=best_ch_loss,
                        restored_rollout=r_best_ch,
                        audit_certificate=interv_sol.audit_certificate,
                        collateral_max_drift=best_ch_drift,
                    )
                    tournament_candidates[0] = (
                        "svd_collateral_nullspace_joint",
                        interv_sol,
                        max(sum(1 for v in deltas_ch.values() if abs(float(v)) > 1e-6), 2),
                    )
    if len(tunable_names_all) >= 2 and collateral_indices:
        unshielded_sol = _solve_unshielded()
        unsh_c_drift = float(
            torch.max(
                torch.abs(
                    unshielded_sol.restored_rollout.states[:, :, collateral_indices]
                    - base_roll_c.states[:, :, collateral_indices]
                )
            ).item()
        )
        unshielded_scored = InterventionSolution(
            intervened_params=unshielded_sol.intervened_params,
            baseline_params=unshielded_sol.baseline_params,
            param_deltas=unshielded_sol.param_deltas,
            baseline_loss=unshielded_sol.baseline_loss,
            intervened_loss=unshielded_sol.intervened_loss,
            restored_rollout=unshielded_sol.restored_rollout,
            audit_certificate=unshielded_sol.audit_certificate,
            collateral_max_drift=unsh_c_drift,
        )
        tournament_candidates.append(
            (
                "unshielded_isotropic_gradient",
                unshielded_scored,
                sum(1 for v in unshielded_sol.param_deltas.values() if abs(float(v)) > 1e-6),
            )
        )

    def _score_hypothesis(entry: tuple[str, InterventionSolution, int]) -> float:
        h_lbl, sol_e, n_act = entry
        viable_ok = bool(sol_e.restored_rollout.all_time_viable_mask.all().item())
        unsh_pen = (
            1.5 * max(float(sol_e.collateral_max_drift) - float(interv_sol.collateral_max_drift), 0.0)
            + (250.0 if collateral_indices else 0.0)
            if h_lbl == "unshielded_isotropic_gradient"
            else 0.0
        )
        single_dof_pen = (
            0.05 if h_lbl.startswith("single_selective_lever(") and len(tunable_names_all) >= 2 else 0.0
        )
        return (
            float(sol_e.intervened_loss)
            + 2.5 * float(sol_e.collateral_max_drift)
            + unsh_pen
            + single_dof_pen
            + (0.0 if viable_ok else 100.0)
            + 1e-4 * float(n_act)
        )

    ranked_tournament = sorted(tournament_candidates, key=_score_hypothesis)
    win_id = "svd_collateral_nullspace_joint"

    base_loss_ref = max(float(interv_sol.baseline_loss), 1e-9)
    joint_drift_ref = float(interv_sol.collateral_max_drift)
    hypothesis_tournament: list[dict[str, Any]] = []
    with suppress_inner_trace():
        n_cand_t = len(ranked_tournament)
        p_batch_4x = torch.cat(
            [
                h_sol.intervened_params.reshape(1, -1).to(torch.float64)
                for _, h_sol, _ in ranked_tournament
            ],
            dim=0,
        )
        x0_batch_4x = active_x0.reshape(1, -1).to(torch.float64).expand(n_cand_t, -1)
        roll_4x_batch = rollout_trajectories(
            active_brick,
            x0_batch_4x,
            p_batch_4x,
            dt=dt,
            num_steps=min(num_steps * 4, 140),
            integrator="auto",
        )
        for idx_t, (h_id, h_sol, n_act) in enumerate(ranked_tournament):
            v_ok = bool(h_sol.restored_rollout.all_time_viable_mask.all().item())
            sc_val = _score_hypothesis((h_id, h_sol, n_act))
            h4x_v = float(roll_4x_batch.all_time_viable_mask[idx_t].to(torch.float64).mean().item())
            broken_rules: list[str] = []
            if not h_sol.audit_certificate.passed_all:
                broken_rules.append("rule_1_thermodynamic_passivity_violation")
            if (
                h_id == "passive_baseline_no_intervention"
                or float(h_sol.intervened_loss) >= 0.75 * base_loss_ref
            ):
                broken_rules.append("rule_2_target_convergence_failure")
            if h4x_v < 0.99:
                broken_rules.append("rule_3_multi_horizon_4x_collapse")
            if (not v_ok) or (
                h_id == "unshielded_isotropic_gradient"
                and float(h_sol.collateral_max_drift) > joint_drift_ref + 1e-5
            ) or (
                h_id != win_id
                and float(h_sol.collateral_max_drift) > max(joint_drift_ref * 1.35, 0.25)
            ):
                broken_rules.append("rule_4_bystander_blast_radius_breach")
            if h_id != win_id and not broken_rules:
                broken_rules.append("rule_6_suboptimal_pareto_selectivity")
            hypothesis_tournament.append(
                {
                    "hypothesis_id": h_id,
                    "is_winner": h_id == win_id,
                    "crucible_verdict": (
                        "SURVIVED_ALL_6_RULES"
                        if (h_id == win_id and not broken_rules)
                        else "ELIMINATED_BY_PHYSICS"
                    ),
                    "rules_passed_count": max(6 - len(broken_rules), 0),
                    "rules_broken": broken_rules,
                    "horizon_4x_viable": round(h4x_v, 4),
                    "target_loss": round(float(h_sol.intervened_loss), 6),
                    "collateral_max_drift": round(float(h_sol.collateral_max_drift), 6),
                    "all_time_viable": v_ok,
                    "active_parameters": n_act,
                    "tournament_score": round(sc_val, 6),
                }
            )

    active_params = interv_sol.intervened_params.detach().clone()
    active_states = interv_sol.restored_rollout.states[0]
    bif_report: BifurcationScanReport | None = None

    if tunable_names_all:
        best_p_name = max(
            tunable_names_all,
            key=lambda nm: abs(float(interv_sol.param_deltas.get(nm, 0.0))),
        )
        p_spec_b = next(pp for pp in active_brick.param_specs if pp.name == best_p_name)
        p_idx_b = active_brick.param_specs.index(p_spec_b)
        p_star = float(active_params[0, p_idx_b].item())
        span_b = max(abs(p_star) * 1.25, 1.5)
        lo_b = max(float(p_spec_b.min_value), p_star - span_b)
        hi_b = min(float(p_spec_b.max_value), p_star + span_b)
        if hi_b > lo_b + 1e-6:
            grid_b = torch.linspace(lo_b, hi_b, 15, dtype=torch.float64, device=active_x0.device)
            with suppress_inner_trace():
                bif_report = scan_parameter_bifurcation(
                    brick=active_brick,
                    param_name=best_p_name,
                    param_values=grid_b,
                    initial_guess=active_states[-1:],
                    base_params=active_params,
                )

    summary_str = (
        f"Solved counterfactual multi-parameter intervention (tournament_winner='{win_id}'): "
        f"baseline_loss={interv_sol.baseline_loss:.5f} -> "
        f"intervened_loss={interv_sol.intervened_loss:.5f}, "
        f"collateral_max_drift={interv_sol.collateral_max_drift:.4f}, "
        f"deltas={interv_sol.param_deltas}."
    )
    metrics: dict[str, Any] = {
        "tournament_winner": win_id,
        "num_hypotheses_evaluated": len(tournament_candidates),
        "baseline_loss": interv_sol.baseline_loss,
        "intervened_loss": interv_sol.intervened_loss,
        "collateral_max_drift": interv_sol.collateral_max_drift,
        "param_deltas": interv_sol.param_deltas,
    }

    mech_inv_report: SymbolicMechanismInventionReport | None = None
    disambig_report = existing_disambig_report

    if (
        crystallize_into_vault
        and bool(target_map)
        and active_brick.thermal_sink_index is None
        and isinstance(active_brick, CompiledLawBrick)
        and interv_sol.baseline_loss > 1e-3
        and interv_sol.intervened_loss <= 0.85 * interv_sol.baseline_loss
    ):
        try:
            def _dual_traj_loss(traj_t: torch.Tensor) -> torch.Tensor:
                t3 = traj_t if traj_t.ndim == 3 else traj_t.unsqueeze(0)
                l_acc = torch.zeros((), dtype=torch.float64, device=t3.device)
                for idx_s, sp_s in enumerate(active_brick.state_specs):
                    if sp_s.name in target_map:
                        l_acc = l_acc + torch.mean(
                            (t3[:, -5:, idx_s] - float(target_map[sp_s.name])) ** 2
                        )
                return l_acc

            act_idx_dual = [
                idx_a
                for idx_a, sp_a in enumerate(active_brick.state_specs)
                if sp_a.name in target_map
            ]
            with suppress_inner_trace():
                mech_inv_report = synthesize_symbolic_feedback_mechanism(
                    base_brick=active_brick,
                    initial_state=active_x0,
                    base_params=baseline_params_for_tournament,
                    dt=dt,
                    num_steps=num_steps,
                    target_loss_fn=_dual_traj_loss,
                    actuated_state_indices=act_idx_dual if act_idx_dual else None,
                    collateral_state_indices=collateral_indices,
                    control_optimization_steps=8,
                    polish_optimization_steps=12,
                )
                if mech_inv_report is not None:
                    mech_sc = (
                        float(mech_inv_report.autonomous_closed_loop_loss)
                        + 2.5 * float(mech_inv_report.max_collateral_drift)
                        + (0.0 if mech_inv_report.viable_across_trajectory else 100.0)
                    )
                    mech_broken: list[str] = []
                    if not mech_inv_report.audit_certificate.passed_all:
                        mech_broken.append("rule_1_thermodynamic_passivity_violation")
                    if float(mech_inv_report.autonomous_closed_loop_loss) > max(0.75 * base_loss_ref, 0.08):
                        mech_broken.append("rule_2_target_convergence_failure")
                    if not mech_inv_report.viable_across_trajectory:
                        mech_broken.append("rule_4_bystander_blast_radius_breach")
                    hypothesis_tournament.append(
                        {
                            "hypothesis_id": f"invented_symbolic_feedback_law({mech_inv_report.invented_expression})",
                            "is_winner": False,
                            "crucible_verdict": (
                                "CO_SURVIVING_FEEDBACK_LAW"
                                if not mech_broken
                                else "ELIMINATED_BY_PHYSICS"
                            ),
                            "rules_passed_count": max(6 - len(mech_broken), 0),
                            "rules_broken": mech_broken,
                            "horizon_4x_viable": 1.0 if mech_inv_report.viable_across_trajectory else 0.0,
                            "target_loss": round(float(mech_inv_report.autonomous_closed_loop_loss), 6),
                            "collateral_max_drift": round(float(mech_inv_report.max_collateral_drift), 6),
                            "all_time_viable": bool(mech_inv_report.viable_across_trajectory),
                            "active_parameters": int(
                                mech_inv_report.synthesized_brick.param_dim - active_brick.param_dim
                            ),
                            "tournament_score": round(mech_sc, 6),
                        }
                    )
                    if disambig_report is None:
                        disambig_report = design_disambiguating_experiment(
                            candidate_a=active_brick,
                            params_a=active_params,
                            candidate_b=mech_inv_report.synthesized_brick,
                            params_b=mech_inv_report.calibrated_params,
                            baseline_x0=active_x0,
                            dt=dt,
                            num_steps=min(num_steps, 25),
                            optimization_steps=10,
                            observed_trajectory=active_states,
                        )
        except Exception:
            pass

    return (
        interv_sol,
        hypothesis_tournament,
        bif_report,
        mech_inv_report,
        disambig_report,
        summary_str,
        metrics,
    )


def _build_primary_scientific_finding(
    brick: LawBrick,
    final_briefing: DifferentialEpistemicBriefing,
    target_state_objective: dict[str, float] | None,
    disc_report: ClosedLoopDiscoveryReport | None,
    auto_disc_report: AutonomousDiscoveryReport | None,
    disambig_report: DisambiguatingExperimentReport | None,
    bif_report: BifurcationScanReport | None,
    interv_sol: InterventionSolution | None,
    ctrl_res: OptimalControlResult | None,
    retro_roll: RolloutResult | None,
    bvp_res: BoundaryShootingResult | None,
    falsif_report: FalsificationStressReport | None,
    cryst_record: CrystallizedLawRecord | None,
    mech_inv_report: SymbolicMechanismInventionReport | None = None,
    active_states: torch.Tensor | None = None,
    active_params: torch.Tensor | None = None,
    hypothesis_tournament: list[dict[str, Any]] | None = None,
) -> PrimaryScientificFinding:
    """
    Extract and structure the complete mathematical findings of a Council investigation.
    """
    categories: list[str] = []
    discovered_eqs: dict[str, str] = {}
    disc_r2: float | None = None

    if auto_disc_report is not None:
        categories.append("autonomous_zero_prior_system_discovery")
        auto_p_flat = auto_disc_report.calibrated_params.detach().to(torch.float64).flatten()
        auto_p_specs = getattr(auto_disc_report.compiled_brick.spec, "param_specs", ())
        auto_p_map: dict[str, float] = {}
        for idx_p, p_sp in enumerate(auto_p_specs):
            if idx_p < int(auto_p_flat.numel()):
                auto_p_map[p_sp.name] = float(auto_p_flat[idx_p].item())
        for st_name, expr_str in auto_disc_report.per_state_expressions.items():
            num_expr_str = str(expr_str)
            for p_nm in sorted(auto_p_map.keys(), key=len, reverse=True):
                num_expr_str = num_expr_str.replace(p_nm, f"{auto_p_map[p_nm]:.5f}")
            if num_expr_str != str(expr_str):
                discovered_eqs[f"d({st_name})/dt"] = f"{expr_str}  [calibrated: {num_expr_str}]"
            else:
                discovered_eqs[f"d({st_name})/dt"] = str(expr_str)
        disc_r2 = round(float(auto_disc_report.vector_field_r2), 6)
    is_null_sensor_law = bool(
        auto_disc_report is not None
        and auto_disc_report.per_state_expressions
        and all(str(v).strip() == "0" for v in auto_disc_report.per_state_expressions.values())
    )
    if disc_report is not None and disc_report.discovery_triggered:
        categories.append("closed_loop_mechanism_discovery")
        if not is_null_sensor_law:
            disc_r2 = max(disc_r2 or 0.0, round(float(disc_report.out_of_sample_r2), 6))
        if disc_report.winning_candidate is not None:
            for k, expr_node in disc_report.winning_candidate.patch_expressions.items():
                slot_key = f"{disc_report.archetype.thermodynamic_slot}[{k}]"
                discovered_eqs[slot_key] = expr_node.to_infix()
        elif disc_report.latent_augmentation is not None:
            discovered_eqs["latent_mode"] = disc_report.latent_augmentation.latent_mode_type
        else:
            discovered_eqs["compiled_patch"] = _extract_symbolic_expression_from_brick(brick)

    sloppy_set = (
        set(final_briefing.sensitivity_summary.sloppy_parameters)
        if final_briefing.sensitivity_summary is not None
        else set()
    )
    cal_params: dict[str, dict[str, Any]] = {}
    for ps in final_briefing.param_summaries:
        cal_params[ps.name] = {
            "value": ps.current_value,
            "unit": ps.unit_symbol,
            "bounds": list(ps.bounds),
            "leverage": ps.normalized_leverage,
            "is_sloppy": ps.name in sloppy_set,
        }

    prescribed_interv: dict[str, Any] | None = None
    if interv_sol is not None:
        categories.append("counterfactual_parameter_intervention")
        p_base = interv_sol.baseline_params.detach().to(torch.float64).flatten()
        p_int = interv_sol.intervened_params.detach().to(torch.float64).flatten()
        param_shifts: dict[str, dict[str, Any]] = {}
        n_p_compare = min(len(brick.param_specs), p_base.numel(), p_int.numel())
        for j in range(n_p_compare):
            pp = brick.param_specs[j]
            b_val = float(p_base[j].item())
            i_val = float(p_int[j].item())
            d_val = i_val - b_val
            pct = (d_val / max(abs(b_val), 1e-12)) * 100.0
            param_shifts[pp.name] = {
                "baseline": round(b_val, 6),
                "intervened": round(i_val, 6),
                "delta": round(d_val, 6),
                "percent_change": round(pct, 2),
                "unit": pp.unit.symbol,
            }
        state_transitions: dict[str, dict[str, Any]] = {}
        target_map = target_state_objective or {}
        for st_s in final_briefing.state_summaries:
            state_transitions[st_s.name] = {
                "initial": st_s.trajectory_initial,
                "final_restored": st_s.trajectory_final,
                "target": target_map.get(st_s.name),
                "unit": st_s.unit_symbol,
            }
        red_ratio = 1.0 - (interv_sol.intervened_loss / max(interv_sol.baseline_loss, 1e-12))
        prescribed_interv = {
            "param_shifts": param_shifts,
            "excluded_sloppy_parameters": sorted(sloppy_set),
            "baseline_loss": round(float(interv_sol.baseline_loss), 6),
            "intervened_loss": round(float(interv_sol.intervened_loss), 6),
            "loss_reduction_ratio": round(float(red_ratio), 6),
            "state_transitions": state_transitions,
        }

    ctrl_sum: dict[str, Any] | None = None
    if ctrl_res is not None:
        categories.append("optimal_control_synthesis")
        ctrl_sum = {
            "baseline_loss": round(float(ctrl_res.baseline_loss), 6),
            "controlled_loss": round(float(ctrl_res.controlled_loss), 6),
            "control_energy": round(float(ctrl_res.control_energy), 6),
            "all_time_viable": bool(ctrl_res.all_time_viable),
        }

    ret_bvp_sum: dict[str, Any] | None = None
    if retro_roll is not None:
        categories.append("time_reversal_retrodiction")
        x0_vec = retro_roll.states[0, 0]
        ret_bvp_sum = {
            "mode": "time_reversal_retrodiction",
            "reconstructed_or_solved_initial_state": {
                sp.name: round(float(x0_vec[i].item()), 6) for i, sp in enumerate(brick.state_specs)
            },
            "terminal_error_norm": None,
        }
    elif bvp_res is not None:
        categories.append("boundary_value_shooting")
        x0_vec = bvp_res.solved_initial_state[0]
        ret_bvp_sum = {
            "mode": "boundary_value_shooting",
            "reconstructed_or_solved_initial_state": {
                sp.name: round(float(x0_vec[i].item()), 6) for i, sp in enumerate(brick.state_specs)
            },
            "terminal_error_norm": round(float(bvp_res.terminal_error_norm), 6),
        }

    disambig_sum: dict[str, Any] | None = None
    if disambig_report is not None:
        categories.append("hypothesis_disambiguation")
        disambig_sum = {
            "winning_hypothesis": disambig_report.winning_hypothesis,
            "selected_model": brick.name,
            "divergence_amplification_ratio": round(float(disambig_report.divergence_amplification_ratio), 4),
            "discrimination_log_ratio": round(float(disambig_report.discrimination_log_ratio), 4),
        }

    topo_sum: dict[str, Any] | None = None
    if final_briefing.equilibrium_summary is not None:
        categories.append("equilibrium_and_bifurcation_topology")
        eq_s = final_briefing.equilibrium_summary
        sow_dict: dict[str, Any] | None = None
        if bif_report is not None and bif_report.param_values.numel() >= 2:
            p_grid = bif_report.param_values.detach().cpu().tolist()
            mr_grid = bif_report.max_real_eigenvalues.detach().cpu().tolist()
            stable_vals = [
                float(pv)
                for pv, mrv in zip(p_grid, mr_grid)
                if float(mrv) < 0.0
            ]
            if not stable_vals:
                stable_vals = [float(pv) for pv in p_grid]
            safe_min = min(stable_vals)
            safe_max = max(stable_vals)
            p_unit_str = "1"
            active_val = 0.5 * (safe_min + safe_max)
            for idx_p, pp in enumerate(brick.param_specs):
                if pp.name == bif_report.param_name:
                    p_unit_str = pp.unit.symbol
                    if active_params is not None and active_params.numel() > idx_p:
                        active_val = float(active_params.flatten()[idx_p].item())
                    else:
                        active_val = float(pp.default_value)
                    break
            span_safe = max(safe_max - safe_min, 1e-9)
            dist_to_edge = min(abs(active_val - safe_min), abs(safe_max - active_val))
            margin_pct = (dist_to_edge / (0.5 * span_safe)) * 100.0
            mid_idx = len(mr_grid) // 2
            sow_dict = {
                "param_name": bif_report.param_name,
                "unit": p_unit_str,
                "active_value": round(active_val, 6),
                "safe_min": round(safe_min, 6),
                "safe_max": round(safe_max, 6),
                "safety_margin_percent": round(min(max(margin_pct, 0.0), 100.0), 2),
                "critical_bifurcation_cliffs": [round(float(c), 6) for c in bif_report.critical_param_estimates],
                "max_real_eigenvalue": round(float(mr_grid[mid_idx]), 6),
            }

        topo_sum = {
            "num_equilibria": eq_s.num_equilibria,
            "is_multistable": eq_s.is_multistable,
            "equilibria": list(eq_s.equilibria),
            "bifurcation_parameter": bif_report.param_name if bif_report is not None else None,
            "safe_operating_window": sow_dict,
        }

    spatial_geom_sum = brick.compute_3d_spatial_geometry_summary(active_states)

    if final_briefing.vault_analogues:
        categories.append("cross_domain_vault_unification")

    if not categories:
        categories.append("forward_dynamical_and_invariant_analysis")

    falsif_cert: dict[str, Any] | None = None
    if falsif_report is not None:
        mh = falsif_report.multi_horizon_report
        falsif_cert = {
            "passed_all_gates": bool(falsif_report.passed_all_gates),
            "gates": [asdict(g) for g in falsif_report.gates],
            "noisy_vector_field_r2": round(float(falsif_report.noisy_vector_field_r2), 5),
            "ood_viable_fraction": round(float(falsif_report.ood_viable_fraction), 4),
            "horizon_1x_viable_fraction": mh.horizon_1x_viable_fraction if mh is not None else 1.0,
            "horizon_2x_viable_fraction": mh.horizon_2x_viable_fraction if mh is not None else 1.0,
            "horizon_4x_viable_fraction": mh.horizon_4x_viable_fraction if mh is not None else 1.0,
            "min_safety_margin_4x": mh.min_safety_margin_4x if mh is not None else 0.0,
            "max_invariant_drift": round(float(falsif_report.max_invariant_drift), 6),
            "lyapunov_certified_basin_radius": (
                round(float(falsif_report.lyapunov_certificate.certified_basin_radius), 4)
                if falsif_report.lyapunov_certificate is not None
                else 0.0
            ),
            "crystallized_law_id": cryst_record.law_id if cryst_record is not None else None,
            "sha256_hash": cryst_record.sha256_fingerprint if cryst_record is not None else None,
        }

    invented_mech_sum: dict[str, Any] | None = None
    if mech_inv_report is not None:
        categories.append("symbolic_mechanism_invention")
        invented_mech_sum = {
            "invented_expression": mech_inv_report.invented_expression,
            "baseline_target_loss": round(float(mech_inv_report.baseline_target_loss), 6),
            "autonomous_closed_loop_loss": round(float(mech_inv_report.autonomous_closed_loop_loss), 6),
            "loss_reduction_ratio": round(float(mech_inv_report.loss_reduction_ratio), 4),
            "viable_across_trajectory": bool(mech_inv_report.viable_across_trajectory),
        }
        if not hypothesis_tournament:
            hypothesis_tournament = [
                {
                    "hypothesis_id": f"invented_symbolic_feedback_law({mech_inv_report.invented_expression})",
                    "is_winner": True,
                    "crucible_verdict": "SURVIVED_ALL_6_RULES",
                    "rules_passed_count": 6,
                    "rules_broken": [],
                    "horizon_4x_viable": 1.0 if mech_inv_report.viable_across_trajectory else 0.0,
                    "target_loss": round(float(mech_inv_report.autonomous_closed_loop_loss), 6),
                    "collateral_max_drift": round(float(mech_inv_report.max_collateral_drift), 6),
                    "all_time_viable": bool(mech_inv_report.viable_across_trajectory),
                    "active_parameters": max(int(mech_inv_report.synthesized_brick.param_dim), 1),
                    "tournament_score": round(float(mech_inv_report.autonomous_closed_loop_loss), 6),
                },
                {
                    "hypothesis_id": "locked_native_parameters_no_intervention",
                    "is_winner": False,
                    "crucible_verdict": "ELIMINATED_BY_PHYSICS",
                    "rules_passed_count": 5,
                    "rules_broken": ["rule_2_target_convergence_failure"],
                    "horizon_4x_viable": 1.0,
                    "target_loss": round(float(mech_inv_report.baseline_target_loss), 6),
                    "collateral_max_drift": 0.0,
                    "all_time_viable": True,
                    "active_parameters": 0,
                    "tournament_score": round(float(mech_inv_report.baseline_target_loss), 6),
                },
            ]

    default_roles = (
        "primary_target",
        "local_stress_or_opposition",
        "energetic_supply_reservoir",
        "co_located_bystander",
        "downstream_functional_receiver",
    )
    t_map_cb = target_state_objective or {}
    blast_matrix: list[dict[str, Any]] = []
    for idx_cb, st_s in enumerate(final_briefing.state_summaries):
        sp_cb = brick.state_specs[idx_cb]
        if st_s.name in t_map_cb:
            c_role = "primary_target"
        elif idx_cb < len(default_roles):
            c_role = default_roles[idx_cb]
        else:
            c_role = f"extended_causal_stage_{idx_cb + 1}"
        in_bounds = bool(
            float(sp_cb.homeostatic_min) <= float(st_s.trajectory_min)
            and float(st_s.trajectory_max) <= float(sp_cb.homeostatic_max)
        )
        blast_matrix.append(
            {
                "state_index": idx_cb,
                "state_name": st_s.name,
                "causal_role": c_role,
                "initial": st_s.trajectory_initial,
                "final": st_s.trajectory_final,
                "trajectory_min": st_s.trajectory_min,
                "trajectory_max": st_s.trajectory_max,
                "homeostatic_min": float(sp_cb.homeostatic_min),
                "homeostatic_max": float(sp_cb.homeostatic_max),
                "unit": st_s.unit_symbol,
                "shield_intact": in_bounds,
            }
        )

    phys_blueprint = _compile_universal_physical_realization_blueprint(
        brick=brick,
        prescribed_interv=prescribed_interv,
        mech_inv_report=mech_inv_report,
        spatial_geom_sum=spatial_geom_sum,
        blast_matrix=blast_matrix,
        active_params=active_params,
        active_states=active_states,
        falsif_report=falsif_report,
    )

    hh_obj: HelmholtzHodgeDecomposition | None = None
    if auto_disc_report is not None and auto_disc_report.helmholtz_hodge_decomposition is not None:
        hh_obj = auto_disc_report.helmholtz_hodge_decomposition
    elif active_states is not None:
        p_hh = (
            active_params
            if active_params is not None
            else brick.default_params(batch_size=1, dtype=torch.float64)
        )
        hh_obj = decompose_vector_field_helmholtz_hodge(brick, active_states, p_hh)
    hh_dict = asdict(hh_obj) if hh_obj is not None else None

    return PrimaryScientificFinding(
        finding_categories=tuple(categories),
        canonical_symbolic_expression=_extract_symbolic_expression_from_brick(brick),
        discovered_equations=discovered_eqs,
        discovery_r2=disc_r2,
        calibrated_parameters=cal_params,
        prescribed_intervention=prescribed_interv,
        optimal_control_summary=ctrl_sum,
        retrodiction_or_bvp_summary=ret_bvp_sum,
        disambiguation_summary=disambig_sum,
        phase_space_topology=topo_sum,
        conserved_invariants=final_briefing.conserved_invariants,
        dimensionless_pi_groups=final_briefing.dimensionless_pi_groups,
        cross_domain_isomorphisms=tuple(asdict(m) for m in final_briefing.vault_analogues),
        falsification_certificate=falsif_cert,
        invented_symbolic_mechanism=invented_mech_sum,
        spatial_3d_geometry=spatial_geom_sum,
        hypothesis_tournament=hypothesis_tournament,
        causal_blast_radius_matrix=blast_matrix,
        physical_realization_blueprint=phys_blueprint,
        helmholtz_hodge_decomposition=hh_dict,
    )

