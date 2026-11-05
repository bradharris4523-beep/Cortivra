"""
Autonomous multi-agent scientific council and closed-loop operator orchestrator
(`cortivra.layer6_agents.council`).
"""

from __future__ import annotations

import html
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Sequence

import torch

try:
    if torch.get_num_threads() > 2:
        torch.set_num_threads(2)
except Exception:
    pass

from cortivra.layer1_axioms.hamiltonian import LawBrick, ParamSpec, SpatialNode3DSpec, StateVarSpec
from cortivra.layer1_axioms.shields import AuditCertificate, audit_law_brick
from cortivra.layer1_axioms.units import (
    DIMENSIONLESS,
    Unit,
    compute_canonical_nondimensional_scaling,
    parse_unit,
    s,
)
from cortivra.layer1_axioms.vault import CrystallizedLawRecord, LawVault
from cortivra.layer2_fabric import register_foundational_laws
from cortivra.layer3_compiler.symbolic_compiler import (
    CompiledLawBrick,
    SymbolicSystemSpec,
    compile_residual_patch_brick,
    compile_symbolic_system,
)
from cortivra.layer4_engine.inverse_solver import (
    BoundaryShootingResult,
    CalibrationAndResidualReport,
    InterventionSolution,
    OptimalControlResult,
    SensitivityAndFisherReport,
    calibrate_and_audit_residual,
    compute_parameter_leverage_and_fisher,
    estimate_trajectory_derivatives,
    solve_boundary_value_shooting,
    solve_optimal_control_waveform,
    solve_target_intervention,
)
from cortivra.layer4_engine.rollout_arena import (
    BifurcationScanReport,
    EquilibriumLandscapeReport,
    RolloutResult,
    discover_equilibrium_landscape,
    retrodict_trajectory,
    rollout_trajectories,
    scan_parameter_bifurcation,
)
from cortivra.layer5_discovery.latent_discovery import (
    ArchetypeFingerprint,
    AutonomousDiscoveryReport,
    ClosedLoopDiscoveryReport,
    HelmholtzHodgeDecomposition,
    NonMarkovianMemoryReport,
    SymbolicMechanismInventionReport,
    classify_mathematical_archetype,
    decompose_vector_field_helmholtz_hodge,
    detect_non_markovian_memory,
    discover_and_compile_law,
    discover_autonomous_symbolic_system,
    synthesize_symbolic_feedback_mechanism,
)
from cortivra.layer5_discovery.sparse_regression import (
    InvariantDiscoveryReport,
    discover_conservation_invariants,
)
from cortivra.layer6_agents.blind_oracle_rescue_suite import (
    OracleClosedLoopTrialMetrics,
    evaluate_brick_mismatched_reality_trial,
)
from cortivra.layer6_agents.blueprint_compiler import (
    _classify_parameter_operator_slot,
    _compile_universal_physical_realization_blueprint,
    _infer_operating_temperature_kelvin,
)
from cortivra.layer6_agents.council_finalizer import (
    finalize_council_investigation_dossier,
)
from cortivra.layer6_agents.council_step_executors import (
    execute_adversarial_falsify_and_crystallize_step,
    execute_discover_symbolic_law_step,
    execute_solve_inverse_problem_step,
)
from cortivra.layer6_agents.deliverable_exporter import (
    compile_fmeca_and_safe_operating_area_matrix,
    export_turnkey_deliverable_bundle,
    format_8stage_execution_flight_recorder_log,
)
from cortivra.layer6_agents.dossier_renderer import (
    _render_dossier_html_report,
    _render_dossier_markdown_report,
    _render_trajectory_svg,
)
from cortivra.layer6_agents.empirical_observatory import (
    EmpiricalTelemetryDataset,
    fetch_open_science_observatory_telemetry,
    load_empirical_telemetry,
)
from cortivra.layer6_agents.diagnostic_briefing import (
    DifferentialEpistemicBriefing,
    VaultAnalogueMatch,
    build_differential_epistemic_briefing,
    format_compiler_or_shield_diagnostic,
    search_vault_isomorphisms,
)
from cortivra.layer6_agents.falsification_and_design import (
    DisambiguatingExperimentReport,
    FalsificationStressReport,
    _extract_symbolic_expression_from_brick,
    crystallize_verified_law,
    design_disambiguating_experiment,
    run_falsification_stress_test,
    verify_parametric_uncertainty_envelope,
)
from cortivra.layer6_agents.hypothesis_compiler import (
    EpistemicStepRecord,
    InverseProblemMode,
    ScientificDiscoveryDossier,
    ScientificInvestigationGoal,
    SelfRepairCompilationReport,
    compile_hypothesis_to_investigation_goal,
    compile_with_self_repair,
)
from cortivra.layer6_agents.meta_clients import (
    LLMInferenceRequest,
    LLMInferenceResponse,
    MultiProviderMetaClient,
    ROLE_SYSTEM_PROMPTS,
    TOOL_MATHEMATICAL_SCHEMAS,
    WORLD_MODEL_SYNTHESIS_PROTOCOL,
)
from cortivra.layer6_agents.multiscale_and_tournament import (
    PrimaryScientificFinding,
    _adapt_candidate_spec_bounds_to_telemetry,
    _build_primary_scientific_finding,
    _compute_subscale_timestep,
    _enforce_universal_world_model_closure,
    _execute_multi_hypothesis_crucible_tournament,
    _formulate_subscale_investigation_prompt,
    _spec_dict_to_symbolic_spec,
    _verify_bidirectional_multiscale_closure,
)
from cortivra.observability.causal_backtracker import (
    CausalContradictionReport,
    analyze_trace_and_backtrack_contradictions,
)
from cortivra.observability.flight_recorder import (
    AlternativeBranchRecord,
    CausalTraceDAG,
    FlightRecorder,
    active_recorder,
    emit_trace_node,
    suppress_inner_trace,
)
from cortivra.observability.forensic_report import (
    format_forensic_xray_report,
    serialize_trace_to_dict,
)


def _select_next_epistemic_operator(payload: dict[str, Any]) -> dict[str, Any]:
    """
    First-principles mathematical precondition selector over the council operator DAG:
    enforces causal ordering across zero-prior discovery, simulation/calibration,
    residual law discovery, active disambiguation, equilibrium/bifurcation analysis,
    Fisher sensitivity filtering, constrained inverse control, Lie conservation
    invariants, and 4-gate Popperian falsification.
    """
    completed: list[str] = list(payload.get("completed_actions", []))
    briefing: dict[str, Any] = dict(payload.get("epistemic_briefing", {}))
    zero_prior = bool(payload.get("zero_prior_model", False))
    has_obs = bool(payload.get("has_observed_data", False))
    num_competing = int(payload.get("num_competing_candidates", 0))
    req_inv = payload.get("requested_inverse_mode")
    target_obj = briefing.get("target_state_objective")
    param_dim = int(briefing.get("param_dim", 0))
    q_text = str(payload.get("question", "")).lower()

    if zero_prior and "discover_symbolic_law_or_latent_state" not in completed:
        return {
            "tool_name": "discover_symbolic_law_or_latent_state",
            "arguments": {"autonomous_from_scratch": True},
            "epistemic_rationale": (
                "Zero prior parametric law is provided; invoking autonomous Layer 5 "
                "symbolic vector-field discovery directly from empirical state telemetry."
            ),
        }

    if not has_obs and "simulate_and_inspect_dynamics" not in completed:
        return {
            "tool_name": "simulate_and_inspect_dynamics",
            "arguments": {},
            "epistemic_rationale": (
                "No empirical trajectory is loaded yet; integrating dx/dt = (J(x) - R(x)) grad H(x) "
                "forward and auditing 1st/2nd Law thermodynamic shields."
            ),
        }

    if has_obs and not zero_prior and "calibrate_and_diagnose_residual" not in completed and param_dim > 0:
        return {
            "tool_name": "calibrate_and_diagnose_residual",
            "arguments": {},
            "epistemic_rationale": (
                "Empirical state telemetry is present with tunable parameters; executing the 4-stage "
                "calibration funnel (Sobol -> Collocation -> Adam -> Levenberg-Marquardt) and auditing "
                "the residual vector field Phi(x) = dx_obs/dt - F_known(x)."
            ),
        }

    res_sum = briefing.get("residual_summary")
    if (
        isinstance(res_sum, dict)
        and bool(res_sum.get("has_unmodeled_mechanism", False))
        and "discover_symbolic_law_or_latent_state" not in completed
    ):
        is_hyst = bool(res_sum.get("is_non_markovian_hysteresis", False))
        t_class = str(res_sum.get("thermodynamic_class", "passive_dissipative"))
        return {
            "tool_name": "discover_symbolic_law_or_latent_state",
            "arguments": {
                "autonomous_from_scratch": False,
                "prefer_latent": is_hyst,
                "thermodynamic_slot": t_class,
            },
            "epistemic_rationale": (
                f"Residual audit detected an unmodeled physical mechanism "
                f"(SNR={res_sum.get('snr_ratio')}, R^2={res_sum.get('coherence_r2')}, "
                f"hysteresis={is_hyst}, class='{t_class}'); invoking Layer 5 Pareto symbolic discovery."
            ),
        }

    if num_competing >= 2 and "design_disambiguating_experiment" not in completed:
        return {
            "tool_name": "design_disambiguating_experiment",
            "arguments": {},
            "epistemic_rationale": (
                f"{num_competing} competing symbolic models remain viable on the Pareto front; "
                "maximizing trajectory divergence ||x_A(t) - x_B(t)||_2^2 to falsify impostors."
            ),
        }

    wants_eq_or_inv = bool(target_obj) or bool(req_inv) or any(
        k in q_text for k in ("equilibrium", "steady", "bifurcation", "restore", "stabilize", "cure", "control")
    )
    if wants_eq_or_inv and "map_equilibria_and_bifurcations" not in completed:
        return {
            "tool_name": "map_equilibria_and_bifurcations",
            "arguments": {},
            "epistemic_rationale": (
                "Mapping steady-state roots F(x*) = 0 on the conserved stoichiometric manifold "
                "and classifying local Jacobian eigenvalues prior to inverse control."
            ),
        }

    if (bool(target_obj) or bool(req_inv)) and param_dim > 0 and "analyze_fisher_sensitivity_and_leverage" not in completed:
        return {
            "tool_name": "analyze_fisher_sensitivity_and_leverage",
            "arguments": {},
            "epistemic_rationale": (
                "Computing Fisher Information Matrix F = J_theta^T J_theta to isolate stiff high-leverage "
                "parameters and filter out structurally sloppy combinations before solving the inverse problem."
            ),
        }

    if (bool(target_obj) or bool(req_inv)) and "solve_inverse_problem" not in completed:
        sens_sum = briefing.get("sensitivity_summary") or {}
        top_params = list(sens_sum.get("top_leverage_parameters", []))
        sloppy = set(sens_sum.get("sloppy_parameters", []))
        selected = [p for p in top_params if p not in sloppy] or top_params
        mode = str(req_inv or "counterfactual_intervention")
        return {
            "tool_name": "solve_inverse_problem",
            "arguments": {
                "mode": mode,
                "selected_parameter_names": selected,
            },
            "epistemic_rationale": (
                f"Solving constrained differentiable inverse problem (mode='{mode}') over "
                f"non-sloppy high-leverage parameters {selected} subject to homeostatic barriers."
            ),
        }

    if "extract_conservation_invariants_and_archetype" not in completed:
        return {
            "tool_name": "extract_conservation_invariants_and_archetype",
            "arguments": {},
            "epistemic_rationale": (
                "Extracting exact Lie-derivative conservation invariants (grad psi . F(x) = 0) "
                "and matching cross-domain LawVault operator isomorphisms."
            ),
        }

    if "adversarial_falsify_and_crystallize" not in completed:
        return {
            "tool_name": "adversarial_falsify_and_crystallize",
            "arguments": {"crystallize": True},
            "epistemic_rationale": (
                "Executing the 4-gate Popperian falsification stress suite (Layer 1 shields, "
                "thermal noise identifiability, OOD + 1x/2x/4x multi-horizon viability, and "
                "time-reversal/invariant stability) before LawVault crystallization."
            ),
        }

    return {
        "tool_name": "conclude_investigation",
        "arguments": {},
        "epistemic_rationale": (
            "All discovery, inverse control, invariant extraction, and 4-gate Popperian "
            "falsification checks have completed; concluding council investigation."
        ),
    }


class AutonomousScientificCouncil:
    """
    Closed-loop multi-agent scientific council orchestrating Layers 1–5 via
    differential epistemic telemetry and composable mathematical operators.
    """

    def __init__(
        self,
        client: MultiProviderMetaClient | None = None,
        vault: LawVault | None = None,
    ) -> None:
        self.client = client if client is not None else MultiProviderMetaClient()
        self.vault = vault

    def investigate(
        self,
        prompt: str,
        domain: str = "multiphysics",
        dt: float = 0.02,
        num_steps: int = 35,
        max_council_steps: int = 8,
        crystallize_into_vault: bool = True,
        observed_states: Any = None,
        observatory_stream: str | None = None,
        recursive_depth: int = 0,
        scale_level: int = 0,
    ) -> ScientificDiscoveryDossier:
        """
        One-Sentence Hypothesis Entry Point:
        Compile a natural-language scientific hypothesis or question (`prompt`) into a
        verified Port-Hamiltonian world model and execute the complete autonomous Council
        investigation loop end-to-end inside a unified `FlightRecorder` session.
        """
        if self.vault is None:
            default_vault = LawVault(sandbox_storage_path="runs/law_vault/crystallized_laws.json")
            with suppress_inner_trace():
                register_foundational_laws(default_vault)
            self.vault = default_vault

        telemetry_ds: EmpiricalTelemetryDataset | None = None
        if observatory_stream is not None and str(observatory_stream).strip():
            s_raw = str(observatory_stream).strip().lower()
            if s_raw in {"auto", "hunt", "public", "universal", "default"}:
                try:
                    from cortivra.layer6_agents.universal_telemetry_hunter import try_hunt_universal_public_telemetry

                    hunted = try_hunt_universal_public_telemetry(
                        prompt=prompt,
                        domain=domain,
                        stream_hint=s_raw,
                        num_steps=num_steps,
                        dt=dt,
                        timeout_sec=3.5,
                    )
                    if hunted is not None and hunted[1].passed_all_5_gates and hunted[0].live_network_fetched:
                        telemetry_ds = hunted[0]
                except Exception:
                    telemetry_ds = None
            else:
                telemetry_ds = fetch_open_science_observatory_telemetry(
                    stream_id=str(observatory_stream).strip(),
                    domain=domain,
                    num_steps=num_steps,
                    dt=dt,
                    prompt=prompt,
                )
        elif observed_states is not None:
            telemetry_ds = load_empirical_telemetry(observed_states, dt=dt)

        existing_rec = active_recorder()
        eff_steps = (
            min(int(telemetry_ds.states.shape[-2]), max(int(num_steps), 10))
            if telemetry_ds is not None
            else num_steps
        )
        if existing_rec is not None:
            goal = compile_hypothesis_to_investigation_goal(
                question=prompt,
                domain=domain,
                client=self.client,
                dt=telemetry_ds.dt if telemetry_ds is not None else dt,
                num_steps=eff_steps,
                max_council_steps=max_council_steps,
                crystallize_into_vault=crystallize_into_vault,
                telemetry_column_names=telemetry_ds.column_names if telemetry_ds is not None else None,
            )
            if telemetry_ds is not None:
                goal.observed_states = telemetry_ds.states
                goal.observed_dx_dt = None
                goal.telemetry_column_names = telemetry_ds.column_names
                goal.telemetry_source_description = telemetry_ds.source_description
                _adapt_candidate_spec_bounds_to_telemetry(goal.candidate_spec_dict, telemetry_ds.states)
            macro_dossier = self._execute_investigation(goal, existing_rec, scale_level=scale_level)
        else:
            with FlightRecorder.session(session_id=f"investigation_{domain}", question=prompt) as rec:
                goal = compile_hypothesis_to_investigation_goal(
                    question=prompt,
                    domain=domain,
                    client=self.client,
                    dt=telemetry_ds.dt if telemetry_ds is not None else dt,
                    num_steps=eff_steps,
                    max_council_steps=max_council_steps,
                    crystallize_into_vault=crystallize_into_vault,
                    telemetry_column_names=telemetry_ds.column_names if telemetry_ds is not None else None,
                )
                if telemetry_ds is not None:
                    goal.observed_states = telemetry_ds.states
                    goal.observed_dx_dt = None
                    goal.telemetry_column_names = telemetry_ds.column_names
                    goal.telemetry_source_description = telemetry_ds.source_description
                    _adapt_candidate_spec_bounds_to_telemetry(goal.candidate_spec_dict, telemetry_ds.states)
                macro_dossier = self._execute_investigation(goal, rec, scale_level=scale_level)

        if recursive_depth > 0:
            sub_prompt, sub_domain = _formulate_subscale_investigation_prompt(
                macro_dossier,
                next_scale_level=scale_level + 1,
            )
            sub_dt, _ = _compute_subscale_timestep(macro_dossier, parent_dt=dt)
            child_dossier = self.investigate(
                prompt=sub_prompt,
                domain=sub_domain,
                dt=sub_dt,
                num_steps=max(min(num_steps, 25), 15),
                max_council_steps=max_council_steps,
                crystallize_into_vault=crystallize_into_vault,
                recursive_depth=recursive_depth - 1,
                scale_level=scale_level + 1,
            )
            homog_cert = _verify_bidirectional_multiscale_closure(
                macro_dossier=macro_dossier,
                sub_dossier=child_dossier,
                macro_dt=dt,
                sub_dt=sub_dt,
                macro_steps=num_steps,
            )
            if (
                macro_dossier.primary_finding is not None
                and macro_dossier.primary_finding.physical_realization_blueprint is not None
            ):
                macro_dossier.primary_finding.physical_realization_blueprint[
                    "multiscale_homogenization_certificate"
                ] = homog_cert
            macro_dossier = replace(macro_dossier, subscale_dossiers=(child_dossier,))

        return macro_dossier

    def run_investigation(self, goal: ScientificInvestigationGoal) -> ScientificDiscoveryDossier:
        """
        Execute an autonomous closed-loop investigation for `goal`.
        """
        existing_rec = active_recorder()
        if existing_rec is not None:
            return self._execute_investigation(goal, existing_rec)
        with FlightRecorder.session(session_id=f"investigation_{goal.domain}", question=goal.question) as rec:
            return self._execute_investigation(goal, rec)

    def _execute_investigation(
        self,
        goal: ScientificInvestigationGoal,
        recorder: FlightRecorder,
        scale_level: int = 0,
    ) -> ScientificDiscoveryDossier:
        self_repair_report: SelfRepairCompilationReport | None = None
        compiler_diags: list[str] = []

        active_brick: LawBrick | None = goal.base_brick
        if active_brick is None and goal.candidate_spec_dict is not None:
            self_repair_report = compile_with_self_repair(
                goal.candidate_spec_dict,
                client=self.client,
            )
            active_brick = self_repair_report.compiled_brick
            compiler_diags.extend(self_repair_report.repair_diagnostics)

        zero_prior_mode = active_brick is None
        if active_brick is None:
            if goal.state_specs_for_autonomous:
                blank_spec = SymbolicSystemSpec(
                    name="autonomous_blank_carrier",
                    domain=goal.domain,
                    state_specs=goal.state_specs_for_autonomous,
                    param_specs=(),
                    time_unit=s,
                    energy_unit=DIMENSIONLESS,
                    hamiltonian_expr="0.0",
                )
                active_brick = compile_symbolic_system(blank_spec)
            elif goal.question.strip():
                synth_goal = compile_hypothesis_to_investigation_goal(
                    question=goal.question,
                    domain=goal.domain,
                    client=self.client,
                    dt=goal.dt,
                    num_steps=goal.num_steps,
                    max_council_steps=goal.max_council_steps,
                    crystallize_into_vault=goal.crystallize_into_vault,
                )
                assert synth_goal.candidate_spec_dict is not None
                self_repair_report = compile_with_self_repair(
                    synth_goal.candidate_spec_dict,
                    client=self.client,
                )
                active_brick = self_repair_report.compiled_brick
                compiler_diags.extend(self_repair_report.repair_diagnostics)
                zero_prior_mode = False
                if goal.target_state_objective is None and synth_goal.target_state_objective is not None:
                    goal.target_state_objective = synth_goal.target_state_objective
                if goal.inverse_mode is None and synth_goal.inverse_mode is not None:
                    goal.inverse_mode = synth_goal.inverse_mode
                if goal.observed_states is None and synth_goal.observed_states is not None:
                    goal.observed_states = synth_goal.observed_states
                if goal.observed_dx_dt is None and synth_goal.observed_dx_dt is not None:
                    goal.observed_dx_dt = synth_goal.observed_dx_dt
            else:
                raise ValueError(
                    "Either base_brick, candidate_spec_dict, or state_specs_for_autonomous must be provided."
                )

        active_params = (
            goal.baseline_params.detach().clone().to(torch.float64)
            if goal.baseline_params is not None
            else active_brick.default_params(batch_size=1, dtype=torch.float64)
        )
        if active_params.ndim == 1:
            active_params = active_params.unsqueeze(0)

        obs_2d: torch.Tensor | None = None
        if goal.observed_states is not None:
            raw_obs = goal.observed_states.detach().clone().to(torch.float64)
            obs_2d = raw_obs[0] if raw_obs.ndim == 3 else (raw_obs.unsqueeze(-1) if raw_obs.ndim == 1 else raw_obs)
            if obs_2d.shape[1] > active_brick.state_dim:
                obs_2d = obs_2d[:, : active_brick.state_dim]
                if goal.observed_dx_dt is not None:
                    goal.observed_dx_dt = goal.observed_dx_dt[..., : active_brick.state_dim]
            goal.observed_states = obs_2d
            if obs_2d.shape[1] < active_brick.state_dim and goal.observed_state_indices is None:
                goal.observed_state_indices = list(range(obs_2d.shape[1]))

        if goal.initial_state is not None:
            active_x0 = goal.initial_state.detach().clone().to(torch.float64)
            if active_x0.ndim == 1:
                active_x0 = active_x0.unsqueeze(0)
        elif obs_2d is not None:
            active_x0 = active_brick.default_state(batch_size=1, dtype=torch.float64)
            if goal.observed_state_indices is not None:
                active_x0[0, list(goal.observed_state_indices)] = obs_2d[0]
            else:
                active_x0[0] = obs_2d[0]
        else:
            active_x0 = active_brick.default_state(batch_size=1, dtype=torch.float64)

        active_states: torch.Tensor | None = None
        if obs_2d is not None:
            if obs_2d.shape[1] == active_brick.state_dim:
                active_states = obs_2d
            else:
                with suppress_inner_trace():
                    init_roll = rollout_trajectories(
                        active_brick,
                        active_x0,
                        active_params,
                        dt=goal.dt,
                        num_steps=int(obs_2d.shape[0]),
                        integrator="auto",
                    )
                full_init = init_roll.states[0].detach().clone()
                idx_obs = list(goal.observed_state_indices or range(obs_2d.shape[1]))
                full_init[: obs_2d.shape[0], idx_obs] = obs_2d
                active_states = full_init
        audit_cert: AuditCertificate | None = audit_law_brick(active_brick)
        cal_report: CalibrationAndResidualReport | None = None
        mem_report: NonMarkovianMemoryReport | None = None
        disc_report: ClosedLoopDiscoveryReport | None = None
        auto_disc_report: AutonomousDiscoveryReport | None = None
        disambig_report: DisambiguatingExperimentReport | None = None
        eq_report: EquilibriumLandscapeReport | None = None
        bif_report: BifurcationScanReport | None = None
        ident_report: SensitivityAndFisherReport | None = None
        interv_sol: InterventionSolution | None = None
        ctrl_res: OptimalControlResult | None = None
        mech_inv_report: SymbolicMechanismInventionReport | None = None
        retro_roll: RolloutResult | None = None
        bvp_res: BoundaryShootingResult | None = None
        inv_report: InvariantDiscoveryReport | None = None
        arch_report: ArchetypeFingerprint | None = None
        falsif_report: FalsificationStressReport | None = None
        cryst_record: CrystallizedLawRecord | None = None
        vault_matches: list[VaultAnalogueMatch] = []
        hypothesis_tournament: list[dict[str, Any]] | None = None

        competing_list: list[tuple[LawBrick, torch.Tensor]] = list(goal.competing_candidates)
        completed_actions: list[str] = []
        step_records: list[EpistemicStepRecord] = []
        pending_step_entries: list[tuple[int, Any, str, dict[str, Any], str, str, dict[str, Any], str]] = []
        is_live_network_client = getattr(self.client, "_http_transport", None) is None

        for step_idx in range(1, goal.max_council_steps + 1):
            with suppress_inner_trace():
                briefing = build_differential_epistemic_briefing(
                    brick=active_brick,
                    states=active_states,
                    params=active_params,
                    audit_certificate=audit_cert,
                    equilibrium_report=eq_report,
                    identifiability_report=ident_report,
                    calibration_report=cal_report,
                    memory_report=mem_report,
                    invariant_report=inv_report,
                    vault=self.vault,
                    compiler_diagnostics=compiler_diags,
                    target_state_objective=goal.target_state_objective,
                    lightweight_step_mode=True,
                )

            b_dict = briefing.to_dict()
            needs_emp_cal = bool(
                goal.observed_states is not None
                and "calibrate_and_diagnose_residual" not in completed_actions
            )
            needs_emp_disc = bool(
                goal.observed_states is not None
                and "calibrate_and_diagnose_residual" in completed_actions
                and "discover_symbolic_law_or_latent_state" not in completed_actions
                and (
                    zero_prior_mode
                    or goal.telemetry_source_description is not None
                    or (cal_report is not None and cal_report.latent_mechanism_required)
                )
            )
            if needs_emp_disc and isinstance(b_dict.get("residual_summary"), dict):
                b_dict["residual_summary"]["has_unmodeled_mechanism"] = True

            prompt_payload = {
                "task_type": "select_next_action",
                "question": goal.question,
                "completed_actions": list(completed_actions),
                "has_observed_data": goal.observed_states is not None,
                "zero_prior_model": zero_prior_mode,
                "num_competing_candidates": len(competing_list),
                "requested_inverse_mode": goal.inverse_mode,
                "has_vault": bool(self.vault is not None and goal.crystallize_into_vault),
                "available_tools": list(TOOL_MATHEMATICAL_SCHEMAS.keys()),
                "tool_schemas": (
                    {k: v.get("arguments", {}) for k, v in TOOL_MATHEMATICAL_SCHEMAS.items()}
                    if is_live_network_client
                    else TOOL_MATHEMATICAL_SCHEMAS
                ),
                "epistemic_briefing": (
                    {
                        "system_name": b_dict.get("system_name"),
                        "domain": b_dict.get("domain"),
                        "state_dim": b_dict.get("state_dim"),
                        "param_dim": b_dict.get("param_dim"),
                        "thermodynamic_audit_passed": b_dict.get("thermodynamic_audit_passed"),
                        "equilibrium_summary": b_dict.get("equilibrium_summary"),
                        "sensitivity_summary": b_dict.get("sensitivity_summary"),
                        "residual_summary": b_dict.get("residual_summary"),
                        "target_state_objective": b_dict.get("target_state_objective"),
                    }
                    if is_live_network_client
                    else b_dict
                ),
            }

            req_step = LLMInferenceRequest(
                role="muse_wonder",
                system_prompt=ROLE_SYSTEM_PROMPTS["muse_wonder"],
                user_prompt=json.dumps(prompt_payload, indent=2),
                temperature=0.0,
                require_json=True,
            )

            if is_live_network_client:
                step_llm_fut = self.client.invoke_background(req_step)
                scheduled_action = _select_next_epistemic_operator(prompt_payload)
                tool_name = str(scheduled_action.get("tool_name", "conclude_investigation"))
                tool_args = dict(scheduled_action.get("arguments", {}))
                rationale = str(
                    scheduled_action.get(
                        "epistemic_rationale",
                        "Proceeding with mathematical evaluation.",
                    )
                )
            else:
                step_llm_fut = None
                llm_resp = self.client.invoke(req_step)
                parsed = llm_resp.parsed_json or {}
                tool_name = str(parsed.get("tool_name", ""))
                tool_args = dict(parsed.get("arguments", {}))
                rationale = str(parsed.get("epistemic_rationale", "Proceeding with mathematical evaluation."))

                valid_tools = set(TOOL_MATHEMATICAL_SCHEMAS.keys())
                if (
                    tool_name not in valid_tools
                    or (tool_name != "conclude_investigation" and tool_name in completed_actions)
                    or (tool_name == "calibrate_and_diagnose_residual" and goal.observed_states is None)
                    or (tool_name == "discover_symbolic_law_or_latent_state" and goal.observed_states is None and not zero_prior_mode)
                    or ("adversarial_falsify_and_crystallize" in completed_actions and tool_name != "conclude_investigation")
                    or (tool_name == "design_disambiguating_experiment" and len(competing_list) < 2)
                    or (needs_emp_cal and tool_name != "calibrate_and_diagnose_residual")
                    or (needs_emp_disc and tool_name != "discover_symbolic_law_or_latent_state")
                ):
                    scheduled_action = _select_next_epistemic_operator(prompt_payload)
                    tool_name = str(scheduled_action.get("tool_name", "conclude_investigation"))
                    tool_args = dict(scheduled_action.get("arguments", {}))
                    rationale = str(scheduled_action.get("epistemic_rationale", rationale))

            if tool_name == "conclude_investigation":
                break

            completed_actions.append(tool_name)
            summary_str = ""
            metrics: dict[str, Any] = {}

            if tool_name == "simulate_and_inspect_dynamics":
                roll = rollout_trajectories(
                    active_brick,
                    active_x0,
                    active_params,
                    dt=goal.dt,
                    num_steps=goal.num_steps,
                    integrator="auto",
                )
                active_states = roll.states[0]
                audit_cert = audit_law_brick(
                    active_brick,
                    sample_state=active_states,
                    sample_params=active_params.expand(active_states.shape[0], -1),
                )
                summary_str = (
                    f"Simulated {goal.num_steps} steps using integrator='{roll.integrator}' "
                    f"(viable_fraction={roll.viable_fraction:.3f}, max_energy_drift={roll.max_energy_drift:.3e})."
                )
                metrics = {
                    "integrator": roll.integrator,
                    "viable_fraction": roll.viable_fraction,
                    "max_energy_drift": roll.max_energy_drift,
                }

            elif tool_name == "calibrate_and_diagnose_residual":
                obs_x = goal.observed_states if goal.observed_states is not None else active_states
                assert obs_x is not None
                x_first = obs_x[0] if obs_x.ndim == 3 else obs_x
                cal_report = calibrate_and_audit_residual(
                    brick=active_brick,
                    observed_states=obs_x,
                    dt=goal.dt,
                    observed_dx_dt=goal.observed_dx_dt,
                    observed_state_indices=goal.observed_state_indices,
                    num_initial_samples=64,
                    num_collocation_survivors=16,
                    num_gradient_starts=4,
                    gradient_steps=45,
                )
                active_params = cal_report.best_params.detach().clone()
                if goal.observed_state_indices is not None and x_first.shape[1] < active_brick.state_dim:
                    full_recon = cal_report.funnel_report.best_rollout.states[0].detach().clone()
                    t_min = min(full_recon.shape[0], x_first.shape[0])
                    idx_obs = list(goal.observed_state_indices)
                    full_recon[:t_min, idx_obs] = x_first[:t_min]
                    active_states = full_recon[:t_min]
                    active_x0 = active_states[0:1].detach().clone()
                res_flow = cal_report.residual_analysis.residual_flow[: x_first.shape[0]]
                res_flow_obs = res_flow[:, : x_first.shape[1]] if res_flow.ndim == 2 else res_flow
                mem_report = detect_non_markovian_memory(x_first, res_flow_obs, dt=goal.dt)
                obs_tag = (
                    f", partial_sensors={len(goal.observed_state_indices)}/{active_brick.state_dim}"
                    if goal.observed_state_indices is not None
                    else ""
                )
                summary_str = (
                    f"Calibrated {active_brick.name}{obs_tag}: trajectory_r2={cal_report.funnel_report.trajectory_r2:.5f}, "
                    f"residual_max_snr={cal_report.residual_analysis.max_snr:.2f}, "
                    f"latent_required={cal_report.latent_mechanism_required}, "
                    f"non_markovian={mem_report.is_non_markovian}."
                )
                metrics = {
                    "trajectory_r2": cal_report.funnel_report.trajectory_r2,
                    "residual_max_snr": cal_report.residual_analysis.max_snr,
                    "latent_mechanism_required": cal_report.latent_mechanism_required,
                    "is_non_markovian": mem_report.is_non_markovian,
                }

            elif tool_name == "discover_symbolic_law_or_latent_state":
                (
                    zero_prior_mode,
                    active_brick,
                    active_x0,
                    active_params,
                    active_states,
                    audit_cert,
                    mem_report,
                    inv_report,
                    arch_report,
                    disc_report,
                    auto_disc_report,
                    summary_str,
                    metrics,
                ) = execute_discover_symbolic_law_step(
                    client=self.client,
                    goal=goal,
                    tool_args=tool_args,
                    zero_prior_mode=zero_prior_mode,
                    active_brick=active_brick,
                    active_x0=active_x0,
                    active_params=active_params,
                    active_states=active_states,
                    audit_cert=audit_cert,
                    cal_report=cal_report,
                    mem_report=mem_report,
                    inv_report=inv_report,
                    arch_report=arch_report,
                    disc_report=disc_report,
                    auto_disc_report=auto_disc_report,
                )

            elif tool_name == "design_disambiguating_experiment":
                if len(competing_list) >= 2:
                    surviving_brick, surviving_params = competing_list[0]
                    for next_cand, next_p in competing_list[1:]:
                        disambig_report = design_disambiguating_experiment(
                            candidate_a=surviving_brick,
                            params_a=surviving_params,
                            candidate_b=next_cand,
                            params_b=next_p,
                            baseline_x0=active_x0,
                            dt=goal.dt,
                            num_steps=min(goal.num_steps, 25),
                            oracle_brick=goal.oracle_brick,
                            oracle_params=goal.oracle_params,
                            observed_trajectory=goal.observed_states,
                        )
                        if disambig_report.winning_hypothesis == "candidate_b":
                            surviving_brick, surviving_params = next_cand, next_p
                    active_brick, active_params = surviving_brick, surviving_params
                    competing_list = [(active_brick, active_params)]
                    assert disambig_report is not None
                    summary_str = (
                        f"Designed disambiguating experiment: amplification={disambig_report.divergence_amplification_ratio:.2f}x, "
                        f"winning_hypothesis='{disambig_report.winning_hypothesis}' "
                        f"(selected '{active_brick.name}')."
                    )
                    metrics = {
                        "divergence_amplification_ratio": disambig_report.divergence_amplification_ratio,
                        "winning_hypothesis": disambig_report.winning_hypothesis,
                        "selected_brick": active_brick.name,
                    }
                    self.client.invoke(
                        LLMInferenceRequest(
                            role="muse_spark_open",
                            system_prompt=ROLE_SYSTEM_PROMPTS["muse_spark_open"],
                            user_prompt=json.dumps(
                                {
                                    "task_type": "adversarial_red_team_review",
                                    "passed_all_gates": True,
                                    "noisy_vector_field_r2": 1.0,
                                    "ood_viable_fraction": 1.0,
                                    "selected_brick": active_brick.name,
                                }
                            ),
                            temperature=0.0,
                            require_json=True,
                        )
                    )

            elif tool_name == "map_equilibria_and_bifurcations":
                eq_report = discover_equilibrium_landscape(
                    brick=active_brick,
                    params=active_params,
                    num_probes=20,
                )
                n_eq = int(eq_report.distinct_equilibria.shape[0])
                tunable_names = [pp.name for pp in active_brick.param_specs if pp.tunable]
                if tunable_names and n_eq >= 1:
                    scan_p = tunable_names[0]
                    if ident_report is not None and ident_report.ranked_param_names:
                        for rk_name in ident_report.ranked_param_names:
                            if rk_name in tunable_names:
                                scan_p = rk_name
                                break
                    p_spec = next(pp for pp in active_brick.param_specs if pp.name == scan_p)
                    p_cur = float(active_params[0, active_brick.param_specs.index(p_spec)].item())
                    lo_g = max(p_spec.min_value, p_cur * 0.25 if p_cur > 0 else p_cur - 2.0)
                    hi_g = min(p_spec.max_value, p_cur * 2.5 if p_cur > 0 else p_cur + 2.0)
                    if hi_g > lo_g + 1e-6:
                        grid = torch.linspace(lo_g, hi_g, 10, dtype=torch.float64, device=active_x0.device)
                        bif_report = scan_parameter_bifurcation(
                            brick=active_brick,
                            param_name=scan_p,
                            param_values=grid,
                            initial_guess=eq_report.distinct_equilibria[0:1],
                            base_params=active_params,
                        )
                summary_str = (
                    f"Mapped equilibrium landscape: found {n_eq} distinct equilibria "
                    f"(stable={len(eq_report.stable_attractor_indices)}, "
                    f"unstable={len(eq_report.unstable_fixed_point_indices)})."
                )
                metrics = {
                    "num_equilibria": n_eq,
                    "num_stable_attractors": len(eq_report.stable_attractor_indices),
                }

            elif tool_name == "analyze_fisher_sensitivity_and_leverage":
                target_map = goal.target_state_objective or {}

                def _obj_fn(roll: RolloutResult) -> torch.Tensor:
                    if target_map:
                        loss_acc = torch.zeros((), dtype=torch.float64, device=roll.states.device)
                        for idx_s, sp_s in enumerate(active_brick.state_specs):
                            if sp_s.name in target_map:
                                t_val = float(target_map[sp_s.name])
                                loss_acc = loss_acc + torch.mean((roll.states[:, -5:, idx_s] - t_val) ** 2)
                        return loss_acc
                    return torch.mean(roll.states[:, -1, :] ** 2)

                ident_report = compute_parameter_leverage_and_fisher(
                    brick=active_brick,
                    initial_state=active_x0,
                    params=active_params,
                    dt=goal.dt,
                    num_steps=min(goal.num_steps, 25),
                    objective_fn=_obj_fn,
                )
                summary_str = (
                    f"Computed Fisher Information Matrix and restorative leverage: "
                    f"ranked_params={ident_report.ranked_param_names}."
                )
                metrics = {
                    "ranked_param_names": list(ident_report.ranked_param_names),
                }

            elif tool_name == "solve_inverse_problem":
                (
                    active_brick,
                    active_x0,
                    active_params,
                    active_states,
                    audit_cert,
                    bif_report,
                    mech_inv_report,
                    disambig_report,
                    interv_sol,
                    ctrl_res,
                    retro_roll,
                    bvp_res,
                    hypothesis_tournament,
                    summary_str,
                    metrics,
                ) = execute_solve_inverse_problem_step(
                    client=self.client,
                    goal=goal,
                    tool_args=tool_args,
                    active_brick=active_brick,
                    active_x0=active_x0,
                    active_params=active_params,
                    active_states=active_states,
                    audit_cert=audit_cert,
                    bif_report=bif_report,
                    mech_inv_report=mech_inv_report,
                    disambig_report=disambig_report,
                    interv_sol=interv_sol,
                    ctrl_res=ctrl_res,
                    retro_roll=retro_roll,
                    bvp_res=bvp_res,
                    hypothesis_tournament=hypothesis_tournament,
                )

            elif tool_name == "extract_conservation_invariants_and_archetype":
                if active_states is None:
                    roll = rollout_trajectories(
                        active_brick,
                        active_x0,
                        active_params,
                        dt=goal.dt,
                        num_steps=goal.num_steps,
                        integrator="auto",
                    )
                    active_states = roll.states[0]
                use_rich_inv_basis = bool(active_states.shape[0] >= active_brick.state_dim * 12)
                inv_report = discover_conservation_invariants(
                    brick=active_brick,
                    states=active_states,
                    params=active_params,
                    max_degree=2 if use_rich_inv_basis else 1,
                    include_cross_terms=use_rich_inv_basis,
                    include_trigonometric=use_rich_inv_basis,
                    include_logarithmic=use_rich_inv_basis,
                )
                if (
                    inv_report.num_invariants == 0
                    and auto_disc_report is not None
                    and auto_disc_report.invariant_report.num_invariants > 0
                ):
                    inv_report = auto_disc_report.invariant_report
                if arch_report is None:
                    arch_report = classify_mathematical_archetype(
                        active_brick,
                        candidate=(disc_report.winning_candidate if disc_report is not None else None),
                        is_latent_augmented=bool(
                            disc_report is not None and disc_report.latent_augmentation is not None
                        ),
                    )
                vault_matches = search_vault_isomorphisms(active_brick, vault=self.vault)
                summary_str = (
                    f"Extracted {inv_report.num_invariants} conservation invariant(s) and "
                    f"{len(vault_matches)} cross-domain LawVault isomorphism(s) "
                    f"(archetype='{arch_report.archetype_name}')."
                )
                metrics = {
                    "num_invariants": inv_report.num_invariants,
                    "archetype_name": arch_report.archetype_name,
                    "num_vault_matches": len(vault_matches),
                }

            elif tool_name == "adversarial_falsify_and_crystallize":
                (
                    active_params,
                    active_states,
                    audit_cert,
                    interv_sol,
                    falsif_report,
                    cryst_record,
                    summary_str,
                    metrics,
                ) = execute_adversarial_falsify_and_crystallize_step(
                    client=self.client,
                    vault=self.vault,
                    goal=goal,
                    active_brick=active_brick,
                    active_x0=active_x0,
                    active_params=active_params,
                    active_states=active_states,
                    inv_report=inv_report,
                    interv_sol=interv_sol,
                    hypothesis_tournament=hypothesis_tournament,
                )

            if is_live_network_client:
                active_provs = self.client.configured_providers()
                live_prov_label = active_provs[0] if active_provs else "baseten"
                emit_trace_node(
                    layer="layer6_agents",
                    operation=f"council_step_{tool_name}",
                    target_name=active_brick.name,
                    selected_branch_id=tool_name,
                    selected_expression=summary_str,
                    metrics={
                        "step_index": step_idx,
                        "agent_role": "muse_wonder",
                        "provider_used": live_prov_label,
                        **metrics,
                    },
                )
                pending_step_entries.append(
                    (
                        step_idx,
                        step_llm_fut,
                        tool_name,
                        tool_args,
                        rationale,
                        summary_str,
                        metrics,
                        active_brick.name,
                    )
                )
            else:
                step_records.append(
                    EpistemicStepRecord(
                        step_index=step_idx,
                        agent_role=llm_resp.role,
                        provider_used=llm_resp.provider_used,
                        model_used=llm_resp.model_used,
                        tool_name=tool_name,
                        tool_arguments=tool_args,
                        epistemic_rationale=rationale,
                        execution_summary=summary_str,
                        key_metrics=metrics,
                    )
                )
                emit_trace_node(
                    layer="layer6_agents",
                    operation=f"council_step_{tool_name}",
                    target_name=active_brick.name,
                    selected_branch_id=tool_name,
                    selected_expression=summary_str,
                    metrics={
                        "step_index": step_idx,
                        "agent_role": llm_resp.role,
                        "provider_used": llm_resp.provider_used,
                        **metrics,
                    },
                )

        return finalize_council_investigation_dossier(
            council=self,
            goal=goal,
            recorder=recorder,
            scale_level=scale_level,
            active_brick=active_brick,
            active_x0=active_x0,
            active_params=active_params,
            active_states=active_states,
            audit_cert=audit_cert,
            self_repair_report=self_repair_report,
            compiler_diags=compiler_diags,
            cal_report=cal_report,
            mem_report=mem_report,
            disc_report=disc_report,
            auto_disc_report=auto_disc_report,
            disambig_report=disambig_report,
            eq_report=eq_report,
            bif_report=bif_report,
            ident_report=ident_report,
            interv_sol=interv_sol,
            ctrl_res=ctrl_res,
            mech_inv_report=mech_inv_report,
            retro_roll=retro_roll,
            bvp_res=bvp_res,
            inv_report=inv_report,
            arch_report=arch_report,
            falsif_report=falsif_report,
            cryst_record=cryst_record,
            vault_matches=vault_matches,
            hypothesis_tournament=hypothesis_tournament,
            step_records=step_records,
            pending_step_entries=pending_step_entries if is_live_network_client else None,
        )


def investigate(
    prompt: str,
    domain: str = "multiphysics",
    client: MultiProviderMetaClient | None = None,
    vault: LawVault | None = None,
    dt: float = 0.02,
    num_steps: int = 35,
    max_council_steps: int = 8,
    crystallize_into_vault: bool = True,
    observed_states: Any = None,
    observatory_stream: str | None = None,
    recursive_depth: int = 0,
) -> ScientificDiscoveryDossier:
    """
    Universal One-Sentence Hypothesis Entry Point (`cortivra.investigate`):
    Takes a single sentence or short paragraph scientific hypothesis (`prompt`),
    translates it into a dimensionally homogeneous Port-Hamiltonian world model,
    compiles & self-repairs the symbolic equations against Layer 1 axioms, executes
    the full `AutonomousScientificCouncil` discovery/intervention/mechanism-invention/falsification
    loop, and returns a machine-verified `ScientificDiscoveryDossier`.
    """
    council = AutonomousScientificCouncil(client=client, vault=vault)
    return council.investigate(
        prompt=prompt,
        domain=domain,
        dt=dt,
        num_steps=num_steps,
        max_council_steps=max_council_steps,
        crystallize_into_vault=crystallize_into_vault,
        observed_states=observed_states,
        observatory_stream=observatory_stream,
        recursive_depth=recursive_depth,
    )
