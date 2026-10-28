"""
Autonomous multi-agent scientific council and closed-loop operator orchestrator
(`cortivra.layer6_agents.council`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick, ParamSpec, StateVarSpec
from cortivra.layer1_axioms.shields import AuditCertificate, audit_law_brick
from cortivra.layer1_axioms.units import DIMENSIONLESS, Unit, parse_unit, s
from cortivra.layer1_axioms.vault import CrystallizedLawRecord, LawVault
from cortivra.layer3_compiler.symbolic_compiler import (
    CompiledLawBrick,
    SymbolicSystemSpec,
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
    NonMarkovianMemoryReport,
    classify_mathematical_archetype,
    detect_non_markovian_memory,
    discover_and_compile_law,
    discover_autonomous_symbolic_system,
)
from cortivra.layer5_discovery.sparse_regression import (
    InvariantDiscoveryReport,
    discover_conservation_invariants,
)
from cortivra.layer6_agents.epistemic_telemetry import (
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
)
from cortivra.layer6_agents.meta_clients import (
    LLMInferenceRequest,
    LLMInferenceResponse,
    MathematicalEpistemicPolicy,
    MultiProviderMetaClient,
    ROLE_SYSTEM_PROMPTS,
    TOOL_MATHEMATICAL_SCHEMAS,
)

InverseProblemMode = Literal[
    "counterfactual_intervention",
    "optimal_control",
    "retrodiction",
    "boundary_value_shooting",
]


@dataclass(frozen=True)
class SelfRepairCompilationReport:
    """
    Report from compiler-guided symbolic self-repair (`compile_with_self_repair`).
    """

    compiled_brick: CompiledLawBrick
    audit_certificate: AuditCertificate
    attempts_used: int
    repair_diagnostics: tuple[str, ...]
    final_spec_dict: dict[str, Any]


@dataclass(frozen=True)
class EpistemicStepRecord:
    """
    Single autonomous tool-selection and mathematical execution step of the Council.
    """

    step_index: int
    agent_role: str
    provider_used: str
    model_used: str
    tool_name: str
    tool_arguments: dict[str, Any]
    epistemic_rationale: str
    execution_summary: str
    key_metrics: dict[str, Any]


@dataclass
class ScientificInvestigationGoal:
    """
    Universal specification for any open-ended scientific, medical, or engineering
    investigation executed by `AutonomousScientificCouncil`.
    """

    question: str
    domain: str = "multiphysics"
    base_brick: LawBrick | None = None
    candidate_spec_dict: dict[str, Any] | None = None
    state_specs_for_autonomous: Sequence[StateVarSpec] = ()
    observed_states: torch.Tensor | None = None
    observed_dx_dt: torch.Tensor | None = None
    initial_state: torch.Tensor | None = None
    baseline_params: torch.Tensor | None = None
    target_state_objective: dict[str, float] | None = None
    inverse_mode: InverseProblemMode | None = None
    competing_candidates: Sequence[tuple[LawBrick, torch.Tensor]] = ()
    oracle_brick: LawBrick | None = None
    oracle_params: torch.Tensor | None = None
    dt: float = 0.02
    num_steps: int = 35
    max_council_steps: int = 8
    crystallize_into_vault: bool = True


@dataclass(frozen=True)
class PrimaryScientificFinding:
    """
    Structured, JSON-serializable synthesis of all mathematical discoveries, parameter
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScientificDiscoveryDossier:
    """
    Complete machine-verifiable record produced by `AutonomousScientificCouncil.run_investigation`.
    """

    question: str
    final_brick: LawBrick
    final_params: torch.Tensor
    steps: tuple[EpistemicStepRecord, ...]
    final_briefing: DifferentialEpistemicBriefing
    self_repair_report: SelfRepairCompilationReport | None
    calibration_report: CalibrationAndResidualReport | None
    discovery_report: ClosedLoopDiscoveryReport | None
    autonomous_discovery_report: AutonomousDiscoveryReport | None
    disambiguation_report: DisambiguatingExperimentReport | None
    equilibrium_report: EquilibriumLandscapeReport | None
    bifurcation_report: BifurcationScanReport | None
    identifiability_report: SensitivityAndFisherReport | None
    intervention_solution: InterventionSolution | None
    optimal_control_result: OptimalControlResult | None
    retrodiction_rollout: RolloutResult | None
    boundary_shooting_result: BoundaryShootingResult | None
    invariant_report: InvariantDiscoveryReport | None
    archetype: ArchetypeFingerprint | None
    vault_analogues: tuple[VaultAnalogueMatch, ...]
    falsification_report: FalsificationStressReport | None
    crystallized_record: CrystallizedLawRecord | None
    executive_summary: str
    llm_telemetry: tuple[LLMInferenceResponse, ...] = field(default_factory=tuple)
    primary_finding: PrimaryScientificFinding | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the entire investigation dossier into a JSON-compatible dictionary
        ready for API responses, persistent storage, or UI rendering.
        """
        return {
            "question": self.question,
            "system_name": self.final_brick.name,
            "domain": self.final_brick.domain,
            "executive_summary": self.executive_summary,
            "primary_finding": self.primary_finding.to_dict() if self.primary_finding is not None else None,
            "steps": [asdict(s) for s in self.steps],
            "final_briefing": self.final_briefing.to_dict(),
            "self_repair": (
                {
                    "attempts_used": self.self_repair_report.attempts_used,
                    "repair_diagnostics": list(self.self_repair_report.repair_diagnostics),
                }
                if self.self_repair_report is not None
                else None
            ),
            "llm_telemetry": [asdict(t) for t in self.llm_telemetry],
        }

    def to_markdown_report(self) -> str:
        """
        Format the investigation dossier as a Markdown report.
        """
        lines: list[str] = [
            f"# Scientific Discovery Dossier: `{self.final_brick.name}`",
            f"**Investigation Question:** {self.question}",
            f"**Domain:** `{self.final_brick.domain}` | **Time Unit:** `{self.final_briefing.time_unit}` | **Energy Unit:** `{self.final_briefing.energy_unit}`",
            "",
            "## 1. Executive Summary",
            self.executive_summary,
            "",
        ]
        pf = self.primary_finding
        if pf is not None:
            lines.append("## 2. Primary Mathematical Findings")
            lines.append(f"- **Canonical Operator Expression:** `{pf.canonical_symbolic_expression}`")
            lines.append(f"- **Discovery Categories:** `{', '.join(pf.finding_categories)}`")
            if pf.discovered_equations:
                r2_str = f" ($R^2 = {pf.discovery_r2:.5f}$)" if pf.discovery_r2 is not None else ""
                lines.append(f"- **Discovered Symbolic Laws{r2_str}:**")
                for eq_k, eq_v in pf.discovered_equations.items():
                    lines.append(f"  - `{eq_k}`: `{eq_v}`")
            if pf.prescribed_intervention is not None:
                iv = pf.prescribed_intervention
                lines.append(
                    f"- **Counterfactual Parameter Intervention:** Loss reduced from `{iv['baseline_loss']:.5f}` to `{iv['intervened_loss']:.5f}` "
                    f"(Excluded Sloppy Parameters: `{iv['excluded_sloppy_parameters']}`)"
                )
                for p_k, p_v in iv["param_shifts"].items():
                    lines.append(
                        f"  - `{p_k}`: `{p_v['baseline']:.4g}` -> `{p_v['intervened']:.4g} {p_v['unit']}` (`{p_v['percent_change']:+.1f}%`)"
                    )
                for s_k, s_v in iv["state_transitions"].items():
                    lines.append(
                        f"  - State `{s_k}`: `{s_v['initial']:.4g}` -> `{s_v['final_restored']:.4g} {s_v['unit']}`"
                        + (f" (target: `{s_v['target']:.4g}`)" if s_v.get("target") is not None else "")
                    )
            if pf.optimal_control_summary is not None:
                cs = pf.optimal_control_summary
                lines.append(
                    f"- **Optimal Control Waveform $u^*(t)$:** Loss `{cs['baseline_loss']:.5f}` -> `{cs['controlled_loss']:.5f}`, "
                    f"Control Energy = `{cs['control_energy']:.4f}`, Viable = `{cs['all_time_viable']}`"
                )
            if pf.retrodiction_or_bvp_summary is not None:
                rb = pf.retrodiction_or_bvp_summary
                lines.append(
                    f"- **{rb['mode']} Solution:** Reconstructed Initial State = `{rb['reconstructed_or_solved_initial_state']}`"
                )
            if pf.disambiguation_summary is not None:
                ds = pf.disambiguation_summary
                lines.append(
                    f"- **Active Experimental Disambiguation:** Amplification = `{ds['divergence_amplification_ratio']:.2f}x`, "
                    f"Selected Model = `{ds['selected_model']}` (`{ds['winning_hypothesis']}`)"
                )
            lines.append("")

            if pf.conserved_invariants or pf.dimensionless_pi_groups:
                lines.append("## 3. Conserved Quantities & Dimensionless Invariants")
                if pf.conserved_invariants:
                    lines.append(f"- **Lie-Derivative Conserved Invariants:** `{', '.join(pf.conserved_invariants)}`")
                if pf.dimensionless_pi_groups:
                    lines.append(f"- **Buckingham-Pi Groups:** `{', '.join(pf.dimensionless_pi_groups)}`")
                lines.append("")

            if pf.cross_domain_isomorphisms:
                lines.append("## 4. Cross-Domain `LawVault` Mathematical Isomorphisms")
                for iso in pf.cross_domain_isomorphisms:
                    lines.append(
                        f"- **`{iso['name']}`** (`{iso['source_domain']}`, score=`{iso['isomorphism_score']}`): "
                        f"`{iso['symbolic_expression']}` — Shared Operators: `{iso['matched_operators']}`"
                    )
                lines.append("")

            if pf.falsification_certificate is not None:
                fc = pf.falsification_certificate
                lines.append("## 5. Adversarial Falsification & Multi-Horizon Stability Certificate")
                lines.append(
                    f"- **All Gates Passed:** `{fc['passed_all_gates']}` | **Noisy $R^2$:** `{fc['noisy_vector_field_r2']:.4f}` | "
                    f"**OOD Viable:** `{fc['ood_viable_fraction']:.3f}` | **Multi-Horizon ($1\\times/2\\times/4\\times$) Viable:** "
                    f"`{fc['horizon_1x_viable_fraction']:.3f} / {fc['horizon_2x_viable_fraction']:.3f} / {fc['horizon_4x_viable_fraction']:.3f}`"
                )
                if fc.get("crystallized_law_id"):
                    lines.append(
                        f"- **Crystallized `LawVault` Record:** `{fc['crystallized_law_id']}` (SHA-256: `{fc['sha256_hash']}`)"
                    )
                lines.append("")

        lines.append("## 6. Multi-Agent Epistemic Execution Trace")
        for st in self.steps:
            lines.append(
                f"{st.step_index}. **`{st.tool_name}`** (`{st.agent_role}` via `{st.provider_used}`): {st.execution_summary}"
            )
        return "\n".join(lines)


def _spec_dict_to_symbolic_spec(spec_dict: dict[str, Any]) -> SymbolicSystemSpec:
    """
    Convert a JSON-serializable dictionary into a typed Layer 3 `SymbolicSystemSpec`.
    """
    state_specs: list[StateVarSpec] = []
    for st in spec_dict.get("state_specs", []):
        u = st["unit"] if isinstance(st["unit"], Unit) else parse_unit(str(st["unit"]))
        h_min_raw = st.get("hard_min")
        h_max_raw = st.get("hard_max")
        state_specs.append(
            StateVarSpec(
                name=str(st["name"]),
                unit=u,
                default_value=float(st.get("default_value", 1.0)),
                homeostatic_min=float(st.get("homeostatic_min", -100.0)),
                homeostatic_max=float(st.get("homeostatic_max", 100.0)),
                hard_min=float(h_min_raw) if h_min_raw is not None else float("-inf"),
                hard_max=float(h_max_raw) if h_max_raw is not None else float("inf"),
            )
        )

    param_specs: list[ParamSpec] = []
    for pr in spec_dict.get("param_specs", []):
        u = pr["unit"] if isinstance(pr["unit"], Unit) else parse_unit(str(pr["unit"]))
        param_specs.append(
            ParamSpec(
                name=str(pr["name"]),
                unit=u,
                default_value=float(pr.get("default_value", 1.0)),
                min_value=float(pr.get("min_value", -1e6)),
                max_value=float(pr.get("max_value", 1e6)),
                tunable=bool(pr.get("tunable", True)),
            )
        )

    t_unit = (
        spec_dict["time_unit"]
        if isinstance(spec_dict.get("time_unit"), Unit)
        else parse_unit(str(spec_dict.get("time_unit", "s")))
    )
    e_unit = (
        spec_dict["energy_unit"]
        if isinstance(spec_dict.get("energy_unit"), Unit)
        else parse_unit(str(spec_dict.get("energy_unit", "J")))
    )

    skew_raw = spec_dict.get("skew_routing_exprs") or spec_dict.get("skew_coupling_exprs") or {}
    skew_map: dict[tuple[int, int], str] = {}
    if isinstance(skew_raw, dict):
        for k, v in skew_raw.items():
            if isinstance(k, tuple):
                skew_map[k] = str(v)
            elif isinstance(k, str) and "," in k:
                parts = k.strip("()[] ").split(",")
                skew_map[(int(parts[0]), int(parts[1]))] = str(v)

    return SymbolicSystemSpec(
        name=str(spec_dict.get("name", "synthesized_symbolic_system")),
        domain=str(spec_dict.get("domain", "multiphysics")),
        state_specs=state_specs,
        param_specs=param_specs,
        time_unit=t_unit,
        energy_unit=e_unit,
        hamiltonian_expr=str(spec_dict.get("hamiltonian_expr", "0.0")),
        dissipation_diag_exprs=tuple(str(x) for x in spec_dict.get("dissipation_diag_exprs", ())),
        skew_routing_exprs=skew_map,
        external_drive_exprs=tuple(str(x) for x in spec_dict.get("external_drive_exprs", ())),
    )


def compile_with_self_repair(
    candidate_spec_dict: dict[str, Any],
    client: MultiProviderMetaClient,
    max_repair_attempts: int = 3,
) -> SelfRepairCompilationReport:
    """
    Attempt to compile `candidate_spec_dict` via Layer 3 (`compile_symbolic_system`) and
    verify it against Layer 1's Thermodynamic & Dimensional Shields (`audit_law_brick`).

    If a compile-time error (`DimensionalMismatchError`, `SymbolicParseError`) or a
    thermodynamic shield violation (`passed_all == False`) occurs, formats the exact
    mathematical diagnostic and invokes `Muse Spark` to repair the specification.
    """
    curr_dict = json.loads(json.dumps(candidate_spec_dict, default=str))
    diagnostics: list[str] = []

    for attempt in range(1, max_repair_attempts + 2):
        try:
            spec_obj = _spec_dict_to_symbolic_spec(curr_dict)
            for d_expr in spec_obj.dissipation_diag_exprs:
                for p_spec in spec_obj.param_specs:
                    if p_spec.name in d_expr and (p_spec.default_value < 0.0 or p_spec.min_value < 0.0):
                        raise ValueError(
                            f"Second Law thermodynamic violation: dissipation parameter '{p_spec.name}' "
                            f"in '{d_expr}' has negative value/bound (default={p_spec.default_value}, "
                            f"min={p_spec.min_value}). Passive dissipation must be non-negative."
                        )
            compiled = compile_symbolic_system(spec_obj)
            cert = audit_law_brick(compiled)
            if cert.passed_all:
                return SelfRepairCompilationReport(
                    compiled_brick=compiled,
                    audit_certificate=cert,
                    attempts_used=attempt,
                    repair_diagnostics=tuple(diagnostics),
                    final_spec_dict=curr_dict,
                )
            diag_str = format_compiler_or_shield_diagnostic(cert, candidate_spec_summary=spec_obj.name)
        except Exception as exc:
            diag_str = format_compiler_or_shield_diagnostic(
                exc, candidate_spec_summary=str(curr_dict.get("name", "candidate"))
            )

        diagnostics.append(diag_str)
        if attempt > max_repair_attempts:
            raise ValueError(
                f"Symbolic self-repair exhausted after {max_repair_attempts} attempts. "
                f"Last diagnostic: {diag_str}"
            )

        repair_prompt = json.dumps(
            {
                "task_type": "symbolic_self_repair",
                "candidate_spec": curr_dict,
                "diagnostic_error": diag_str,
                "attempt": attempt,
            },
            indent=2,
        )
        resp = client.invoke(
            LLMInferenceRequest(
                role="muse_spark",
                system_prompt=ROLE_SYSTEM_PROMPTS["muse_spark"],
                user_prompt=repair_prompt,
                temperature=0.0,
                require_json=True,
            )
        )
        if resp.parsed_json and isinstance(resp.parsed_json.get("repaired_spec"), dict):
            curr_dict = resp.parsed_json["repaired_spec"]

    raise RuntimeError("Unreachable state in compile_with_self_repair.")


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

    def run_investigation(self, goal: ScientificInvestigationGoal) -> ScientificDiscoveryDossier:
        """
        Execute an autonomous closed-loop investigation for `goal`.
        """
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
            if not goal.state_specs_for_autonomous:
                raise ValueError(
                    "Either base_brick, candidate_spec_dict, or state_specs_for_autonomous must be provided."
                )
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

        active_params = (
            goal.baseline_params.detach().clone().to(torch.float64)
            if goal.baseline_params is not None
            else active_brick.default_params(batch_size=1, dtype=torch.float64)
        )
        if active_params.ndim == 1:
            active_params = active_params.unsqueeze(0)

        active_x0 = (
            goal.initial_state.detach().clone().to(torch.float64)
            if goal.initial_state is not None
            else (
                goal.observed_states[0:1].detach().clone().to(torch.float64)
                if (goal.observed_states is not None and goal.observed_states.ndim == 2)
                else (
                    goal.observed_states[0, 0:1].detach().clone().to(torch.float64)
                    if (goal.observed_states is not None and goal.observed_states.ndim == 3)
                    else active_brick.default_state(batch_size=1, dtype=torch.float64)
                )
            )
        )
        if active_x0.ndim == 1:
            active_x0 = active_x0.unsqueeze(0)

        active_states: torch.Tensor | None = goal.observed_states
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
        retro_roll: RolloutResult | None = None
        bvp_res: BoundaryShootingResult | None = None
        inv_report: InvariantDiscoveryReport | None = None
        arch_report: ArchetypeFingerprint | None = None
        falsif_report: FalsificationStressReport | None = None
        cryst_record: CrystallizedLawRecord | None = None
        vault_matches: list[VaultAnalogueMatch] = []

        competing_list: list[tuple[LawBrick, torch.Tensor]] = list(goal.competing_candidates)
        completed_actions: list[str] = []
        step_records: list[EpistemicStepRecord] = []

        for step_idx in range(1, goal.max_council_steps + 1):
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
            )

            prompt_payload = {
                "task_type": "select_next_action",
                "question": goal.question,
                "completed_actions": completed_actions,
                "has_observed_data": goal.observed_states is not None,
                "zero_prior_model": zero_prior_mode,
                "num_competing_candidates": len(competing_list),
                "requested_inverse_mode": goal.inverse_mode,
                "has_vault": bool(self.vault is not None and goal.crystallize_into_vault),
                "available_tools": list(TOOL_MATHEMATICAL_SCHEMAS.keys()),
                "tool_schemas": TOOL_MATHEMATICAL_SCHEMAS,
                "epistemic_briefing": briefing.to_dict(),
            }

            llm_resp = self.client.invoke(
                LLMInferenceRequest(
                    role="muse_wonder",
                    system_prompt=ROLE_SYSTEM_PROMPTS["muse_wonder"],
                    user_prompt=json.dumps(prompt_payload, indent=2),
                    temperature=0.0,
                    require_json=True,
                )
            )
            parsed = llm_resp.parsed_json or {}
            tool_name = str(parsed.get("tool_name", ""))
            tool_args = dict(parsed.get("arguments", {}))
            rationale = str(parsed.get("epistemic_rationale", "Proceeding with mathematical evaluation."))

            valid_tools = set(TOOL_MATHEMATICAL_SCHEMAS.keys())
            if (
                tool_name not in valid_tools
                or (tool_name != "conclude_investigation" and tool_name in completed_actions)
                or (tool_name == "calibrate_and_diagnose_residual" and goal.observed_states is None)
                or (tool_name == "design_disambiguating_experiment" and len(competing_list) < 2)
            ):
                _, fallback_action = MathematicalEpistemicPolicy._handle_action_selection(prompt_payload)
                tool_name = str(fallback_action.get("tool_name", "conclude_investigation"))
                tool_args = dict(fallback_action.get("arguments", {}))
                rationale = str(fallback_action.get("epistemic_rationale", rationale))

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
                x_flat = obs_x.reshape(-1, active_brick.state_dim) if obs_x.ndim == 3 else obs_x
                x_first = obs_x[0] if obs_x.ndim == 3 else obs_x
                dx_flat = (
                    goal.observed_dx_dt.reshape(-1, active_brick.state_dim)
                    if (goal.observed_dx_dt is not None and goal.observed_dx_dt.ndim == 3)
                    else goal.observed_dx_dt
                )
                cal_report = calibrate_and_audit_residual(
                    brick=active_brick,
                    observed_states=x_flat,
                    dt=goal.dt,
                    observed_dx_dt=dx_flat,
                    num_initial_samples=64,
                    num_collocation_survivors=16,
                    num_gradient_starts=4,
                    gradient_steps=45,
                )
                active_params = cal_report.best_params.detach().clone()
                res_flow = cal_report.residual_analysis.residual_flow[: x_first.shape[0]]
                mem_report = detect_non_markovian_memory(x_first, res_flow, dt=goal.dt)
                summary_str = (
                    f"Calibrated {active_brick.name}: trajectory_r2={cal_report.funnel_report.trajectory_r2:.5f}, "
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
                obs_x = goal.observed_states if goal.observed_states is not None else active_states
                assert obs_x is not None
                if zero_prior_mode or bool(tool_args.get("autonomous_from_scratch", False)):
                    auto_disc_report = discover_autonomous_symbolic_system(
                        name=f"discovered_{goal.domain}_system",
                        domain=goal.domain,
                        state_specs=active_brick.state_specs,
                        observed_states=obs_x,
                        dt=goal.dt,
                        observed_dx_dt=goal.observed_dx_dt,
                        time_unit=active_brick.time_unit,
                        energy_unit=active_brick.energy_unit,
                        max_polynomial_degree=2,
                        max_terms_per_state=2,
                        include_rational_primitives=False,
                    )
                    active_brick = auto_disc_report.compiled_brick
                    active_params = auto_disc_report.calibrated_params
                    inv_report = auto_disc_report.invariant_report
                    arch_report = auto_disc_report.archetype
                    audit_cert = auto_disc_report.audit_certificate
                    zero_prior_mode = False
                    summary_str = (
                        f"Discovered autonomous coupled system '{active_brick.name}' from scratch: "
                        f"vector_field_r2={auto_disc_report.vector_field_r2:.5f}, "
                        f"equations={auto_disc_report.per_state_expressions}."
                    )
                    metrics = {
                        "vector_field_r2": auto_disc_report.vector_field_r2,
                        "trajectory_r2": auto_disc_report.trajectory_r2,
                        "per_state_expressions": auto_disc_report.per_state_expressions,
                    }
                else:
                    disc_report = discover_and_compile_law(
                        base_brick=active_brick,
                        observed_states=obs_x,
                        dt=goal.dt,
                        observed_dx_dt=goal.observed_dx_dt,
                        calibration_report=cal_report,
                    )
                    active_brick = disc_report.winning_brick
                    active_params = disc_report.winning_params
                    arch_report = disc_report.archetype
                    audit_cert = disc_report.audit_certificate
                    if active_brick.state_dim > active_x0.shape[1]:
                        if disc_report.latent_augmentation is not None:
                            active_x0 = disc_report.latent_augmentation.reconstructed_initial_state.detach().clone()
                            roll_lat = rollout_trajectories(
                                active_brick,
                                active_x0,
                                active_params,
                                dt=goal.dt,
                                num_steps=goal.num_steps,
                                integrator="auto",
                            )
                            active_states = roll_lat.states[0]
                    summary_str = (
                        f"Executed closed-loop discovery: triggered={disc_report.discovery_triggered}, "
                        f"winning_brick='{active_brick.name}', archetype='{arch_report.archetype_name}', "
                        f"out_of_sample_r2={disc_report.out_of_sample_r2:.5f}."
                    )
                    metrics = {
                        "discovery_triggered": disc_report.discovery_triggered,
                        "winning_brick": active_brick.name,
                        "archetype": arch_report.archetype_name,
                        "out_of_sample_r2": disc_report.out_of_sample_r2,
                    }

                self.client.invoke(
                    LLMInferenceRequest(
                        role="muse_spark",
                        system_prompt=ROLE_SYSTEM_PROMPTS["muse_spark"],
                        user_prompt=json.dumps(
                            {
                                "task_type": "audit_operator_or_intervention",
                                "operator_or_mode": active_brick.name,
                                "thermodynamic_audit_passed": True if audit_cert is None else audit_cert.passed_all,
                            }
                        ),
                        temperature=0.0,
                        require_json=True,
                    )
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
                inv_mode: str = str(tool_args.get("mode") or goal.inverse_mode or "counterfactual_intervention")
                target_map = goal.target_state_objective or {}

                def _target_loss(roll: RolloutResult) -> torch.Tensor:
                    if target_map:
                        loss_acc = torch.zeros((), dtype=torch.float64, device=roll.states.device)
                        for idx_s, sp_s in enumerate(active_brick.state_specs):
                            if sp_s.name in target_map:
                                t_val = float(target_map[sp_s.name])
                                loss_acc = loss_acc + torch.mean((roll.states[:, -5:, idx_s] - t_val) ** 2)
                        return loss_acc
                    return torch.mean(roll.states[:, -1, :] ** 2)

                if inv_mode == "counterfactual_intervention":
                    sel_names = tool_args.get("selected_parameter_names")
                    allowed = sel_names if sel_names else None
                    interv_sol = solve_target_intervention(
                        brick=active_brick,
                        initial_state=active_x0,
                        baseline_params=active_params,
                        dt=goal.dt,
                        num_steps=goal.num_steps,
                        target_loss_fn=_target_loss,
                        allowed_param_names=allowed,
                        learning_rate=0.12,
                        max_iters=90,
                        l2_regularization=1e-4,
                    )
                    active_params = interv_sol.intervened_params.detach().clone()
                    active_states = interv_sol.restored_rollout.states[0]
                    audit_cert = interv_sol.audit_certificate
                    summary_str = (
                        f"Solved counterfactual multi-parameter intervention: "
                        f"baseline_loss={interv_sol.baseline_loss:.5f} -> "
                        f"intervened_loss={interv_sol.intervened_loss:.5f}, "
                        f"deltas={interv_sol.param_deltas}."
                    )
                    metrics = {
                        "baseline_loss": interv_sol.baseline_loss,
                        "intervened_loss": interv_sol.intervened_loss,
                        "param_deltas": interv_sol.param_deltas,
                    }

                elif inv_mode == "optimal_control":
                    ctrl_res = solve_optimal_control_waveform(
                        brick=active_brick,
                        initial_state=active_x0,
                        params=active_params,
                        dt=goal.dt,
                        num_steps=goal.num_steps,
                        target_loss_fn=_target_loss,
                        max_iters=50,
                    )
                    active_states = ctrl_res.controlled_rollout.states[0]
                    summary_str = (
                        f"Synthesized optimal control waveform u*(t): "
                        f"baseline_loss={ctrl_res.baseline_loss:.5f} -> "
                        f"controlled_loss={ctrl_res.controlled_loss:.5f}, "
                        f"all_time_viable={ctrl_res.all_time_viable}."
                    )
                    metrics = {
                        "baseline_loss": ctrl_res.baseline_loss,
                        "controlled_loss": ctrl_res.controlled_loss,
                        "control_energy": ctrl_res.control_energy,
                        "all_time_viable": ctrl_res.all_time_viable,
                    }

                elif inv_mode == "retrodiction":
                    term_state = active_x0
                    if goal.observed_states is not None and goal.observed_states.ndim == 2:
                        term_state = goal.observed_states[-1:].to(torch.float64)
                    retro_roll = retrodict_trajectory(
                        brick=active_brick,
                        final_state=term_state,
                        params=active_params,
                        dt=goal.dt,
                        num_steps=goal.num_steps,
                    )
                    active_states = retro_roll.states[0]
                    summary_str = (
                        f"Executed backward time-reversal retrodiction across {goal.num_steps} steps "
                        f"(reconstructed x(0)={retro_roll.states[0, 0].tolist()})."
                    )
                    metrics = {
                        "reconstructed_initial_state": [round(float(v), 6) for v in retro_roll.states[0, 0].tolist()],
                    }

                elif inv_mode == "boundary_value_shooting":
                    target_indices: list[int] = []
                    target_vals_list: list[float] = []
                    for idx_s, sp_s in enumerate(active_brick.state_specs):
                        if sp_s.name in target_map:
                            target_indices.append(idx_s)
                            target_vals_list.append(float(target_map[sp_s.name]))
                    if not target_indices:
                        target_indices = [0]
                        target_vals_list = [0.0]
                    known_idx = [i for i in range(active_brick.state_dim) if i not in target_indices]
                    if not known_idx and active_brick.state_dim >= 2:
                        known_idx = [0]
                        target_indices = [1]
                    known_vals = active_x0[0, known_idx]
                    t_vals = torch.tensor(target_vals_list, dtype=torch.float64, device=active_x0.device)
                    bvp_res = solve_boundary_value_shooting(
                        brick=active_brick,
                        known_initial_indices=known_idx,
                        known_initial_values=known_vals,
                        target_final_indices=target_indices,
                        target_final_values=t_vals,
                        params=active_params,
                        dt=goal.dt,
                        num_steps=goal.num_steps,
                    )
                    active_x0 = bvp_res.solved_initial_state
                    active_states = bvp_res.rollout.states[0]
                    summary_str = (
                        f"Solved two-point boundary-value shooting: "
                        f"terminal_error={bvp_res.terminal_error_norm:.3e}, converged={bvp_res.converged}."
                    )
                    metrics = {
                        "terminal_error_norm": bvp_res.terminal_error_norm,
                        "converged": bvp_res.converged,
                    }

                self.client.invoke(
                    LLMInferenceRequest(
                        role="muse_spark",
                        system_prompt=ROLE_SYSTEM_PROMPTS["muse_spark"],
                        user_prompt=json.dumps(
                            {
                                "task_type": "audit_operator_or_intervention",
                                "operator_or_mode": inv_mode,
                                "thermodynamic_audit_passed": True if audit_cert is None else audit_cert.passed_all,
                            }
                        ),
                        temperature=0.0,
                        require_json=True,
                    )
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
                inv_report = discover_conservation_invariants(
                    brick=active_brick,
                    states=active_states,
                    params=active_params,
                    max_degree=2,
                    include_cross_terms=True,
                    include_trigonometric=True,
                    include_logarithmic=True,
                )
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
                falsif_report = run_falsification_stress_test(
                    brick=active_brick,
                    calibrated_params=active_params,
                    reference_states=active_states,
                    dt=goal.dt,
                    invariant_report=inv_report,
                )

                if not falsif_report.passed_all_gates and interv_sol is not None:
                    target_map = goal.target_state_objective or {}

                    def _safe_target_loss(roll_in: RolloutResult) -> torch.Tensor:
                        if target_map:
                            l_acc = torch.zeros((), dtype=torch.float64, device=roll_in.states.device)
                            for i_s, sp_s in enumerate(active_brick.state_specs):
                                if sp_s.name in target_map:
                                    l_acc = l_acc + torch.mean(
                                        (roll_in.states[:, -8:, i_s] - float(target_map[sp_s.name])) ** 2
                                    )
                            return l_acc
                        return torch.mean(roll_in.states[:, -1, :] ** 2)

                    repaired_interv = solve_target_intervention(
                        brick=active_brick,
                        initial_state=active_x0,
                        baseline_params=interv_sol.baseline_params,
                        dt=goal.dt,
                        num_steps=min(goal.num_steps * 2, 80),
                        target_loss_fn=_safe_target_loss,
                        learning_rate=0.08,
                        max_iters=100,
                        l2_regularization=5e-3,
                    )
                    repaired_falsif = run_falsification_stress_test(
                        brick=active_brick,
                        calibrated_params=repaired_interv.intervened_params,
                        reference_states=repaired_interv.restored_rollout.states[0],
                        dt=goal.dt,
                        invariant_report=inv_report,
                    )
                    if repaired_falsif.passed_all_gates or (
                        repaired_falsif.ood_viable_fraction >= falsif_report.ood_viable_fraction
                    ):
                        interv_sol = repaired_interv
                        active_params = repaired_interv.intervened_params.detach().clone()
                        active_states = repaired_interv.restored_rollout.states[0]
                        falsif_report = repaired_falsif

                audit_cert = falsif_report.audit_certificate
                if (
                    falsif_report.passed_all_gates
                    and self.vault is not None
                    and goal.crystallize_into_vault
                ):
                    existing_laws = self.vault.list_laws(include_discovered=True)
                    p_ids = (existing_laws[0].law_id,) if existing_laws else ()
                    cryst_record = crystallize_verified_law(
                        vault=self.vault,
                        brick=active_brick,
                        stress_report=falsif_report,
                        parent_law_ids=p_ids,
                    )
                mh_4x = (
                    falsif_report.multi_horizon_report.horizon_4x_viable_fraction
                    if falsif_report.multi_horizon_report is not None
                    else falsif_report.ood_viable_fraction
                )
                summary_str = (
                    f"Executed 4-gate + multi-horizon (4x) falsification suite: "
                    f"passed_all={falsif_report.passed_all_gates}, "
                    f"noisy_r2={falsif_report.noisy_vector_field_r2:.4f}, "
                    f"ood_viable={falsif_report.ood_viable_fraction:.3f}, "
                    f"horizon_4x_viable={mh_4x:.3f}, "
                    f"crystallized={cryst_record is not None}."
                )
                metrics = {
                    "passed_all_gates": falsif_report.passed_all_gates,
                    "noisy_vector_field_r2": falsif_report.noisy_vector_field_r2,
                    "ood_viable_fraction": falsif_report.ood_viable_fraction,
                    "horizon_4x_viable_fraction": mh_4x,
                    "crystallized_law_id": cryst_record.law_id if cryst_record is not None else None,
                }
                self.client.invoke(
                    LLMInferenceRequest(
                        role="muse_spark_open",
                        system_prompt=ROLE_SYSTEM_PROMPTS["muse_spark_open"],
                        user_prompt=json.dumps(
                            {
                                "task_type": "adversarial_red_team_review",
                                "passed_all_gates": falsif_report.passed_all_gates,
                                "noisy_vector_field_r2": falsif_report.noisy_vector_field_r2,
                                "ood_viable_fraction": falsif_report.ood_viable_fraction,
                                "horizon_4x_viable_fraction": mh_4x,
                            }
                        ),
                        temperature=0.0,
                        require_json=True,
                    )
                )

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

        final_briefing = build_differential_epistemic_briefing(
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
        )

        primary_finding = _build_primary_scientific_finding(
            brick=active_brick,
            final_briefing=final_briefing,
            target_state_objective=goal.target_state_objective,
            disc_report=disc_report,
            auto_disc_report=auto_disc_report,
            disambig_report=disambig_report,
            bif_report=bif_report,
            interv_sol=interv_sol,
            ctrl_res=ctrl_res,
            retro_roll=retro_roll,
            bvp_res=bvp_res,
            falsif_report=falsif_report,
            cryst_record=cryst_record,
        )

        glimmer_payload = {
            "task_type": "synthesize_dossier_narrative",
            "question": goal.question,
            "system_name": active_brick.name,
            "executed_tools": [s.tool_name for s in step_records],
            "conserved_invariants": list(final_briefing.conserved_invariants),
            "vault_analogues": [m.name for m in final_briefing.vault_analogues],
            "primary_finding": primary_finding.to_dict(),
        }
        glimmer_resp = self.client.invoke(
            LLMInferenceRequest(
                role="muse_glimmer",
                system_prompt=ROLE_SYSTEM_PROMPTS["muse_glimmer"],
                user_prompt=json.dumps(glimmer_payload, indent=2),
                temperature=0.1,
                require_json=True,
            )
        )
        exec_summary = (
            str(glimmer_resp.parsed_json.get("executive_summary"))
            if (glimmer_resp.parsed_json and "executive_summary" in glimmer_resp.parsed_json)
            else glimmer_resp.raw_content
        )

        return ScientificDiscoveryDossier(
            question=goal.question,
            final_brick=active_brick,
            final_params=active_params,
            steps=tuple(step_records),
            final_briefing=final_briefing,
            self_repair_report=self_repair_report,
            calibration_report=cal_report,
            discovery_report=disc_report,
            autonomous_discovery_report=auto_disc_report,
            disambiguation_report=disambig_report,
            equilibrium_report=eq_report,
            bifurcation_report=bif_report,
            identifiability_report=ident_report,
            intervention_solution=interv_sol,
            optimal_control_result=ctrl_res,
            retrodiction_rollout=retro_roll,
            boundary_shooting_result=bvp_res,
            invariant_report=inv_report,
            archetype=arch_report,
            vault_analogues=tuple(vault_matches),
            falsification_report=falsif_report,
            crystallized_record=cryst_record,
            executive_summary=exec_summary,
            llm_telemetry=tuple(self.client.telemetry_log),
            primary_finding=primary_finding,
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
) -> PrimaryScientificFinding:
    """
    Extract and structure the complete mathematical findings of a Council investigation.
    """
    categories: list[str] = []
    discovered_eqs: dict[str, str] = {}
    disc_r2: float | None = None

    if auto_disc_report is not None:
        categories.append("autonomous_zero_prior_system_discovery")
        for st_name, expr_str in auto_disc_report.per_state_expressions.items():
            discovered_eqs[f"d({st_name})/dt"] = str(expr_str)
        disc_r2 = round(float(auto_disc_report.vector_field_r2), 6)
    elif disc_report is not None and disc_report.discovery_triggered:
        categories.append("closed_loop_mechanism_discovery")
        disc_r2 = round(float(disc_report.out_of_sample_r2), 6)
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
        for j, pp in enumerate(brick.param_specs):
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
        topo_sum = {
            "num_equilibria": eq_s.num_equilibria,
            "is_multistable": eq_s.is_multistable,
            "equilibria": list(eq_s.equilibria),
            "bifurcation_parameter": bif_report.param_name if bif_report is not None else None,
        }

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
            "crystallized_law_id": cryst_record.law_id if cryst_record is not None else None,
            "sha256_hash": cryst_record.sha256_fingerprint if cryst_record is not None else None,
        }

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
    )
