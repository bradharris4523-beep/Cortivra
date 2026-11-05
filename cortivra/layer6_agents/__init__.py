"""
Layer 6 (`cortivra.layer6_agents`) — Multi-agent scientific council, empirical
telemetry validation, formal stability verification, and 3-stage physical realization.

Architectural Subsystems:
  1. Council & Epistemic Orchestration:
     `council`, `council_finalizer`, `meta_clients`, `diagnostic_briefing`,
     `hypothesis_tournament`, `multiscale_and_tournament`
  2. Empirical Telemetry Ingestion & Out-of-Sample Validation:
     `empirical_observatory`, `open_science_harvester`,
     `universal_telemetry_hunter`, `autonomous_frontier_explorer`
  3. Stability Proofs, Experimental Design & Falsification:
     `falsification_and_design`, `blind_oracle_rescue_suite`,
     `blind_reconstruction_benchmarks`
  4. 3-Stage Physical Realization Pipeline:
     - Stage 1 (Continuum & 3D CAD Blueprint): `blueprint_compiler`, `hardware_artifact_synthesizer`
     - Stage 2 (Synthesis Reactor Schedule): `recursive_synthesis_engine`
     - Stage 3 (Molecular Graph SMILES & 3D Lattice): `pass3_matter_and_lattice_compiler`
  5. Dossier Rendering & Deliverable Export:
     `dossier_renderer`, `html_command_center_renderer`,
     `deliverable_exporter`, `flight_recorder_exporter`
"""

from cortivra.layer6_agents.blind_oracle_rescue_suite import (
    BlindOracleRescueResult,
    BlindOracleRescueSuiteReport,
    OracleClosedLoopTrialMetrics,
    SealedGroundTruthOracle,
    build_five_flagship_sealed_oracles,
    evaluate_brick_mismatched_reality_trial,
    run_blind_oracle_rescue_suite,
    run_single_blind_oracle_rescue,
)
from cortivra.layer6_agents.blind_reconstruction_benchmarks import (
    BlindReconstructionBenchmarkResult,
    BlindReconstructionSuiteReport,
    run_benchmark_celestial_3d_unseen_perturber_inversion,
    run_benchmark_excitable_bioelectrical_and_collateral_cure,
    run_benchmark_locked_pathway_autonomous_catalyst_invention,
    run_benchmark_non_markovian_hidden_state_and_3d_multifield_pde,
    run_benchmark_quantum_symplectic_lindblad_and_kramers_escape,
    run_blind_scientific_reconstruction_suite,
)
from cortivra.layer6_agents.blueprint_compiler import (
    _classify_parameter_operator_slot,
    _compile_universal_physical_realization_blueprint,
    _derive_7d_si_constitutive_tensor_scaling,
    _infer_operating_temperature_kelvin,
)
from cortivra.layer6_agents.council import (
    AutonomousScientificCouncil,
    EpistemicStepRecord,
    InverseProblemMode,
    PrimaryScientificFinding,
    ScientificDiscoveryDossier,
    ScientificInvestigationGoal,
    SelfRepairCompilationReport,
    compile_hypothesis_to_investigation_goal,
    compile_with_self_repair,
    investigate,
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
    OutOfSampleExtrapolationReport,
    fetch_open_science_observatory_telemetry,
    load_empirical_telemetry,
    run_out_of_sample_and_ood_shock_benchmark,
)
from cortivra.layer6_agents.diagnostic_briefing import (
    DifferentialEpistemicBriefing,
    EquilibriumTelemetrySummary,
    MultiHorizonTelemetrySummary,
    ParamTelemetrySummary,
    ResidualTelemetrySummary,
    SensitivityTelemetrySummary,
    StateTelemetrySummary,
    VaultAnalogueMatch,
    VaultOperatorTransferReport,
    build_differential_epistemic_briefing,
    format_compiler_or_shield_diagnostic,
    search_vault_isomorphisms,
    transfer_isomorphic_operator_from_vault,
)
from cortivra.layer6_agents.falsification_and_design import (
    DisambiguatingExperimentReport,
    FalsificationGateResult,
    FalsificationStressReport,
    LyapunovStabilityCertificate,
    MultiHorizonStabilityReport,
    compute_lyapunov_stability_certificate,
    crystallize_verified_law,
    design_disambiguating_experiment,
    run_falsification_stress_test,
    verify_multi_horizon_stability,
)
from cortivra.layer6_agents.meta_clients import (
    AgentRole,
    LLMInferenceRequest,
    LLMInferenceResponse,
    MultiProviderMetaClient,
    ProviderEndpointConfig,
    ProviderName,
    ROLE_SYSTEM_PROMPTS,
    TOOL_MATHEMATICAL_SCHEMAS,
)
from cortivra.layer6_agents.multiscale_and_tournament import (
    _adapt_candidate_spec_bounds_to_telemetry,
    _build_primary_scientific_finding,
    _compute_subscale_timestep,
    _enforce_universal_world_model_closure,
    _execute_multi_hypothesis_crucible_tournament,
    _formulate_subscale_investigation_prompt,
    _spec_dict_to_symbolic_spec,
    _verify_bidirectional_multiscale_closure,
)
from cortivra.layer6_agents.universal_telemetry_hunter import (
    ChronologicalSplitCrucibleResult,
    DualBlindTemporalCrucibleReport,
    FiveGateDataQualificationReport,
    LiteratureCitationRecord,
    LiteratureGroundingReport,
    NextSpotForecastComparison,
    audit_five_gate_telemetry_qualification,
    extract_scientific_search_ontology,
    harvest_open_science_literature_priors,
    hunt_universal_public_telemetry,
    run_dual_blind_temporal_forecast_crucible,
    try_hunt_universal_public_telemetry,
)
from cortivra.layer6_agents.autonomous_frontier_explorer import (
    AutonomousCuriosityChainReport,
    AutonomousFrontierExplorationCertificate,
    BeyondTextbookResidualAnomalyReport,
    CrossObservatoryStreamFusionReport,
    CuriosityGenerationStep,
    MultiEventReplicationEpochResult,
    MultiEventZeroShotReplicationReport,
    Spatial3DReactionDiffusionPDEReport,
    execute_autonomous_frontier_exploration_suite,
    fuse_cross_observatory_public_streams,
    hunt_beyond_textbook_residual_anomalies,
    run_autonomous_curiosity_question_chain,
    run_multi_event_zero_shot_replication_crucible,
    run_spatial_3d_reaction_diffusion_pde_escalation,
)
from cortivra.layer6_agents.recursive_synthesis_engine import (
    RecursivePass2SynthesisReport,
    SynthesisProcessStageSpec,
    execute_recursive_pass2_synthesis_and_fabrication_engine,
)
from cortivra.layer6_agents.pass3_matter_and_lattice_compiler import (
    CandidateMolecularScaffoldSpec,
    MetamaterialUnitCellLatticeSpec,
    Pass3MatterAndLatticeReport,
    execute_pass3_matter_and_lattice_compiler,
)

__all__ = [
    "ProviderName",
    "AgentRole",
    "ProviderEndpointConfig",
    "LLMInferenceRequest",
    "LLMInferenceResponse",
    "MultiProviderMetaClient",
    "ROLE_SYSTEM_PROMPTS",
    "TOOL_MATHEMATICAL_SCHEMAS",
    "StateTelemetrySummary",
    "ParamTelemetrySummary",
    "EquilibriumTelemetrySummary",
    "SensitivityTelemetrySummary",
    "ResidualTelemetrySummary",
    "MultiHorizonTelemetrySummary",
    "VaultAnalogueMatch",
    "DifferentialEpistemicBriefing",
    "build_differential_epistemic_briefing",
    "search_vault_isomorphisms",
    "format_compiler_or_shield_diagnostic",
    "MultiHorizonStabilityReport",
    "LyapunovStabilityCertificate",
    "DisambiguatingExperimentReport",
    "FalsificationGateResult",
    "FalsificationStressReport",
    "verify_multi_horizon_stability",
    "compute_lyapunov_stability_certificate",
    "design_disambiguating_experiment",
    "run_falsification_stress_test",
    "crystallize_verified_law",
    "InverseProblemMode",
    "PrimaryScientificFinding",
    "SelfRepairCompilationReport",
    "EpistemicStepRecord",
    "ScientificInvestigationGoal",
    "ScientificDiscoveryDossier",
    "compile_hypothesis_to_investigation_goal",
    "compile_with_self_repair",
    "AutonomousScientificCouncil",
    "investigate",
    "BlindReconstructionBenchmarkResult",
    "BlindReconstructionSuiteReport",
    "run_benchmark_excitable_bioelectrical_and_collateral_cure",
    "run_benchmark_celestial_3d_unseen_perturber_inversion",
    "run_benchmark_locked_pathway_autonomous_catalyst_invention",
    "run_benchmark_quantum_symplectic_lindblad_and_kramers_escape",
    "run_benchmark_non_markovian_hidden_state_and_3d_multifield_pde",
    "run_blind_scientific_reconstruction_suite",
    "_infer_operating_temperature_kelvin",
    "_classify_parameter_operator_slot",
    "_compile_universal_physical_realization_blueprint",
    "_render_trajectory_svg",
    "_render_dossier_html_report",
    "_render_dossier_markdown_report",
    "compile_fmeca_and_safe_operating_area_matrix",
    "format_8stage_execution_flight_recorder_log",
    "export_turnkey_deliverable_bundle",
    "SealedGroundTruthOracle",
    "OracleClosedLoopTrialMetrics",
    "BlindOracleRescueResult",
    "BlindOracleRescueSuiteReport",
    "build_five_flagship_sealed_oracles",
    "run_single_blind_oracle_rescue",
    "run_blind_oracle_rescue_suite",
    "FiveGateDataQualificationReport",
    "NextSpotForecastComparison",
    "ChronologicalSplitCrucibleResult",
    "DualBlindTemporalCrucibleReport",
    "audit_five_gate_telemetry_qualification",
    "hunt_universal_public_telemetry",
    "run_dual_blind_temporal_forecast_crucible",
    "CrossObservatoryStreamFusionReport",
    "BeyondTextbookResidualAnomalyReport",
    "MultiEventReplicationEpochResult",
    "MultiEventZeroShotReplicationReport",
    "CuriosityGenerationStep",
    "AutonomousCuriosityChainReport",
    "Spatial3DReactionDiffusionPDEReport",
    "AutonomousFrontierExplorationCertificate",
    "fuse_cross_observatory_public_streams",
    "hunt_beyond_textbook_residual_anomalies",
    "run_multi_event_zero_shot_replication_crucible",
    "run_autonomous_curiosity_question_chain",
    "run_spatial_3d_reaction_diffusion_pde_escalation",
    "execute_autonomous_frontier_exploration_suite",
    "SynthesisProcessStageSpec",
    "RecursivePass2SynthesisReport",
    "execute_recursive_pass2_synthesis_and_fabrication_engine",
    "CandidateMolecularScaffoldSpec",
    "MetamaterialUnitCellLatticeSpec",
    "Pass3MatterAndLatticeReport",
    "execute_pass3_matter_and_lattice_compiler",
]
