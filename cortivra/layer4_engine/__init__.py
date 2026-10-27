"""
Layer 4 (`cortivra.layer4_engine`) — Trajectory integration, parameter calibration,
and inverse boundary/control solvers.
"""

from cortivra.layer4_engine.elimination_funnel import (
    EliminationFunnelReport,
    HypothesisRankingEntry,
    rank_hypothesis_portfolio,
    run_elimination_funnel,
    sample_parameter_universes,
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
    EquilibriumAnalysis,
    EquilibriumLandscapeReport,
    RolloutResult,
    discover_equilibrium_landscape,
    estimate_trajectory_derivatives,
    find_steady_state_and_stability,
    retrodict_trajectory,
    rollout_trajectories,
    scan_parameter_bifurcation,
    select_optimal_integrator,
)

__all__ = [
    "RolloutResult",
    "EquilibriumAnalysis",
    "EquilibriumLandscapeReport",
    "BifurcationScanReport",
    "select_optimal_integrator",
    "rollout_trajectories",
    "retrodict_trajectory",
    "find_steady_state_and_stability",
    "discover_equilibrium_landscape",
    "scan_parameter_bifurcation",
    "estimate_trajectory_derivatives",
    "EliminationFunnelReport",
    "HypothesisRankingEntry",
    "sample_parameter_universes",
    "run_elimination_funnel",
    "rank_hypothesis_portfolio",
    "CalibrationAndResidualReport",
    "BoundaryShootingResult",
    "SensitivityAndFisherReport",
    "InterventionSolution",
    "OptimalControlResult",
    "calibrate_and_audit_residual",
    "solve_boundary_value_shooting",
    "compute_parameter_leverage_and_fisher",
    "solve_target_intervention",
    "solve_optimal_control_waveform",
]
