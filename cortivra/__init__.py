"""
Cortivra — Multi-scale differentiable physics discovery, verification, and inverse design engine.

Architecture:
    - layer1_axioms:      7D SI dimensional algebra, Port-Hamiltonian operators, thermodynamic
                          and GKSL quantum coherence shields, and cryptographic LawVault.
    - layer2_fabric:      Vectorized PyTorch differential primitives across mechanics,
                          electrodynamics, chemical kinetics, and spatial transport.
    - layer3_compiler:    Symbolic equation AST parser, 7D SI dimensional type-checker,
                          spatial PDE discretizer, and Port-Hamiltonian tensor compiler.
    - layer4_engine:      Symplectic and stiff implicit trajectory integrators, equilibrium
                          and bifurcation solvers, and SVD nullspace inverse design solvers.
    - layer5_discovery:   Implicit SINDy-PI sparse Pareto symbolic regression, Noether
                          invariant extraction, and Helmholtz-Hodge vector-field decomposition.
    - layer6_agents:      Neuro-symbolic discovery pipeline, multi-hypothesis physical filter,
                          5-scale blueprint compiler, Sim-to-Real wet-lab/HIL verifier, and
                          20-file engineering deliverable exporter.
    - observability:      Causal computation trace graph, SHA-256 Merkle provenance, root-cause
                          contradiction backtracker, and structured execution telemetry.
"""

__version__ = "0.1.0"
__author__ = "Brad Harris"

from typing import Any

from cortivra.layer6_agents.council import (
    AutonomousScientificCouncil,
    ScientificDiscoveryDossier,
    ScientificInvestigationGoal,
    compile_hypothesis_to_investigation_goal,
    investigate,
)


def __getattr__(name: str) -> Any:
    if name in ("serve_workbench", "start_workbench_server"):
        from cortivra import ui_server

        return getattr(ui_server, name)
    raise AttributeError(f"module 'cortivra' has no attribute {name!r}")

__all__ = [
    "__version__",
    "__author__",
    "investigate",
    "compile_hypothesis_to_investigation_goal",
    "AutonomousScientificCouncil",
    "ScientificInvestigationGoal",
    "ScientificDiscoveryDossier",
    "serve_workbench",
    "start_workbench_server",
]
