"""
Cortivra — Multi-scale physics and biology simulation sandbox.

7-Layer Structure:
    - layer1_axioms:      Units, energy/mass conservation, safety limits, and Law Vault.
    - layer2_fabric:      12 reusable math building blocks for chemistry, cells, networks, and mechanics.
    - layer3_compiler:    Turns equations and SAM 3.1 cell shapes into fast GPU tensors.
    - layer4_engine:      Runs parallel GPU sweeps and inverse gradients to find working solutions.
    - layer5_proof:       Stress-tests solutions with noise and checks blind rediscovery from scratch.
    - layer6_agents:      Meta AI models (Muse Wonder, Muse Spark, Muse Spark Open, Muse Glimmer, SAM 3.1).
    - layer7_observatory: Live server streaming simulation results to the UI.
"""

__version__ = "0.1.0"
__author__ = "Brad Harris"

from cortivra.layer6_agents.council import (
    AutonomousScientificCouncil,
    ScientificDiscoveryDossier,
    ScientificInvestigationGoal,
    compile_hypothesis_to_investigation_goal,
    investigate,
)

__all__ = [
    "__version__",
    "__author__",
    "investigate",
    "compile_hypothesis_to_investigation_goal",
    "AutonomousScientificCouncil",
    "ScientificInvestigationGoal",
    "ScientificDiscoveryDossier",
]
