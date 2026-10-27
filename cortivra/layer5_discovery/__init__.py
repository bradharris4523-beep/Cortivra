"""
Layer 5 (`cortivra.layer5_discovery`) — Dimensionally constrained symbolic basis generation,
weak-integral sparse Pareto regression, rational Padé discovery, non-Markovian latent
state augmentation, and closed-loop law compilation.
"""

from cortivra.layer5_discovery.basis_library import (
    CandidateBasisTerm,
    SymbolicBasisLibrary,
    estimate_characteristic_scales,
    generate_symbolic_basis_library,
)
from cortivra.layer5_discovery.latent_discovery import (
    ArchetypeFingerprint,
    AutonomousDiscoveryReport,
    ClosedLoopDiscoveryReport,
    HamiltonianPatchedLawBrick,
    LatentAugmentationResult,
    LatentAugmentedLawBrick,
    LatentOscillatorAugmentedLawBrick,
    NonMarkovianMemoryReport,
    classify_mathematical_archetype,
    compile_symbolic_candidate,
    detect_non_markovian_memory,
    discover_and_compile_law,
    discover_autonomous_symbolic_system,
    discover_latent_state_augmentation,
)
from cortivra.layer5_discovery.sparse_regression import (
    ConservationInvariant,
    InvariantDiscoveryReport,
    ParetoDiscoveryReport,
    SymbolicCandidate,
    build_weak_integral_system,
    discover_conservation_invariants,
    discover_rational_pade_candidates,
    solve_sparse_pareto_front,
)

__all__ = [
    "CandidateBasisTerm",
    "SymbolicBasisLibrary",
    "estimate_characteristic_scales",
    "generate_symbolic_basis_library",
    "SymbolicCandidate",
    "ParetoDiscoveryReport",
    "ConservationInvariant",
    "InvariantDiscoveryReport",
    "build_weak_integral_system",
    "discover_rational_pade_candidates",
    "solve_sparse_pareto_front",
    "discover_conservation_invariants",
    "HamiltonianPatchedLawBrick",
    "LatentAugmentedLawBrick",
    "LatentOscillatorAugmentedLawBrick",
    "NonMarkovianMemoryReport",
    "LatentAugmentationResult",
    "ArchetypeFingerprint",
    "ClosedLoopDiscoveryReport",
    "AutonomousDiscoveryReport",
    "detect_non_markovian_memory",
    "discover_latent_state_augmentation",
    "compile_symbolic_candidate",
    "classify_mathematical_archetype",
    "discover_and_compile_law",
    "discover_autonomous_symbolic_system",
]
