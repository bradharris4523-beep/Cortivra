"""
Layer 2 (`cortivra.layer2_fabric`) — Differential primitives and physical domain bricks.

Provides:
    - `primitives`: 16 mathematical operators across 4 families.
    - `mechanics_and_fields`: `CentralForceHamiltonianBrick` (N-body orbital / power-law fields)
      and `NonlinearPendulumAndViscoelasticBrick`.
    - `electrodynamics_and_circuits`: `NonlinearConductanceCircuitBrick` (capacitive-conductive
      voltage-gated circuits with Rush-Larsen exponential-Euler integration).
    - `kinetics_and_thermodynamics`: `StoichiometricReactionBrick` and `CooperativeBindingBrick`.
    - `transport_and_fields`: `SpatialAdvectionDiffusionBrick` (O(N) matrix-free PDE field)
      and `EventImpulseCouplingBrick`.
    - `register_foundational_laws`: Populates a Layer 1 `LawVault` with Tier 1 (`CORE_IMMUTABLE`)
      axioms across mechanics, electrodynamics, chemistry, and continuum transport.
"""

from __future__ import annotations

from cortivra.layer1_axioms.vault import CrystallizedLawRecord, LawVault
from cortivra.layer2_fabric.electrodynamics_and_circuits import NonlinearConductanceCircuitBrick
from cortivra.layer2_fabric.kinetics_and_thermodynamics import (
    CooperativeBindingBrick,
    StoichiometricReactionBrick,
)
from cortivra.layer2_fabric.mechanics_and_fields import (
    CentralForceHamiltonianBrick,
    NonlinearPendulumAndViscoelasticBrick,
)
from cortivra.layer2_fabric.primitives import (
    PRIMITIVE_REGISTRY,
    PrimitiveSpec,
    boltzmann_partition_switch,
    conservative_advection_1d,
    cooperative_hill_saturation,
    cubic_bistable_feedback,
    eyring_barrier_rate,
    get_primitives_by_role,
    ghk_electrodiffusion_flux,
    graph_laplacian_diffusion_1d,
    graph_laplacian_sparse_edges,
    inverse_distance_potential_and_force,
    linear_relaxation_rate,
    nernst_equilibrium_potential_mv,
    ohmic_dissipative_driving_force,
    quadratic_energy_storage,
    reversible_mass_action_rates,
    skew_symmetric_poisson_flow,
    smooth_bernoulli_quotient,
    step_rush_larsen,
    surrogate_event_impulse,
    trigonometric_phase_interaction,
)
from cortivra.layer2_fabric.transport_and_fields import (
    EventImpulseCouplingBrick,
    SpatialAdvectionDiffusionBrick,
)


def register_foundational_laws(vault: LawVault) -> list[CrystallizedLawRecord]:
    """
    Register the 8 core mathematical-physics laws into Tier 1 (`CORE_IMMUTABLE`) of `vault`
    if they are not already present.
    """
    core_specs = [
        (
            "core_newton_coulomb_inverse_square",
            "Inverse-Square Central Force Law",
            "mechanics",
            "F = -G * m1 * m2 * (q1 - q2) / |q1 - q2|^3",
            "Conservative pairwise inverse-distance potential and central force.",
        ),
        (
            "core_trig_pendulum_kuramoto",
            "Trigonometric Phase & Pendulum Restoring Law",
            "mechanics",
            "tau_i = -m*g*l*sin(theta_i) + sum_j K_ij * sin(theta_j - theta_i)",
            "Periodic restoring torque and phase synchronization interaction.",
        ),
        (
            "core_kirchhoff_capacitive_balance",
            "Kirchhoff Capacitive Current Conservation",
            "electrodynamics",
            "C_m * dV/dt + sum_k G_k(V, t) * (V - E_k) = I_ext",
            "Charge conservation across a capacitive boundary with conductive pathways.",
        ),
        (
            "core_boltzmann_eyring_gating",
            "Boltzmann-Eyring Thermodynamic Barrier Partition",
            "electrodynamics",
            "x_inf(V) = 1 / (1 + exp((V_h - V) / k)), dx/dt = (x_inf - x) / tau(V)",
            "Two-state statistical-mechanical occupancy and first-order barrier relaxation.",
        ),
        (
            "core_nernst_ghk_electrodiffusion",
            "Nernst-Planck & Goldman-Hodgkin-Katz Electrodiffusion",
            "electrodynamics",
            "E_eq = (R*T / (z*F)) * ln(c_out / c_in), I_GHK = P*z*F*(c_in*B(-u) - c_out*B(u))",
            "Thermodynamic equilibrium potential and electro-diffusive flux across a membrane.",
        ),
        (
            "core_mass_action_detailed_balance",
            "Stoichiometric Mass-Action & Wegscheider Detailed Balance",
            "chemistry",
            "dc/dt = S * (k_f * c^nu_f - k_r * c^nu_r), R*T*ln(prod(k_f)/prod(k_r)) = -dG_ext",
            "Reversible chemical reaction kinetics conserving mass, charge, and cycle affinity.",
        ),
        (
            "core_cooperative_hill_allostery",
            "Cooperative Hill-Langmuir Binding Saturation",
            "chemistry",
            "theta(c) = c^n / (K_d^n + c^n)",
            "Cooperative receptor-ligand binding and saturating enzyme kinetics.",
        ),
        (
            "core_continuum_advection_diffusion",
            "Continuum Advection-Diffusion Conservation PDE",
            "transport",
            "du/dt = -v * du/dx + D * d^2u/dx^2",
            "Spatial transport combining skew-symmetric advection and Laplacian diffusion.",
        ),
    ]

    registered: list[CrystallizedLawRecord] = []
    existing_ids = {r.law_id for r in vault.list_laws(include_discovered=False)}
    for law_id, name, domain, expr, desc in core_specs:
        if law_id in existing_ids:
            registered.append(vault.get_law(law_id, include_discovered=False))
        else:
            rec = vault.register_core_law(
                law_id=law_id,
                name=name,
                domain=domain,
                symbolic_expression=expr,
                description=desc,
            )
            registered.append(rec)
    return registered


__all__ = [
    "PrimitiveSpec",
    "PRIMITIVE_REGISTRY",
    "get_primitives_by_role",
    "quadratic_energy_storage",
    "inverse_distance_potential_and_force",
    "trigonometric_phase_interaction",
    "skew_symmetric_poisson_flow",
    "boltzmann_partition_switch",
    "eyring_barrier_rate",
    "nernst_equilibrium_potential_mv",
    "smooth_bernoulli_quotient",
    "ghk_electrodiffusion_flux",
    "reversible_mass_action_rates",
    "cooperative_hill_saturation",
    "ohmic_dissipative_driving_force",
    "cubic_bistable_feedback",
    "linear_relaxation_rate",
    "step_rush_larsen",
    "graph_laplacian_diffusion_1d",
    "graph_laplacian_sparse_edges",
    "conservative_advection_1d",
    "surrogate_event_impulse",
    "CentralForceHamiltonianBrick",
    "NonlinearPendulumAndViscoelasticBrick",
    "NonlinearConductanceCircuitBrick",
    "StoichiometricReactionBrick",
    "CooperativeBindingBrick",
    "SpatialAdvectionDiffusionBrick",
    "EventImpulseCouplingBrick",
    "register_foundational_laws",
]
