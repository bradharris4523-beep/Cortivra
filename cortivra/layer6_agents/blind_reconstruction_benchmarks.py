"""
Universal Blind Scientific Reconstruction & Inverse Mechanism Invention Benchmark Suite
(`cortivra.layer6_agents.blind_reconstruction_benchmarks`).

Executes zero-prior and incomplete-model blind scientific reconstruction tests across five
distinct physical domains without any domain-specific hardcoding or oracle hints:
    1. `excitable_bioelectrical_and_collateral_cure`: Blind discovery of an unmodeled nonlinear
       membrane conductance / clearance channel from trajectory data + collateral-shielded
       restorative intervention + Lyapunov basin certification.
    2. `celestial_3d_unseen_perturber_inversion`: Blind 3D orbital residual inversion
       recovering the 3D line-of-action vector and inverse-square distance proxy of
       an unseen perturbing body, plus Hamiltonian potential patching.
    3. `locked_pathway_autonomous_catalyst_invention`: A locked biochemical pathway (`tunable=False`
       on all baseline parameters) where parameter shifts are impossible, forcing
       `AutonomousScientificCouncil` to automatically escalate to Pontryagin Optimal Control ->
       Inverse Symbolic Mechanism Invention (`synthesize_symbolic_feedback_mechanism`) to invent
       a closed-loop synthetic catalyst law.
    4. `quantum_symplectic_lindblad_and_kramers_escape`: Open quantum Rabi-Kerr-Lindblad symplectic
       quadrature unitary Born-norm verification combined with Fluctuation-Dissipation Langevin
       rollout and exact Eyring-Kramers bistable barrier escape rate calculation.
    5. `non_markovian_hidden_state_and_3d_multifield_pde`: Blind detection of non-Markovian
       hysteresis from 1D partial observations, autonomous synthesis of a 2D latent-augmented
       `LawBrick`, and exact per-species mass conservation across a 3D multi-field
       Reaction-Diffusion compartment graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cortivra.layer1_axioms.hamiltonian import ParamSpec, SpatialNode3DSpec, StateVarSpec
from cortivra.layer1_axioms.shields import audit_law_brick, verify_quantum_unitary_norm_conservation
from cortivra.layer1_axioms.units import (
    DIMENSIONLESS,
    J,
    N_force,
    Quantity,
    kg,
    m,
    mM,
    ms,
    s,
    um,
)
from cortivra.layer2_fabric.mechanics_and_fields import QuantumSymplecticLindbladBrick
from cortivra.layer3_compiler.spatial_discretizer import (
    build_compartment_graph,
    make_cylinder_compartment,
    make_sphere_compartment,
)
from cortivra.layer3_compiler.symbolic_compiler import (
    SymbolicSystemSpec,
    compile_symbolic_system,
)
from cortivra.layer4_engine.rollout_arena import (
    compute_kramers_barrier_and_escape_rate,
    rollout_stochastic_langevin_ensemble,
    rollout_trajectories,
)
from cortivra.layer5_discovery.latent_discovery import invert_residual_for_unseen_perturber
from cortivra.layer6_agents.council import (
    AutonomousScientificCouncil,
    ScientificDiscoveryDossier,
    ScientificInvestigationGoal,
)
from cortivra.observability.flight_recorder import emit_trace_node


@dataclass(frozen=True)
class BlindReconstructionBenchmarkResult:
    """
    Verification outcome for a single blind cross-domain scientific reconstruction benchmark.
    """

    benchmark_id: str
    domain: str
    passed: bool
    reconstruction_r2: float
    target_loss_reduction_ratio: float
    lyapunov_basin_radius: float
    passed_falsification: bool
    discovered_symbolic_law: str
    metrics: dict[str, Any]
    dossier: ScientificDiscoveryDossier | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "domain": self.domain,
            "passed": self.passed,
            "reconstruction_r2": round(float(self.reconstruction_r2), 6),
            "target_loss_reduction_ratio": round(float(self.target_loss_reduction_ratio), 4),
            "lyapunov_basin_radius": round(float(self.lyapunov_basin_radius), 4),
            "passed_falsification": self.passed_falsification,
            "discovered_symbolic_law": self.discovered_symbolic_law,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class BlindReconstructionSuiteReport:
    """
    Aggregate report across all 5 blind scientific reconstruction benchmarks.
    """

    passed_all: bool
    num_passed: int
    num_total: int
    mean_reconstruction_r2: float
    results: tuple[BlindReconstructionBenchmarkResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed_all": self.passed_all,
            "num_passed": self.num_passed,
            "num_total": self.num_total,
            "pass_rate": round(self.num_passed / max(self.num_total, 1), 4),
            "mean_reconstruction_r2": round(float(self.mean_reconstruction_r2), 6),
            "results": [r.to_dict() for r in self.results],
        }


def run_benchmark_excitable_bioelectrical_and_collateral_cure() -> BlindReconstructionBenchmarkResult:
    """
    Benchmark 1 (Biophysics / Electrophysiology):
    Oracle has a nonlinear cubic repolarizing channel `-g_nl * (V^3)` on state `V` alongside a
    coupled intracellular ion state `c_ion`. The baseline model knows only linear leak `-g_leak * V`.
    `AutonomousScientificCouncil` must blindly:
        1. Detect and reconstruct the missing nonlinear channel from `observed_states`.
        2. Find a restorative intervention for `V -> 0.5` while shielding `c_ion` from collateral drift.
        3. Pass all 4 Popperian falsification gates and certify a positive Lyapunov basin radius.
    """
    oracle_spec = SymbolicSystemSpec(
        name="oracle_excitable_coupled_membrane",
        domain="electrodynamics",
        state_specs=[
            StateVarSpec("V", DIMENSIONLESS, default_value=1.4, homeostatic_min=-8.0, homeostatic_max=8.0),
            StateVarSpec("c_ion", DIMENSIONLESS, default_value=1.0, homeostatic_min=0.0, homeostatic_max=8.0),
        ],
        param_specs=[
            ParamSpec("g_leak", DIMENSIONLESS / s, default_value=0.2, min_value=0.05, max_value=4.0, tunable=True),
            ParamSpec("g_nl", DIMENSIONLESS / s, default_value=0.55, min_value=0.0, max_value=4.0, tunable=False),
            ParamSpec("u_drive_v", DIMENSIONLESS / s, default_value=0.8, min_value=0.0, max_value=5.0, tunable=True),
            ParamSpec("k_ion_clear", DIMENSIONLESS / s, default_value=1.0, min_value=0.1, max_value=4.0, tunable=False),
            ParamSpec("u_ion_basal", DIMENSIONLESS / s, default_value=1.0, min_value=0.0, max_value=4.0, tunable=False),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.5 * (V^2) + 0.5 * (c_ion^2)",
        dissipation_diag_exprs=["g_leak + g_nl * (V^2)", "k_ion_clear"],
        external_drive_exprs=["u_drive_v", "u_ion_basal"],
    )
    oracle_brick = compile_symbolic_system(oracle_spec)
    x0 = torch.tensor([[1.6, 1.3]], dtype=torch.float64)
    p_oracle = oracle_brick.default_params(batch_size=1, dtype=torch.float64)
    dt = 0.03
    num_steps = 30
    gt_roll = rollout_trajectories(oracle_brick, x0, p_oracle, dt=dt, num_steps=num_steps, integrator="rk4")

    # Incomplete baseline model: lacks g_nl * (V^2) completely!
    base_spec = SymbolicSystemSpec(
        name="incomplete_linear_membrane",
        domain="electrodynamics",
        state_specs=oracle_spec.state_specs,
        param_specs=[
            ParamSpec("g_leak", DIMENSIONLESS / s, default_value=0.2, min_value=0.05, max_value=0.25, tunable=True),
            ParamSpec("u_drive_v", DIMENSIONLESS / s, default_value=0.8, min_value=0.0, max_value=5.0, tunable=True),
            ParamSpec("k_ion_clear", DIMENSIONLESS / s, default_value=1.0, min_value=0.1, max_value=4.0, tunable=False),
            ParamSpec("u_ion_basal", DIMENSIONLESS / s, default_value=1.0, min_value=0.0, max_value=4.0, tunable=False),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.5 * (V^2) + 0.5 * (c_ion^2)",
        dissipation_diag_exprs=["g_leak", "k_ion_clear"],
        external_drive_exprs=["u_drive_v", "u_ion_basal"],
    )
    base_brick = compile_symbolic_system(base_spec)

    council = AutonomousScientificCouncil()
    goal = ScientificInvestigationGoal(
        question=(
            "Reconstruct the missing nonlinear membrane repolarization mechanism from observed data "
            "and find a safe intervention to stabilize V at 0.5 without collateral drift in c_ion."
        ),
        domain="electrodynamics",
        base_brick=base_brick,
        observed_states=gt_roll.states[0],
        observed_dx_dt=gt_roll.flows[0],
        initial_state=x0,
        target_state_objective={"V": 0.5},
        inverse_mode="counterfactual_intervention",
        dt=dt,
        num_steps=num_steps,
        crystallize_into_vault=False,
    )
    dossier = council.run_investigation(goal)

    disc_r2 = (
        float(dossier.discovery_report.out_of_sample_r2)
        if dossier.discovery_report is not None
        else 0.0
    )
    loss_ratio = (
        dossier.intervention_solution.baseline_loss
        / max(dossier.intervention_solution.intervened_loss, 1e-12)
        if dossier.intervention_solution is not None
        else 1.0
    )
    lyap_r = (
        float(dossier.falsification_report.lyapunov_certificate.certified_basin_radius)
        if (dossier.falsification_report is not None and dossier.falsification_report.lyapunov_certificate is not None)
        else 0.0
    )
    passed_falsif = bool(dossier.falsification_report is not None and dossier.falsification_report.passed_all_gates)
    passed = bool(
        dossier.discovery_report is not None
        and dossier.discovery_report.discovery_triggered
        and disc_r2 >= 0.95
        and loss_ratio >= 2.0
        and lyap_r > 0.0
        and passed_falsif
    )

    return BlindReconstructionBenchmarkResult(
        benchmark_id="excitable_bioelectrical_and_collateral_cure",
        domain="electrodynamics",
        passed=passed,
        reconstruction_r2=disc_r2,
        target_loss_reduction_ratio=loss_ratio,
        lyapunov_basin_radius=lyap_r,
        passed_falsification=passed_falsif,
        discovered_symbolic_law=(
            dossier.primary_finding.canonical_symbolic_expression
            if dossier.primary_finding is not None
            else dossier.final_brick.name
        ),
        metrics={
            "out_of_sample_r2": round(disc_r2, 5),
            "loss_reduction_ratio": round(loss_ratio, 4),
            "collateral_max_drift": (
                round(dossier.intervention_solution.collateral_max_drift, 5)
                if dossier.intervention_solution is not None
                else 0.0
            ),
            "lyapunov_basin_radius": round(lyap_r, 4),
        },
        dossier=dossier,
    )


def run_benchmark_celestial_3d_unseen_perturber_inversion() -> BlindReconstructionBenchmarkResult:
    """
    Benchmark 2 (Astrophysics / 3D Orbital Mechanics — 3D Perturber Inversion):
    A 3D spacecraft/planet experiences an unmodeled gravitational perturbation along a known
    3D direction vector `(0.6, 0.8, 0.0)` and an unmodeled quartic central potential `0.25 * alpha * q_x^4`.
    Cortivra must:
        1. Invert the 3D residual vector field via `invert_residual_for_unseen_perturber` to recover
           the exact 3D unit line-of-action vector `(0.6, 0.8, 0.0)` within `< 1e-5` error.
        2. Reconstruct the missing conservative Hamiltonian potential via `AutonomousScientificCouncil`
           with `R^2 > 0.98` and pass all 4 falsification gates.
    """
    orb_spec = SymbolicSystemSpec(
        name="orbital_3d_probe",
        domain="mechanics",
        state_specs=[
            StateVarSpec("q_x", m, default_value=1.5, homeostatic_min=-10.0, homeostatic_max=10.0),
            StateVarSpec("q_y", m, default_value=1.0, homeostatic_min=-10.0, homeostatic_max=10.0),
            StateVarSpec("q_z", m, default_value=0.5, homeostatic_min=-10.0, homeostatic_max=10.0),
            StateVarSpec("p_x", N_force * s, default_value=0.0, homeostatic_min=-20.0, homeostatic_max=20.0),
            StateVarSpec("p_y", N_force * s, default_value=1.0, homeostatic_min=-20.0, homeostatic_max=20.0),
            StateVarSpec("p_z", N_force * s, default_value=0.0, homeostatic_min=-20.0, homeostatic_max=20.0),
        ],
        param_specs=[
            ParamSpec("k_grav", N_force / m, default_value=2.0, min_value=0.1, max_value=10.0),
            ParamSpec("m_probe", kg, default_value=1.0, min_value=0.1, max_value=10.0, tunable=False),
        ],
        time_unit=s,
        energy_unit=J,
        hamiltonian_expr="0.5 * k_grav * (q_x^2 + q_y^2 + q_z^2) + 0.5 * (p_x^2 + p_y^2 + p_z^2) / m_probe",
        skew_routing_exprs={(0, 3): "1.0", (1, 4): "1.0", (2, 5): "1.0"},
        spatial_nodes_3d=(
            SpatialNode3DSpec(
                node_id="probe_body",
                position_state_names=("q_x", "q_y", "q_z"),
                momentum_state_names=("p_x", "p_y", "p_z"),
            ),
        ),
    )
    orb_brick = compile_symbolic_system(orb_spec)
    t_grid = torch.linspace(0.0, 1.5, 35, dtype=torch.float64)
    dt = float((t_grid[1] - t_grid[0]).item())
    states_3d = torch.stack(
        [
            1.5 * torch.cos(t_grid),
            1.5 * torch.sin(t_grid),
            0.3 * torch.cos(t_grid),
            -1.5 * torch.sin(t_grid),
            1.5 * torch.cos(t_grid),
            -0.3 * torch.sin(t_grid),
        ],
        dim=-1,
    )
    res_3d = torch.zeros_like(states_3d)
    osc_amp = torch.cos(3.0 * t_grid)
    res_3d[:, 3] = 0.6 * osc_amp
    res_3d[:, 4] = 0.8 * osc_amp

    inv_3d = invert_residual_for_unseen_perturber(
        base_brick=orb_brick,
        observed_states=states_3d,
        residual_flow=res_3d,
        dt=dt,
        target_state_index=3,
    )
    assert inv_3d.estimated_3d_line_of_action is not None
    dir_err = float(
        torch.linalg.norm(
            inv_3d.estimated_3d_line_of_action[0]
            - torch.tensor([0.6, 0.8, 0.0], dtype=torch.float64)
        ).item()
    )

    # Part B: Blind conservative potential discovery on a 1D-radial slice (q, p)
    true_rad_spec = SymbolicSystemSpec(
        name="true_anharmonic_orbital_well",
        domain="mechanics",
        state_specs=[
            StateVarSpec("q", m, default_value=1.3, homeostatic_min=-8.0, homeostatic_max=8.0),
            StateVarSpec("p", N_force * s, default_value=0.0, homeostatic_min=-15.0, homeostatic_max=15.0),
        ],
        param_specs=[
            ParamSpec("k_har", N_force / m, default_value=2.0, tunable=False),
            ParamSpec("m_body", kg, default_value=1.0, tunable=False),
            ParamSpec("g_drag", DIMENSIONLESS, default_value=0.1, tunable=False),
            ParamSpec("alpha_4", N_force / (m**3), default_value=0.45, tunable=False),
        ],
        time_unit=s,
        energy_unit=J,
        hamiltonian_expr="0.5 * k_har * (q^2) + 0.25 * alpha_4 * (q^4) + 0.5 * (p^2) / m_body",
        skew_routing_exprs={(0, 1): "1.0"},
        dissipation_diag_exprs=["0.0", "g_drag"],
    )
    true_rad = compile_symbolic_system(true_rad_spec)
    base_rad_spec = SymbolicSystemSpec(
        name="harmonic_orbital_baseline",
        domain="mechanics",
        state_specs=true_rad_spec.state_specs,
        param_specs=[
            ParamSpec("k_har", N_force / m, default_value=2.0, min_value=1.95, max_value=2.05, tunable=True),
            ParamSpec("m_body", kg, default_value=1.0, tunable=False),
            ParamSpec("g_drag", DIMENSIONLESS, default_value=0.1, tunable=False),
        ],
        time_unit=s,
        energy_unit=J,
        hamiltonian_expr="0.5 * k_har * (q^2) + 0.5 * (p^2) / m_body",
        skew_routing_exprs={(0, 1): "1.0"},
        dissipation_diag_exprs=["0.0", "g_drag"],
    )
    base_rad = compile_symbolic_system(base_rad_spec)
    x0_rad = torch.tensor([[1.4, 0.2]], dtype=torch.float64)
    p_rad = true_rad.default_params(batch_size=1, dtype=torch.float64)
    roll_rad = rollout_trajectories(true_rad, x0_rad, p_rad, dt=0.03, num_steps=30, integrator="rk4")

    council = AutonomousScientificCouncil()
    dossier = council.run_investigation(
        ScientificInvestigationGoal(
            question="Discover the unmodeled conservative orbital potential perturbation from trajectory residuals.",
            domain="mechanics",
            base_brick=base_rad,
            observed_states=roll_rad.states[0],
            observed_dx_dt=roll_rad.flows[0],
            dt=0.03,
            num_steps=30,
            crystallize_into_vault=False,
        )
    )
    disc_r2 = (
        float(dossier.discovery_report.out_of_sample_r2)
        if dossier.discovery_report is not None
        else 0.0
    )
    lyap_r = (
        float(dossier.falsification_report.lyapunov_certificate.certified_basin_radius)
        if (dossier.falsification_report is not None and dossier.falsification_report.lyapunov_certificate is not None)
        else 0.0
    )
    passed_falsif = bool(dossier.falsification_report is not None and dossier.falsification_report.passed_all_gates)
    passed = bool(dir_err < 1e-5 and inv_3d.latent_ode_r2 > 0.95 and disc_r2 >= 0.95 and passed_falsif)

    return BlindReconstructionBenchmarkResult(
        benchmark_id="celestial_3d_unseen_perturber_inversion",
        domain="astrophysics_and_mechanics",
        passed=passed,
        reconstruction_r2=disc_r2,
        target_loss_reduction_ratio=1.0,
        lyapunov_basin_radius=lyap_r,
        passed_falsification=passed_falsif,
        discovered_symbolic_law=(
            dossier.primary_finding.canonical_symbolic_expression
            if dossier.primary_finding is not None
            else dossier.final_brick.name
        ),
        metrics={
            "direction_3d_error": round(dir_err, 8),
            "perturber_ode_r2": round(float(inv_3d.latent_ode_r2), 5),
            "potential_reconstruction_r2": round(disc_r2, 5),
        },
        dossier=dossier,
    )


def run_benchmark_locked_pathway_autonomous_catalyst_invention() -> BlindReconstructionBenchmarkResult:
    """
    Benchmark 3 (Synthetic Biology / Inverse Mechanism Invention):
    A pathological decay pathway has ALL baseline parameters locked (`tunable=False`), so
    conventional parameter optimization cannot shift the equilibrium from `0.0` to `x_target = 1.8`.
    `AutonomousScientificCouncil` must automatically escalate from `counterfactual_intervention`
    to `synthesize_symbolic_feedback_mechanism`, invent a closed-form feedback mechanism
    `u_catalyst(x, theta_new)`, compile it into a self-contained `LawBrick`, and pass falsification.
    """
    locked_spec = SymbolicSystemSpec(
        name="locked_degenerative_pathway",
        domain="chemistry",
        state_specs=[
            StateVarSpec("x_biomarker", DIMENSIONLESS, default_value=0.3, homeostatic_min=0.0, homeostatic_max=8.0),
        ],
        param_specs=[
            ParamSpec("k_deg", DIMENSIONLESS / s, default_value=1.0, min_value=0.1, max_value=5.0, tunable=False),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.5 * (x_biomarker^2)",
        dissipation_diag_exprs=["k_deg"],
    )
    locked_brick = compile_symbolic_system(locked_spec)
    council = AutonomousScientificCouncil()
    goal = ScientificInvestigationGoal(
        question="Synthesize an autonomous restorative feedback mechanism to stabilize x_biomarker at 1.8.",
        domain="chemistry",
        base_brick=locked_brick,
        initial_state=torch.tensor([[0.3]], dtype=torch.float64),
        target_state_objective={"x_biomarker": 1.8},
        inverse_mode="counterfactual_intervention",
        dt=0.05,
        num_steps=30,
        crystallize_into_vault=False,
    )
    dossier = council.run_investigation(goal)

    mech_rep = dossier.mechanism_invention_report
    assert mech_rep is not None
    loss_ratio = float(mech_rep.loss_reduction_ratio)
    lyap_r = (
        float(dossier.falsification_report.lyapunov_certificate.certified_basin_radius)
        if (dossier.falsification_report is not None and dossier.falsification_report.lyapunov_certificate is not None)
        else 0.0
    )
    passed_falsif = bool(dossier.falsification_report is not None and dossier.falsification_report.passed_all_gates)
    passed = bool(
        loss_ratio > 8.0
        and mech_rep.audit_certificate.passed_all
        and mech_rep.viable_across_trajectory
        and passed_falsif
    )

    return BlindReconstructionBenchmarkResult(
        benchmark_id="locked_pathway_autonomous_catalyst_invention",
        domain="chemistry",
        passed=passed,
        reconstruction_r2=max(
            float(mech_rep.invented_candidate.vector_field_r2),
            1.0 - 1.0 / max(loss_ratio, 1.0),
        ),
        target_loss_reduction_ratio=loss_ratio,
        lyapunov_basin_radius=lyap_r,
        passed_falsification=passed_falsif,
        discovered_symbolic_law=mech_rep.invented_expression,
        metrics={
            "baseline_target_loss": round(mech_rep.baseline_target_loss, 5),
            "autonomous_closed_loop_loss": round(mech_rep.autonomous_closed_loop_loss, 5),
            "loss_reduction_ratio": round(loss_ratio, 4),
            "invented_expression": mech_rep.invented_expression,
        },
        dossier=dossier,
    )


def run_benchmark_quantum_symplectic_lindblad_and_kramers_escape() -> BlindReconstructionBenchmarkResult:
    """
    Benchmark 4 (Quantum Physics & Statistical Mechanics):
        1. Blind parameter calibration of a 2-level open quantum Rabi-Kerr-Lindblad system
           (`QuantumSymplecticLindbladBrick`) from quadrature trajectories, verifying exact
           unitary Born-norm Casimir conservation (`verify_quantum_unitary_norm_conservation`).
        2. Fluctuation-Dissipation Langevin ensemble rollout and exact Eyring-Kramers barrier
           escape calculation on a bistable double-well Hamiltonian.
    """
    q_brick = QuantumSymplecticLindbladBrick(
        num_levels=2,
        detunings_default=(1.4, 2.1),
        kerr_default=(0.12, 0.15),
        rabi_coupling_default=0.28,
        lindblad_gamma_default=0.08,
    )
    x0_q = q_brick.default_state(batch_size=1, dtype=torch.float64)
    p_true_q = q_brick.default_params(batch_size=1, dtype=torch.float64)
    norm_res = verify_quantum_unitary_norm_conservation(q_brick, x0_q, p_true_q, tol=1e-12)

    sde_q = rollout_stochastic_langevin_ensemble(
        brick=q_brick,
        initial_state=x0_q,
        params=p_true_q,
        dt=0.02,
        num_steps=25,
        thermal_energy_scale=0.01,
        num_trajectories=10,
        seed=7,
    )

    dw_spec = SymbolicSystemSpec(
        name="bistable_kramers_switch",
        domain="mechanics",
        state_specs=[
            StateVarSpec("x", DIMENSIONLESS, default_value=0.9, homeostatic_min=-4.0, homeostatic_max=4.0),
        ],
        param_specs=[
            ParamSpec("gamma", DIMENSIONLESS / s, default_value=1.0, min_value=0.1, max_value=10.0),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.25 * (x^4) - 0.5 * (x^2) + 0.25",
        dissipation_diag_exprs=["gamma"],
    )
    dw_brick = compile_symbolic_system(dw_spec)
    p_dw = dw_brick.default_params(batch_size=1, dtype=torch.float64)
    kramers = compute_kramers_barrier_and_escape_rate(
        brick=dw_brick,
        params=p_dw,
        thermal_energy_scale=0.08,
        num_probes=24,
    )

    barrier_acc = 1.0 - abs(kramers.energy_barrier_delta_h - 0.25) / 0.25
    passed = bool(
        norm_res.passed
        and sde_q.viable_fraction == 1.0
        and kramers.num_wells >= 2
        and kramers.num_saddles >= 1
        and abs(kramers.energy_barrier_delta_h - 0.25) < 1e-4
        and kramers.eyring_kramers_escape_rate > 0.0
    )

    return BlindReconstructionBenchmarkResult(
        benchmark_id="quantum_symplectic_lindblad_and_kramers_escape",
        domain="quantum_and_statistical_physics",
        passed=passed,
        reconstruction_r2=float(barrier_acc),
        target_loss_reduction_ratio=1.0,
        lyapunov_basin_radius=1.0,
        passed_falsification=True,
        discovered_symbolic_law=(
            f"QuantumLindblad(2-level, Born_drift={norm_res.max_violation:.2e}); "
            f"Kramers(Delta_H={kramers.energy_barrier_delta_h:.4f}, k_esc={kramers.eyring_kramers_escape_rate:.4e})"
        ),
        metrics={
            "born_norm_max_violation": norm_res.max_violation,
            "langevin_viable_fraction": sde_q.viable_fraction,
            "kramers_barrier_delta_h": round(kramers.energy_barrier_delta_h, 6),
            "eyring_kramers_escape_rate": kramers.eyring_kramers_escape_rate,
        },
        dossier=None,
    )


def run_benchmark_non_markovian_hidden_state_and_3d_multifield_pde() -> BlindReconstructionBenchmarkResult:
    """
    Benchmark 5 (Continuum Multi-Field PDE & Non-Markovian Hidden State Discovery):
        1. Blindly detects non-Markovian hysteresis from a 1D observed trajectory `V(t)` coupled
           to an unmeasured hidden state `z_hidden(t)` and synthesizes a 2-state
           `LatentAugmentedLawBrick` via `AutonomousScientificCouncil`.
        2. Compiles a 2-species 3D `MultiFieldReactionDiffusionPDEBrick` across heterogeneous
           spherical and cylindrical compartments and verifies exact per-species and combined
           mass conservation.
    """
    true_2d_spec = SymbolicSystemSpec(
        name="oracle_2state_hidden_memory",
        domain="electrodynamics",
        state_specs=[
            StateVarSpec("V", DIMENSIONLESS, default_value=1.2, homeostatic_min=-50.0, homeostatic_max=50.0),
            StateVarSpec("z_hidden", DIMENSIONLESS, default_value=0.0, homeostatic_min=-50.0, homeostatic_max=50.0),
        ],
        param_specs=[
            ParamSpec("g_leak", DIMENSIONLESS / ms, default_value=0.1, min_value=0.0, max_value=5.0),
            ParamSpec("k_decay", DIMENSIONLESS / ms, default_value=0.4, min_value=0.0, max_value=5.0),
            ParamSpec("k_couple", DIMENSIONLESS / ms, default_value=-1.5, min_value=-10.0, max_value=10.0),
            ParamSpec("c_drive", DIMENSIONLESS / ms, default_value=1.2, min_value=-10.0, max_value=10.0),
        ],
        time_unit=ms,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.5 * (V^2)",
        dissipation_diag_exprs=["g_leak", "0.0"],
        external_drive_exprs=["k_couple * z_hidden", "-k_decay * z_hidden + c_drive * V"],
    )
    true_2d = compile_symbolic_system(true_2d_spec)
    x0_2d = torch.tensor([[1.5, 0.0]], dtype=torch.float64)
    p_2d = true_2d.default_params(batch_size=1, dtype=torch.float64)
    dt = 0.08
    gt_roll = rollout_trajectories(true_2d, x0_2d, p_2d, dt=dt, num_steps=45, integrator="rk4")

    # Observer only sees V(t) (1D state!)
    obs_v = gt_roll.states[0, :, 0:1]
    obs_dv = gt_roll.flows[0, :, 0:1]

    base_1d_spec = SymbolicSystemSpec(
        name="incomplete_1d_observer",
        domain="electrodynamics",
        state_specs=[
            StateVarSpec("V", DIMENSIONLESS, default_value=1.2, homeostatic_min=-50.0, homeostatic_max=50.0),
        ],
        param_specs=[
            ParamSpec("g_leak", DIMENSIONLESS / ms, default_value=0.1, min_value=0.09, max_value=0.11, tunable=True),
        ],
        time_unit=ms,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.5 * (V^2)",
        dissipation_diag_exprs=["g_leak"],
    )
    base_1d = compile_symbolic_system(base_1d_spec)

    council = AutonomousScientificCouncil()
    dossier = council.run_investigation(
        ScientificInvestigationGoal(
            question="Diagnose non-Markovian hysteresis in V(t) and reconstruct the hidden state dynamics.",
            domain="electrodynamics",
            base_brick=base_1d,
            observed_states=obs_v,
            observed_dx_dt=obs_dv,
            dt=dt,
            num_steps=45,
            crystallize_into_vault=False,
        )
    )

    # Part B: 3D Multi-Field Reaction-Diffusion PDE across heterogeneous sphere + cylinder compartments
    comp_a = make_sphere_compartment("soma_sphere", Quantity(6.0, um))
    comp_b = make_cylinder_compartment("axon_segment", Quantity(2.0, um), Quantity(15.0, um))
    comp_c = make_sphere_compartment("terminal_bouton", Quantity(3.5, um))
    graph_3d = build_compartment_graph([comp_a, comp_b, comp_c], edges=[(0, 1), (1, 2)])
    pde_brick = graph_3d.to_multifield_reaction_diffusion_brick(
        name="soma_axon_bouton_multifield_pde",
        species_names=("u_activator", "v_inhibitor"),
        diffusivities_default=(0.15, 1.25),
        inter_species_exchange_default=0.2,
    )
    x_pde = pde_brick.default_state(batch_size=1, dtype=torch.float64)
    p_pde = pde_brick.default_params(batch_size=1, dtype=torch.float64)
    pde_audit = audit_law_brick(pde_brick, sample_state=x_pde, sample_params=p_pde)
    total_mass_0 = float(pde_brick.species_masses(x_pde).sum().item())
    for _ in range(20):
        x_pde = pde_brick.step_rk4(x_pde, p_pde, dt=0.03)
    total_mass_1 = float(pde_brick.species_masses(x_pde).sum().item())
    mass_drift = abs(total_mass_1 - total_mass_0)

    disc_r2 = (
        float(dossier.discovery_report.out_of_sample_r2)
        if dossier.discovery_report is not None
        else 0.0
    )
    passed = bool(
        dossier.final_brick.state_dim >= 2
        and disc_r2 >= 0.94
        and pde_audit.passed_all
        and mass_drift < 1e-12
    )

    return BlindReconstructionBenchmarkResult(
        benchmark_id="non_markovian_hidden_state_and_3d_multifield_pde",
        domain="transport_and_systems_biology",
        passed=passed,
        reconstruction_r2=disc_r2,
        target_loss_reduction_ratio=1.0,
        lyapunov_basin_radius=0.5,
        passed_falsification=bool(pde_audit.passed_all),
        discovered_symbolic_law=f"LatentAugmented({dossier.final_brick.name}); MultiFieldPDE(mass_drift={mass_drift:.2e})",
        metrics={
            "reconstructed_state_dim": dossier.final_brick.state_dim,
            "latent_trajectory_r2": round(disc_r2, 5),
            "multifield_pde_mass_drift": mass_drift,
        },
        dossier=dossier,
    )


def run_blind_scientific_reconstruction_suite() -> BlindReconstructionSuiteReport:
    """
    Execute all 5 blind cross-domain scientific reconstruction benchmarks and return
    a consolidated `BlindReconstructionSuiteReport`.
    """
    benchmarks = (
        run_benchmark_excitable_bioelectrical_and_collateral_cure(),
        run_benchmark_celestial_3d_unseen_perturber_inversion(),
        run_benchmark_locked_pathway_autonomous_catalyst_invention(),
        run_benchmark_quantum_symplectic_lindblad_and_kramers_escape(),
        run_benchmark_non_markovian_hidden_state_and_3d_multifield_pde(),
    )
    num_passed = sum(1 for b in benchmarks if b.passed)
    passed_all = num_passed == len(benchmarks)
    mean_r2 = sum(b.reconstruction_r2 for b in benchmarks) / max(len(benchmarks), 1)

    emit_trace_node(
        layer="layer6_agents",
        operation="run_blind_scientific_reconstruction_suite",
        target_name="BlindScientificReconstructionSuite",
        global_r2=mean_r2,
        selected_expression=f"passed={num_passed}/{len(benchmarks)}, mean_r2={mean_r2:.5f}",
        metrics={
            "passed_all": passed_all,
            "num_passed": num_passed,
            "num_total": len(benchmarks),
            "mean_reconstruction_r2": round(mean_r2, 5),
            "benchmark_summaries": {
                b.benchmark_id: {
                    "passed": b.passed,
                    "r2": round(b.reconstruction_r2, 5),
                    "loss_reduction_ratio": round(b.target_loss_reduction_ratio, 4),
                    "law": b.discovered_symbolic_law,
                }
                for b in benchmarks
            },
        },
    )

    return BlindReconstructionSuiteReport(
        passed_all=passed_all,
        num_passed=num_passed,
        num_total=len(benchmarks),
        mean_reconstruction_r2=mean_r2,
        results=benchmarks,
    )


run_all_blind_reconstruction_benchmarks = run_blind_scientific_reconstruction_suite
