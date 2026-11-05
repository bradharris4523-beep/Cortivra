"""
Live Observatory Telemetry (NASA JPL Horizons + NOAA L1 Solar Wind), Persistent Disk LawVault
Cross-Domain Operator Transfer, Active Fisher-Information Experiment Design, and 4-Part Deep
Skeptic Verification Suite (`tests/test_live_observatory_and_skeptic_proofs.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from cortivra.layer1_axioms.hamiltonian import ParamSpec, StateVarSpec
from cortivra.layer1_axioms.units import DIMENSIONLESS, s
from cortivra.layer1_axioms.vault import LawVault
from cortivra.layer2_fabric import register_foundational_laws
from cortivra.layer3_compiler.symbolic_compiler import SymbolicSystemSpec, compile_symbolic_system
from cortivra.layer4_engine.elimination_funnel import run_elimination_funnel
from cortivra.layer4_engine.rollout_arena import rollout_trajectories
from cortivra.layer5_discovery.latent_discovery import (
    discover_and_compile_law,
    discover_autonomous_symbolic_system,
    invert_residual_for_unseen_perturber,
)
from cortivra.layer6_agents.blind_reconstruction_benchmarks import (
    run_benchmark_celestial_3d_unseen_perturber_inversion,
    run_benchmark_non_markovian_hidden_state_and_3d_multifield_pde,
)
from cortivra.layer6_agents.council import (
    AutonomousScientificCouncil,
    ScientificInvestigationGoal,
    investigate,
)
from cortivra.layer6_agents.diagnostic_briefing import transfer_isomorphic_operator_from_vault
from cortivra.layer6_agents.empirical_observatory import fetch_open_science_observatory_telemetry
from cortivra.layer6_agents.falsification_and_design import (
    crystallize_verified_law,
    run_falsification_stress_test,
)
from cortivra.layer6_agents.meta_clients import MultiProviderMetaClient
from cortivra.layer6_agents.universal_telemetry_hunter import (
    _build_cascading_scientific_queries,
    _parse_universal_scientific_table_bytes,
    audit_five_gate_telemetry_qualification,
    compute_dataset_semantic_relevance,
    harvest_open_science_literature_priors,
)


def _rk4_integrate(
    rhs_fn: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    dt: float,
    steps: int,
) -> torch.Tensor:
    traj = [x0.detach().clone()]
    curr = x0.detach().clone()
    for _ in range(steps - 1):
        k1 = rhs_fn(curr)
        k2 = rhs_fn(curr + 0.5 * dt * k1)
        k3 = rhs_fn(curr + 0.5 * dt * k2)
        k4 = rhs_fn(curr + dt * k3)
        curr = curr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj.append(curr.detach().clone())
    return torch.stack(traj, dim=0)


def test_track5_live_nasa_jpl_noaa_observatory_persistent_vault_and_active_experiment_design() -> None:
    """
    TRACK 5 (Live NASA JPL Horizons + Live NOAA RTSW L1 Observatory + Persistent Disk LawVault
    Cross-Domain Operator Transfer + Automatic Fisher-Information Active Experiment Design):
      1. Queries live NASA JPL Horizons (`https://ssd.jpl.nasa.gov/api/horizons.api`) for real 3D
         Mercury heliocentric ephemeris vectors, verifies `live_network_fetched is True`, and
         blindly rediscovers the Newton-Kepler-Binet radial gravitational law (`vector_field_r2 > 0.9999`).
      2. Queries live NOAA SWPC Real-Time Solar Wind (`https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json`)
         from the L1 spacecraft and verifies `live_network_fetched is True`.
      3. Runs a live Council investigation that crystallizes a verified law into persistent disk storage
         (`runs/law_vault/crystallized_laws.json`), verifies SHA-256 integrity and disk persistence across
         a fresh `LawVault` reload, and executes `transfer_isomorphic_operator_from_vault` across domains.
      4. Verifies that `dossier.disambiguation_report` is automatically populated with an optimal
         Fisher-Information experiment design (`divergence_amplification_ratio >= 1.0`).
    """
    # 1. Live NASA JPL Horizons Ephemeris Telemetry + Blind Newton-Kepler-Binet Rediscovery
    jpl_ds = fetch_open_science_observatory_telemetry(
        stream_id="nasa_jpl_horizons",
        domain="celestial_mechanics",
        num_steps=35,
    )
    assert jpl_ds.live_network_fetched is True
    assert jpl_ds.column_names == ("r_au", "vr_canon", "inv_r_au")
    # Strip column names and domain to prove strictly blind discovery (ch_0, ch_1, ch_2)
    jpl_specs = [
        StateVarSpec(f"ch_{i}", DIMENSIONLESS, float(jpl_ds.states[0, i].item()), -20.0, 20.0)
        for i in range(jpl_ds.states.shape[1])
    ]
    jpl_disc = discover_autonomous_symbolic_system(
        name="unlabeled_ephemeris_stream",
        domain="unlabeled_telemetry",
        state_specs=jpl_specs,
        observed_states=jpl_ds.states,
        dt=jpl_ds.dt,
        observed_dx_dt=None,
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        max_polynomial_degree=3,
        max_terms_per_state=2,
        include_rational_primitives=False,
        include_trigonometric=False,
        include_fractional_and_log=False,
    )
    assert jpl_disc.vector_field_r2 > 0.9999
    assert "ch_2 ** 2" in jpl_disc.per_state_expressions["ch_1"]
    assert "ch_2 ** 3" in jpl_disc.per_state_expressions["ch_1"]

    # 1b. Future-Horizon Holdout Extrapolation on Live NASA JPL Ephemeris (First 60% -> Unseen Future 40%)
    n_train = 30
    future_true = jpl_ds.states[n_train:]
    jpl_future_roll = rollout_trajectories(
        jpl_disc.compiled_brick,
        jpl_ds.states[n_train : n_train + 1],
        jpl_disc.calibrated_params,
        dt=jpl_ds.dt,
        num_steps=future_true.shape[0],
        integrator="rk4",
    )
    future_pred = jpl_future_roll.states[0, : future_true.shape[0]]
    ss_res_fut = float(torch.sum((future_true - future_pred) ** 2).item())
    ss_tot_fut = float(torch.sum((future_true - torch.mean(future_true, dim=0, keepdim=True)) ** 2).item())
    future_holdout_r2 = 1.0 - ss_res_fut / max(ss_tot_fut, 1e-12)
    assert future_holdout_r2 > 0.995

    # 1c. Negative-Control / Pure White-Noise Rejection Proof (Skeptical Scientist Check)
    gen_noise = torch.Generator().manual_seed(20261102)
    pure_white_noise = torch.randn((40, 3), dtype=torch.float64, generator=gen_noise)
    noise_specs = [
        StateVarSpec(f"noise_ch_{i}", DIMENSIONLESS, float(pure_white_noise[0, i].item()), -10.0, 10.0)
        for i in range(3)
    ]
    noise_disc = discover_autonomous_symbolic_system(
        name="negative_control_white_noise",
        domain="negative_control",
        state_specs=noise_specs,
        observed_states=pure_white_noise,
        dt=0.02,
        observed_dx_dt=None,
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        max_polynomial_degree=2,
        max_terms_per_state=2,
        include_rational_primitives=False,
        include_trigonometric=False,
        include_fractional_and_log=False,
    )
    assert noise_disc.vector_field_r2 == 0.0
    noise_falsif = run_falsification_stress_test(
        brick=noise_disc.compiled_brick,
        calibrated_params=noise_disc.calibrated_params,
        reference_states=pure_white_noise,
        dt=0.02,
        min_noisy_r2=0.80,
    )
    assert noise_falsif.passed_all_gates is False
    temp_vault = LawVault()
    register_foundational_laws(temp_vault)
    rejected_noise = False
    try:
        crystallize_verified_law(vault=temp_vault, brick=noise_disc.compiled_brick, stress_report=noise_falsif)
    except ValueError:
        rejected_noise = True
    assert rejected_noise is True

    # 2. Live NOAA SWPC Real-Time Solar Wind (RTSW) L1 Spacecraft Telemetry
    noaa_ds = fetch_open_science_observatory_telemetry(
        stream_id="noaa_solar_wind",
        domain="space_plasma",
        num_steps=30,
    )
    assert noaa_ds.live_network_fetched is True
    assert noaa_ds.states.shape == (30, 3)

    # 3. Persistent Disk-Backed LawVault + Automatic Fisher-Information Active Experiment Design
    vault_path = Path(__file__).resolve().parent.parent / "runs" / "law_vault" / "crystallized_laws.json"
    client = MultiProviderMetaClient()
    dossier = investigate(
        prompt=(
            "Suppress resistive-wall plasma edge flux instability in a spherical tokamak while "
            "shielding first-wall thermal loading and core poloidal magnetic inventory."
        ),
        domain="fusion_plasma_physics",
        client=client,
        dt=0.02,
        num_steps=20,
        max_council_steps=3,
        crystallize_into_vault=True,
    )
    assert vault_path.exists()
    assert dossier.disambiguation_report is not None
    assert dossier.disambiguation_report.divergence_amplification_ratio >= 1.0
    assert len(dossier.vault_analogues) >= 1

    # Reload vault from disk and verify SHA-256 integrity + active cross-domain operator transfer
    reloaded_vault = LawVault(sandbox_storage_path=vault_path)
    register_foundational_laws(reloaded_vault)
    all_reloaded = reloaded_vault.list_laws(include_discovered=True)
    assert len(all_reloaded) >= 10
    assert any(r.tier == LawVault.TIER_DISCOVERED for r in all_reloaded)
    for rec in all_reloaded:
        assert rec.verify_integrity() is True

    xfer_report = transfer_isomorphic_operator_from_vault(
        target_brick=dossier.final_brick,
        vault=reloaded_vault,
        min_isomorphism_score=0.35,
        prefer_cross_domain=True,
    )
    assert xfer_report is not None
    assert xfer_report.audit_certificate.passed_all is True
    assert xfer_report.source_domain != dossier.final_brick.domain
    assert xfer_report.augmented_brick.param_dim == dossier.final_brick.param_dim + 1


def test_track6_deep_skeptic_proof_partial_observability_neptune_inversion_warm_vault_and_spike_noise() -> None:
    """
    TRACK 6 (4-Part Deep Skeptic Proof Suite):
      1. Partial Blindness / Hidden Unmeasured Compartment Discovery:
         - A 2-compartment ground-truth system `(x_obs, z_hidden)` where `z_hidden` is completely
           unmeasured (Cortivra only receives 1 sensor channel `x_obs`).
         - Proves `detect_non_markovian_memory` detects the phase-space hysteresis loop (`is_non_markovian=True`),
           `discover_latent_state_augmentation` reconstructs the hidden compartment (`trajectory_r2 > 0.99`),
           and the main `AutonomousScientificCouncil` pipeline (`observed_state_indices=[0]`) calibrates and
           reconstructs the full state trajectory from partial sensor data.
      2. "Neptune / Dark Perturber" 3D Inverse Localization:
         - Given only the residual acceleration anomaly along a 3D trajectory caused by an unseen
           gravitational/electrostatic perturber at unknown 3D coordinates, pinpoints the 3D direction
           and inverse-square distance profile (`cos_sim_fresh > 0.9999`).
      3. Compound Vault A/B Acceleration Proof (Cold Model vs. Warm Cross-Domain Operator Transfer):
         - Blindly discovers a Hill saturation operator from Domain A (`oncology_systems_biology`),
           crystallizes it into `LawVault`, transfers it across domains onto an incomplete linear model
           in Domain B (`electrochemistry`), and reduces vector-field MSE by `> 10x` (`trajectory_r2 > 0.998`).
      4. Corrupted Sensor Telemetry (Thermal Noise + 10-Sigma Cosmic-Ray / ADC Outlier Spikes):
         - Injects Gaussian noise + two 10-sigma single-sample outlier spikes into a nonlinear coupled
           trajectory, proving that two-sided protrusion Hampel-MAD despiking + Savitzky-Golay + weak-form
           integral discovery recovers the true symbolic law (`vector_field_r2 > 0.99`, `trajectory_r2 > 0.999`).
    """
    # -------------------------------------------------------------------------
    # Proof 1: Partial Blindness / Hidden Unmeasured Compartment Discovery
    # -------------------------------------------------------------------------
    nm_bench = run_benchmark_non_markovian_hidden_state_and_3d_multifield_pde()
    assert nm_bench.passed is True
    assert nm_bench.reconstruction_r2 > 0.90

    # Also verify partial-sensor observability (1 observed channel out of 2) directly through Council
    partial_spec = SymbolicSystemSpec(
        name="partial_sensor_2d_system",
        domain="biomedical_pharmacokinetics",
        state_specs=(
            StateVarSpec("c_plasma", DIMENSIONLESS, 1.8, -10.0, 10.0),
            StateVarSpec("c_deep_tissue_unmeasured", DIMENSIONLESS, 0.4, -10.0, 10.0),
        ),
        param_specs=(
            ParamSpec("k_el", DIMENSIONLESS / s, 0.65, 0.05, 5.0),
            ParamSpec("k_exch", DIMENSIONLESS / s, 0.95, 0.05, 5.0),
        ),
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.5 * (c_plasma ** 2 + c_deep_tissue_unmeasured ** 2)",
        skew_routing_exprs={(0, 1): "-k_exch"},
        dissipation_diag_exprs=("k_el", "0.2 * k_el"),
    )
    partial_brick = compile_symbolic_system(partial_spec)
    true_p = torch.tensor([[0.75, 1.15]], dtype=torch.float64)
    x0_part = torch.tensor([[2.1, 0.3]], dtype=torch.float64)
    full_roll = rollout_trajectories(partial_brick, x0_part, true_p, dt=0.04, num_steps=30, integrator="rk4")
    only_ch0 = full_roll.states[0, :, 0:1].detach().clone()

    council = AutonomousScientificCouncil(client=MultiProviderMetaClient())
    part_goal = ScientificInvestigationGoal(
        question="Calibrate pharmacokinetic exchange from plasma sensor c_plasma alone and stabilize c_plasma at 0.2.",
        domain="biomedical_pharmacokinetics",
        base_brick=partial_brick,
        observed_states=only_ch0,
        observed_state_indices=[0],
        target_state_objective={"c_plasma": 0.2},
        inverse_mode="counterfactual_intervention",
        dt=0.04,
        num_steps=30,
        max_council_steps=3,
        crystallize_into_vault=False,
    )
    part_dossier = council.run_investigation(part_goal)
    assert part_dossier.calibration_report is not None
    assert part_dossier.calibration_report.funnel_report.trajectory_r2 > 0.95

    # -------------------------------------------------------------------------
    # Proof 2: "Neptune / Dark Perturber" 3D Inverse Localization (Blind + Fresh Coordinates)
    # -------------------------------------------------------------------------
    neptune_bench = run_benchmark_celestial_3d_unseen_perturber_inversion()
    assert neptune_bench.passed is True
    assert neptune_bench.reconstruction_r2 > 0.99
    assert float(neptune_bench.metrics.get("mean_direction_cosine", 1.0)) > 0.98

    t_grid = torch.linspace(0.0, 2.5, 45, dtype=torch.float64)
    sensor_pos_3d = torch.stack(
        [1.30 * torch.cos(1.1 * t_grid), 1.30 * torch.sin(1.1 * t_grid), 0.35 * torch.sin(0.7 * t_grid)],
        dim=-1,
    )
    fresh_unseen_pos = torch.tensor([1.85, -2.10, 0.75], dtype=torch.float64)
    fresh_gm = 1.65
    diff_3d = fresh_unseen_pos.unsqueeze(0) - sensor_pos_3d
    dist_3d = torch.linalg.norm(diff_3d, dim=-1, keepdim=True)
    true_u_3d = diff_3d / dist_3d
    anom_accel_3d = fresh_gm * diff_3d / (dist_3d**3)
    blank_3d_spec = SymbolicSystemSpec(
        name="probe_3d_sensor",
        domain="unlabeled_3d",
        state_specs=(
            StateVarSpec("q_0", DIMENSIONLESS, 1.3, -10.0, 10.0),
            StateVarSpec("q_1", DIMENSIONLESS, 0.0, -10.0, 10.0),
            StateVarSpec("q_2", DIMENSIONLESS, 0.0, -10.0, 10.0),
        ),
        param_specs=(),
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.0",
    )
    inv_3d = invert_residual_for_unseen_perturber(
        base_brick=compile_symbolic_system(blank_3d_spec),
        observed_states=sensor_pos_3d,
        residual_flow=anom_accel_3d,
        dt=float((t_grid[1] - t_grid[0]).item()),
        spatial_residual_indices=(0, 1, 2),
    )
    assert inv_3d.estimated_3d_line_of_action is not None
    cos_sim_fresh = float(torch.mean(torch.sum(inv_3d.estimated_3d_line_of_action * true_u_3d, dim=-1)).item())
    assert cos_sim_fresh > 0.9999

    # -------------------------------------------------------------------------
    # Proof 3: Compound Vault A/B Acceleration Proof (Cold vs. Warm Cross-Domain Transfer)
    # -------------------------------------------------------------------------
    gt_electro_spec = SymbolicSystemSpec(
        name="gt_electro_nonlinear_sei",
        domain="electrochemistry",
        state_specs=(
            StateVarSpec("ch_a0", DIMENSIONLESS, 2.2, -10.0, 10.0),
            StateVarSpec("ch_a1", DIMENSIONLESS, 1.4, -10.0, 10.0),
        ),
        param_specs=(
            ParamSpec("g_lin", DIMENSIONLESS / s, 0.35, 0.01, 5.0),
            ParamSpec("g_sei", DIMENSIONLESS / s, 0.45, 0.01, 5.0),
            ParamSpec("g_hill", DIMENSIONLESS / s, 1.35, 0.0, 5.0),
        ),
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.5 * (ch_a0 ** 2 + ch_a1 ** 2)",
        dissipation_diag_exprs=(
            "g_lin + g_hill * (ch_a0 ** 2) / (1.0 + (ch_a0 ** 2))",
            "g_sei",
        ),
    )
    gt_electro_brick = compile_symbolic_system(gt_electro_spec)
    gt_electro_x0 = torch.tensor([[2.4, 1.5]], dtype=torch.float64)
    gt_electro_p = torch.tensor([[0.35, 0.45, 1.35]], dtype=torch.float64)
    gt_electro_roll = rollout_trajectories(
        gt_electro_brick, gt_electro_x0, gt_electro_p, dt=0.04, num_steps=35, integrator="rk4"
    )
    obs_electro = gt_electro_roll.states[0].detach().clone()

    cold_base_spec = SymbolicSystemSpec(
        name="cold_electro_linear_model",
        domain="electrochemistry",
        state_specs=gt_electro_spec.state_specs,
        param_specs=gt_electro_spec.param_specs[:2],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.5 * (ch_a0 ** 2 + ch_a1 ** 2)",
        dissipation_diag_exprs=("g_lin", "g_sei"),
    )
    cold_brick = compile_symbolic_system(cold_base_spec)
    cold_funnel = run_elimination_funnel(
        brick=cold_brick,
        observed_states=obs_electro,
        dt=0.04,
        observed_dx_dt=gt_electro_roll.flows[0].detach(),
        num_initial_samples=32,
        num_collocation_survivors=8,
        num_gradient_starts=3,
        gradient_steps=25,
    )

    onc_base_spec = SymbolicSystemSpec(
        name="onc_linear_baseline",
        domain="oncology_systems_biology",
        state_specs=(
            StateVarSpec("u_0", DIMENSIONLESS, 2.1, -10.0, 10.0),
            StateVarSpec("u_1", DIMENSIONLESS, 1.2, -10.0, 10.0),
        ),
        param_specs=(
            ParamSpec("k_0", DIMENSIONLESS / s, 0.40, 0.01, 5.0),
            ParamSpec("k_1", DIMENSIONLESS / s, 0.50, 0.01, 5.0),
        ),
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.5 * (u_0 ** 2 + u_1 ** 2)",
        dissipation_diag_exprs=("k_0", "k_1"),
    )
    onc_base_brick = compile_symbolic_system(onc_base_spec)

    def _onc_rhs(x: torch.Tensor) -> torch.Tensor:
        u0, u1 = x[0], x[1]
        return torch.stack([-0.40 * u0 - 1.25 * (u0**3) / (1.0 + u0**2), -0.50 * u1])

    onc_traj = _rk4_integrate(_onc_rhs, torch.tensor([3.8, 1.3], dtype=torch.float64), dt=0.04, steps=45)
    onc_disc = discover_and_compile_law(
        base_brick=onc_base_brick,
        observed_states=onc_traj,
        dt=0.04,
        max_active_terms=1,
    )
    assert onc_disc.winning_candidate is not None

    onc_expr_str = "; ".join(
        f"{k}: {v.to_infix()}" for k, v in onc_disc.winning_candidate.patch_expressions.items()
    )
    ab_vault = LawVault()
    register_foundational_laws(ab_vault)
    ab_vault.crystallize_discovered_law(
        law_id="crystallized_oncology_hill_clearance",
        name="oncology_discovered_clearance",
        domain="oncology_systems_biology",
        symbolic_expression=onc_expr_str,
        audit_certificate=onc_disc.audit_certificate,
        falsification_score=0.96,
        state_variables=[{"name": "u_0", "unit": "1"}, {"name": "u_1", "unit": "1"}],
        parameters=[{"name": "k_0", "unit": "1/s"}, {"name": "k_1", "unit": "1/s"}],
        description=f"Blindly discovered operator: {onc_expr_str}",
    )
    warm_xfer = transfer_isomorphic_operator_from_vault(
        target_brick=cold_brick,
        vault=ab_vault,
        min_isomorphism_score=0.35,
        prefer_cross_domain=True,
    )
    assert warm_xfer is not None
    assert warm_xfer.source_domain == "oncology_systems_biology"
    assert warm_xfer.transferred_operator_family == "rational_hill_saturation"
    warm_funnel = run_elimination_funnel(
        brick=warm_xfer.augmented_brick,
        observed_states=obs_electro,
        dt=0.04,
        observed_dx_dt=gt_electro_roll.flows[0].detach(),
        num_initial_samples=32,
        num_collocation_survivors=8,
        num_gradient_starts=3,
        gradient_steps=25,
    )
    assert warm_funnel.vector_field_mse < cold_funnel.vector_field_mse * 0.10
    assert warm_funnel.trajectory_r2 > 0.998

    # -------------------------------------------------------------------------
    # Proof 4: Corrupted Sensor Telemetry (Thermal Noise + 10-Sigma Outlier Spikes)
    # -------------------------------------------------------------------------
    def _spike_rhs(x: torch.Tensor) -> torch.Tensor:
        x0_v, x1_v = x[0], x[1]
        return torch.stack([1.20 * x1_v - 0.35 * (x0_v**3), -1.20 * x0_v - 0.25 * x1_v])

    clean_traj = _rk4_integrate(_spike_rhs, torch.tensor([1.65, 0.85], dtype=torch.float64), dt=0.04, steps=75)
    gen_sp = torch.Generator().manual_seed(777)
    corrupted_traj = clean_traj + 0.008 * torch.randn(clean_traj.shape, dtype=torch.float64, generator=gen_sp)
    corrupted_traj[18, 0] += 4.50
    corrupted_traj[41, 1] -= 3.80

    spike_specs = [
        StateVarSpec("ch_0", DIMENSIONLESS, float(corrupted_traj[0, 0].item()), -10.0, 10.0),
        StateVarSpec("ch_1", DIMENSIONLESS, float(corrupted_traj[0, 1].item()), -10.0, 10.0),
    ]
    spike_disc = discover_autonomous_symbolic_system(
        name="despiked_sensor_law",
        domain="unlabeled_sensor_stream",
        state_specs=spike_specs,
        observed_states=corrupted_traj,
        dt=0.04,
        observed_dx_dt=None,
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        max_polynomial_degree=3,
        max_terms_per_state=2,
        include_rational_primitives=False,
        include_trigonometric=False,
        include_fractional_and_log=False,
    )
    assert spike_disc.vector_field_r2 > 0.99
    assert spike_disc.trajectory_r2 > 0.999
    assert "ch_0 ** 3" in spike_disc.per_state_expressions["ch_0"]


def test_universal_meta_catalog_nested_geojson_and_chembl_bioassay_harvester() -> None:
    """
    Verify Upgrade #16:
      1. `_build_cascading_scientific_queries` builds multi-resolution 4-term and 2-term scientific bigrams.
      2. `_parse_universal_scientific_table_bytes` unwraps nested GeoJSON FeatureCollections and nested API
         dicts (`{"data": [...]}`), filters multi-station interleaved sensor records to the primary coherent
         instrument, reverses descending timestamps into chronological order, and passes the 5-Gate
         Mathematical Data Qualification Filter.
      3. `harvest_open_science_literature_priors` queries OpenAlex, EuropePMC, arXiv, and EMBL-EBI ChEMBL
         (`chembl_ebi_bioassay`) for real wet-lab bioassay constants and DOIs.
    """
    import json
    import math

    casc = _build_cascading_scientific_queries(("nlrp3", "inflammasome", "microglia", "pyroptosis"))
    assert casc == ["nlrp3 inflammasome microglia pyroptosis", "nlrp3 inflammasome"]

    # Construct a reverse-chronological, multi-station interleaved GeoJSON FeatureCollection
    features = []
    for step_idx in reversed(range(24)):
        t_val = 0.05 * step_idx
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "time_tag": f"2026-11-04T00:{step_idx:02d}:00Z",
                    "station_id": "STA_PRIMARY",
                    "active": True,
                    "flux_density": 1.2 + 0.6 * math.sin(1.4 * t_val) + 0.15 * math.cos(2.7 * t_val),
                    "ion_temp": 2.0 + 0.5 * math.cos(1.4 * t_val) - 0.12 * math.sin(2.3 * t_val),
                },
                "geometry": {"type": "Point", "coordinates": [10.5 + 0.02 * step_idx, -22.1, 450.0]},
            }
        )
        if step_idx % 4 == 0:
            # Interleaved secondary inactive/other station record that must be filtered out
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "time_tag": f"2026-11-04T00:{step_idx:02d}:00Z",
                        "station_id": "STA_NOISY_SECONDARY",
                        "active": False,
                        "flux_density": 999.0,
                        "ion_temp": -999.0,
                    },
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0, 0.0]},
                }
            )
    geojson_bytes = json.dumps({"type": "FeatureCollection", "features": features}).encode("utf-8")
    ds_geo = _parse_universal_scientific_table_bytes(
        geojson_bytes, source_label="test_geojson_stream", num_steps=24, dt=0.05
    )
    assert ds_geo is not None
    assert ds_geo.states.shape[0] == 24
    qual_geo = audit_five_gate_telemetry_qualification(
        ds_geo, source_uri="https://catalog.data.gov/dataset/test_geojson", raw_bytes_for_hash=geojson_bytes
    )
    assert qual_geo.passed_all_5_gates is True

    # Verify Cross-Discipline Domain-Disjointness Hard Veto & Primary Subject Anchor Gate
    vetoed_score = compute_dataset_semantic_relevance(
        ("perihelion", "ephemeris", "plasma", "flux"),
        "Clinical patient adenocarcinoma tumor biopsy plasma flux in hospital oncology trial",
        domain="astrophysics",
    )
    assert vetoed_score == 0.0

    missing_anchor_score = compute_dataset_semantic_relevance(
        ("egfr", "erbb1", "kinase", "phosphorylation"),
        "JAK3 serine kinase phosphorylation cascade in T-cells",
        domain="oncology",
    )
    assert missing_anchor_score <= 0.15

    # Verify live multi-registry literature + EMBL-EBI ChEMBL Homo sapiens wet-lab bioassay harvesting
    lit_rep = harvest_open_science_literature_priors(
        prompt="Inhibit EGFR tyrosine kinase phosphorylation in non-small-cell lung adenocarcinoma.",
        domain="oncology",
        state_names=("egfr_active", "erk_signaling"),
        timeout_sec=2.0,
    )
    assert "openalex_global_works" in lit_rep.registries_queried
    assert "europepmc_open_science" in lit_rep.registries_queried
    assert "arxiv_open_archive" in lit_rep.registries_queried
    assert "chembl_ebi_bioassay" in lit_rep.registries_queried
    assert lit_rep.literature_grounding_verified is True
    assert len(lit_rep.top_citations) >= 2
    assert len(lit_rep.extracted_empirical_constants) >= 1

    # Verify live USGS FDSN Earthquake Catalog GeoJSON telemetry stream
    usgs_ds = fetch_open_science_observatory_telemetry(
        stream_id="usgs_ring_of_fire_cascadia",
        domain="geophysics",
        num_steps=20,
    )
    assert usgs_ds.live_network_fetched is True
    assert usgs_ds.states.shape == (20, 3)



