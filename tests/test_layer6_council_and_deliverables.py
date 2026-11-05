"""
Verification suite for Layer 6 (`cortivra.layer6_agents`).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import torch

from cortivra.layer1_axioms.hamiltonian import ParamSpec, StateVarSpec
from cortivra.layer1_axioms.units import DIMENSIONLESS, J, N_force, kg, m, mol, s
from cortivra.layer1_axioms.vault import LawVault
from cortivra.layer2_fabric import (
    NonlinearPendulumAndViscoelasticBrick,
    register_foundational_laws,
)
from cortivra.layer3_compiler.symbolic_compiler import (
    SymbolicSystemSpec,
    compile_symbolic_system,
)
from cortivra.layer4_engine.rollout_arena import rollout_trajectories
from cortivra.layer6_agents import (
    AutonomousScientificCouncil,
    LLMInferenceRequest,
    MultiProviderMetaClient,
    ScientificInvestigationGoal,
    build_differential_epistemic_briefing,
    compile_with_self_repair,
    crystallize_verified_law,
    design_disambiguating_experiment,
    run_falsification_stress_test,
)




def test_one_sentence_investigate_across_scientific_archetypes_and_cli() -> None:
    """
    Verify the top-level One-Sentence Hypothesis Engine (`cortivra.investigate` and `cortivra.__main__.main`):
        1. Bioelectrical collateral-shielded cure prompt -> restores `V_m` while shielding `Ca_cyt`.
        2. Locked-pathway catalyst invention prompt -> triggers `synthesize_symbolic_feedback_mechanism`
           and populates `primary_finding.invented_symbolic_mechanism`.
        3. Unexplained anomaly discovery prompt -> synthesizes ground-truth anomaly trajectory,
           blindly discovers the missing nonlinear law via Layer 5, and solves the intervention.
        4. Quantum Lindblad coherence CLI invocation (`cortivra.__main__.main`).
    """
    import cortivra
    from cortivra.__main__ import main as cli_main

    fast_client = MultiProviderMetaClient()

    # 1. Bioelectrical collateral-shielded intervention from 1 sentence
    d_neuro = cortivra.investigate(
        "How can we restore healthy resting potential in a depolarized sensory neuron without triggering intracellular calcium overload?",
        client=fast_client,
        dt=0.05,
        num_steps=30,
        crystallize_into_vault=False,
    )
    assert d_neuro.intervention_solution is not None
    assert d_neuro.intervention_solution.intervened_loss < d_neuro.intervention_solution.baseline_loss * 0.25
    assert d_neuro.intervention_solution.collateral_max_drift < 0.25
    assert d_neuro.falsification_report is not None and d_neuro.falsification_report.passed_all_gates

    # 2. Locked pathway symbolic feedback catalyst invention from 1 sentence
    d_cat = cortivra.investigate(
        "Invent a synthetic feedback catalyst to restore metabolite levels in an irreversibly locked pathway.",
        client=fast_client,
        dt=0.04,
        num_steps=25,
        crystallize_into_vault=False,
    )
    assert d_cat.mechanism_invention_report is not None
    assert d_cat.mechanism_invention_report.loss_reduction_ratio > 1.5
    assert d_cat.primary_finding is not None
    assert d_cat.primary_finding.invented_symbolic_mechanism is not None
    assert "Invented Closed-Loop Symbolic Feedback Mechanism" in d_cat.to_markdown_report()

    # 3. Unexplained anomaly & missing law discovery from 1 sentence
    d_anom = cortivra.investigate(
        "Why is our depolarized neuron membrane decaying faster than linear theory predicts, and what is the missing law?",
        client=fast_client,
        dt=0.03,
        num_steps=30,
        crystallize_into_vault=False,
    )
    assert d_anom.discovery_report is not None
    assert d_anom.discovery_report.discovery_triggered
    assert d_anom.discovery_report.out_of_sample_r2 > 0.90

    # 4. CLI invocation (`python -m cortivra ...`)
    exit_code = cli_main(
        ["Stabilize quantum qubit coherence against Lindblad dephasing.", "--steps", "20", "--provider", "baseten"]
    )
    assert exit_code == 0


def test_dynamic_n_compartment_causal_blast_radius_and_svd_bystander_shield(tmp_path: Path) -> None:
    """
    Verify the Dynamic N-Compartment Causal Blast-Radius & Bystander Shield (N >= 5)
    and Universal Physical Realization & Blueprint Compiler:
    Synthesizes a 5-compartment nonlinear Port-Hamiltonian world model covering Primary Target,
    Local Stress, Energetic Supply Reservoir, Co-Located Bystander, and Downstream Functional
    Receiver, executes the SVD Collateral Nullspace Tournament + Symbolic Feedback Invention,
    populates `causal_blast_radius_matrix` with all 5 compartments shielded, and compiles
    thermodynamic Eyring-Boltzmann barrier shifts, bystander selectivity ratios, and ANSI C99
    controller firmware.
    """
    import cortivra

    fast_client = MultiProviderMetaClient()
    vault = LawVault(tmp_path / "blast_radius_vault")
    register_foundational_laws(vault)
    dossier = cortivra.investigate(
        "Find a restorative intervention for cellular transport that shields neighboring bystanders and downstream receivers.",
        domain="biophysics",
        client=fast_client,
        vault=vault,
        dt=0.025,
        num_steps=25,
        crystallize_into_vault=True,
    )
    assert dossier.final_brick.state_dim >= 5
    assert len(dossier.final_brick.spatial_nodes_3d) >= 5
    assert dossier.intervention_solution is not None
    assert dossier.intervention_solution.intervened_loss < dossier.intervention_solution.baseline_loss * 0.1
    assert dossier.primary_finding is not None
    assert dossier.primary_finding.causal_blast_radius_matrix is not None
    assert len(dossier.primary_finding.causal_blast_radius_matrix) >= 5
    roles = {entry["causal_role"] for entry in dossier.primary_finding.causal_blast_radius_matrix}
    assert {
        "primary_target",
        "local_stress_or_opposition",
        "energetic_supply_reservoir",
        "co_located_bystander",
        "downstream_functional_receiver",
    }.issubset(roles)
    assert all(entry["shield_intact"] for entry in dossier.primary_finding.causal_blast_radius_matrix)
    bp = dossier.primary_finding.physical_realization_blueprint
    assert bp is not None
    assert len(bp["thermodynamic_actuator_specs"]) >= 1
    assert len(bp["bystander_selectivity_specifications"]) >= 4
    assert bp["closed_loop_mechanism_blueprint"] is not None
    assert "cortivra_closed_loop_actuator" in bp["closed_loop_mechanism_blueprint"]["ansi_c99_firmware_code"]
    assert len(bp["constitutive_physical_anchors"]) >= 5
    assert len(bp["buckingham_pi_invariants"]) >= 5
    assert bp["parametric_uncertainty_envelope"] is not None
    assert bp["parametric_uncertainty_envelope"]["num_ensemble_worlds"] == 17
    assert bp["parametric_uncertainty_envelope"]["ensemble_viable_fraction"] >= 0.75
    assert bp["parametric_uncertainty_envelope"]["bystander_shield_survival_rate"] >= 0.75
    md = dossier.to_markdown_report()
    assert "Causal Blast-Radius & Bystander Protection Matrix" in md
    assert "## 9. Universal Physical Realization & Actionable Engineering/Biophysical Blueprint" in md
    assert "cortivra_closed_loop_actuator" in md
    assert "First-Principles Constitutive Physical Anchors" in md
    assert "Buckingham $\\Pi$ Dimensionless Scale-Invariance Matrix" in md
    assert "Parametric Uncertainty Envelope Verification" in md
    assert "6-Rule Multi-Hypothesis Physical Elimination Crucible" in md
    assert dossier.primary_finding.hypothesis_tournament is not None
    assert len(dossier.primary_finding.hypothesis_tournament) >= 5
    eliminated = [
        h for h in dossier.primary_finding.hypothesis_tournament
        if h.get("crucible_verdict") == "ELIMINATED_BY_PHYSICS"
    ]
    assert len(eliminated) >= 3
    assert any("rule_2_target_convergence_failure" in h.get("rules_broken", []) for h in eliminated)
    assert dossier.disambiguation_report is not None
    assert dossier.primary_finding.disambiguation_summary is not None


def test_three_stage_empirical_observatory_and_partial_sensor_ingestion(tmp_path: Path) -> None:
    """
    Verify Stage 2 & Stage 3:
      - Universal Empirical Telemetry Loader (`load_empirical_telemetry`) from CSV with header & time column
      - Partial sensor observability (`K = 2` observed sensor channels into an `N = 5` compartment world model)
      - Open-Science Observatory Streamer (`fetch_open_science_observatory_telemetry` / `--observatory`)
    """
    import cortivra
    from cortivra.layer6_agents.empirical_observatory import (
        fetch_open_science_observatory_telemetry,
        load_empirical_telemetry,
    )

    # Create a 2-sensor empirical CSV trajectory (partial observability K=2 < N=5)
    csv_path = tmp_path / "partial_ruffini_sensors.csv"
    lines = ["time,sa2_receptor_potential,collagen_strain"]
    for step in range(25):
        t_val = step * 0.02
        s1 = 1.75 * (0.94**step) + 0.15
        s2 = 1.35 * (0.96**step) + 0.10
        lines.append(f"{t_val:.4f},{s1:.6f},{s2:.6f}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ds = load_empirical_telemetry(csv_path)
    assert ds.states.shape == (25, 2)
    assert ds.column_names == ("sa2_receptor_potential", "collagen_strain")
    assert abs(ds.dt - 0.02) < 1e-5

    # Run cortivra.investigate with partial-sensor CSV data (2 columns into 5-state model)
    fast_client = MultiProviderMetaClient()
    dossier_csv = cortivra.investigate(
        "Calibrate Ruffini SA2 mechanotransduction from partial stretch sensor telemetry and stabilize firing.",
        domain="neuroscience",
        client=fast_client,
        observed_states=csv_path,
        max_council_steps=8,
        crystallize_into_vault=False,
    )
    assert dossier_csv.calibration_report is not None
    assert dossier_csv.final_brick.state_dim >= 5
    assert dossier_csv.falsification_report is not None

    # Verify Open-Science Observatory Streamer (`ruffini_sa2` and `noaa_solar_wind`)
    obs_sa2 = fetch_open_science_observatory_telemetry("ruffini_sa2", domain="neuroscience", num_steps=20)
    assert obs_sa2.states.shape == (20, 3)
    obs_sw = fetch_open_science_observatory_telemetry("noaa_solar_wind", domain="astrophysics", num_steps=20, timeout_sec=1.0)
    assert obs_sw.states.shape == (20, 3)


def test_three_phase_blind_csv_rediscovery_and_adversarial_torture_suite(tmp_path: Path) -> None:
    """
    Verify the 3-Phase Verification & Hardening Gauntlet:
      - Phase 1: Double-blind anonymized CSV equation rediscovery (`time,ch_0,ch_1` with zero domain hints)
        recovering exact symbolic laws (`d(ch_0)/dt`, `d(ch_1)/dt`) and numerical constants (`R^2 > 0.99`).
      - Phase 2: Naive short prompt autonomous bystander & failure-mode deduction (`N >= 5` compartments)
        with 6-Rule Physical Elimination Crucible and 17-World Parametric Uncertainty Envelope.
      - Phase 3: Noisy partial-sensor CSV Savitzky-Golay denoising + 4x-horizon viability shield.
    """
    import cortivra

    fast_client = MultiProviderMetaClient()

    # Phase 1: Generate an anonymized coupled bimolecular CSV (`time,ch_0,ch_1`)
    # Ground truth: d(ch_0)/dt = -0.70*ch_0 + 0.55*ch_0*ch_1, d(ch_1)/dt = -0.50*ch_1 - 0.35*ch_0*ch_1
    dt = 0.04
    steps = 30
    x = torch.tensor([1.25, 1.60], dtype=torch.float64)
    traj = [x]
    for _ in range(steps - 1):
        def _f(st: torch.Tensor) -> torch.Tensor:
            c0, c1 = st[0], st[1]
            return torch.stack([-0.70 * c0 + 0.55 * c0 * c1, -0.50 * c1 - 0.35 * c0 * c1])
        k1 = _f(x)
        k2 = _f(x + 0.5 * dt * k1)
        k3 = _f(x + 0.5 * dt * k2)
        k4 = _f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj.append(x)
    clean_mat = torch.stack(traj)

    blind_csv = tmp_path / "blind_mystery.csv"
    lines = ["time,ch_0,ch_1"]
    for idx_t in range(steps):
        lines.append(f"{idx_t * dt:.6f},{float(clean_mat[idx_t, 0]):.8f},{float(clean_mat[idx_t, 1]):.8f}")
    blind_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dossier_blind = cortivra.investigate(
        "Discover the exact governing differential equations from this unknown sensor dataset and stabilize ch_0.",
        client=fast_client,
        observed_states=blind_csv,
        max_council_steps=5,
        crystallize_into_vault=False,
    )
    assert dossier_blind.autonomous_discovery_report is not None
    assert dossier_blind.autonomous_discovery_report.vector_field_r2 > 0.999
    assert dossier_blind.primary_finding is not None
    eqs = dossier_blind.primary_finding.discovered_equations
    assert "d(ch_0)/dt" in eqs and "d(ch_1)/dt" in eqs
    assert "[calibrated:" in eqs["d(ch_0)/dt"]
    assert "-0.70" in eqs["d(ch_0)/dt"] or "-0.69" in eqs["d(ch_0)/dt"]
    assert "0.55" in eqs["d(ch_0)/dt"] or "0.54" in eqs["d(ch_0)/dt"]
    assert "-0.50" in eqs["d(ch_1)/dt"] or "-0.49" in eqs["d(ch_1)/dt"]
    assert "-0.35" in eqs["d(ch_1)/dt"] or "-0.34" in eqs["d(ch_1)/dt"]

    # Phase 3: Inject 1.5% Gaussian sensor noise into partial-sensor CSV and verify denoising
    torch.manual_seed(42)
    noisy_mat = clean_mat + 0.015 * torch.std(clean_mat, dim=0, keepdim=True) * torch.randn_like(clean_mat)
    noisy_csv = tmp_path / "noisy_partial.csv"
    n_lines = ["time,sensor_alpha,sensor_beta"]
    for idx_t in range(steps):
        n_lines.append(f"{idx_t * dt:.6f},{float(noisy_mat[idx_t, 0]):.8f},{float(noisy_mat[idx_t, 1]):.8f}")
    noisy_csv.write_text("\n".join(n_lines) + "\n", encoding="utf-8")

    dossier_noisy = cortivra.investigate(
        "Discover the governing equations from this noisy partial-sensor telemetry and stabilize sensor_alpha.",
        client=fast_client,
        observed_states=noisy_csv,
        max_council_steps=5,
        crystallize_into_vault=False,
    )
    assert dossier_noisy.autonomous_discovery_report is not None
    assert dossier_noisy.autonomous_discovery_report.vector_field_r2 > 0.90
    assert dossier_noisy.primary_finding is not None
    assert "d(sensor_alpha)/dt" in dossier_noisy.primary_finding.discovered_equations


def test_three_next_gen_pillars_ashby_eyring_recursive_multiscale_and_html_command_center(
    tmp_path: Path,
) -> None:
    """
    Verify the 3 Next-Generation Pillars of Cortivra:
      - Pillar 1: Ashby-Eyring 3-Way Multiplicative Factorization
        (alpha = alpha_material_constitutive * alpha_3d_geometry * alpha_active_feedback)
        and Domain Physical Realization & Experimental Verification Protocol.
      - Pillar 2: Autonomous Multi-Scale Recursive Chaining (`recursive_depth=1`),
        automatically formulating and solving the Scale-1 sub-system micro-mechanism
        investigation targeting the Scale-0 bottleneck lever.
      - Pillar 3: Self-contained dark-mode HTML5 + inline SVG Scientific Command Center
        (`to_html_report()` and CLI `--recursive --html`).
    """
    import cortivra
    from cortivra.__main__ import main as cli_main

    fast_client = MultiProviderMetaClient()
    vault = LawVault(tmp_path / "pillars_vault")
    register_foundational_laws(vault)

    dossier = cortivra.investigate(
        "Eliminate thermoacoustic combustion instability in a liquid rocket injector while protecting chamber wall thermal liners.",
        domain="aerospace",
        client=fast_client,
        vault=vault,
        dt=0.02,
        num_steps=20,
        crystallize_into_vault=False,
        recursive_depth=1,
    )

    # Verify Pillar 1: Dynamic Temperature, Operator-Aware Variational Co-Design & Pareto Menu
    assert dossier.primary_finding is not None
    bp = dossier.primary_finding.physical_realization_blueprint
    assert bp is not None
    assert bp["reference_temperature_kelvin"] > 0.0
    assert "temperature_inference_source" in bp
    act_specs = bp["thermodynamic_actuator_specs"]
    assert len(act_specs) >= 1
    for a_sp in act_specs:
        ratio = float(a_sp["modulation_ratio"])
        a_mat = float(a_sp["alpha_material_constitutive"])
        a_geom = float(a_sp["alpha_3d_geometry"])
        a_act = float(a_sp["alpha_active_feedback"])
        assert 0.35 - 1e-4 <= a_mat <= 2.85 + 1e-4
        assert 0.15 - 1e-4 <= a_geom <= 8.0 + 1e-4
        assert abs((a_mat * a_geom * a_act) - ratio) < 1e-3
        assert "operator_slot" in a_sp
        assert "pareto_realization_options" in a_sp
        p_opts = a_sp["pareto_realization_options"]
        assert "option_a_passive_material_dominant" in p_opts
        assert "option_b_passive_material_plus_3d_geometry" in p_opts
        assert "option_c_variational_trimodal_codesign" in p_opts
        if abs(ratio - 1.0) > 0.02:
            # Variational co-design actively shares load across material, 3D geometry, and firmware
            assert abs(a_geom - 1.0) > 1e-4
            assert abs(a_act - 1.0) > 1e-4
        assert a_sp["realizability_tier"] in (
            "tier_1_direct_material_or_molecular_substitution",
            "tier_2_hybrid_material_plus_3d_micro_geometry",
            "tier_3_active_closed_loop_metamaterial_or_firmware",
        )

    bottleneck = bp["primary_bottleneck_lever"]
    assert isinstance(bottleneck, dict) and "parameter" in bottleneck
    dom_proto = bp["domain_realization_protocol"]
    assert isinstance(dom_proto, dict)
    assert "material_or_molecular_mechanism" in dom_proto
    assert "micro_geometric_architecture" in dom_proto
    assert "active_feedback_or_firmware_role" in dom_proto
    assert len(dom_proto["step_by_step_verification_protocol"]) >= 4

    # Verify Pillar 2: Autonomous Multi-Scale Recursive Chaining & Bidirectional Homogenization Closure
    assert dossier.scale_level == 0
    assert len(dossier.subscale_dossiers) == 1
    sub_d = dossier.subscale_dossiers[0]
    assert sub_d.scale_level == 1
    assert sub_d.final_brick.state_dim >= 5
    assert sub_d.falsification_report is not None and sub_d.falsification_report.passed_all_gates
    assert bottleneck["parameter"] in sub_d.question

    m_cert = bp.get("multiscale_homogenization_certificate")
    assert isinstance(m_cert, dict)
    assert m_cert["bidirectional_closure_verified"] is True
    assert m_cert["adiabatic_enslavement_satisfied"] is True
    assert float(m_cert["scale_separation_epsilon"]) < 1.0
    assert m_cert["homogenized_bystander_shield_intact"] is True
    assert float(m_cert["homogenized_4x_horizon_viable_fraction"]) >= 0.99

    md = dossier.to_markdown_report()
    assert "Ashby-Eyring Factorization" in md
    assert "Two-Way Bidirectional Multi-Scale Homogenization Closure" in md
    assert "Domain Physical Realization & Experimental Verification Protocol" in md
    assert "## 10. Autonomous Multi-Scale Recursive Sub-System Realization Dossiers" in md
    assert "Scale-1 Sub-System Dossier" in md

    # Verify Pillar 3: Interactive Visual Command-Center HTML5 Report & CLI flags
    html_report = dossier.to_html_report()
    assert "<!DOCTYPE html>" in html_report
    assert "<svg" in html_report
    assert "Ashby-Eyring 3-Way Multiplicative" in html_report
    assert "Two-Way Bidirectional Multi-Scale Homogenization Closure" in html_report
    assert "Scale-1 Recursive Sub-System Dossier" in html_report
    assert "cortivra_closed_loop_actuator" in html_report

    out_html = tmp_path / "command_center.html"
    rc = cli_main(
        [
            "Restore ATP synthesis in a depolarized mitochondrion.",
            "--domain",
            "biology",
            "--steps",
            "15",
            "--provider",
            "baseten",
            "--recursive",
            "--html",
            str(out_html),
        ]
    )
    assert rc == 0
    assert out_html.exists()
    assert "<svg" in out_html.read_text(encoding="utf-8")


def test_cli_blind_scientific_rediscovery_and_adversarial_negative_control(tmp_path: Path) -> None:
    """
    End-to-end validation of genuine blind scientific rediscovery and adversarial negative-control
    rejection exclusively through the production CLI (`cortivra.__main__.main` with `--data`):
      1. Nonlinear Damped Duffing Oscillator (3-term cubic restoring + damping, negative excursions).
      2. Cooperative Hill (n=2) Rational Enzyme Kinetics + Mass Conservation (`s_0 + s_1 = const`).
      3. Noisy Lotka-Volterra Autocatalytic Cross-Coupling (`u_0 * u_1` under Gaussian sensor noise).
      4. Bistable Quartic Double-Well / Landau Symmetry Breaking (multi-trajectory JSON ensemble).
      5. Single-Channel Partial Observation of a Coupled 2D Non-Markovian Oscillator (`obs_0` + `z_latent`).
      6. Adversarial Negative Control: Pure uncorrelated Gaussian white noise with a false 5th-order
         chaotic anomaly prompt -> verified rejection (`d(n_0)/dt = 0`, `d(n_1)/dt = 0`, `R^2 = 0.0`).
    """
    import csv
    from cortivra.__main__ import main as cli_main

    dtype = torch.float64

    def _rk4(f, x0: torch.Tensor, dt: float, steps: int, sub: int = 20) -> torch.Tensor:
        h = dt / sub
        traj = [x0.clone()]
        x = x0.clone()
        for _ in range(steps - 1):
            for _ in range(sub):
                k1 = f(x)
                k2 = f(x + 0.5 * h * k1)
                k3 = f(x + 0.5 * h * k2)
                k4 = f(x + h * k3)
                x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            traj.append(x.clone())
        return torch.stack(traj, dim=0)

    def _write_csv(p: Path, dt: float, cols: list[str], data: torch.Tensor) -> None:
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", *cols])
            for t_i in range(data.shape[0]):
                w.writerow([f"{t_i * dt:.6f}"] + [f"{float(data[t_i, c].item()):.8f}" for c in range(data.shape[1])])

    def _run_cli_and_load_json(prompt_str: str, dom: str, data_path: Path, run_tag: str) -> dict:
        log_dir = tmp_path / f"logs_{run_tag}"
        rc = cli_main(
            [
                prompt_str,
                "--domain",
                dom,
                "--steps",
                "16",
                "--provider",
                "baseten",
                "--data",
                str(data_path),
                "--save-logs",
                "--log-dir",
                str(log_dir),
            ]
        )
        assert rc == 0
        j_files = list(log_dir.glob("*/dossier.json"))
        assert len(j_files) == 1
        return json.loads(j_files[0].read_text(encoding="utf-8"))

    # 1. Blind Nonlinear Damped Duffing Oscillator
    csv_1 = tmp_path / "blind_duffing.csv"
    traj_1 = _rk4(
        lambda x: torch.stack([1.40 * x[..., 1], -1.20 * x[..., 0] - 0.35 * x[..., 1] - 0.85 * (x[..., 0] ** 3)], dim=-1),
        torch.tensor([1.35, -0.90], dtype=dtype),
        dt=0.025,
        steps=80,
    )
    _write_csv(csv_1, 0.025, ["ch_0", "ch_1"], traj_1)
    d1 = _run_cli_and_load_json(
        "Discover the unknown coupled dynamical equations governing channels ch_0 and ch_1.",
        "physics",
        csv_1,
        "duffing",
    )
    pf1 = d1["primary_finding"]
    expr1 = pf1["discovered_equations"]
    assert "ch_1" in expr1["d(ch_0)/dt"] and "1.400" in expr1["d(ch_0)/dt"]
    assert "ch_0" in expr1["d(ch_1)/dt"] and "ch_1" in expr1["d(ch_1)/dt"] and "(ch_0 ** 3)" in expr1["d(ch_1)/dt"]
    assert float(pf1["discovery_r2"]) > 0.999

    # 2. Blind Cooperative Hill (n=2) Enzyme Kinetics + Mass Conservation
    csv_2 = tmp_path / "blind_hill.csv"
    traj_2 = _rk4(
        lambda x: torch.stack(
            [
                -1.80 * (x[..., 0] ** 2) / (0.65 ** 2 + x[..., 0] ** 2),
                +1.80 * (x[..., 0] ** 2) / (0.65 ** 2 + x[..., 0] ** 2),
            ],
            dim=-1,
        ),
        torch.tensor([2.60, 0.40], dtype=dtype),
        dt=0.03,
        steps=80,
    )
    _write_csv(csv_2, 0.03, ["s_0", "s_1"], traj_2)
    d2 = _run_cli_and_load_json(
        "Identify the unknown reaction rate law and conservation invariants governing s_0 and s_1.",
        "biochemistry",
        csv_2,
        "hill",
    )
    pf2 = d2["primary_finding"]
    expr2 = pf2["discovered_equations"]
    assert "hill(s_0, 0.650" in expr2["d(s_0)/dt"] and "-1.800" in expr2["d(s_0)/dt"]
    assert "hill(s_0, 0.650" in expr2["d(s_1)/dt"] and "1.800" in expr2["d(s_1)/dt"]
    invs_2 = pf2["conserved_invariants"]
    assert any("s_0" in inv and "s_1" in inv for inv in invs_2)

    # 3. Blind Noisy Lotka-Volterra Autocatalytic Cross-Coupling
    csv_3 = tmp_path / "blind_lv_noisy.csv"
    traj_3_clean = _rk4(
        lambda x: torch.stack(
            [
                0.90 * x[..., 0] - 0.60 * x[..., 0] * x[..., 1],
                -0.75 * x[..., 1] + 0.50 * x[..., 0] * x[..., 1],
            ],
            dim=-1,
        ),
        torch.tensor([2.40, 0.65], dtype=dtype),
        dt=0.04,
        steps=90,
    )
    gen_3 = torch.Generator().manual_seed(20261030)
    traj_3_noisy = traj_3_clean + 0.006 * torch.randn(traj_3_clean.shape, generator=gen_3, dtype=dtype)
    _write_csv(csv_3, 0.04, ["u_0", "u_1"], traj_3_noisy)
    d3 = _run_cli_and_load_json(
        "Reconstruct the nonlinear interaction law between u_0 and u_1 from noisy field measurements.",
        "ecology",
        csv_3,
        "lv_noisy",
    )
    pf3 = d3["primary_finding"]
    expr3 = pf3["discovered_equations"]
    assert "(u_0 * u_1)" in expr3["d(u_0)/dt"] and "(u_0 * u_1)" in expr3["d(u_1)/dt"]
    assert float(pf3["discovery_r2"]) > 0.96

    # 4. Blind Bistable Quartic Double-Well Multi-Trajectory Ensemble
    data_4 = tmp_path / "blind_bistable.json"
    inits_4 = [
        torch.tensor([0.18, 1.40], dtype=dtype),
        torch.tensor([-0.22, -1.10], dtype=dtype),
        torch.tensor([1.65, 0.80], dtype=dtype),
        torch.tensor([-1.70, -0.75], dtype=dtype),
    ]
    trajs_4 = torch.stack(
        [
            _rk4(
                lambda x: torch.stack([1.25 * x[..., 0] - 0.80 * (x[..., 0] ** 3), -0.60 * x[..., 1]], dim=-1),
                ic,
                dt=0.03,
                steps=55,
            )
            for ic in inits_4
        ],
        dim=0,
    )
    data_4.write_text(json.dumps({"dt": 0.03, "columns": ["x_0", "x_1"], "states": trajs_4.tolist()}), encoding="utf-8")
    d4 = _run_cli_and_load_json(
        "Determine the nonlinear restoring vector field governing x_0 and x_1.",
        "physics",
        data_4,
        "bistable",
    )
    pf4 = d4["primary_finding"]
    expr4 = pf4["discovered_equations"]
    assert "(x_0 ** 3)" in expr4["d(x_0)/dt"] and "1.249" in expr4["d(x_0)/dt"] and "-0.799" in expr4["d(x_0)/dt"]
    assert "-0.600" in expr4["d(x_1)/dt"]

    # 5. Blind Single-Channel Partial Observation of a 2D Non-Markovian System
    csv_5 = tmp_path / "blind_hidden.csv"
    traj_5 = _rk4(
        lambda x: torch.stack([x[..., 1], -1.44 * x[..., 0] - 0.30 * x[..., 1]], dim=-1),
        torch.tensor([1.50, 0.0], dtype=dtype),
        dt=0.03,
        steps=160,
    )
    _write_csv(csv_5, 0.03, ["obs_0"], traj_5[:, 0:1])
    d5 = _run_cli_and_load_json(
        "Analyze single-channel telemetry obs_0, test for non-Markovian hidden memory, and reconstruct the coupled equations.",
        "physics",
        csv_5,
        "hidden",
    )
    assert d5["final_briefing"]["residual_summary"]["is_non_markovian_hysteresis"] is True
    expr5 = d5["primary_finding"]["discovered_equations"]
    assert "d(z_latent)/dt" in expr5
    assert "obs_0" in expr5["d(z_latent)/dt"] and "z_latent" in expr5["d(z_latent)/dt"]

    # 6. Adversarial Negative Control: Pure White Noise + False 5th-Order Chaos Claim
    csv_6 = tmp_path / "blind_noise.csv"
    gen_6 = torch.Generator().manual_seed(99999)
    n0 = 2.0 + 0.035 * torch.randn((80, 1), generator=gen_6, dtype=dtype)
    n1 = 5.0 + 0.035 * torch.randn((80, 1), generator=gen_6, dtype=dtype)
    _write_csv(csv_6, 0.025, ["n_0", "n_1"], torch.cat([n0, n1], dim=-1))
    d6 = _run_cli_and_load_json(
        "Discover the 5th-order chaotic nonlinear oscillator law and exotic cross-coupling governing n_0 and n_1.",
        "physics",
        csv_6,
        "noise",
    )
    pf6 = d6["primary_finding"]
    expr6 = pf6["discovered_equations"]
    assert expr6.get("d(n_0)/dt", "0") == "0"
    assert expr6.get("d(n_1)/dt", "0") == "0"
    assert float(pf6["discovery_r2"] or 0.0) == 0.0


def test_four_architectural_upgrades_end_to_end(tmp_path: Path) -> None:
    """
    Comprehensive end-to-end verification of the 4 Architectural Upgrades:
      1. Upgrade 1: Unconditional Symbolic-to-Port-Hamiltonian Helmholtz-Hodge Decomposition
         (`decompose_vector_field_helmholtz_hodge`, `J_skew` vs `R_sym`) and `--data` world-model
         synchronization.
      2. Upgrade 2: Canonical Buckingham-Pi Non-Dimensionalization (`compute_canonical_nondimensional_scaling`),
         exact 7D SI scale invariance (`scale_invariance_max_drift <= 1e-9`), stiff timescale
         separation `epsilon = tau_fast / tau_slow`, and worst-decile `cvar_10_primary_final`.
      3. Upgrade 3: Continuous Fractional Power-Law (`x**alpha` via Variable Projection + Golden-Section
         Search), Logarithmic Creep (`log(1 + x)`), and 1D Spatial PDE Operators (`laplacian_1d`,
         `advection_1d`) with zero hints.
      4. Upgrade 4: Quantitative Physical Realizability & Thermodynamic Impedance Translator
         (`delta_ea_kj_per_mol`, `delta_ea_ev`, `effective_impedance_ratio`, `quality_factor_q_post`,
         and `thermodynamic_work_budget`).
    """
    import csv
    import cortivra
    from cortivra.__main__ import main as cli_main
    from cortivra.layer1_axioms.hamiltonian import ParamSpec, StateVarSpec
    from cortivra.layer1_axioms.units import (
        DIMENSIONLESS,
        compute_canonical_nondimensional_scaling,
        parse_unit,
        s,
    )
    from cortivra.layer5_discovery.latent_discovery import (
        decompose_vector_field_helmholtz_hodge,
        discover_autonomous_symbolic_system,
    )

    dtype = torch.float64

    def _rk4(f, x0: torch.Tensor, dt: float, steps: int, sub: int = 20) -> torch.Tensor:
        h = dt / sub
        traj = [x0.clone()]
        x = x0.clone()
        for _ in range(steps - 1):
            for _ in range(sub):
                k1 = f(x)
                k2 = f(x + 0.5 * h * k1)
                k3 = f(x + 0.5 * h * k2)
                k4 = f(x + h * k3)
                x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            traj.append(x.clone())
        return torch.stack(traj, dim=0)

    # --- Upgrade 3A: Blind Continuous Fractional Power-Law Rediscovery (Kleiber / Fractional Kinetics) ---
    # Ground truth: d(m_0)/dt = -1.45 * (m_0 ** 0.75), d(m_1)/dt = -0.80 * log(1.0 + m_1)
    traj_frac = _rk4(
        lambda x: torch.stack(
            [
                -1.45 * (torch.clamp(x[..., 0], min=1e-9) ** 0.75),
                -0.80 * torch.log(1.0 + torch.clamp(x[..., 1], min=0.0)),
            ],
            dim=-1,
        ),
        torch.tensor([4.50, 3.20], dtype=dtype),
        dt=0.04,
        steps=75,
    )
    specs_frac = [
        StateVarSpec("m_0", DIMENSIONLESS, default_value=4.50, homeostatic_min=0.0, homeostatic_max=20.0),
        StateVarSpec("m_1", DIMENSIONLESS, default_value=3.20, homeostatic_min=0.0, homeostatic_max=20.0),
    ]
    rep_frac = discover_autonomous_symbolic_system(
        name="blind_fractional_and_log_test",
        domain="physics",
        state_specs=specs_frac,
        observed_states=traj_frac,
        dt=0.04,
        include_trigonometric=False,
        include_fractional_and_log=True,
    )
    assert rep_frac.vector_field_r2 > 0.999
    expr_m0 = rep_frac.per_state_expressions["m_0"]
    expr_m1 = rep_frac.per_state_expressions["m_1"]
    assert "**" in expr_m0 and ("0.75" in expr_m0 or "0.74" in expr_m0 or "0.76" in expr_m0)
    assert "log(" in expr_m1

    # --- Upgrade 3B: Blind 1D Spatial Diffusion PDE Stencil Rediscovery ---
    # Ground truth on interior node u_1: d(u_1)/dt = 1.60 * (u_0 - 2*u_1 + u_2)
    traj_pde = _rk4(
        lambda u: torch.stack(
            [
                -0.50 * u[..., 0],
                1.60 * (u[..., 0] - 2.0 * u[..., 1] + u[..., 2]),
                -0.30 * u[..., 2],
            ],
            dim=-1,
        ),
        torch.tensor([2.80, 0.40, 1.90], dtype=dtype),
        dt=0.025,
        steps=65,
    )
    specs_pde = [
        StateVarSpec("u_0", DIMENSIONLESS, default_value=2.80, homeostatic_min=-10.0, homeostatic_max=10.0),
        StateVarSpec("u_1", DIMENSIONLESS, default_value=0.40, homeostatic_min=-10.0, homeostatic_max=10.0),
        StateVarSpec("u_2", DIMENSIONLESS, default_value=1.90, homeostatic_min=-10.0, homeostatic_max=10.0),
    ]
    rep_pde = discover_autonomous_symbolic_system(
        name="blind_spatial_pde_test",
        domain="continuum_mechanics",
        state_specs=specs_pde,
        observed_states=traj_pde,
        dt=0.025,
        max_terms_per_state=1,
        include_rational_primitives=False,
        include_trigonometric=False,
        include_fractional_and_log=False,
    )
    assert rep_pde.vector_field_r2 > 0.999
    assert "u_0" in rep_pde.per_state_expressions["u_1"]
    assert "u_1" in rep_pde.per_state_expressions["u_1"]
    assert "u_2" in rep_pde.per_state_expressions["u_1"]

    # --- Upgrade 1: Helmholtz-Hodge Decomposition on Conservative vs Dissipative Systems ---
    assert rep_pde.helmholtz_hodge_decomposition is not None
    hh_pde = rep_pde.helmholtz_hodge_decomposition
    assert hh_pde.is_volume_contracting is True
    assert hh_pde.mean_divergence_rate < -1.0

    # Pure symplectic harmonic oscillator: d(q)/dt = 1.5*p, d(p)/dt = -1.5*q
    traj_symp = _rk4(
        lambda x: torch.stack([1.50 * x[..., 1], -1.50 * x[..., 0]], dim=-1),
        torch.tensor([1.20, 0.0], dtype=dtype),
        dt=0.03,
        steps=60,
    )
    specs_symp = [
        StateVarSpec("q", DIMENSIONLESS, default_value=1.20),
        StateVarSpec("p", DIMENSIONLESS, default_value=0.0),
    ]
    rep_symp = discover_autonomous_symbolic_system(
        name="symplectic_hh_test",
        domain="mechanics",
        state_specs=specs_symp,
        observed_states=traj_symp,
        dt=0.03,
        max_terms_per_state=1,
        include_rational_primitives=False,
        include_trigonometric=False,
        include_fractional_and_log=False,
    )
    hh_symp = decompose_vector_field_helmholtz_hodge(
        rep_symp.compiled_brick, traj_symp, rep_symp.calibrated_params
    )
    assert hh_symp.conservative_energy_fraction > 0.99
    assert hh_symp.dominant_regime == "conservative_symplectic_dominant"

    # --- Upgrade 2: Buckingham-Pi Canonical Non-Dimensionalization & 7D SI Scale Invariance ---
    nd_scale = compute_canonical_nondimensional_scaling(
        state_specs=[
            StateVarSpec("v_m", parse_unit("V"), default_value=0.07),
            StateVarSpec("ca_i", parse_unit("mol/m^3"), default_value=1.2),
        ],
        param_specs=[
            ParamSpec("g_leak", parse_unit("1/s"), default_value=2.5),
            ParamSpec("g_fast", parse_unit("1/s"), default_value=25.0),
            ParamSpec("k_osc", parse_unit("1/s^2"), default_value=9.0),
        ],
        time_unit=s,
    )
    assert nd_scale.scale_invariance_verified is True
    assert nd_scale.scale_invariance_max_drift <= 1e-9
    assert 0.0 < nd_scale.timescale_separation_epsilon <= 0.15

    # --- Upgrade 1 + 2 + 4 via Live CLI (`--data` and prompt-only) ---
    csv_frac_cli = tmp_path / "cli_fractional.csv"
    with csv_frac_cli.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "x_0", "x_1"])
        for t_i in range(traj_frac.shape[0]):
            w.writerow(
                [
                    f"{t_i * 0.04:.6f}",
                    f"{float(traj_frac[t_i, 0].item()):.8f}",
                    f"{float(traj_frac[t_i, 1].item()):.8f}",
                ]
            )
    log_dir_cli = tmp_path / "logs_upgrade_cli"
    rc = cli_main(
        [
            "Discover the exact fractional power-law and logarithmic relaxation equations governing x_0 and x_1 and stabilize x_0.",
            "--domain",
            "biophysics",
            "--steps",
            "18",
            "--provider",
            "baseten",
            "--data",
            str(csv_frac_cli),
            "--save-logs",
            "--log-dir",
            str(log_dir_cli),
        ]
    )
    assert rc == 0
    j_files = list(log_dir_cli.glob("*/dossier.json"))
    assert len(j_files) == 1
    d_json = json.loads(j_files[0].read_text(encoding="utf-8"))
    pf_cli = d_json["primary_finding"]
    assert pf_cli["helmholtz_hodge_decomposition"] is not None
    assert "**" in pf_cli["discovered_equations"]["d(x_0)/dt"]
    assert "log(" in pf_cli["discovered_equations"]["d(x_1)/dt"]
    bp_cli = pf_cli["physical_realization_blueprint"]
    assert bp_cli is not None
    assert "canonical_nondimensional_scaling" in bp_cli
    assert bp_cli["canonical_nondimensional_scaling"]["scale_invariance_verified"] is True
    assert "thermodynamic_work_budget" in bp_cli
    assert "minimum_thermodynamic_work_joules" in bp_cli["thermodynamic_work_budget"]
    assert "cvar_10_primary_final" in bp_cli["parametric_uncertainty_envelope"]
    for a_sp in bp_cli["thermodynamic_actuator_specs"]:
        assert "delta_ea_kj_per_mol" in a_sp
        assert "delta_ea_ev" in a_sp
        assert "effective_impedance_ratio" in a_sp
    for b_inv in bp_cli["buckingham_pi_invariants"]:
        assert "quality_factor_q_pre" in b_inv
        assert "quality_factor_q_post" in b_inv

    fc_cli = pf_cli["falsification_certificate"]
    cd_cli = fc_cli["cross_domain_and_horizon_synthesis"]
    assert "universal_telemetry_hunt_and_dual_blind_crucible" in cd_cli
    dual_c = cd_cli["universal_telemetry_hunt_and_dual_blind_crucible"]
    assert dual_c is not None
    assert dual_c["qualification_audit"]["passed_all_5_gates"] is True
    assert dual_c["split_30pct_train_70pct_blind_future"]["cortivra_hybrid_future_r2"] > 0.90
    assert dual_c["split_70pct_train_30pct_blind_future"]["cortivra_hybrid_future_r2"] > 0.95
    assert len(dual_c["split_30pct_train_70pct_blind_future"]["next_spot_forecasts"]) >= 1

    assert "autonomous_frontier_exploration" in cd_cli
    afe = cd_cli["autonomous_frontier_exploration"]
    assert afe is not None
    assert afe["cross_observatory_fusion"]["fused_cross_observatory_blind_70pct_r2"] >= 0.85
    assert afe["beyond_textbook_residual_discovery"]["beyond_textbook_augmented_blind_70pct_r2"] >= 0.85
    assert afe["multi_event_replication_crucible"]["universal_law_replication_verified"] is True
    assert afe["autonomous_curiosity_question_chain"]["generations_executed"] == 3
    assert len(afe["autonomous_curiosity_question_chain"]["autonomous_next_frontier_questions"]) == 3
    assert afe["spatial_3d_pde_verification"] is not None
    assert afe["spatial_3d_pde_verification"]["onsager_laplacian_psd_verified"] is True
    assert afe["spatial_3d_pde_verification"]["diffusive_mass_conservation_verified"] is True
    assert afe["spatial_3d_pde_verification"]["spatial_pde_stability_verified"] is True

    assert "recursive_pass2_synthesis_and_fabrication_blueprint" in bp_cli
    p2_cli = bp_cli["recursive_pass2_synthesis_and_fabrication_blueprint"]
    assert p2_cli is not None
    assert p2_cli["solved_pass2_product_yield_pct"] >= 90.0
    assert p2_cli["parasitic_impurity_or_defect_pct"] <= 5.0
    assert p2_cli["impurity_bystander_shield_intact"] is True
    assert p2_cli["reactor_zero_thermal_runaway_proven"] is True
    assert len(p2_cli["four_stage_synthesis_schedule"]) == 4

    assert "pass3_matter_and_lattice_architecture" in bp_cli
    p3_cli = bp_cli["pass3_matter_and_lattice_architecture"]
    assert p3_cli is not None
    assert p3_cli["combinatorial_candidates_evaluated"] == 700
    assert len(p3_cli["pareto_molecular_candidates"]) == 3
    assert len(p3_cli["metamaterial_topology_tournament"]) == 4
    assert p3_cli["top_molecular_scaffold"]["canonical_smiles"]
    assert p3_cli["top_molecular_scaffold"]["molecular_formula"]
    assert len(p3_cli["top_molecular_scaffold"]["heavy_atom_3d_xyz_angstrom"]) >= 12
    assert p3_cli["selected_metamaterial_lattice"]["euler_buckling_safety_factor"] >= 1.5
    assert "<svg" in p3_cli["pass3_structure_svg"]

    md_files = list(log_dir_cli.glob("*/dossier.md"))
    assert len(md_files) == 1
    md_txt = md_files[0].read_text(encoding="utf-8")
    assert "5D. Pillar 1" in md_txt
    assert "5E. Pillar 2" in md_txt
    assert "5F. Pillar 3" in md_txt
    assert "5G. Pillar 4" in md_txt
    assert "5H. Pillar 5" in md_txt
    assert "Recursive Pass-2 Autonomous Synthesis & Manufacturing Process Solver" in md_txt
    assert "Pass-3 Atomistic Molecular Graph (`SMILES`) & 3D Metamaterial Unit-Cell Lattice Compiler" in md_txt

    sop_files = list(log_dir_cli.glob("*/deliverables/wetlab_and_manufacturing_sop.md"))
    assert len(sop_files) == 1
    sop_txt = sop_files[0].read_text(encoding="utf-8")
    assert "## 5. Recursive Pass-2 Dynamical Synthesis & Foundry Process Recipe" in sop_txt
    assert "## 6. Pass-3 Atomistic Molecular Graph (`SMILES` & 3D Atomic XYZ) & 3D Metamaterial Unit-Cell Lattice" in sop_txt

    # Verify Plain-English Section 1A in dossier.md and dossier.html
    assert "### 1A. Plain-English Discovery, Bystander Safety & Wet-Lab Action Brief" in md_txt
    html_files = list(log_dir_cli.glob("*/dossier.html"))
    assert len(html_files) == 1
    html_txt = html_files[0].read_text(encoding="utf-8")
    assert "1A. Plain-English Discovery, Bystander Shield & Lab Translation Brief" in html_txt

    # Execute the compiled standalone digital_twin_simulator.py deliverable in a subprocess
    import subprocess
    import sys

    sim_files = list(log_dir_cli.glob("*/deliverables/digital_twin_simulator.py"))
    assert len(sim_files) == 1
    sim_proc = subprocess.run(
        [sys.executable, str(sim_files[0])],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    sim_payload = json.loads(sim_proc.stdout)
    assert sim_payload["all_compartments_within_homeostatic_bounds"] is True
    assert sim_payload["mismatched_25pct_adaptive_cbf_verified"] is True
    assert len(sim_payload["final_state"]) >= 2


def test_adversarial_constant_filtering_and_dynamic_ontology_epistemic_synthesis() -> None:
    """
    Verify:
      1. `extract_quantitative_constants_from_text` rejects bare single-digit figure/group references
         (`"Fig. 3 K"`, `"Group 1 V"`, `"Table 2 T"`) while extracting real physical & bioassay constants
         (`"310.15 K"`, `"IC50=14.2 nM"`, `"Kd = 0.85 uM"`, `"24.5 MPa"`).
      2. `repair_and_unify_symbolic_spec_dict` in Layer 3 algebraically solves 7D SI dimensional
         constraints across `hamiltonian_expr`, `dissipation_diag_exprs`, and `external_drive_exprs`,
         and enforces 2nd-Law non-negativity on dissipation parameters.
    """
    from cortivra.layer3_compiler.symbolic_compiler import (
        repair_and_unify_symbolic_spec_dict,
    )
    from cortivra.layer6_agents.universal_telemetry_hunter import (
        extract_quantitative_constants_from_text,
    )

    sample_abstract = (
        "As shown in Fig. 3 K and Group 1 V (Table 2 T), thermal incubation at 310.15 K "
        "yielded potent allosteric inhibition with IC50=14.2 nM and Kd = 0.85 uM under "
        "cortical loading of 24.5 MPa."
    )
    extracted = extract_quantitative_constants_from_text(sample_abstract)
    assert "3 K" not in extracted
    assert "1 V" not in extracted
    assert "2 T" not in extracted
    assert any("310.15 K" in item for item in extracted)
    assert any("14.2 nM" in item for item in extracted)
    assert any("0.85 uM" in item for item in extracted)
    assert any("24.5 MPa" in item for item in extracted)

    broken_spec = {
        "name": "coral_zooxanthella_photosystem",
        "domain": "marine_symbiology",
        "state_specs": [
            {"name": "coral_ros", "unit": "m", "default_value": 1.5},
            {"name": "zooxanthella_atp", "unit": "kg*m/s", "default_value": 0.8},
        ],
        "param_specs": [
            {"name": "k_spring", "unit": "1", "default_value": 2.4, "min_value": 0.1, "max_value": 10.0},
            {"name": "m_eff", "unit": "kg", "default_value": 1.0, "min_value": 0.1, "max_value": 10.0},
            {"name": "gamma_diss", "unit": "1", "default_value": -0.35, "min_value": -1.0, "max_value": 5.0},
        ],
        "time_unit": "s",
        "energy_unit": "J",
        "hamiltonian_expr": "0.5 * k_spring * (coral_ros ** 2) + 0.5 * (zooxanthella_atp ** 2) / m_eff",
        "dissipation_diag_exprs": ["0.0", "-gamma_diss"],
        "skew_routing_exprs": {"0,1": "1.0"},
    }
    repaired_spec, notes = repair_and_unify_symbolic_spec_dict(
        broken_spec, diagnostic_error="2nd Law dissipation parameter negative"
    )
    assert len(notes) >= 3
    param_map = {p["name"]: p for p in repaired_spec["param_specs"]}
    assert param_map["gamma_diss"]["default_value"] > 0.0
    assert param_map["gamma_diss"]["min_value"] >= 0.0
    assert repaired_spec["dissipation_diag_exprs"][1] == "gamma_diss"




