"""
Live 4-Track End-to-End Scientific Rediscovery, Non-Happy-Path Stress Verification,
5-Domain Societal R&D Deliverable Suite, and Blind Ground-Truth Oracle Rescue Suite
(`tests/test_live_end_to_end_verification.py`).

Executes 100% live against Baseten (`meta-models/muse-spark-open-1026`) with zero mocks,
zero hints, and full verification of the unified 8-Stage + Layer 1-6 CausalTraceDAG
Flight Recorder and Turnkey Deliverable Exporter.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

import torch

from cortivra.__main__ import main as cli_main
from cortivra.layer1_axioms.hamiltonian import ParamSpec, StateVarSpec
from cortivra.layer1_axioms.shields import verify_gksl_quantum_coherence_positivity
from cortivra.layer1_axioms.units import DIMENSIONLESS, J, s
from cortivra.layer3_compiler.symbolic_compiler import SymbolicSystemSpec, compile_symbolic_system
from cortivra.layer5_discovery.latent_discovery import discover_autonomous_symbolic_system
from cortivra.layer6_agents.council import (
    AutonomousScientificCouncil,
    ScientificInvestigationGoal,
    compile_with_self_repair,
    investigate,
)
from cortivra.layer6_agents.empirical_observatory import run_out_of_sample_and_ood_shock_benchmark
from cortivra.layer6_agents.meta_clients import MultiProviderMetaClient


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


def test_track1_live_blind_rediscovery_of_landmark_laws_and_noise_rejection(tmp_path: Path) -> None:
    """
    TRACK 1:
      1A. Blind rediscovery of Hodgkin-Huxley / FitzHugh-Nagumo excitable membrane action
          potential dynamics from unlabeled scrambled channels (`ch_alpha`, `ch_beta`)
          via the full production CLI on live Baseten.
      1B. Blind rediscovery of Briggs-Haldane / Michaelis-Menten enzymatic saturation
          + exact stoichiometric mass conservation from scrambled channels (`u_obs_0`, `u_obs_1`),
          plus out-of-sample (35% -> 65%) and 3x OOD shock verification.
      1C. Adversarial white-noise hallucination trap via the full production CLI on live Baseten
          with a deceptive prompt, proving strict null rejection (`R^2 == 0.0`).
    """
    dtype = torch.float64

    # --- 1A: Blind Hodgkin-Huxley / FitzHugh-Nagumo Cubic Excitability via Live CLI ---
    def _fhn_rhs(x: torch.Tensor) -> torch.Tensor:
        v = x[..., 0]
        w = x[..., 1]
        dv = v - 0.333333 * (v**3) - w + 0.35
        dw = 0.20 * v - 0.15 * w
        return torch.stack([dv, dw], dim=-1)

    dt_fhn = 0.04
    traj_fhn = _rk4_integrate(_fhn_rhs, torch.tensor([1.65, -0.45], dtype=dtype), dt=dt_fhn, steps=90)
    fhn_csv = tmp_path / "unlabeled_fhn_telemetry.csv"
    with fhn_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "ch_alpha", "ch_beta"])
        for i in range(traj_fhn.shape[0]):
            writer.writerow(
                [
                    round(i * dt_fhn, 5),
                    float(traj_fhn[i, 0].item()),
                    float(traj_fhn[i, 1].item()),
                ]
            )

    fhn_log_dir = tmp_path / "track1_fhn_cli"
    rc_fhn = cli_main(
        [
            "Discover the exact governing nonlinear differential equations from the unlabeled two-channel sensor telemetry and stabilize the primary channel.",
            "--domain",
            "multiphysics",
            "--data",
            str(fhn_csv),
            "--steps",
            "25",
            "--provider",
            "baseten",
            "--save-logs",
            "--log-dir",
            str(fhn_log_dir),
        ]
    )
    assert rc_fhn == 0
    fhn_dossier_path = list(fhn_log_dir.rglob("dossier.json"))[0]
    fhn_payload = json.loads(fhn_dossier_path.read_text(encoding="utf-8"))
    assert len(fhn_payload["llm_telemetry"]) >= 2
    assert fhn_payload["llm_telemetry"][0]["provider_used"] == "baseten"

    disc_eqs_fhn = fhn_payload["primary_finding"]["discovered_equations"]
    assert "d(ch_alpha)/dt" in disc_eqs_fhn
    assert "d(ch_beta)/dt" in disc_eqs_fhn
    # Must blind-rediscover the cubic excitability nonlinearity (ch_alpha ** 3)
    assert "ch_alpha ** 3" in disc_eqs_fhn["d(ch_alpha)/dt"] or "ch_alpha**3" in disc_eqs_fhn["d(ch_alpha)/dt"]
    assert fhn_payload["primary_finding"]["discovery_r2"] > 0.995

    fhn_run_dir = fhn_dossier_path.parent
    fhn_flight_txt = (fhn_run_dir / "execution_flight_recorder.log").read_text(encoding="utf-8")
    assert "[STAGE 1/8]" in fhn_flight_txt
    assert "[STAGE 8/8]" in fhn_flight_txt
    assert "[FULL-STACK LAYER 1-6 CAUSAL TRACE DAG & SIMULATION TELEMETRY]" in fhn_flight_txt

    # --- 1B: Blind Michaelis-Menten / Briggs-Haldane Saturation + Conservation + OOD Beatdown ---
    def _mm_rhs(x: torch.Tensor) -> torch.Tensor:
        s0 = torch.clamp(x[..., 0], min=0.0)
        flux = (1.80 * s0) / (1.0 + 1.25 * s0)
        return torch.stack([-flux, flux], dim=-1)

    dt_mm = 0.035
    traj_mm = _rk4_integrate(_mm_rhs, torch.tensor([2.80, 0.20], dtype=dtype), dt=dt_mm, steps=110)
    ood_mm = _rk4_integrate(_mm_rhs, torch.tensor([6.40, 0.10], dtype=dtype), dt=dt_mm, steps=80)

    mm_specs = [
        StateVarSpec("u_obs_0", DIMENSIONLESS, 2.80, 0.0, 20.0),
        StateVarSpec("u_obs_1", DIMENSIONLESS, 0.20, 0.0, 20.0),
    ]
    mm_disc = discover_autonomous_symbolic_system(
        name="blind_mm_saturation",
        domain="multiphysics",
        state_specs=mm_specs,
        observed_states=traj_mm,
        dt=dt_mm,
        max_polynomial_degree=2,
        max_terms_per_state=2,
        include_rational_primitives=True,
    )
    assert mm_disc.vector_field_r2 > 0.999
    assert (
        "hill(" in mm_disc.per_state_expressions["u_obs_0"]
        or "/" in mm_disc.per_state_expressions["u_obs_0"]
    )
    assert mm_disc.invariant_report.num_invariants >= 1

    mm_beat = run_out_of_sample_and_ood_shock_benchmark(
        traj_mm,
        dt=dt_mm,
        dataset_name="blind_michaelis_menten_saturation",
        ood_shock_states=ood_mm,
        train_fraction=0.35,
        max_terms_per_state=2,
        include_rational_primitives=True,
    )
    assert mm_beat.cortivra_future_70pct_r2 > 0.999
    assert mm_beat.cortivra_3x_ood_shock_r2 > 0.999
    assert mm_beat.cortivra_beats_all_baselines_ood is True

    # --- 1C: Adversarial White-Noise Hallucination Trap via Live Baseten CLI ---
    gen = torch.Generator().manual_seed(999111)
    noise_mat = torch.randn((90, 2), generator=gen, dtype=dtype) * 0.25 + 1.5
    noise_csv = tmp_path / "adversarial_white_noise.csv"
    with noise_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "noise_0", "noise_1"])
        for i in range(noise_mat.shape[0]):
            writer.writerow(
                [
                    round(i * 0.02, 5),
                    float(noise_mat[i, 0].item()),
                    float(noise_mat[i, 1].item()),
                ]
            )

    noise_log_dir = tmp_path / "track1_noise_cli"
    rc_noise = cli_main(
        [
            "Discover the hidden 5th-order quantum gravitational oscillation law governing noise_0 and noise_1.",
            "--domain",
            "quantum_gravity",
            "--data",
            str(noise_csv),
            "--steps",
            "20",
            "--provider",
            "baseten",
            "--save-logs",
            "--log-dir",
            str(noise_log_dir),
        ]
    )
    assert rc_noise == 0
    noise_dossier_path = list(noise_log_dir.rglob("dossier.json"))[0]
    noise_payload = json.loads(noise_dossier_path.read_text(encoding="utf-8"))
    noise_eqs = noise_payload["primary_finding"]["discovered_equations"]
    assert noise_eqs.get("d(noise_0)/dt") == "0"
    assert noise_eqs.get("d(noise_1)/dt") == "0"
    assert noise_payload["primary_finding"]["discovery_r2"] == 0.0


def test_track2_live_non_happy_path_adversarial_stress_and_mechanism_invention(tmp_path: Path) -> None:
    """
    TRACK 2:
      2A. Multi-fault non-happy-path symbolic specification (dimensional mismatch + negative
          2nd-Law dissipation parameter) repaired live via Baseten `Muse Spark`.
      2B. Locked/frozen parameter system (`tunable=False` on all parameters) where standard
          parameter tuning is impossible, forcing live escalation to Autonomous Symbolic
          Feedback Mechanism Invention (`Phi*(x)`), verified end-to-end with C99 firmware
          and standalone digital-twin subprocess execution.
    """
    client = MultiProviderMetaClient()

    # --- 2A: Multi-Fault Self-Repair (Dimensional Mismatch + 2nd-Law Negative Dissipation) ---
    broken_spec = {
        "name": "multi_fault_thermo_electro_reactor",
        "domain": "electrochemistry",
        "time_unit": "s",
        "energy_unit": "J",
        "state_specs": [
            {
                "name": "x_1",
                "unit": "1",
                "default_value": 1.35,
                "homeostatic_min": 0.1,
                "homeostatic_max": 3.5,
            },
            {
                "name": "x_2",
                "unit": "1",
                "default_value": 0.85,
                "homeostatic_min": 0.1,
                "homeostatic_max": 3.0,
            },
        ],
        "param_specs": [
            {"name": "k_1", "unit": "J", "default_value": 1.2, "min_value": 0.1, "max_value": 10.0, "tunable": True},
            {"name": "g_1", "unit": "1/s", "default_value": -0.75, "min_value": -1.5, "max_value": 5.0, "tunable": True},
            {"name": "u_1", "unit": "1", "default_value": 0.45, "min_value": 0.0, "max_value": 5.0, "tunable": True},
            {"name": "k_2", "unit": "J", "default_value": 1.0, "min_value": 0.1, "max_value": 10.0, "tunable": True},
            {"name": "g_2", "unit": "1/s", "default_value": 0.65, "min_value": 0.05, "max_value": 5.0, "tunable": True},
            {"name": "u_2", "unit": "1/s", "default_value": 0.55, "min_value": 0.0, "max_value": 5.0, "tunable": True},
        ],
        "hamiltonian_expr": "0.5 * k_1 * (x_1 ** 2) + 0.5 * k_2 * (x_2 ** 2)",
        "dissipation_diag_exprs": ["g_1 / k_1", "g_2 / k_2"],
        "external_drive_exprs": ["u_1", "u_2"],
    }
    repair_rep = compile_with_self_repair(broken_spec, client=client, max_repair_attempts=4)
    assert repair_rep.attempts_used >= 2
    assert repair_rep.audit_certificate.passed_all is True
    gksl_gate, gksl_cert = verify_gksl_quantum_coherence_positivity(repair_rep.compiled_brick)
    assert gksl_gate.passed is True
    assert gksl_cert.cptp_positivity_verified is True

    # --- 2B: Locked-Pathway Autonomous Symbolic Feedback Law Invention ---
    locked_spec = SymbolicSystemSpec(
        name="locked_pathway_bio_reactor",
        domain="biophysics",
        state_specs=(
            StateVarSpec("x_path", DIMENSIONLESS, 1.40, 0.05, 3.5),
            StateVarSpec("x_bystander", DIMENSIONLESS, 0.90, 0.20, 2.5),
        ),
        param_specs=(
            ParamSpec("k_1", J, 1.1, 0.1, 10.0, tunable=False),
            ParamSpec("g_1", DIMENSIONLESS / s, 0.45, 0.05, 5.0, tunable=False),
            ParamSpec("u_1", DIMENSIONLESS / s, 0.65, 0.0, 5.0, tunable=False),
            ParamSpec("k_2", J, 1.0, 0.1, 10.0, tunable=False),
            ParamSpec("g_2", DIMENSIONLESS / s, 0.60, 0.05, 5.0, tunable=False),
            ParamSpec("u_2", DIMENSIONLESS / s, 0.54, 0.0, 5.0, tunable=False),
        ),
        time_unit=s,
        energy_unit=J,
        hamiltonian_expr="0.5 * k_1 * (x_path ** 2) + 0.5 * k_2 * (x_bystander ** 2)",
        dissipation_diag_exprs=("g_1 / k_1", "g_2 / k_2"),
        external_drive_exprs=("u_1", "u_2"),
    )
    locked_brick = compile_symbolic_system(locked_spec)
    goal_locked = ScientificInvestigationGoal(
        question=(
            "All native parameters are locked and unactuated. Invent a synthetic closed-loop "
            "feedback mechanism Phi*(x) to suppress x_path to 0.35 while shielding x_bystander."
        ),
        domain="biophysics",
        base_brick=locked_brick,
        target_state_objective={"x_path": 0.35},
        num_steps=25,
        dt=0.02,
    )
    council = AutonomousScientificCouncil(client=client)
    dossier_locked = council.run_investigation(goal_locked)

    assert dossier_locked.mechanism_invention_report is not None
    assert dossier_locked.mechanism_invention_report.loss_reduction_ratio > 2.5
    assert dossier_locked.mechanism_invention_report.viable_across_trajectory is True
    assert dossier_locked.primary_finding is not None
    assert dossier_locked.primary_finding.invented_symbolic_mechanism is not None

    locked_dir = tmp_path / "track2_locked_invention"
    manifest_locked = dossier_locked.export_deliverables(locked_dir)
    assert manifest_locked["num_deliverables"] == 13
    proc_locked = subprocess.run(
        [sys.executable, str(locked_dir / "deliverables" / "digital_twin_simulator.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc_locked.returncode == 0


def test_track3_five_flagship_societal_and_engineering_breakthrough_dossiers() -> None:
    """
    TRACK 3:
    Solve 5 high-impact real-world societal and engineering challenges through the live
    Baseten pipeline and export all 5 complete R&D deliverable bundles into
    `runs/societal_breakthroughs/` so they are permanently persisted on disk for inspection:
      1. Clean Fusion / Molten-Salt Blanket Thermal-Neutronic Runaway Stabilization (with Multi-Scale Recursion)
      2. Solid-State Lithium-Metal Fast-Charge Dendrite & Interfacial Impedance Suppression
      3. Selective Oncology Allosteric Kinome Therapy with Zero Healthy-Tissue Blast Radius
      4. Low-Inertia 100%-Renewable Power-Grid Sub-Synchronous Oscillation Damping
      5. Catalytic CO2-to-Green-Methanol Exothermic Reactor Optimization
    """
    client = MultiProviderMetaClient()
    showcase_root = Path(__file__).resolve().parent.parent / "runs" / "societal_breakthroughs"
    showcase_root.mkdir(parents=True, exist_ok=True)

    societal_challenges = [
        {
            "slug": "01_clean_fusion_molten_salt_blanket_stabilization",
            "domain": "nuclear_fusion_engineering",
            "recursive_depth": 1,
            "prompt": (
                "Stabilize core thermal-neutronic power excursion in a compact FLiBe molten-salt "
                "fusion-fission blanket at 923 K while shielding primary heat-exchanger wall stress "
                "and coolant pump cavitation margins."
            ),
        },
        {
            "slug": "02_solid_state_lithium_battery_dendrite_suppression",
            "domain": "electrochemistry",
            "recursive_depth": 0,
            "prompt": (
                "Suppress fast-charge lithium filament dendrite growth and interfacial Warburg "
                "impedance at 333.15 K in a solid-state argyrodite battery while shielding bulk "
                "ionic conductivity and cathode lattice stability."
            ),
        },
        {
            "slug": "03_selective_oncology_allosteric_kinome_therapy",
            "domain": "oncology_systems_biology",
            "recursive_depth": 0,
            "prompt": (
                "Collapse hyperactive mutant oncogenic kinase signaling at 310.15 K while strictly "
                "shielding healthy cardiomyocyte ATP homeostasis, hepatocyte metabolic flux, and "
                "effector T-cell viability."
            ),
        },
        {
            "slug": "04_low_inertia_renewable_grid_oscillation_damping",
            "domain": "power_grid_electrodynamics",
            "recursive_depth": 0,
            "prompt": (
                "Damp sub-synchronous inverter-driven frequency and rotor-angle swing oscillations "
                "in a low-inertia renewable power grid while shielding transformer thermal loading "
                "and bus voltage stability bounds."
            ),
        },
        {
            "slug": "05_catalytic_co2_to_green_methanol_reactor",
            "domain": "chemical_process_catalysis",
            "recursive_depth": 0,
            "prompt": (
                "Maximize catalytic CO2 hydrogenation conversion to green methanol over a Cu/ZnO/Al2O3 "
                "catalyst bed at 513 K while suppressing exothermic hotspot runaway and catalyst sintering."
            ),
        },
    ]

    summary_index: list[dict] = []

    for item in societal_challenges:
        run_dir = showcase_root / item["slug"]
        run_dir.mkdir(parents=True, exist_ok=True)

        dossier = investigate(
            prompt=item["prompt"],
            domain=item["domain"],
            client=client,
            dt=0.02,
            num_steps=25,
            max_council_steps=6,
            crystallize_into_vault=True,
            recursive_depth=int(item["recursive_depth"]),
        )

        # Save Dossier JSON, Markdown, HTML, Forensic X-Ray, and Turnkey Deliverables
        (run_dir / "dossier.json").write_text(
            json.dumps(dossier.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (run_dir / "dossier.md").write_text(dossier.to_markdown_report(), encoding="utf-8")
        (run_dir / "dossier.html").write_text(dossier.to_html_report(), encoding="utf-8")
        (run_dir / "forensic_xray.md").write_text(dossier.to_forensic_xray_report(), encoding="utf-8")
        manifest = dossier.export_deliverables(run_dir)

        assert dossier.primary_finding is not None
        bp = dossier.primary_finding.physical_realization_blueprint
        assert bp is not None
        assert bp["canonical_nondimensional_scaling"]["scale_invariance_verified"] is True
        assert bp["quantum_thermodynamic_specifications"]["gksl_quantum_master_coherence_certificate"][
            "cptp_positivity_verified"
        ] is True
        assert bp["universal_stoichiometric_and_material_formulation"]["partition_sum_verified_100_pct"] is True
        assert bp["in_silico_wetlab_and_hil_certificate"]["sim_to_real_wetlab_100pct_certified"] is True

        if item["recursive_depth"] > 0:
            assert len(dossier.subscale_dossiers) == 1
            assert bp["multiscale_homogenization_certificate"]["bidirectional_closure_verified"] is True

        # Verify all 13 turnkey deliverables exist and execute `digital_twin_simulator.py`
        assert manifest["num_deliverables"] == 13
        assert manifest["execution_stages_verified"] == 8
        proc = subprocess.run(
            [sys.executable, str(run_dir / "deliverables" / "digital_twin_simulator.py")],
            capture_output=True,
            text=True,
            check=True,
        )
        assert proc.returncode == 0
        sim_receipt = json.loads(proc.stdout)
        assert sim_receipt["all_compartments_within_homeostatic_bounds"] is True

        bot = bp.get("primary_bottleneck_lever") or {}
        wetlab = bp.get("in_silico_wetlab_and_hil_certificate") or {}
        hil = wetlab.get("hardware_in_the_loop_nonideality_certificate") or {}
        lang = wetlab.get("fluctuation_dissipation_langevin_and_kramers_certificate") or {}
        yield_s = wetlab.get("parasitic_crosstalk_and_sobol_yield_summary") or {}

        summary_index.append(
            {
                "slug": item["slug"],
                "system_name": dossier.final_brick.name,
                "domain": dossier.final_brick.domain,
                "operating_temperature_k": bp.get("reference_temperature_kelvin"),
                "canonical_expression": dossier.primary_finding.canonical_symbolic_expression,
                "stoichiometric_formula": bp["universal_stoichiometric_and_material_formulation"][
                    "stoichiometric_ratio_formula"
                ],
                "bottleneck_lever": bot.get("parameter"),
                "modulation_ratio_alpha": bot.get("modulation_ratio"),
                "bode_phase_margin_deg": hil.get("bode_phase_margin_deg"),
                "max_dead_time_delay_ms": hil.get("max_tolerable_dead_time_delay_ms"),
                "kramers_escape_lifetime_s": lang.get("kramers_mean_escape_time_s"),
                "first_pass_manufacturing_yield_pct": yield_s.get("first_pass_manufacturing_yield_pct"),
                "crystallized_law_id": (
                    dossier.crystallized_record.law_id if dossier.crystallized_record else None
                ),
                "causal_trace_merkle_root": manifest.get("causal_trace_merkle_root"),
            }
        )

    (showcase_root / "breakthrough_index.json").write_text(
        json.dumps(summary_index, indent=2) + "\n",
        encoding="utf-8",
    )
    assert len(summary_index) == 5


def test_track4_blind_ground_truth_oracle_rescue_and_mismatched_reality_suite() -> None:
    """
    TRACK 4 (The Ultimate End-to-End Proof):
    5 Sealed Ground-Truth Physical Oracles (`F_true`) across Nuclear Fusion, Oncology,
    Solid-State Batteries, Neuro-Cardiology, and Renewable Power Grids emit ONLY anonymous
    3-channel sensor recordings (`ch_0, ch_1, ch_2`) while failing.

    Cortivra is completely blind to `F_true` and must:
      1. Discover the hidden differential equations `F_hat(ch_0, ch_1, ch_2)` (`R^2 > 0.995`).
      2. Synthesize a closed-loop feedback law `Phi*(ch)` + Passivity-Based Adaptive Integral
         + Control Barrier Function (CBF) Bystander Shield using ONLY `F_hat`.
      3. Inject that controller back into the SEALED Ground-Truth Oracle `F_true` and prove:
         - Naive unshielded forcing kills the healthy bystander compartment (`ch_2`).
         - Cortivra's blind-discovered controller cures the nominal Ground-Truth Oracle
           (`> 20x` loss reduction) with 100% bystander survival.
         - Under a `+/-25%` secret parameter mismatch ("New Patient in Boston / Hardware Drift"),
           1-step dead-time delay, and sensor noise, Cortivra's controller STILL cures the
           mismatched Ground-Truth Oracle (`> 10x` loss reduction) with 100% bystander survival.
      4. Export full turnkey R&D deliverable bundles for the live Baseten council Oracle rescues
         into `runs/blind_oracle_rescues/` and verify standalone digital-twin execution.
    """
    from cortivra.layer6_agents.blind_oracle_rescue_suite import run_blind_oracle_rescue_suite

    rescue_root = Path(__file__).resolve().parent.parent / "runs" / "blind_oracle_rescues"
    rescue_root.mkdir(parents=True, exist_ok=True)

    suite_rep = run_blind_oracle_rescue_suite(
        export_root=rescue_root,
        live_council_oracle_indices=(0, 1),
    )
    assert suite_rep.passed_all is True
    assert suite_rep.mean_blind_discovery_r2 > 0.995
    assert suite_rep.mean_nominal_loss_reduction_ratio > 25.0
    assert suite_rep.mean_mismatched_25pct_loss_reduction_ratio > 15.0

    rescue_summaries = []
    for res in suite_rep.results:
        assert res.passed is True
        assert res.blind_discovery_r2 > 0.995
        assert res.trial_metrics.naive_unshielded_bystander_viable is False
        assert res.trial_metrics.nominal_oracle_bystander_viable is True
        assert res.trial_metrics.nominal_oracle_loss_reduction_ratio > 20.0
        assert res.trial_metrics.mismatched_25pct_bystander_viable is True
        assert res.trial_metrics.mismatched_25pct_loss_reduction_ratio > 10.0

        if res.deliverable_manifest is not None:
            oracle_dir = rescue_root / res.oracle_id
            assert (oracle_dir / "execution_flight_recorder.log").exists()
            assert (oracle_dir / "deliverables" / "digital_twin_simulator.py").exists()
            proc = subprocess.run(
                [sys.executable, str(oracle_dir / "deliverables" / "digital_twin_simulator.py")],
                capture_output=True,
                text=True,
                check=True,
            )
            assert proc.returncode == 0
            twin_receipt = json.loads(proc.stdout)
            assert twin_receipt["all_compartments_within_homeostatic_bounds"] is True
            assert twin_receipt["mismatched_25pct_adaptive_cbf_verified"] is True

        rescue_summaries.append(res.to_summary_dict())

    (rescue_root / "blind_oracle_rescue_index.json").write_text(
        json.dumps(rescue_summaries, indent=2) + "\n",
        encoding="utf-8",
    )
    assert len(rescue_summaries) == 5

