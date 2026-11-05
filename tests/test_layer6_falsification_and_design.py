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


def test_multi_provider_fireworks_baseten_meta_routing_and_failover() -> None:
    """
    Verify `MultiProviderMetaClient` routes requests across Fireworks AI, Baseten, and Meta API,
    sends proper Bearer headers and JSON bodies, and automatically fails over when a primary
    provider returns a transient 503 error.
    """
    captured_requests: list[tuple[str, str, dict]] = []

    def _mock_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        auth = request.headers.get("Authorization", "")
        body = json.loads(request.content.decode("utf-8"))
        captured_requests.append((url_str, auth, body))

        # Simulate a transient 503 on Fireworks AI so client fails over to Baseten
        if "fireworks.ai" in url_str:
            return httpx.Response(status_code=503, json={"error": "service_unavailable"})

        if "baseten.co" in url_str:
            return httpx.Response(
                status_code=200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "```json\n{\"tool_name\": \"simulate_and_inspect_dynamics\", \"arguments\": {}, \"epistemic_rationale\": \"Verified Baseten failover.\"}\n```"
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 140, "completion_tokens": 28},
                },
            )

        return httpx.Response(status_code=400, json={"error": "unexpected_host"})

    transport = httpx.MockTransport(_mock_handler)
    client = MultiProviderMetaClient(
        env_overrides={
            "FIREWORKS_API_KEY": "fw_live_test_key_123",
            "BASETEN_API_KEY": "baseten_live_test_key_456",
            "META_API_KEY": "your_meta_model_api_key_here",  # placeholder should be ignored
        },
        http_transport=transport,
    )

    assert client.configured_providers() == ["baseten", "fireworks"]

    resp = client.invoke(
        LLMInferenceRequest(
            role="muse_spark_open",
            system_prompt="System test",
            user_prompt="{\"task_type\": \"select_next_action\"}", preferred_provider="fireworks",
            require_json=True,
        )
    )

    assert len(captured_requests) == 2
    assert "fireworks.ai/inference/v1/chat/completions" in captured_requests[0][0]
    assert captured_requests[0][1] == "Bearer fw_live_test_key_123"
    assert "baseten.co/v1/chat/completions" in captured_requests[1][0]
    assert captured_requests[1][1] == "Bearer baseten_live_test_key_456"

    assert resp.provider_used == "baseten"
    assert resp.failover_attempts == 1
    assert resp.prompt_tokens == 140
    assert resp.completion_tokens == 28
    assert resp.parsed_json is not None
    assert resp.parsed_json["tool_name"] == "simulate_and_inspect_dynamics"


def test_compiler_guided_symbolic_self_repair_unit_and_second_law_violations() -> None:
    """
    Verify `compile_with_self_repair` catches both:
        1. Layer 3 `DimensionalMismatchError` (wrong parameter unit in rate equation)
        2. Layer 1 Second Law violation (negative dissipation parameter)
    and automatically repairs the specification via `Muse Spark` until `audit_law_brick` passes.
    """
    client = MultiProviderMetaClient()

    # Case 1: Dimensional mismatch (k_rate has unit "1" instead of "1/s" in dx/dt = -k_rate * c_A)
    bad_unit_spec = {
        "name": "bad_unit_reaction",
        "domain": "chemistry",
        "state_specs": [
            {
                "name": "c_A",
                "unit": "mol/m^3",
                "default_value": 2.0,
                "homeostatic_min": 0.0,
                "homeostatic_max": 20.0,
            }
        ],
        "param_specs": [
            {
                "name": "k_rate",
                "unit": "1",  # Intentional unit bug: should be 1/s
                "default_value": 0.5,
                "min_value": 0.0,
                "max_value": 10.0,
            }
        ],
        "time_unit": "s",
        "energy_unit": "J",
        "hamiltonian_expr": "0.0",
        "external_drive_exprs": ["-k_rate * c_A"],
    }

    rep_unit = compile_with_self_repair(bad_unit_spec, client=client, max_repair_attempts=3)
    assert rep_unit.attempts_used == 2
    assert len(rep_unit.repair_diagnostics) == 1
    assert "DimensionError" in rep_unit.repair_diagnostics[0]
    assert rep_unit.audit_certificate.passed_all

    # Case 2: Second Law violation (negative dissipation coefficient in passive oscillator)
    bad_thermo_spec = {
        "name": "bad_second_law_oscillator",
        "domain": "mechanics",
        "state_specs": [
            {"name": "q", "unit": "m", "default_value": 1.0, "homeostatic_min": -10.0, "homeostatic_max": 10.0},
            {"name": "p", "unit": "N*s", "default_value": 0.5, "homeostatic_min": -10.0, "homeostatic_max": 10.0},
        ],
        "param_specs": [
            {"name": "k_s", "unit": "N/m", "default_value": 2.0, "min_value": 0.1, "max_value": 20.0},
            {"name": "m_osc", "unit": "kg", "default_value": 1.0, "min_value": 0.1, "max_value": 10.0},
            {"name": "gamma_bad", "unit": "N*s/m", "default_value": -0.8, "min_value": -5.0, "max_value": 5.0},
        ],
        "time_unit": "s",
        "energy_unit": "J",
        "hamiltonian_expr": "0.5 * k_s * (q ** 2) + 0.5 * (p ** 2) / m_osc",
        "dissipation_diag_exprs": ["0.0", "gamma_bad"],
        "skew_coupling_exprs": {"0,1": "1.0"},
    }

    rep_thermo = compile_with_self_repair(bad_thermo_spec, client=client, max_repair_attempts=3)
    assert rep_thermo.attempts_used == 2
    assert len(rep_thermo.repair_diagnostics) == 1
    assert rep_thermo.audit_certificate.passed_all

    # Case 3: Quartic Duffing Hamiltonian unit mismatch (alpha has unit "N/m" instead of "N/m^3")
    # combined with unbalanced parentheses in hamiltonian_expr
    bad_hamiltonian_spec = {
        "name": "bad_quartic_hamiltonian_oscillator",
        "domain": "mechanics",
        "state_specs": [
            {"name": "q", "unit": "m", "default_value": 0.8, "homeostatic_min": -10.0, "homeostatic_max": 10.0},
            {"name": "p", "unit": "N*s", "default_value": 0.2, "homeostatic_min": -10.0, "homeostatic_max": 10.0},
        ],
        "param_specs": [
            {"name": "k_s", "unit": "N/m", "default_value": 2.0, "min_value": 0.1, "max_value": 20.0},
            {"name": "alpha_q", "unit": "N/m", "default_value": 1.2, "min_value": 0.0, "max_value": 20.0},
            {"name": "m_osc", "unit": "kg", "default_value": 1.0, "min_value": 0.1, "max_value": 10.0},
        ],
        "time_unit": "s",
        "energy_unit": "J",
        "hamiltonian_expr": "0.5 * k_s * (q ** 2) + 0.25 * alpha_q * (q ** 4) + 0.5 * (p ** 2) / m_osc",
        "dissipation_diag_exprs": ["0.0", "0.0"],
        "skew_coupling_exprs": {"0,1": "1.0"},
    }
    rep_ham = compile_with_self_repair(bad_hamiltonian_spec, client=client, max_repair_attempts=3)
    assert rep_ham.attempts_used == 2
    assert rep_ham.audit_certificate.passed_all


def test_active_adversarial_disambiguating_experiment_design() -> None:
    """
    Verify `design_disambiguating_experiment` finds an initial state `x0*` and stimulus `u*(t)`
    within homeostatic bounds that amplifies the trajectory divergence between a harmonic
    oscillator (`candidate_a`) and a quartic Duffing oscillator (`candidate_b`), and correctly
    identifies `candidate_b` when tested against the oracle.
    """
    spec_linear = SymbolicSystemSpec(
        name="harmonic_candidate_a",
        domain="mechanics",
        state_specs=[
            StateVarSpec("q", m, default_value=0.1, homeostatic_min=-5.0, homeostatic_max=5.0),
            StateVarSpec("p", N_force * s, default_value=0.0, homeostatic_min=-10.0, homeostatic_max=10.0),
        ],
        param_specs=[
            ParamSpec("k", N_force / m, default_value=2.0),
            ParamSpec("m_osc", kg, default_value=1.0, tunable=False),
            ParamSpec("damping", N_force * s / m, default_value=0.1),
        ],
        time_unit=s,
        energy_unit=J,
        hamiltonian_expr="0.5 * k * (q ** 2) + 0.5 * (p ** 2) / m_osc",
        dissipation_diag_exprs=("0.0", "damping"),
        skew_routing_exprs={(0, 1): "1.0"},
    )
    cand_a = compile_symbolic_system(spec_linear)

    spec_duffing = SymbolicSystemSpec(
        name="duffing_candidate_b",
        domain="mechanics",
        state_specs=spec_linear.state_specs,
        param_specs=[
            ParamSpec("k", N_force / m, default_value=2.0),
            ParamSpec("alpha", N_force / (m ** 3), default_value=1.8),
            ParamSpec("m_osc", kg, default_value=1.0, tunable=False),
            ParamSpec("damping", N_force * s / m, default_value=0.1),
        ],
        time_unit=s,
        energy_unit=J,
        hamiltonian_expr="0.5 * k * (q ** 2) + 0.25 * alpha * (q ** 4) + 0.5 * (p ** 2) / m_osc",
        dissipation_diag_exprs=("0.0", "damping"),
        skew_routing_exprs={(0, 1): "1.0"},
    )
    cand_b = compile_symbolic_system(spec_duffing)

    # At tiny initial amplitude x0 = [0.05, 0.0], q^4 is ~6e-6 so models A and B are nearly indistinguishable
    x0_small = torch.tensor([[0.05, 0.0]], dtype=torch.float64)
    pa = cand_a.default_params(batch_size=1, dtype=torch.float64)
    pb = cand_b.default_params(batch_size=1, dtype=torch.float64)

    rep = design_disambiguating_experiment(
        candidate_a=cand_a,
        params_a=pa,
        candidate_b=cand_b,
        params_b=pb,
        baseline_x0=x0_small,
        dt=0.03,
        num_steps=25,
        max_stimulus_amplitude=2.5,
        oracle_brick=cand_b,
        oracle_params=pb,
    )

    assert rep.maximized_divergence_mse > rep.initial_divergence_mse * 20.0
    assert rep.divergence_amplification_ratio > 20.0
    assert rep.winning_hypothesis == "candidate_b"
    assert rep.error_mse_b < 1e-12
    assert rep.error_mse_a > 1e-2


def test_four_gate_falsification_stress_suite_and_vault_crystallization(tmp_path: Path) -> None:
    """
    Verify `run_falsification_stress_test` evaluates all 4 falsification gates and
    `crystallize_verified_law` promotes passing laws into `LawVault` with SHA-256 lineage.
    """
    vault = LawVault(tmp_path / "test_vault_l6")
    register_foundational_laws(vault)

    pendulum = NonlinearPendulumAndViscoelasticBrick(damping_default=0.05)
    x0 = torch.tensor([[0.6, 0.2]], dtype=torch.float64)
    p = pendulum.default_params(batch_size=1, dtype=torch.float64)
    roll = rollout_trajectories(pendulum, x0, p, dt=0.02, num_steps=30, integrator="rk4")

    stress_rep = run_falsification_stress_test(
        brick=pendulum,
        calibrated_params=p,
        reference_states=roll.states[0],
        dt=0.02,
    )
    assert stress_rep.passed_all_gates
    assert len(stress_rep.gates) == 4
    assert stress_rep.noisy_vector_field_r2 > 0.95
    assert stress_rep.ood_viable_fraction == 1.0
    assert stress_rep.multi_horizon_report is not None
    assert stress_rep.multi_horizon_report.passed_multi_horizon
    assert stress_rep.multi_horizon_report.horizon_1x_viable_fraction == 1.0
    assert stress_rep.multi_horizon_report.horizon_2x_viable_fraction == 1.0
    assert stress_rep.multi_horizon_report.horizon_4x_viable_fraction == 1.0
    assert stress_rep.multi_horizon_report.min_safety_margin_4x > 0.0

    core_laws = vault.list_laws()
    rec = crystallize_verified_law(
        vault=vault,
        brick=pendulum,
        stress_report=stress_rep,
        law_id="verified_damped_pendulum",
        parent_law_ids=(core_laws[0].law_id,),
    )
    assert rec.law_id == "verified_damped_pendulum"
    assert rec.verify_integrity()
    assert len(vault.get_ancestry_chain("verified_damped_pendulum")) == 2


def test_council_autonomous_multi_target_intervention_cure_discovery(tmp_path: Path) -> None:
    """
    Verify `AutonomousScientificCouncil` autonomously diagnoses a pathological steady state
    in a multi-parameter biochemical homeostasis network, computes the Fisher leverage spectrum
    (filtering out a structurally sloppy parameter), solves the counterfactual parameter
    intervention to restore healthy homeostasis, and stress-tests the result.
    """
    # Biochemical homeostasis system:
    # dx/dt = k_prod - k_clear * x + 0.0 * k_sloppy * x
    # Baseline has pathological clearance deficit (k_clear = 0.25 -> equilibrium x* = 8.0,
    # whereas healthy target is x = 2.0)
    spec = SymbolicSystemSpec(
        name="metabolic_homeostasis_pathway",
        domain="chemistry",
        state_specs=[
            StateVarSpec("x_metab", mol / (m ** 3), default_value=6.5, homeostatic_min=0.1, homeostatic_max=15.0),
        ],
        param_specs=[
            ParamSpec("k_sloppy", DIMENSIONLESS / s, default_value=1.0, min_value=0.0, max_value=10.0, tunable=True),
            ParamSpec("k_prod", (mol / (m ** 3)) / s, default_value=2.0, min_value=0.2, max_value=10.0, tunable=True),
            ParamSpec("k_clear", DIMENSIONLESS / s, default_value=0.25, min_value=0.1, max_value=5.0, tunable=True),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.0",
        external_drive_exprs=("k_prod - k_clear * x_metab + (0.0001 * k_sloppy * x_metab)",),
    )
    pathway_brick = compile_symbolic_system(spec)

    vault = LawVault(tmp_path / "cure_vault")
    register_foundational_laws(vault)
    council = AutonomousScientificCouncil(vault=vault)

    goal = ScientificInvestigationGoal(
        question="Find a safe parameter intervention to restore x_metab from pathological level 6.5 to healthy target 2.0.",
        domain="chemistry",
        base_brick=pathway_brick,
        initial_state=torch.tensor([[6.5]], dtype=torch.float64),
        target_state_objective={"x_metab": 2.0},
        inverse_mode="counterfactual_intervention",
        dt=0.04,
        num_steps=40,
    )

    dossier = council.run_investigation(goal)

    executed_tools = [s.tool_name for s in dossier.steps]
    assert "simulate_and_inspect_dynamics" in executed_tools
    assert "map_equilibria_and_bifurcations" in executed_tools
    assert "analyze_fisher_sensitivity_and_leverage" in executed_tools
    assert "solve_inverse_problem" in executed_tools
    assert "adversarial_falsify_and_crystallize" in executed_tools

    # Verify Fisher sensitivity correctly flagged k_sloppy as sloppy and targeted k_prod / k_clear
    assert dossier.final_briefing.sensitivity_summary is not None
    assert "k_sloppy" in dossier.final_briefing.sensitivity_summary.sloppy_parameters
    assert dossier.intervention_solution is not None
    assert dossier.intervention_solution.intervened_loss < dossier.intervention_solution.baseline_loss * 0.05
    final_x = float(dossier.intervention_solution.restored_rollout.states[0, -1, 0].item())
    assert abs(final_x - 2.0) < 0.25
    assert dossier.falsification_report is not None and dossier.falsification_report.passed_all_gates
    assert dossier.primary_finding is not None
    assert dossier.primary_finding.prescribed_intervention is not None
    assert "k_sloppy" in dossier.primary_finding.prescribed_intervention["excluded_sloppy_parameters"]
    assert len(dossier.executive_summary) > 30
    json_str = json.dumps(dossier.to_dict())
    assert "counterfactual_parameter_intervention" in json_str
    md_report = dossier.to_markdown_report()
    assert "## 2. Primary Mathematical Findings" in md_report
    assert "Counterfactual Parameter Intervention" in md_report


def test_council_closed_loop_mechanism_discovery_and_cross_domain_vault_transfer(tmp_path: Path) -> None:
    """
    Verify `AutonomousScientificCouncil` detects an unmodeled cubic damping mechanism from
    observed trajectory data, triggers Layer 5 symbolic discovery, matches cross-domain
    `LawVault` analogues, passes the 4-gate falsification suite, and crystallizes the law.
    """
    true_spec = SymbolicSystemSpec(
        name="true_nonlinear_damped_oscillator",
        domain="mechanics",
        state_specs=[
            StateVarSpec("q", m, default_value=1.4, homeostatic_min=-10.0, homeostatic_max=10.0),
            StateVarSpec("p", N_force * s, default_value=0.0, homeostatic_min=-15.0, homeostatic_max=15.0),
        ],
        param_specs=[
            ParamSpec("k_s", N_force / m, default_value=3.0),
            ParamSpec("m_osc", kg, default_value=1.0, tunable=False),
            ParamSpec("gamma_lin", N_force * s / m, default_value=0.1),
            ParamSpec("gamma_cub", (N_force * s / m) / ((N_force * s) ** 2), default_value=0.45),
        ],
        time_unit=s,
        energy_unit=J,
        hamiltonian_expr="0.5 * k_s * (q ** 2) + 0.5 * (p ** 2) / m_osc",
        dissipation_diag_exprs=("0.0", "gamma_lin + gamma_cub * (p ** 2)"),
        skew_routing_exprs={(0, 1): "1.0"},
    )
    true_brick = compile_symbolic_system(true_spec)
    x0 = torch.tensor([[1.5, 0.0]], dtype=torch.float64)
    p_true = true_brick.default_params(batch_size=1, dtype=torch.float64)
    dt = 0.025
    gt_roll = rollout_trajectories(true_brick, x0, p_true, dt=dt, num_steps=45, integrator="rk4")

    # Incomplete baseline brick with only linear damping
    base_spec = SymbolicSystemSpec(
        name="incomplete_linear_oscillator",
        domain="mechanics",
        state_specs=true_spec.state_specs,
        param_specs=[
            ParamSpec("k_s", N_force / m, default_value=3.0, tunable=False),
            ParamSpec("m_osc", kg, default_value=1.0, tunable=False),
            ParamSpec("gamma_lin", N_force * s / m, default_value=0.1, min_value=0.0, max_value=5.0, tunable=True),
        ],
        time_unit=s,
        energy_unit=J,
        hamiltonian_expr="0.5 * k_s * (q ** 2) + 0.5 * (p ** 2) / m_osc",
        dissipation_diag_exprs=("0.0", "gamma_lin"),
        skew_routing_exprs={(0, 1): "1.0"},
    )
    base_brick = compile_symbolic_system(base_spec)

    vault = LawVault(tmp_path / "discovery_vault")
    register_foundational_laws(vault)
    council = AutonomousScientificCouncil(vault=vault)

    goal = ScientificInvestigationGoal(
        question="Calibrate oscillator and discover any unmodeled nonlinear damping operator.",
        domain="mechanics",
        base_brick=base_brick,
        observed_states=gt_roll.states[0],
        observed_dx_dt=gt_roll.flows[0],
        dt=dt,
        num_steps=45,
    )

    dossier = council.run_investigation(goal)
    assert dossier.discovery_report is not None
    assert dossier.discovery_report.discovery_triggered
    assert dossier.discovery_report.out_of_sample_r2 > 0.99
    assert len(dossier.vault_analogues) >= 1
    assert len(dossier.vault_analogues[0].matched_operators) >= 1
    assert dossier.falsification_report is not None and dossier.falsification_report.passed_all_gates
    assert dossier.crystallized_record is not None
    active_roles = {t.role for t in dossier.llm_telemetry}
    assert {"muse_wonder", "muse_spark", "muse_spark_open", "muse_glimmer"}.issubset(active_roles)
    assert dossier.primary_finding is not None
    assert len(dossier.primary_finding.discovered_equations) >= 1
    assert len(dossier.executive_summary) > 30
    assert "## 5. Adversarial Falsification & Multi-Horizon Stability Certificate" in dossier.to_markdown_report()


def test_council_autonomous_zero_prior_discovery_and_optimal_control_and_retrodiction() -> None:
    """
    Verify `AutonomousScientificCouncil` handles:
        1. Zero-prior autonomous system discovery from raw trajectories (`base_brick=None`).
        2. Optimal control waveform synthesis (`inverse_mode="optimal_control"`).
        3. Backward time-reversal retrodiction (`inverse_mode="retrodiction"`).
    """
    # Part 1: Zero-prior autonomous discovery + optimal control
    gen_spec = SymbolicSystemSpec(
        name="true_coupled_relaxation",
        domain="electrodynamics",
        state_specs=[
            StateVarSpec("x", DIMENSIONLESS, default_value=1.2, homeostatic_min=-10.0, homeostatic_max=10.0),
            StateVarSpec("y", DIMENSIONLESS, default_value=0.8, homeostatic_min=-10.0, homeostatic_max=10.0),
        ],
        param_specs=[
            ParamSpec("a", DIMENSIONLESS / s, default_value=-0.8),
            ParamSpec("b", DIMENSIONLESS / s, default_value=0.5),
            ParamSpec("c", DIMENSIONLESS / s, default_value=-1.1),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.0",
        external_drive_exprs=("a * x + b * y", "c * y"),
    )
    gen_brick = compile_symbolic_system(gen_spec)
    x0 = torch.tensor([[2.0, 1.5]], dtype=torch.float64)
    p_gen = gen_brick.default_params(batch_size=1, dtype=torch.float64)
    dt = 0.03
    roll = rollout_trajectories(gen_brick, x0, p_gen, dt=dt, num_steps=35, integrator="rk4")

    council = AutonomousScientificCouncil()
    goal_auto = ScientificInvestigationGoal(
        question="Discover the coupled system equations from raw data and synthesize a control waveform to drive x to 0.0.",
        domain="electrodynamics",
        base_brick=None,
        state_specs_for_autonomous=gen_spec.state_specs,
        observed_states=roll.states[0],
        observed_dx_dt=roll.flows[0],
        initial_state=x0,
        target_state_objective={"x": 0.0, "y": 0.0},
        inverse_mode="optimal_control",
        dt=dt,
        num_steps=25,
        crystallize_into_vault=False,
    )

    dossier_auto = council.run_investigation(goal_auto)
    assert dossier_auto.autonomous_discovery_report is not None
    assert dossier_auto.autonomous_discovery_report.vector_field_r2 > 0.999
    assert dossier_auto.optimal_control_result is not None
    assert dossier_auto.optimal_control_result.controlled_loss < dossier_auto.optimal_control_result.baseline_loss

    # Part 2: Time-reversal retrodiction on a conservative Hamiltonian system
    cons_pend = NonlinearPendulumAndViscoelasticBrick(damping_default=0.0)
    x0_pend = torch.tensor([[0.5, 0.2]], dtype=torch.float64)
    p_pend = cons_pend.default_params(batch_size=1, dtype=torch.float64)
    roll_pend = rollout_trajectories(cons_pend, x0_pend, p_pend, dt=0.02, num_steps=30, integrator="symplectic_midpoint")

    goal_retro = ScientificInvestigationGoal(
        question="Reconstruct the initial state x(0) from the terminal trajectory via time-reversal retrodiction.",
        domain="mechanics",
        base_brick=cons_pend,
        initial_state=roll_pend.states[:, -1, :],
        inverse_mode="retrodiction",
        dt=0.02,
        num_steps=30,
        crystallize_into_vault=False,
    )
    dossier_retro = council.run_investigation(goal_retro)
    assert dossier_retro.retrodiction_rollout is not None
    reconstructed_x0 = dossier_retro.retrodiction_rollout.states[0, 0]
    assert float(torch.linalg.norm(reconstructed_x0 - x0_pend[0]).item()) < 5e-3

    # Part 3: Two-point boundary-value shooting through the Council
    goal_bvp = ScientificInvestigationGoal(
        question="Solve the two-point boundary-value shooting problem to reach target position q=0.35.",
        domain="mechanics",
        base_brick=cons_pend,
        initial_state=torch.tensor([[0.0, 0.1]], dtype=torch.float64),
        target_state_objective={"p": 0.0},
        inverse_mode="boundary_value_shooting",
        dt=0.02,
        num_steps=25,
        crystallize_into_vault=False,
    )
    dossier_bvp = council.run_investigation(goal_bvp)
    assert dossier_bvp.boundary_shooting_result is not None
    assert dossier_bvp.boundary_shooting_result.terminal_error_norm < 1e-2


def test_council_bistable_bifurcation_tipping_points_and_competing_disambiguation() -> None:
    """
    Verify `AutonomousScientificCouncil` maps multistable equilibria + parameter bifurcations
    on a bistable double-well potential system, and autonomously executes active experimental
    disambiguation when `competing_candidates` are supplied to the investigation goal.
    """
    bistable_spec = SymbolicSystemSpec(
        name="bistable_double_well_switch",
        domain="chemistry",
        state_specs=[
            StateVarSpec("x", DIMENSIONLESS, default_value=1.1, homeostatic_min=-5.0, homeostatic_max=5.0),
        ],
        param_specs=[
            ParamSpec("a_lin", DIMENSIONLESS / s, default_value=1.5, min_value=0.2, max_value=4.0, tunable=True),
            ParamSpec("b_cub", DIMENSIONLESS / s, default_value=1.0, min_value=0.2, max_value=4.0, tunable=True),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.0",
        external_drive_exprs=("a_lin * x - b_cub * (x ** 3)",),
    )
    bistable_brick = compile_symbolic_system(bistable_spec)

    monostable_spec = SymbolicSystemSpec(
        name="monostable_linear_impostor",
        domain="chemistry",
        state_specs=bistable_spec.state_specs,
        param_specs=[
            ParamSpec("a_lin", DIMENSIONLESS / s, default_value=-0.5, min_value=-4.0, max_value=4.0, tunable=True),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.0",
        external_drive_exprs=("a_lin * x",),
    )
    impostor_brick = compile_symbolic_system(monostable_spec)

    p_bi = bistable_brick.default_params(batch_size=1, dtype=torch.float64)
    p_im = impostor_brick.default_params(batch_size=1, dtype=torch.float64)

    cubic_decay_spec = SymbolicSystemSpec(
        name="pure_cubic_decay_impostor",
        domain="chemistry",
        state_specs=bistable_spec.state_specs,
        param_specs=[
            ParamSpec("b_cub", DIMENSIONLESS / s, default_value=1.0, min_value=0.1, max_value=4.0, tunable=True),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.0",
        external_drive_exprs=("-b_cub * (x ** 3)",),
    )
    cubic_impostor_brick = compile_symbolic_system(cubic_decay_spec)
    p_cub_im = cubic_impostor_brick.default_params(batch_size=1, dtype=torch.float64)

    council = AutonomousScientificCouncil()
    goal = ScientificInvestigationGoal(
        question="Disambiguate between the competing candidates and bistable switch, then map its equilibria and bifurcations.",
        domain="chemistry",
        base_brick=impostor_brick,
        competing_candidates=((impostor_brick, p_im), (cubic_impostor_brick, p_cub_im), (bistable_brick, p_bi)),
        oracle_brick=bistable_brick,
        oracle_params=p_bi,
        initial_state=torch.tensor([[0.05]], dtype=torch.float64),
        dt=0.04,
        num_steps=30,
        crystallize_into_vault=False,
    )

    dossier = council.run_investigation(goal)
    assert dossier.disambiguation_report is not None
    assert dossier.disambiguation_report.winning_hypothesis == "candidate_b"
    assert dossier.final_brick.name == "bistable_double_well_switch"
    assert dossier.equilibrium_report is not None
    assert dossier.equilibrium_report.distinct_equilibria.shape[0] >= 2
    assert dossier.bifurcation_report is not None
    assert len(dossier.bifurcation_report.param_values) == 10


def test_open_ended_hypothesis_to_physics_world_model_synthesis_across_domains() -> None:
    """
    Verify Upgrade 5b (Universal Hypothesis-to-Physics Compiler):
    `AutonomousScientificCouncil.run_investigation` accepts an open-ended natural-language
    scientific question with `base_brick=None, candidate_spec_dict=None, state_specs_for_autonomous=()`,
    synthesizes a formal Port-Hamiltonian `SymbolicSystemSpec` (including 3D spatial nodes,
    GENERIC thermal sinks, or Onsager cross-transport), compiles it via `compile_with_self_repair`,
    executes the full Council investigation, shields collateral states, and passes falsification.
    """
    council = AutonomousScientificCouncil()

    # 1. Open-ended 3D orbital/spaceflight hypothesis prompt
    goal_space = ScientificInvestigationGoal(
        question="Synthesize a 3D orbital spacecraft trajectory model and find a parameter intervention to reach target orbit.",
        domain="astrophysics_and_aerospace",
        dt=0.03,
        num_steps=25,
        crystallize_into_vault=False,
    )
    dossier_space = council.run_investigation(goal_space)
    assert len(dossier_space.final_brick.spatial_nodes_3d) >= 1
    assert dossier_space.self_repair_report is not None
    assert dossier_space.intervention_solution is not None
    assert dossier_space.falsification_report is not None and dossier_space.falsification_report.passed_all_gates

    # 2. Open-ended GENERIC thermal/entropy conservation prompt
    goal_thermo = ScientificInvestigationGoal(
        question="Model thermal heat dissipation and entropy conservation in a mechanical engine and optimize damping.",
        domain="thermodynamics",
        dt=0.03,
        num_steps=25,
        crystallize_into_vault=False,
    )
    dossier_thermo = council.run_investigation(goal_thermo)
    assert dossier_thermo.final_brick.state_dim >= 2
    assert dossier_thermo.falsification_report is not None and dossier_thermo.falsification_report.passed_all_gates

    # 3. Open-ended coupled transport / cellular homeostasis prompt with collateral shielding
    goal_bio = ScientificInvestigationGoal(
        question="Figure out a selective restorative intervention for cellular transport without collateral drift.",
        domain="biophysics",
        dt=0.03,
        num_steps=25,
        crystallize_into_vault=False,
    )
    dossier_bio = council.run_investigation(goal_bio)
    assert dossier_bio.intervention_solution is not None
    assert dossier_bio.intervention_solution.intervened_loss < dossier_bio.intervention_solution.baseline_loss
    assert dossier_bio.falsification_report is not None and dossier_bio.falsification_report.passed_all_gates
    assert dossier_bio.falsification_report.lyapunov_certificate is not None
    assert dossier_bio.falsification_report.lyapunov_certificate.certified_basin_radius > 0.0


def test_formal_lyapunov_basin_of_attraction_certificate() -> None:
    """
    Verify Horizon 5 (`compute_lyapunov_stability_certificate`):
    Solves `A_eff^T P + P A_eff = -I` (`P = P^T > 0`) at the equilibrium `x^*` and verifies
    `dV/dt(x) <= 0` across concentric radial shells to certify `certified_basin_radius > 0`.
    """
    from cortivra.layer6_agents import compute_lyapunov_stability_certificate

    pend = NonlinearPendulumAndViscoelasticBrick(damping_default=0.4)
    p = pend.default_params(batch_size=1, dtype=torch.float64)
    cert = compute_lyapunov_stability_certificate(
        brick=pend,
        params=p,
        num_radial_shells=5,
        max_radius=1.2,
    )
    assert cert.is_locally_asymptotically_stable
    assert cert.is_dissipative_across_shells
    assert cert.certified_basin_radius >= 1.0
    assert cert.min_eigenvalue_p > 0.0
    assert all(dv <= 1e-5 for dv in cert.shell_max_dv_dt)

