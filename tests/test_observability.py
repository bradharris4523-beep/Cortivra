"""
Verification suite for `cortivra.observability` (FlightRecorder, CausalTraceDAG,
Merkle Provenance Chain, Native Layer 3 AST Force & Adjoint Attribution,
Universal Causal Contradiction Backtracker, and Forensic X-Ray Report).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cortivra.layer1_axioms.hamiltonian import ParamSpec, StateVarSpec
from cortivra.layer1_axioms.shields import analyze_unexplained_residual
from cortivra.layer1_axioms.units import (
    DIMENSIONLESS,
    J,
    N_force,
    discover_dimensionless_groups,
    kg,
    m,
    s,
)
from cortivra.layer1_axioms.vault import LawVault
from cortivra.layer2_fabric import register_foundational_laws
from cortivra.layer3_compiler.symbolic_compiler import (
    SymbolicSystemSpec,
    compile_residual_patch_brick,
    compile_symbolic_system,
)
from cortivra.layer4_engine.inverse_solver import (
    solve_boundary_value_shooting,
    solve_optimal_control_waveform,
)
from cortivra.layer4_engine.rollout_arena import (
    discover_equilibrium_landscape,
    retrodict_trajectory,
    rollout_trajectories,
    scan_parameter_bifurcation,
)
from cortivra.layer5_discovery.latent_discovery import (
    detect_non_markovian_memory,
    discover_autonomous_symbolic_system,
    discover_latent_state_augmentation,
)
from cortivra.layer6_agents import (
    AutonomousScientificCouncil,
    MultiProviderMetaClient,
    ScientificInvestigationGoal,
    compile_with_self_repair,
)
from cortivra.observability import (
    AlternativeBranchRecord,
    CausalTraceNode,
    FlightRecorder,
    analyze_trace_and_backtrack_contradictions,
    attribute_state_space_contradiction,
    emit_trace_node,
    format_forensic_xray_report,
    serialize_trace_to_dict,
    serialize_trace_to_jsonl,
    summarize_singular_spectrum,
    trace_span,
)


def test_ambient_flight_recorder_context_and_merkle_provenance_integrity() -> None:
    """
    Verify `emit_trace_node` is a no-op outside a `FlightRecorder.session` and builds a
    cryptographically chained `CausalTraceDAG` with tamper-evident SHA-256 Merkle hashes
    inside a session.
    """
    assert emit_trace_node("layer1_axioms", "noop_check", "target") is None

    streamed: list[CausalTraceNode] = []
    with FlightRecorder.session(
        session_id="test_dag_session",
        question="Verify hierarchical span, sequential DAG edges, and Merkle hashes.",
        on_node_recorded=lambda node: streamed.append(node),
    ) as recorder:
        with trace_span("layer6_agents", "outer_council_step", "oscillator") as span_node:
            assert span_node is not None
            n1 = emit_trace_node(
                layer="layer3_compiler",
                operation="compile_symbolic_system",
                target_name="oscillator",
                selected_expression="0.5 * k * q^2",
            )
            n2 = emit_trace_node(
                layer="layer1_axioms",
                operation="audit_law_brick",
                target_name="oscillator",
                physical_violation_metric=0.0,
            )

    assert len(recorder.dag.nodes) == 3
    assert len(streamed) == 3
    assert n1 is not None and n2 is not None
    assert span_node.node_id in n1.parent_ids
    assert n1.node_id in n2.parent_ids
    ancestors = recorder.dag.ancestors_of(n2.node_id)
    assert [a.node_id for a in ancestors] == [n1.node_id, span_node.node_id]

    assert len(n1.merkle_hash) == 64
    assert len(n2.merkle_hash) == 64
    assert len(recorder.dag.merkle_root_hash) == 64
    assert recorder.dag.verify_merkle_integrity()

    original_expr = n1.selected_expression
    n1.selected_expression = "tampered_expression"
    assert not recorder.dag.verify_merkle_integrity()
    n1.selected_expression = original_expr
    assert recorder.dag.verify_merkle_integrity()


def test_native_layer3_ast_force_and_cumulative_adjoint_attribution() -> None:
    """
    Verify `summarize_singular_spectrum` extracts exact condition numbers and nullspaces,
    and `attribute_state_space_contradiction` decomposes native Layer 3 `ExprNode` trees
    across `CompiledLawBrick` and `ResidualPatchedLawBrick` while computing trajectory-integrated
    adjoint parameter sensitivities.
    """
    mat = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1e-7],
        ],
        dtype=torch.float64,
    )
    spec_sum = summarize_singular_spectrum(mat, operator_name="TestOperator", null_abs_tol=1e-4)
    assert spec_sum.matrix_shape == (3, 3)
    assert spec_sum.effective_rank == 2
    assert spec_sum.nullspace_dimension == 1
    assert spec_sum.condition_number > 1e6

    # Build a base linear reactor and patch it with an explosive cubic term via ResidualPatchedLawBrick
    base_spec = SymbolicSystemSpec(
        name="base_linear_reactor",
        domain="chemistry",
        state_specs=[
            StateVarSpec("x", DIMENSIONLESS, default_value=1.2, homeostatic_min=-5.0, homeostatic_max=5.0),
        ],
        param_specs=[
            ParamSpec("k_lin", DIMENSIONLESS / s, default_value=0.2),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.5 * (x ** 2)",
        dissipation_diag_exprs=("k_lin",),
    )
    base_brick = compile_symbolic_system(base_spec)
    patched_brick = compile_residual_patch_brick(
        name="patched_unstable_cubic_reactor",
        base_brick=base_brick,
        patch_param_specs=[ParamSpec("k_explosion", DIMENSIONLESS / s, default_value=2.5)],
        thermodynamic_mode="active_driven",
        patch_expressions={0: "k_explosion * (x ** 3)"},
    )
    x0 = torch.tensor([[1.4]], dtype=torch.float64)
    p = patched_brick.default_params(batch_size=1, dtype=torch.float64)
    roll = rollout_trajectories(patched_brick, x0, p, dt=0.02, num_steps=20, integrator="rk4")

    attr = attribute_state_space_contradiction(
        brick=patched_brick,
        states=roll.states[0],
        params=p,
        dt=0.02,
    )
    assert attr.contradiction_state_name == "x"
    assert "k_explosion" in attr.dominant_term_or_parameter
    assert attr.dominant_contribution_fraction > 0.85
    assert attr.local_jacobian_max_real_eigenvalue > 5.0
    assert attr.parameter_sensitivities["k_explosion"] > attr.parameter_sensitivities["k_lin"] * 5.0
    assert "k_explosion" in attr.cumulative_trajectory_sensitivities
    assert (
        attr.cumulative_trajectory_sensitivities["k_explosion"]
        > attr.cumulative_trajectory_sensitivities["k_lin"]
    )
    assert attr.max_energy_power_residual < 1e-9


def test_causal_backtracker_traces_downstream_failure_to_upstream_ambiguous_decision() -> None:
    """
    Verify `analyze_trace_and_backtrack_contradictions` detects a downstream global rollout
    contradiction and traces backward up the `CausalTraceDAG` to pinpoint the upstream
    ill-conditioned / narrow-margin symbolic selection node and its runner-up alternative.
    """
    impostor_spec = SymbolicSystemSpec(
        name="impostor_divergent_polynomial",
        domain="mechanics",
        state_specs=[
            StateVarSpec("x", DIMENSIONLESS, default_value=1.5, homeostatic_min=-4.0, homeostatic_max=4.0),
        ],
        param_specs=[
            ParamSpec("c_bad", DIMENSIONLESS / s, default_value=1.8),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.0",
        external_drive_exprs=("c_bad * (x ** 3)",),
    )
    impostor_brick = compile_symbolic_system(impostor_spec)
    p = impostor_brick.default_params(batch_size=1, dtype=torch.float64)
    roll = rollout_trajectories(
        impostor_brick,
        torch.tensor([[1.3]], dtype=torch.float64),
        p,
        dt=0.02,
        num_steps=15,
        integrator="rk4",
    )

    with FlightRecorder.session(
        session_id="synthetic_contradiction_trace",
        question="Trace why trajectory diverged at 4x horizon.",
    ) as rec:
        emit_trace_node(
            layer="layer4_engine",
            operation="estimate_derivatives",
            target_name="raw_trajectory",
            local_r2=0.999,
        )
        ill_cond_mat = torch.tensor([[1.0, 1.0], [1.0, 1.0 + 1e-6]], dtype=torch.float64)
        op_spec = summarize_singular_spectrum(ill_cond_mat, operator_name="WeakIntegralDesign")
        pareto_node = emit_trace_node(
            layer="layer5_discovery",
            operation="solve_sparse_pareto_front",
            target_name="active_driven",
            local_error_metric=1e-4,
            local_r2=0.992,
            decision_margin=0.22,
            selected_branch_id="poly_cubic_cand",
            selected_expression="c_bad * (x ** 3)",
            operator_spectrum=op_spec,
            alternatives=[
                AlternativeBranchRecord(
                    branch_id="pade_saturating_cand",
                    expression_or_summary="-v_max * x / (1 + k_m * x^2)",
                    objective_score=-142.1,
                    local_r2=0.991,
                    complexity=5,
                )
            ],
        )
        fail_node = emit_trace_node(
            layer="layer6_agents",
            operation="run_falsification_stress_test",
            target_name="impostor_divergent_polynomial",
            status="failed",
            local_r2=0.985,
            global_r2=0.12,
            global_error_metric=45.0,
            physical_violation_metric=12.4,
        )

    assert pareto_node is not None and fail_node is not None
    report = analyze_trace_and_backtrack_contradictions(
        dag=rec.dag,
        brick=impostor_brick,
        states=roll.states[0],
        params=p,
    )

    assert report.contradiction_detected
    assert report.failing_node_id == fail_node.node_id
    assert len(report.pipeline_fractures) >= 1
    assert report.primary_root_cause is not None
    assert report.primary_root_cause.node_id == pareto_node.node_id
    assert report.primary_root_cause.recommended_alternative is not None
    assert report.primary_root_cause.recommended_alternative.branch_id == "pade_saturating_cand"
    assert "c_bad" in report.causal_chain_summary
    assert "narrow decision margin" in report.causal_chain_summary

    xray_md = format_forensic_xray_report(rec.dag, report)
    assert "## 1. Causal Computation Graph Timeline" in xray_md
    assert "## 2. Operator Spectral Conditioning & Nullspace Geometry" in xray_md
    assert "## 3. Epistemic Decision Branches & Runner-Up Alternatives" in xray_md
    assert "## 4. Causal Contradiction & Root-Cause Backtracker" in xray_md
    assert "pade_saturating_cand" in xray_md
    assert "Verified=`True`" in xray_md

    jsonl_str = serialize_trace_to_jsonl(rec.dag, report)
    lines = jsonl_str.strip().splitlines()
    assert len(lines) == 4
    parsed_last = json.loads(lines[-1])
    assert parsed_last["record_type"] == "causal_contradiction_report"
    assert parsed_last["merkle_root_hash"] == rec.dag.merkle_root_hash


def test_all_layers_emit_connected_telemetry_into_flight_recorder(tmp_path: Path) -> None:
    """
    Verify that operations across Layer 1 (Buckingham-Pi, residual analysis), Layer 2 (LawVault
    foundational registration), Layer 3 (symbolic & residual patch compilation), Layer 4
    (equilibria, bifurcations, retrodiction, BVP shooting, optimal control), Layer 5
    (non-Markovian hysteresis, latent augmentation, autonomous discovery), and Layer 6
    (symbolic self-repair) all emit connected `CausalTraceNode`s when a session is active.
    """
    with FlightRecorder.session(
        session_id="all_layers_connected_session",
        question="Verify 100% layer connectivity across Layers 1-6.",
    ) as rec:
        groups = discover_dimensionless_groups([("m", kg), ("k", N_force / m), ("t", s)])
        assert len(groups) == 1

        vault = LawVault(tmp_path / "layer2_vault")
        register_foundational_laws(vault)

        broken_spec_dict = {
            "name": "self_repaired_oscillator",
            "domain": "mechanics",
            "state_specs": [
                {"name": "q", "unit": "m", "default_value": 1.0, "homeostatic_min": -10.0, "homeostatic_max": 10.0},
                {"name": "p", "unit": "N*s", "default_value": 0.0, "homeostatic_min": -10.0, "homeostatic_max": 10.0},
            ],
            "param_specs": [
                {"name": "k_s", "unit": "1", "default_value": 2.0, "min_value": 0.1, "max_value": 10.0},
                {"name": "m_osc", "unit": "kg", "default_value": 1.0, "min_value": 0.1, "max_value": 10.0},
                {"name": "gamma", "unit": "s/kg", "default_value": 0.2, "min_value": 0.0, "max_value": 5.0},
            ],
            "time_unit": "s",
            "energy_unit": "J",
            "hamiltonian_expr": "0.5 * k_s * (q ** 2) + 0.5 * (p ** 2) / m_osc",
            "dissipation_diag_exprs": ["0.0", "gamma"],
            "skew_routing_exprs": {"0,1": "1.0"},
        }
        repair_rep = compile_with_self_repair(broken_spec_dict, client=MultiProviderMetaClient())
        brick = repair_rep.compiled_brick
        p = brick.default_params(batch_size=1, dtype=torch.float64)
        x0 = torch.tensor([[1.2, 0.0]], dtype=torch.float64)

        roll = rollout_trajectories(brick, x0, p, dt=0.05, num_steps=15, integrator="rk4")
        analyze_unexplained_residual(
            brick=brick,
            state=roll.states[0],
            params=p.expand(roll.states.shape[1], -1),
            observed_dx_dt=roll.flows[0] + 0.5 * (roll.states[0] ** 2),
            parameter_fit_exhausted=True,
        )

        discover_equilibrium_landscape(brick, p, num_probes=6)
        scan_parameter_bifurcation(
            brick,
            param_name="k_s",
            param_values=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            initial_guess=torch.zeros((1, 2), dtype=torch.float64),
            base_params=p,
        )
        retrodict_trajectory(brick, roll.states[:, -1, :], p, dt=0.05, num_steps=5)
        solve_boundary_value_shooting(
            brick=brick,
            known_initial_indices=[0],
            known_initial_values=torch.tensor([1.0], dtype=torch.float64),
            target_final_indices=[0],
            target_final_values=torch.tensor([0.5], dtype=torch.float64),
            params=p,
            dt=0.05,
            num_steps=8,
            max_iters=12,
        )
        solve_optimal_control_waveform(
            brick=brick,
            initial_state=x0,
            params=p,
            dt=0.05,
            num_steps=8,
            target_loss_fn=lambda r: torch.mean(r.states[:, -1, :] ** 2),
            max_iters=10,
        )

        t_grid = torch.linspace(0.0, 1.0, 20, dtype=torch.float64).unsqueeze(-1)
        states_nm = torch.cat([torch.sin(6.0 * t_grid), torch.cos(6.0 * t_grid)], dim=-1)
        res_nm = torch.cat([torch.exp(-t_grid), torch.zeros_like(t_grid)], dim=-1)
        detect_non_markovian_memory(states_nm, res_nm, dt=0.05)
        discover_latent_state_augmentation(
            base_brick=brick,
            observed_states=states_nm,
            base_params=p,
            residual_flow=res_nm,
            dt=0.05,
            target_state_index=0,
        )
        discover_autonomous_symbolic_system(
            name="auto_test_sys",
            domain="mechanics",
            state_specs=[StateVarSpec("x", DIMENSIONLESS, default_value=1.0)],
            observed_states=torch.exp(-0.5 * t_grid),
            dt=0.05,
            max_polynomial_degree=1,
            max_terms_per_state=1,
            include_rational_primitives=False,
            include_trigonometric=False,
        )

    ops = {n.operation for n in rec.dag.nodes}
    expected_ops = {
        "discover_dimensionless_groups",
        "register_foundational_laws",
        "compile_with_self_repair",
        "compile_symbolic_system",
        "audit_law_brick",
        "analyze_unexplained_residual",
        "discover_equilibrium_landscape",
        "scan_parameter_bifurcation",
        "retrodict_trajectory",
        "solve_boundary_value_shooting",
        "solve_optimal_control_waveform",
        "detect_non_markovian_memory",
        "discover_latent_state_augmentation",
        "discover_autonomous_symbolic_system",
    }
    assert expected_ops.issubset(ops)
    assert rec.dag.verify_merkle_integrity()


def test_end_to_end_council_investigation_populates_full_causal_trace_dag(tmp_path: Path) -> None:
    """
    Verify running `AutonomousScientificCouncil.run_investigation` automatically records
    causal nodes across Layers 1, 2, 3, 4, 5, and 6, attaches `causal_trace_dag` and
    `contradiction_report` to the `ScientificDiscoveryDossier`, and renders the Forensic X-Ray.
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
    gt_roll = rollout_trajectories(true_brick, x0, p_true, dt=dt, num_steps=40, integrator="rk4")

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

    vault = LawVault(tmp_path / "obs_vault")
    register_foundational_laws(vault)
    council = AutonomousScientificCouncil(vault=vault)

    goal = ScientificInvestigationGoal(
        question="Calibrate oscillator and discover unmodeled damping with full causal trace.",
        domain="mechanics",
        base_brick=base_brick,
        observed_states=gt_roll.states[0],
        observed_dx_dt=gt_roll.flows[0],
        dt=dt,
        num_steps=40,
        crystallize_into_vault=True,
    )

    dossier = council.run_investigation(goal)
    assert dossier.causal_trace_dag is not None
    assert dossier.contradiction_report is not None
    assert dossier.causal_trace_dag.verify_merkle_integrity()

    recorded_layers = {n.layer for n in dossier.causal_trace_dag.nodes}
    assert {"layer1_axioms", "layer3_compiler", "layer4_engine", "layer5_discovery", "layer6_agents"}.issubset(
        recorded_layers
    )

    xray = dossier.to_forensic_xray_report()
    assert "Forensic Scientific Flight Recorder X-Ray" in xray
    assert "solve_sparse_pareto_front" in xray
    assert "run_elimination_funnel" in xray
    assert "crystallize_verified_law" in xray

    serialized = serialize_trace_to_dict(dossier.causal_trace_dag, dossier.contradiction_report)
    assert serialized["num_nodes"] == len(dossier.causal_trace_dag.nodes)
    assert serialized["merkle_integrity_verified"] is True
