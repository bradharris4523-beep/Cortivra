"""
JSONL machine serializer and Markdown Forensic X-Ray Report generator
(`cortivra.observability.forensic_report`).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from cortivra.observability.causal_backtracker import CausalContradictionReport
from cortivra.observability.flight_recorder import CausalTraceDAG


def serialize_trace_to_dict(
    dag: CausalTraceDAG,
    contradiction_report: CausalContradictionReport | None = None,
) -> dict[str, Any]:
    """
    Serialize `dag` and its optional `contradiction_report` into a JSON-compatible dictionary.
    """
    total_ms = sum(n.duration_ms for n in dag.nodes)
    return {
        "session_id": dag.session_id,
        "question": dag.question,
        "created_at_epoch_s": dag.created_at_epoch_s,
        "merkle_root_hash": dag.merkle_root_hash,
        "merkle_integrity_verified": dag.verify_merkle_integrity(),
        "num_nodes": len(dag.nodes),
        "total_duration_ms": round(total_ms, 3),
        "nodes": [n.to_dict() for n in dag.nodes],
        "contradiction_report": (
            contradiction_report.to_dict() if contradiction_report is not None else None
        ),
    }


def serialize_trace_to_jsonl(
    dag: CausalTraceDAG,
    contradiction_report: CausalContradictionReport | None = None,
) -> str:
    """
    Serialize every node in `dag` (plus the summary `contradiction_report`) as JSON Lines.
    """
    lines: list[str] = []
    for node in dag.nodes:
        payload = {"record_type": "causal_trace_node", **node.to_dict()}
        lines.append(json.dumps(payload, default=str))
    if contradiction_report is not None:
        lines.append(
            json.dumps(
                {
                    "record_type": "causal_contradiction_report",
                    "merkle_root_hash": dag.merkle_root_hash,
                    **asdict(contradiction_report),
                },
                default=str,
            )
        )
    return "\n".join(lines)


def format_forensic_xray_report(
    dag: CausalTraceDAG,
    contradiction_report: CausalContradictionReport | None = None,
) -> str:
    """
    Generate a human-readable Markdown Forensic X-Ray Report detailing:
        1. Causal Computation Graph timeline across Layers 1–6 (with SHA-256 Merkle provenance).
        2. Operator Spectral Conditioning & Nullspace Geometry table.
        3. Decision Branches & Runner-Up Alternatives table.
        4. Universal Causal Contradiction & Root-Cause Backtracking diagnosis.
    """
    total_ms = sum(n.duration_ms for n in dag.nodes)
    integrity = dag.verify_merkle_integrity()
    lines: list[str] = [
        f"# Forensic Scientific Flight Recorder X-Ray: `{dag.session_id}`",
        (
            f"**Question:** {dag.question or 'N/A'} | **Causal Nodes:** `{len(dag.nodes)}` | "
            f"**Total Time:** `{total_ms:.2f} ms` | **Merkle Root:** `{dag.merkle_root_hash[:16]}...` "
            f"(Verified=`{integrity}`)"
        ),
        "",
        "## 1. Causal Computation Graph Timeline",
        "| Node ID | Layer | Operation | Target | Status | Local $R^2$ | Global $R^2$ | Violation | Margin | Merkle | Time (ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for n in dag.nodes:
        loc_r2 = f"{n.local_r2:.5f}" if n.local_r2 is not None else "-"
        glo_r2 = f"{n.global_r2:.5f}" if n.global_r2 is not None else "-"
        viol = f"{n.physical_violation_metric:.2e}" if n.physical_violation_metric > 0.0 else "0.0"
        marg = f"{n.decision_margin:.4f}" if n.decision_margin is not None else "-"
        short_hash = n.merkle_hash[:8] if n.merkle_hash else "-"
        lines.append(
            f"| `{n.node_id}` | `{n.layer}` | `{n.operation}` | `{n.target_name}` | "
            f"`{n.status}` | `{loc_r2}` | `{glo_r2}` | `{viol}` | `{marg}` | `{short_hash}` | `{n.duration_ms:.2f}` |"
        )
    lines.append("")

    spectral_nodes = [n for n in dag.nodes if n.operator_spectrum is not None]
    if spectral_nodes:
        lines.append("## 2. Operator Spectral Conditioning & Nullspace Geometry")
        lines.append(
            "| Node ID | Operator | Shape | Condition $\\kappa$ | Effective Rank | Nullspace Dim | Top Singular Values |"
        )
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for n in spectral_nodes:
            sp = n.operator_spectrum
            assert sp is not None
            top_sv = ", ".join(f"{v:.3g}" for v in sp.singular_values[:5])
            lines.append(
                f"| `{n.node_id}` | `{sp.operator_name}` | `{sp.matrix_shape}` | "
                f"`{sp.condition_number:.3e}` | `{sp.effective_rank}` | `{sp.nullspace_dimension}` | `[{top_sv}]` |"
            )
        lines.append("")

    branch_nodes = [n for n in dag.nodes if n.selected_expression or n.alternatives]
    if branch_nodes:
        lines.append("## 3. Epistemic Decision Branches & Runner-Up Alternatives")
        for n in branch_nodes:
            marg_txt = (
                f" (Decision Margin $\\Delta = {n.decision_margin:.4f}$)"
                if n.decision_margin is not None
                else ""
            )
            lines.append(
                f"- **`{n.node_id}` (`{n.operation}` on `{n.target_name}`){marg_txt}:** "
                f"Selected `{n.selected_branch_id or 'winner'}` -> `{n.selected_expression or 'N/A'}`"
            )
            for idx_a, alt in enumerate(n.alternatives[:4], start=1):
                lines.append(
                    f"  - *Runner-up #{idx_a}* (`{alt.branch_id}`): `{alt.expression_or_summary}` "
                    f"(score=`{alt.objective_score:.4f}`, $R^2$=`{alt.local_r2:.5f}`, complexity=`{alt.complexity}`)"
                )
        lines.append("")

    if contradiction_report is not None:
        lines.append("## 4. Causal Contradiction & Root-Cause Backtracker")
        lines.append(f"- **Contradiction Detected:** `{contradiction_report.contradiction_detected}`")
        lines.append(f"- **Contradiction Classification:** `{contradiction_report.contradiction_type}`")
        lines.append(f"- **Causal Chain Summary:** {contradiction_report.causal_chain_summary}")

        fa = contradiction_report.force_attribution
        if fa is not None:
            lines.append(
                f"- **State-Space Force Attribution at $t^* = {fa.contradiction_step_index}$ "
                f"(State `{fa.contradiction_state_name}`, $x(t^*) = {fa.state_vector_at_contradiction}$):**"
            )
            lines.append(
                f"  - Dominant Term / Parameter: **`{fa.dominant_term_or_parameter}`** "
                f"(`{fa.dominant_contribution_fraction * 100.0:.1f}%` of force, "
                f"Max Real Jacobian Eigenvalue $\\text{{Re}}(\\lambda_{{\\max}}) = {fa.local_jacobian_max_real_eigenvalue:+.4f}$)"
            )
            lines.append(
                f"  - Thermodynamic Flow Split: Conservative=`{fa.conservative_flow_share * 100.0:.1f}%`, "
                f"Dissipative=`{fa.dissipative_flow_share * 100.0:.1f}%`, "
                f"External=`{fa.external_drive_flow_share * 100.0:.1f}%` "
                f"(Max Power Residual=`{fa.max_energy_power_residual:.2e}`)"
            )
            if fa.term_contributions:
                lines.append(f"  - Sub-Term Forces: `{fa.term_contributions}`")
            if fa.parameter_sensitivities:
                lines.append(
                    f"  - Instantaneous Parameter Sensitivities $|\\theta_j \\partial F / \\partial \\theta_j|$: `{fa.parameter_sensitivities}`"
                )
            if fa.cumulative_trajectory_sensitivities:
                lines.append(
                    f"  - Cumulative Trajectory Adjoint Sensitivities $|\\theta_j \\partial x(t^*) / \\partial \\theta_j|$: `{fa.cumulative_trajectory_sensitivities}`"
                )

        if contradiction_report.pipeline_fractures:
            lines.append("- **Stage-Transition Monotonicity Fractures:**")
            for fr in contradiction_report.pipeline_fractures:
                lines.append(f"  - {fr.fracture_summary}")

        if contradiction_report.root_cause_nodes:
            lines.append("- **Backtracked Upstream Decision Vulnerabilities (Ranked):**")
            for rc in contradiction_report.root_cause_nodes[:4]:
                lines.append(
                    f"  - `[score={rc.vulnerability_score:.3f}]` {rc.causal_explanation}"
                )

    return "\n".join(lines)
