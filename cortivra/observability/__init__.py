"""
Causal provenance graph, ambient flight recorder, and universal contradiction backtracker
(`cortivra.observability`).
"""

from cortivra.observability.causal_backtracker import (
    CausalContradictionReport,
    DecisionVulnerabilityNode,
    PipelineFractureEdge,
    TermForceAttribution,
    analyze_trace_and_backtrack_contradictions,
    attribute_state_space_contradiction,
)
from cortivra.observability.flight_recorder import (
    AlternativeBranchRecord,
    CausalTraceDAG,
    CausalTraceNode,
    FlightRecorder,
    LayerId,
    OperatorSpectrumSummary,
    active_recorder,
    emit_trace_node,
    summarize_singular_spectrum,
    suppress_inner_trace,
    trace_span,
)
from cortivra.observability.forensic_report import (
    format_forensic_xray_report,
    serialize_trace_to_dict,
    serialize_trace_to_jsonl,
)

__all__ = [
    "LayerId",
    "OperatorSpectrumSummary",
    "AlternativeBranchRecord",
    "CausalTraceNode",
    "CausalTraceDAG",
    "FlightRecorder",
    "active_recorder",
    "emit_trace_node",
    "suppress_inner_trace",
    "trace_span",
    "summarize_singular_spectrum",
    "TermForceAttribution",
    "PipelineFractureEdge",
    "DecisionVulnerabilityNode",
    "CausalContradictionReport",
    "attribute_state_space_contradiction",
    "analyze_trace_and_backtrack_contradictions",
    "format_forensic_xray_report",
    "serialize_trace_to_dict",
    "serialize_trace_to_jsonl",
]
