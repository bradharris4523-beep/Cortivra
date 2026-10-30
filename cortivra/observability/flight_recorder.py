"""
Ambient context-propagated scientific flight recorder and causal computation graph
(`cortivra.observability.flight_recorder`).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Literal, Sequence

import torch

LayerId = Literal[
    "layer1_axioms",
    "layer2_fabric",
    "layer3_compiler",
    "layer4_engine",
    "layer5_discovery",
    "layer6_agents",
    "layer7_interface",
]


@dataclass(frozen=True)
class OperatorSpectrumSummary:
    """
    Spectral conditioning and singular/eigenvalue geometry of a linear or linearized
    mathematical operator (`A^T A`, Fisher `J_theta^T J_theta`, Jacobian `dF/dx`, Lie `L_F`,
    or SI Dimensional Matrix `D`).
    """

    operator_name: str
    matrix_shape: tuple[int, int]
    singular_values: tuple[float, ...]
    condition_number: float
    effective_rank: int
    spectral_gap_ratio: float
    nullspace_dimension: int


@dataclass(frozen=True)
class AlternativeBranchRecord:
    """
    Runner-up hypothesis, parameter vector, or symbolic expression evaluated at a decision node.
    """

    branch_id: str
    expression_or_summary: str
    objective_score: float
    local_r2: float
    complexity: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalTraceNode:
    """
    Single mathematical operation or epistemic decision node in the `CausalTraceDAG`.
    """

    node_id: str
    parent_ids: tuple[str, ...]
    layer: LayerId
    operation: str
    target_name: str
    started_at_ns: int
    duration_ms: float = 0.0
    status: str = "completed"
    local_error_metric: float | None = None
    local_r2: float | None = None
    global_error_metric: float | None = None
    global_r2: float | None = None
    physical_violation_metric: float = 0.0
    decision_margin: float | None = None
    selected_branch_id: str | None = None
    selected_expression: str | None = None
    operator_spectrum: OperatorSpectrumSummary | None = None
    alternatives: tuple[AlternativeBranchRecord, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()
    merkle_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_node_merkle_hash(
    node: CausalTraceNode,
    parent_hashes: Sequence[str],
) -> str:
    """
    Compute deterministic SHA-256 Merkle provenance hash chaining `parent_hashes` with
    the mathematical content of `node`.
    """
    payload = {
        "node_id": node.node_id,
        "parent_ids": list(node.parent_ids),
        "parent_hashes": list(parent_hashes),
        "layer": node.layer,
        "operation": node.operation,
        "target_name": node.target_name,
        "status": node.status,
        "local_error_metric": node.local_error_metric,
        "local_r2": node.local_r2,
        "global_error_metric": node.global_error_metric,
        "global_r2": node.global_r2,
        "physical_violation_metric": node.physical_violation_metric,
        "decision_margin": node.decision_margin,
        "selected_branch_id": node.selected_branch_id,
        "selected_expression": node.selected_expression,
        "operator_spectrum": asdict(node.operator_spectrum) if node.operator_spectrum else None,
        "alternatives": [asdict(a) for a in node.alternatives],
        "metrics": node.metrics,
        "diagnostics": list(node.diagnostics),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass
class SimulationTelemetryCounters:
    """
    Real-time hardware simulation, Quasi-Monte Carlo (Sobol) universe sampling,
    batched ODE rollout, Newton-Raphson Jacobian, and adjoint backpropagation counters.
    """

    ode_rollout_calls: int = 0
    trajectories_integrated: int = 0
    integration_steps_executed: int = 0
    sobol_qmc_universes_sampled: int = 0
    newton_jacobian_solves: int = 0
    adjoint_gradient_passes: int = 0
    stochastic_langevin_trajectories: int = 0
    integrator_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CausalTraceDAG:
    """
    Directed Acyclic Causal Computation Graph with SHA-256 Merkle provenance chain.
    """

    session_id: str
    question: str = ""
    created_at_epoch_s: float = field(default_factory=time.time)
    nodes: list[CausalTraceNode] = field(default_factory=list)
    simulation_counters: SimulationTelemetryCounters = field(
        default_factory=SimulationTelemetryCounters
    )
    _node_index: dict[str, CausalTraceNode] = field(default_factory=dict, repr=False)

    def add_node(self, node: CausalTraceNode) -> CausalTraceNode:
        self.refresh_node_hash(node)
        self.nodes.append(node)
        self._node_index[node.node_id] = node
        return node

    def refresh_node_hash(self, node: CausalTraceNode) -> str:
        parent_hashes = [
            self._node_index[p_id].merkle_hash
            for p_id in node.parent_ids
            if p_id in self._node_index
        ]
        node.merkle_hash = _compute_node_merkle_hash(node, parent_hashes)
        return node.merkle_hash

    @property
    def merkle_root_hash(self) -> str:
        """Return the aggregate SHA-256 Merkle root hash across all nodes in the DAG."""
        if not self.nodes:
            return hashlib.sha256(b"").hexdigest()
        combined = ":".join(n.merkle_hash for n in self.nodes).encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def verify_merkle_integrity(self) -> bool:
        """
        Recompute every node's SHA-256 Merkle hash from its parents and verify that no
        node or causal link in the DAG has been tampered with.
        """
        verified_hashes: dict[str, str] = {}
        for node in self.nodes:
            p_hashes = [verified_hashes.get(p_id, "") for p_id in node.parent_ids if p_id in verified_hashes]
            expected = _compute_node_merkle_hash(node, p_hashes)
            if node.merkle_hash != expected:
                return False
            verified_hashes[node.node_id] = expected
        return True

    def get_node(self, node_id: str) -> CausalTraceNode | None:
        return self._node_index.get(node_id)

    def ancestors_of(self, node_id: str) -> list[CausalTraceNode]:
        """
        Return all upstream causal ancestors of `node_id` ordered from immediate parent
        backward to the root nodes.
        """
        visited: set[str] = set()
        ordered: list[CausalTraceNode] = []
        queue: list[str] = []
        start_node = self.get_node(node_id)
        if start_node is None:
            return []
        queue.extend(start_node.parent_ids)

        while queue:
            curr_id = queue.pop(0)
            if curr_id in visited:
                continue
            visited.add(curr_id)
            curr_node = self.get_node(curr_id)
            if curr_node is not None:
                ordered.append(curr_node)
                for p_id in curr_node.parent_ids:
                    if p_id not in visited:
                        queue.append(p_id)
        return ordered

    def children_of(self, node_id: str) -> list[CausalTraceNode]:
        return [n for n in self.nodes if node_id in n.parent_ids]


def summarize_singular_spectrum(
    matrix_or_svals: torch.Tensor,
    operator_name: str = "operator",
    matrix_shape: tuple[int, int] | None = None,
    rank_rel_tol: float = 1e-4,
    null_abs_tol: float = 1e-3,
) -> OperatorSpectrumSummary:
    """
    Compute the singular value spectrum, condition number, effective numerical rank,
    spectral gap ratio, and nullspace dimension of a tensor operator or singular value vector.
    """
    t = torch.nan_to_num(matrix_or_svals.detach().to(torch.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    if t.ndim == 1:
        s_vals = torch.sort(torch.abs(t), descending=True).values
        shape = matrix_shape if matrix_shape is not None else (int(t.numel()), int(t.numel()))
    elif t.ndim == 2:
        shape = (int(t.shape[0]), int(t.shape[1]))
        if t.numel() == 0:
            s_vals = torch.zeros((0,), dtype=torch.float64, device=t.device)
        else:
            s_vals = torch.linalg.svdvals(t)
    else:
        flat = t.reshape(t.shape[0], -1)
        shape = (int(flat.shape[0]), int(flat.shape[1]))
        s_vals = torch.linalg.svdvals(flat)

    if s_vals.numel() == 0:
        return OperatorSpectrumSummary(
            operator_name=operator_name,
            matrix_shape=shape,
            singular_values=(),
            condition_number=1.0,
            effective_rank=0,
            spectral_gap_ratio=1.0,
            nullspace_dimension=0,
        )

    s_list = [round(float(v), 8) for v in s_vals.tolist()]
    s_max = max(float(s_vals[0].item()), 0.0)
    s_min = max(float(s_vals[-1].item()), 0.0)
    cond = (s_max / max(s_min, 1e-15)) if s_max > 1e-15 else 1.0
    eff_rank = int(torch.sum(s_vals > (s_max * rank_rel_tol)).item()) if s_max > 1e-15 else 0
    max_possible_rank = min(shape[0], shape[1])
    null_from_sv = int(torch.sum(s_vals <= null_abs_tol).item())
    structural_null = max(shape[1] - max_possible_rank, 0)
    null_dim = null_from_sv + structural_null

    if s_vals.numel() >= 2 and s_vals[0].item() > 1e-15:
        ratios = s_vals[:-1] / torch.clamp(s_vals[1:], min=1e-15)
        gap_ratio = float(torch.max(ratios).item())
    else:
        gap_ratio = 1.0

    return OperatorSpectrumSummary(
        operator_name=operator_name,
        matrix_shape=shape,
        singular_values=tuple(s_list[:16]),
        condition_number=round(min(cond, 1e15), 4),
        effective_rank=eff_rank,
        spectral_gap_ratio=round(min(gap_ratio, 1e12), 4),
        nullspace_dimension=null_dim,
    )


_ACTIVE_RECORDER: contextvars.ContextVar[FlightRecorder | None] = contextvars.ContextVar(
    "_ACTIVE_RECORDER", default=None
)


class FlightRecorder:
    """
    Ambient scientific flight recorder capturing a `CausalTraceDAG` across Layers 1–7.
    """

    def __init__(
        self,
        session_id: str = "cortivra_session",
        question: str = "",
        on_node_recorded: Callable[[CausalTraceNode], None] | None = None,
    ) -> None:
        self.dag = CausalTraceDAG(session_id=session_id, question=question)
        self._on_node_recorded = on_node_recorded
        self._span_stack: list[str] = []
        self._last_completed_node_id: str | None = None
        self._counter: int = 0
        self._inner_loop_depth: int = 0

    @classmethod
    @contextmanager
    def session(
        cls,
        session_id: str = "cortivra_session",
        question: str = "",
        on_node_recorded: Callable[[CausalTraceNode], None] | None = None,
    ) -> Iterator[FlightRecorder]:
        recorder = cls(
            session_id=session_id,
            question=question,
            on_node_recorded=on_node_recorded,
        )
        token = _ACTIVE_RECORDER.set(recorder)
        try:
            yield recorder
        finally:
            _ACTIVE_RECORDER.reset(token)

    @property
    def in_inner_loop(self) -> bool:
        return self._inner_loop_depth > 0

    @contextmanager
    def suppress_inner_loop(self) -> Iterator[None]:
        self._inner_loop_depth += 1
        try:
            yield
        finally:
            self._inner_loop_depth = max(0, self._inner_loop_depth - 1)

    def _next_node_id(self, layer: LayerId, operation: str) -> str:
        self._counter += 1
        short_layer = layer.split("_")[0]
        return f"node_{self._counter:03d}_{short_layer}_{operation}"

    def _resolve_parent_ids(self, explicit_parents: Sequence[str] | None = None) -> tuple[str, ...]:
        if explicit_parents:
            return tuple(explicit_parents)
        parents: list[str] = []
        if self._last_completed_node_id:
            parents.append(self._last_completed_node_id)
        if self._span_stack and self._span_stack[-1] not in parents:
            parents.append(self._span_stack[-1])
        return tuple(parents)

    def record_node(
        self,
        layer: LayerId,
        operation: str,
        target_name: str,
        duration_ms: float = 0.0,
        status: str = "completed",
        local_error_metric: float | None = None,
        local_r2: float | None = None,
        global_error_metric: float | None = None,
        global_r2: float | None = None,
        physical_violation_metric: float = 0.0,
        decision_margin: float | None = None,
        selected_branch_id: str | None = None,
        selected_expression: str | None = None,
        operator_spectrum: OperatorSpectrumSummary | None = None,
        alternatives: Sequence[AlternativeBranchRecord] = (),
        metrics: dict[str, Any] | None = None,
        diagnostics: Sequence[str] = (),
        parent_ids: Sequence[str] | None = None,
    ) -> CausalTraceNode:
        node = CausalTraceNode(
            node_id=self._next_node_id(layer, operation),
            parent_ids=self._resolve_parent_ids(parent_ids),
            layer=layer,
            operation=operation,
            target_name=target_name,
            started_at_ns=time.perf_counter_ns(),
            duration_ms=round(float(duration_ms), 3),
            status=status,
            local_error_metric=(
                round(float(local_error_metric), 8)
                if (local_error_metric is not None and math.isfinite(local_error_metric))
                else local_error_metric
            ),
            local_r2=(
                round(float(local_r2), 6)
                if (local_r2 is not None and math.isfinite(local_r2))
                else local_r2
            ),
            global_error_metric=(
                round(float(global_error_metric), 8)
                if (global_error_metric is not None and math.isfinite(global_error_metric))
                else global_error_metric
            ),
            global_r2=(
                round(float(global_r2), 6)
                if (global_r2 is not None and math.isfinite(global_r2))
                else global_r2
            ),
            physical_violation_metric=round(float(physical_violation_metric), 8)
            if math.isfinite(physical_violation_metric)
            else float("inf"),
            decision_margin=(
                round(float(decision_margin), 6)
                if (decision_margin is not None and math.isfinite(decision_margin))
                else decision_margin
            ),
            selected_branch_id=selected_branch_id,
            selected_expression=selected_expression,
            operator_spectrum=operator_spectrum,
            alternatives=tuple(alternatives),
            metrics=dict(metrics) if metrics else {},
            diagnostics=tuple(diagnostics),
        )
        self.dag.add_node(node)
        self._last_completed_node_id = node.node_id
        if self._on_node_recorded is not None:
            self._on_node_recorded(node)
        return node

    @contextmanager
    def span(
        self,
        layer: LayerId,
        operation: str,
        target_name: str,
        parent_ids: Sequence[str] | None = None,
    ) -> Iterator[CausalTraceNode]:
        t0 = time.perf_counter_ns()
        node = CausalTraceNode(
            node_id=self._next_node_id(layer, operation),
            parent_ids=self._resolve_parent_ids(parent_ids),
            layer=layer,
            operation=operation,
            target_name=target_name,
            started_at_ns=t0,
            status="running",
        )
        self.dag.add_node(node)
        self._span_stack.append(node.node_id)
        try:
            yield node
            node.status = "completed"
        except Exception as exc:
            node.status = "failed"
            node.diagnostics = (*node.diagnostics, f"{type(exc).__name__}: {exc}")
            raise
        finally:
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            node.duration_ms = round(elapsed_ms, 3)
            if self._span_stack and self._span_stack[-1] == node.node_id:
                self._span_stack.pop()
            for n in self.dag.nodes:
                self.dag.refresh_node_hash(n)
            self._last_completed_node_id = node.node_id
            if self._on_node_recorded is not None:
                self._on_node_recorded(node)


def active_recorder() -> FlightRecorder | None:
    """Return the active ambient `FlightRecorder` in the current execution context, if any."""
    return _ACTIVE_RECORDER.get()


@contextmanager
def suppress_inner_trace() -> Iterator[None]:
    """
    Suppress inner-loop helper trace emissions when inside iterative solvers.
    """
    rec = _ACTIVE_RECORDER.get()
    if rec is None:
        yield
        return
    with rec.suppress_inner_loop():
        yield


def emit_trace_node(
    layer: LayerId,
    operation: str,
    target_name: str,
    skip_if_inner_loop: bool = False,
    **kwargs: Any,
) -> CausalTraceNode | None:
    """
    Emit a `CausalTraceNode` to the ambient `FlightRecorder` if a session is active.
    Zero-overhead no-op when no session is active or when `skip_if_inner_loop=True` inside an inner loop.
    """
    rec = _ACTIVE_RECORDER.get()
    if rec is None:
        return None
    if skip_if_inner_loop and rec.in_inner_loop:
        return None
    return rec.record_node(layer=layer, operation=operation, target_name=target_name, **kwargs)


@contextmanager
def trace_span(
    layer: LayerId,
    operation: str,
    target_name: str,
    parent_ids: Sequence[str] | None = None,
) -> Iterator[CausalTraceNode | None]:
    """
    Open a hierarchical causal span in the ambient `FlightRecorder` if active.
    """
    rec = _ACTIVE_RECORDER.get()
    if rec is None:
        yield None
        return
    with rec.span(layer=layer, operation=operation, target_name=target_name, parent_ids=parent_ids) as node:
        yield node


def record_simulation_activity(
    *,
    ode_rollout_calls: int = 0,
    trajectories_integrated: int = 0,
    integration_steps_executed: int = 0,
    sobol_qmc_universes_sampled: int = 0,
    newton_jacobian_solves: int = 0,
    adjoint_gradient_passes: int = 0,
    stochastic_langevin_trajectories: int = 0,
    integrator: str | None = None,
) -> None:
    """
    Increment ambient hardware simulation and QMC universe counters on the active `FlightRecorder`.
    Zero-overhead no-op when no `FlightRecorder` session is active.
    """
    rec = _ACTIVE_RECORDER.get()
    if rec is None:
        return
    c = rec.dag.simulation_counters
    c.ode_rollout_calls += int(ode_rollout_calls)
    c.trajectories_integrated += int(trajectories_integrated)
    c.integration_steps_executed += int(integration_steps_executed)
    c.sobol_qmc_universes_sampled += int(sobol_qmc_universes_sampled)
    c.newton_jacobian_solves += int(newton_jacobian_solves)
    c.adjoint_gradient_passes += int(adjoint_gradient_passes)
    c.stochastic_langevin_trajectories += int(stochastic_langevin_trajectories)
    if integrator:
        c.integrator_usage[integrator] = c.integrator_usage.get(integrator, 0) + 1

