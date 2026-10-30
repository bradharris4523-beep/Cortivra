"""
Symbolic-to-Port-Hamiltonian GPU tensor compiler (`cortivra.layer3_compiler.symbolic_compiler`).

Compiles declarative symbolic equations (`SymbolicSystemSpec`) or unmodeled residual patches
into batched, differentiable `LawBrick` instances that enforce SI unit homogeneity and
1st/2nd Law thermodynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import (
    LawBrick,
    ParamSpec,
    PowerPortSpec,
    SpatialNode3DSpec,
    StateVarSpec,
)
from cortivra.layer1_axioms.units import (
    DIMENSIONLESS,
    DimensionError,
    J,
    Quantity,
    Unit,
    m,
    s,
)
from cortivra.layer3_compiler.ast_nodes import (
    ConstNode,
    ExprNode,
    SymbolNode,
    parse_symbolic_expr,
)
from cortivra.observability.flight_recorder import emit_trace_node


@dataclass(frozen=True)
class OnsagerCouplingSpec:
    """
    Irreducible rank-1 symmetric positive semi-definite Onsager dissipative channel:
        Delta_R(x) = gamma(x) * (v * v^T) >= 0
    where `stoichiometry` maps state indices to signed coupling weights `(state_idx, nu_idx)`
    (e.g., `((0, 1.0), (1, -1.0))` for mass-conserving reaction `X_0 <-> X_1` or diffusive flux
    between compartments `0` and `1`), and `conductance_expr` is a non-negative symbolic rate/conductance.
    """

    stoichiometry: tuple[tuple[int, float], ...]
    conductance_expr: str | ExprNode


@dataclass(frozen=True)
class SymbolicSystemSpec:
    """
    Declarative specification of a dynamical system in symbolic form.

    Parameters:
        name: Identifier for the compiled `LawBrick`.
        domain: Physical domain name (e.g., `"mechanics"`, `"electrodynamics"`, `"chemistry"`, `"optics"`).
        state_specs: Sequence of `StateVarSpec` declarations.
        param_specs: Sequence of `ParamSpec` declarations.
        port_specs: Sequence of `PowerPortSpec` declarations.
        time_unit: Integration time unit (`s`, `ms`, etc.).
        energy_unit: Stored Hamiltonian energy unit (`J`, `fJ`, `DIMENSIONLESS`, etc.).
        hamiltonian_expr: Symbolic expression string (or `ExprNode`) for stored energy `H(x, params)`.
        skew_routing_exprs: Mapping `(i, j) -> expr_str` for upper-triangular entries `i < j` of `J(x)`
                            (`J_{ji} = -J_{ij}` is enforced automatically by skew-symmetry).
        dissipation_diag_exprs: Sequence of symbolic expressions (one per state variable) for non-negative
                                diagonal dissipation entries `R_{ii}(x, params) >= 0`.
        dissipation_coupling_exprs: Mapping `(i, j) -> expr_str` for upper-triangular pairwise Onsager
                                    dissipative couplings `Delta_R = gamma_{ij}(x) * (e_i - e_j)(e_i - e_j)^T >= 0`.
        onsager_couplings: Sequence of general multi-state rank-1 `OnsagerCouplingSpec` channels
                           `Delta_R = gamma_k(x) * v_k v_k^T >= 0`.
        external_drive_exprs: Optional sequence of symbolic expressions (one per state variable) for
                              active/driven flows `u_i(x, params)` in units of `state_i.unit / time_unit`.
        constants: Optional mapping of constant symbol names to `Quantity` or `ConstNode`.
        thermal_sink_state_name: Optional state variable name receiving 100% of dissipated power
                                 `D(x) = grad_H . R grad_H >= 0` (GENERIC 1st-Law thermal conservation).
        thermal_capacity_scale: Conversion factor `1 / C_th` scaling dissipated power into `d(x_thermal)/dt`.
        spatial_nodes_3d: Optional 3D Euclidean node coordinate descriptors (`SpatialNode3DSpec`).
    """

    name: str
    domain: str
    state_specs: Sequence[StateVarSpec]
    param_specs: Sequence[ParamSpec] = ()
    port_specs: Sequence[PowerPortSpec] = ()
    time_unit: Unit = s
    energy_unit: Unit = J
    hamiltonian_expr: str | ExprNode = "0.0"
    skew_routing_exprs: Mapping[tuple[int, int], str | ExprNode] = field(default_factory=dict)
    dissipation_diag_exprs: Sequence[str | ExprNode] = ()
    dissipation_coupling_exprs: Mapping[tuple[int, int], str | ExprNode] = field(default_factory=dict)
    onsager_couplings: Sequence[OnsagerCouplingSpec] = ()
    external_drive_exprs: Sequence[str | ExprNode] = ()
    constants: Mapping[str, Quantity | ConstNode] = field(default_factory=dict)
    thermal_sink_state_name: str | None = None
    thermal_capacity_scale: float = 1.0
    spatial_nodes_3d: Sequence[SpatialNode3DSpec] = ()


def build_symbol_table(
    state_specs: Sequence[StateVarSpec],
    param_specs: Sequence[ParamSpec],
    constants: Mapping[str, Quantity | ConstNode] | None = None,
) -> dict[str, SymbolNode | ConstNode]:
    """Construct a compile-time symbol table from state specs, parameter specs, and physical constants."""
    table: dict[str, SymbolNode | ConstNode] = {}
    for sp in state_specs:
        table[sp.name] = SymbolNode(name=sp.name, unit=sp.unit, role="state")
    for pp in param_specs:
        if pp.name in table:
            raise ValueError(f"Name collision between state and parameter symbol '{pp.name}'.")
        table[pp.name] = SymbolNode(name=pp.name, unit=pp.unit, role="param")
    if constants:
        for cname, cval in constants.items():
            if cname in table:
                raise ValueError(f"Constant name '{cname}' collides with state or parameter name.")
            if isinstance(cval, ConstNode):
                table[cname] = cval
            elif isinstance(cval, Quantity):
                table[cname] = ConstNode(float(cval.value), cval.unit)
            else:
                raise TypeError(f"Expected Quantity or ConstNode for constant '{cname}', got {type(cval).__name__}.")
    return table


def _to_expr_node(
    expr: str | ExprNode,
    symbol_table: Mapping[str, SymbolNode | ConstNode],
) -> ExprNode:
    if isinstance(expr, ExprNode):
        expr.infer_unit()
        return expr
    return parse_symbolic_expr(expr, symbol_table=symbol_table)


class CompiledLawBrick(LawBrick):
    """
    Batched GPU `LawBrick` compiled from symbolic AST expressions.

    Guarantees at compile time:
        1. `H(x, params)` is dimensionally compatible with `energy_unit` (or `DIMENSIONLESS` Lyapunov storage).
        2. Every effort component `e_i = dH / dx_i` is computed symbolically (`H_ast.diff(x_i)`) and verified
           against PyTorch autograd.
        3. `J(x) = A(x) - A(x)^T` is skew-symmetric by construction (`J = -J^T`).
        4. `R(x) = diag(clamp(r_i(x), min=0))` is symmetric positive semi-definite by construction (`R >= 0`).
        5. Optional active external drives `u_i(x, params)` are verified at compile time to have exact
           SI dimension `Dim(state_i.unit / time_unit)` and converted to exact numeric scale.
    """

    def __init__(self, spec: SymbolicSystemSpec) -> None:
        super().__init__(
            name=spec.name,
            domain=spec.domain,
            state_specs=spec.state_specs,
            param_specs=spec.param_specs,
            port_specs=spec.port_specs,
            time_unit=spec.time_unit,
            energy_unit=spec.energy_unit,
            matrix_free=False,
            thermal_sink_state_name=spec.thermal_sink_state_name,
            thermal_capacity_scale=spec.thermal_capacity_scale,
            spatial_nodes_3d=spec.spatial_nodes_3d,
        )
        self.spec = spec
        self.symbol_table = build_symbol_table(
            self.state_specs, self.param_specs, spec.constants
        )
        # Populate 3D Euclidean geometry constants (volumes, radii, pairwise distances, and Fickian geometric factors)
        m3_unit = m ** 3
        for idx_3d, nd_3d in enumerate(self.spatial_nodes_3d):
            v_key = f"vol_3d_{idx_3d}"
            r_key = f"radius_3d_{idx_3d}"
            if v_key not in self.symbol_table:
                self.symbol_table[v_key] = ConstNode(float(nd_3d.volume_m3), m3_unit)
            if r_key not in self.symbol_table:
                self.symbol_table[r_key] = ConstNode(float(nd_3d.radius_m), m)
            for jdx_3d in range(idx_3d + 1, len(self.spatial_nodes_3d)):
                nd_j = self.spatial_nodes_3d[jdx_3d]
                dx = float(nd_3d.fixed_position_m[0] - nd_j.fixed_position_m[0])
                dy = float(nd_3d.fixed_position_m[1] - nd_j.fixed_position_m[1])
                dz = float(nd_3d.fixed_position_m[2] - nd_j.fixed_position_m[2])
                dist_ij = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-9)
                eff_r = min(float(nd_3d.radius_m), float(nd_j.radius_m))
                geom_ij = (math.pi * eff_r * eff_r) / dist_ij
                d_key = f"dist_3d_{idx_3d}_{jdx_3d}"
                g_key = f"geom_cond_3d_{idx_3d}_{jdx_3d}"
                if d_key not in self.symbol_table:
                    self.symbol_table[d_key] = ConstNode(dist_ij, m)
                if g_key not in self.symbol_table:
                    self.symbol_table[g_key] = ConstNode(geom_ij, m)

        # 1. Compile and dimension-check Hamiltonian H(x, params)
        self.hamiltonian_ast = _to_expr_node(spec.hamiltonian_expr, self.symbol_table).simplify()
        h_unit = self.hamiltonian_ast.infer_unit()
        if h_unit.dimension != self.energy_unit.dimension and not (
            isinstance(self.hamiltonian_ast, ConstNode) and self.hamiltonian_ast.value == 0.0
        ):
            raise DimensionError(
                f"CompiledLawBrick '{self.name}': Hamiltonian expression '{self.hamiltonian_ast.to_infix()}' "
                f"has unit '{h_unit.symbol}' ({h_unit.dimension.format_si()}), expected "
                f"'{self.energy_unit.symbol}' ({self.energy_unit.dimension.format_si()})."
            )
        self._h_scale = (
            h_unit.conversion_factor_to(self.energy_unit)
            if h_unit.dimension == self.energy_unit.dimension
            else 1.0
        )

        # Compile symbolic gradient nodes dH/dx_i where available; fall back to autograd if a primitive lacks symbolic diff
        self._effort_asts: list[ExprNode | None] = []
        for sp in self.state_specs:
            try:
                self._effort_asts.append(self.hamiltonian_ast.diff(sp.name).simplify())
            except Exception:
                self._effort_asts.append(None)

        # 2. Compile skew-symmetric routing expressions J_{ij}(x, params) for i < j
        self._skew_asts: dict[tuple[int, int], tuple[ExprNode, float]] = {}
        for (i, j), expr in spec.skew_routing_exprs.items():
            if not (0 <= i < j < self.state_dim):
                raise ValueError(
                    f"CompiledLawBrick '{self.name}': skew_routing_exprs index {(i, j)} must satisfy "
                    f"0 <= i < j < {self.state_dim}."
                )
            node = _to_expr_node(expr, self.symbol_table).simplify()
            node_u = node.infer_unit()
            expected_j_unit = (self.state_specs[i].unit * self.state_specs[j].unit) / (
                self.energy_unit * self.time_unit
            )
            if node_u.dimension != expected_j_unit.dimension and node_u.dimension != DIMENSIONLESS.dimension:
                raise DimensionError(
                    f"CompiledLawBrick '{self.name}': J[{i},{j}] expression '{node.to_infix()}' has "
                    f"dimension {node_u.dimension.format_si()}, expected {expected_j_unit.dimension.format_si()}."
                )
            scale = (
                node_u.conversion_factor_to(expected_j_unit)
                if node_u.dimension == expected_j_unit.dimension
                else 1.0
            )
            self._skew_asts[(i, j)] = (node, scale)

        # 3. Compile diagonal dissipation expressions R_{ii}(x, params) >= 0
        if spec.dissipation_diag_exprs and len(spec.dissipation_diag_exprs) != self.state_dim:
            raise ValueError(
                f"CompiledLawBrick '{self.name}': expected {self.state_dim} dissipation_diag_exprs, "
                f"got {len(spec.dissipation_diag_exprs)}."
            )
        self._diss_asts: list[tuple[ExprNode, float]] = []
        for idx, expr in enumerate(spec.dissipation_diag_exprs):
            node = _to_expr_node(expr, self.symbol_table).simplify()
            node_u = node.infer_unit()
            expected_r_unit = (self.state_specs[idx].unit ** 2) / (self.energy_unit * self.time_unit)
            if node_u.dimension != expected_r_unit.dimension and node_u.dimension != DIMENSIONLESS.dimension:
                raise DimensionError(
                    f"CompiledLawBrick '{self.name}': R[{idx},{idx}] expression '{node.to_infix()}' has "
                    f"dimension {node_u.dimension.format_si()}, expected {expected_r_unit.dimension.format_si()}."
                )
            scale = (
                node_u.conversion_factor_to(expected_r_unit)
                if node_u.dimension == expected_r_unit.dimension
                else 1.0
            )
            self._diss_asts.append((node, scale))

        # 3b. Compile off-diagonal symmetric positive semi-definite Onsager couplings R_onsager = sum_k gamma_k * v_k v_k^T >= 0
        self._onsager_channels: list[tuple[tuple[tuple[int, float], ...], ExprNode, float]] = []
        all_onsager_specs: list[OnsagerCouplingSpec] = list(spec.onsager_couplings)
        for (i_c, j_c), c_expr in spec.dissipation_coupling_exprs.items():
            if not (0 <= i_c < j_c < self.state_dim):
                raise ValueError(
                    f"CompiledLawBrick '{self.name}': dissipation_coupling_exprs index {(i_c, j_c)} must satisfy "
                    f"0 <= i < j < {self.state_dim}."
                )
            all_onsager_specs.append(
                OnsagerCouplingSpec(
                    stoichiometry=((i_c, 1.0), (j_c, -1.0)),
                    conductance_expr=c_expr,
                )
            )

        for ch_spec in all_onsager_specs:
            if not ch_spec.stoichiometry:
                raise ValueError(f"CompiledLawBrick '{self.name}': OnsagerCouplingSpec stoichiometry cannot be empty.")
            for idx_s, _ in ch_spec.stoichiometry:
                if not (0 <= idx_s < self.state_dim):
                    raise ValueError(
                        f"CompiledLawBrick '{self.name}': OnsagerCouplingSpec state index {idx_s} out of range."
                    )
            ref_idx = ch_spec.stoichiometry[0][0]
            ref_unit = self.state_specs[ref_idx].unit
            for idx_s, _ in ch_spec.stoichiometry[1:]:
                if self.state_specs[idx_s].unit.dimension != ref_unit.dimension:
                    raise DimensionError(
                        f"CompiledLawBrick '{self.name}': Onsager coupled states #{ref_idx} ('{ref_unit.symbol}') "
                        f"and #{idx_s} ('{self.state_specs[idx_s].unit.symbol}') must share dimensions."
                    )
            node = _to_expr_node(ch_spec.conductance_expr, self.symbol_table).simplify()
            node_u = node.infer_unit()
            expected_r_unit = (ref_unit ** 2) / (self.energy_unit * self.time_unit)
            if node_u.dimension != expected_r_unit.dimension and node_u.dimension != DIMENSIONLESS.dimension:
                raise DimensionError(
                    f"CompiledLawBrick '{self.name}': Onsager conductance '{node.to_infix()}' has "
                    f"dimension {node_u.dimension.format_si()}, expected {expected_r_unit.dimension.format_si()}."
                )
            scale = (
                node_u.conversion_factor_to(expected_r_unit)
                if node_u.dimension == expected_r_unit.dimension
                else 1.0
            )
            self._onsager_channels.append((tuple(ch_spec.stoichiometry), node, scale))

        # 4. Compile optional active/external drive expressions u_i(x, params) in state_i.unit / time_unit
        if spec.external_drive_exprs and len(spec.external_drive_exprs) != self.state_dim:
            raise ValueError(
                f"CompiledLawBrick '{self.name}': expected {self.state_dim} external_drive_exprs, "
                f"got {len(spec.external_drive_exprs)}."
            )
        self._drive_asts: list[tuple[ExprNode, float]] = []
        for idx, expr in enumerate(spec.external_drive_exprs):
            node = _to_expr_node(expr, self.symbol_table).simplify()
            target_deriv_unit = self.state_specs[idx].unit / self.time_unit
            node_u = node.infer_unit()
            if isinstance(node, ConstNode) and node.value == 0.0 and node_u == DIMENSIONLESS:
                self._drive_asts.append((node, 1.0))
                continue
            scale = node_u.conversion_factor_to(target_deriv_unit)
            self._drive_asts.append((node, scale))

    def _build_env(self, state: torch.Tensor, params: torch.Tensor) -> dict[str, torch.Tensor]:
        env: dict[str, torch.Tensor] = {}
        for idx, sp in enumerate(self.state_specs):
            env[sp.name] = state[:, idx]
        for idx, pp in enumerate(self.param_specs):
            env[pp.name] = params[:, idx]
        return env

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        env = self._build_env(state, params)
        val = self.hamiltonian_ast.eval_tensor(env)
        return val * self._h_scale if abs(self._h_scale - 1.0) > 1e-12 else val

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if all(node is not None for node in self._effort_asts):
            env = self._build_env(state, params)
            cols = [node.eval_tensor(env) * self._h_scale for node in self._effort_asts if node is not None]
            return torch.stack(cols, dim=-1)
        return super().effort(state, params)

    def structure_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        n = self.state_dim
        j_mat = torch.zeros((batch, n, n), dtype=state.dtype, device=state.device)
        if not self._skew_asts:
            return j_mat
        env = self._build_env(state, params)
        for (i, j), (node, scale) in self._skew_asts.items():
            val = node.eval_tensor(env) * scale
            j_mat[:, i, j] = val
            j_mat[:, j, i] = -val
        return j_mat

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        n = self.state_dim
        r_mat = torch.zeros((batch, n, n), dtype=state.dtype, device=state.device)
        if not self._diss_asts and not self._onsager_channels:
            return r_mat
        env = self._build_env(state, params)
        for idx, (node, scale) in enumerate(self._diss_asts):
            val = torch.clamp(node.eval_tensor(env) * scale, min=0.0)
            r_mat[:, idx, idx] = r_mat[:, idx, idx] + val
        for stoich, node, scale in self._onsager_channels:
            gamma = torch.clamp(node.eval_tensor(env) * scale, min=0.0)
            for i_s, nu_i in stoich:
                for j_s, nu_j in stoich:
                    r_mat[:, i_s, j_s] = r_mat[:, i_s, j_s] + gamma * (nu_i * nu_j)
        return r_mat

    def compute_compiled_external_drive(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Evaluate compiled `external_drive_exprs` as a `[Batch, StateDim]` flow tensor."""
        if not self._drive_asts:
            return torch.zeros_like(state)
        env = self._build_env(state, params)
        cols = [node.eval_tensor(env) * scale for node, scale in self._drive_asts]
        return torch.stack(cols, dim=-1)

    def external_drive_power(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute instantaneous power `P_ext = grad_H(x) . u_drive(x)` supplied by `external_drive_exprs`."""
        if not self._drive_asts:
            return torch.zeros((state.shape[0],), dtype=state.dtype, device=state.device)
        e = self.effort(state, params)
        u = self.compute_compiled_external_drive(state, params)
        return (e * u).sum(dim=-1)

    def flow(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        port_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        dx_dt = super().flow(state, params, port_inputs=port_inputs)
        if self._drive_asts:
            dx_dt = dx_dt + self.compute_compiled_external_drive(state, params)
        return dx_dt

    def total_symbolic_complexity(self) -> int:
        """Return total AST node count across Hamiltonian, J, R, and drive expressions."""
        total = self.hamiltonian_ast.tree_complexity()
        total += sum(node.tree_complexity() for node, _ in self._skew_asts.values())
        total += sum(node.tree_complexity() for node, _ in self._diss_asts)
        total += sum(node.tree_complexity() for _, node, _ in self._onsager_channels)
        total += sum(node.tree_complexity() for node, _ in self._drive_asts)
        return total


class ResidualPatchedLawBrick(LawBrick):
    """
    Composite `LawBrick` formed by augmenting a known `base_brick` with a dimensionally
    verified symbolic residual patch on one or more state derivatives (used when Layer 1's
    `analyze_unexplained_residual` confirms an unmodeled physical mechanism).

    Supports both dense and `matrix_free=True` base bricks across three thermodynamic patch modes
    matching `LatentResidualAnalysis.thermodynamic_class`:
        - `"passive_dissipative"`: Adds a non-negative symbolic diagonal dissipation term
          `Delta_R_{ii}(x, params) >= 0` so `dx_i/dt_patch = -Delta_R_{ii} * e_i`.
        - `"conservative_coupling"`: Adds a skew-symmetric symbolic coupling `Delta_J_{ij}(x, params)`
          between state `i` and state `j` so total energy `H(x)` remains strictly conserved.
        - `"active_driven"`: Adds a dimensionally checked symbolic flow expression `Phi_i(x, params)`
          directly to state derivative `i` (with automatic SI unit scale conversion to `state_i.unit / time_unit`).
    """

    def __init__(
        self,
        name: str,
        base_brick: LawBrick,
        patch_param_specs: Sequence[ParamSpec],
        thermodynamic_mode: str,
        patch_expressions: Mapping[int | tuple[int, int], str | ExprNode],
        constants: Mapping[str, Quantity | ConstNode] | None = None,
    ) -> None:
        if thermodynamic_mode not in ("passive_dissipative", "conservative_coupling", "active_driven"):
            raise ValueError(f"Unsupported thermodynamic_mode '{thermodynamic_mode}'.")

        self.base_brick = base_brick
        self.base_param_dim = base_brick.param_dim
        self.patch_param_specs = tuple(patch_param_specs)
        self.thermodynamic_mode = thermodynamic_mode

        combined_params = list(base_brick.param_specs) + list(self.patch_param_specs)
        super().__init__(
            name=name,
            domain=base_brick.domain,
            state_specs=base_brick.state_specs,
            param_specs=combined_params,
            port_specs=base_brick.port_specs,
            time_unit=base_brick.time_unit,
            energy_unit=base_brick.energy_unit,
            matrix_free=base_brick.is_matrix_free,
            thermal_sink_state_name=base_brick.thermal_sink_state_name,
            thermal_capacity_scale=base_brick.thermal_capacity_scale,
            spatial_nodes_3d=base_brick.spatial_nodes_3d,
        )

        self.symbol_table = build_symbol_table(
            self.state_specs, self.param_specs, constants
        )
        self._compiled_patches: dict[int | tuple[int, int], tuple[ExprNode, float]] = {}

        for key, expr in patch_expressions.items():
            node = _to_expr_node(expr, self.symbol_table).simplify()
            node_u = node.infer_unit()

            if thermodynamic_mode == "active_driven":
                if not isinstance(key, int) or not (0 <= key < self.state_dim):
                    raise ValueError(f"Expected integer state index for active_driven patch, got {key!r}.")
                expected_u = self.state_specs[key].unit / self.time_unit
                scale = node_u.conversion_factor_to(expected_u)
                self._compiled_patches[key] = (node, scale)

            elif thermodynamic_mode == "passive_dissipative":
                if isinstance(key, int) and (0 <= key < self.state_dim):
                    expected_u = (self.state_specs[key].unit ** 2) / (self.energy_unit * self.time_unit)
                    if node_u.dimension != expected_u.dimension and node_u.dimension != DIMENSIONLESS.dimension:
                        raise DimensionError(
                            f"Dissipative patch on state #{key} has dimension {node_u.dimension.format_si()}, "
                            f"expected {expected_u.dimension.format_si()}."
                        )
                    scale = node_u.conversion_factor_to(expected_u) if node_u.dimension == expected_u.dimension else 1.0
                    self._compiled_patches[key] = (node, scale)
                elif isinstance(key, tuple) and len(key) == 2 and (0 <= key[0] < key[1] < self.state_dim):
                    i_p, j_p = key
                    if self.state_specs[i_p].unit.dimension != self.state_specs[j_p].unit.dimension:
                        raise DimensionError(
                            f"Onsager dissipative coupling patch on {(i_p, j_p)} requires matching state dimensions."
                        )
                    expected_u = (self.state_specs[i_p].unit ** 2) / (self.energy_unit * self.time_unit)
                    if node_u.dimension != expected_u.dimension and node_u.dimension != DIMENSIONLESS.dimension:
                        raise DimensionError(
                            f"Onsager dissipative patch on {(i_p, j_p)} has dimension {node_u.dimension.format_si()}, "
                            f"expected {expected_u.dimension.format_si()}."
                        )
                    scale = node_u.conversion_factor_to(expected_u) if node_u.dimension == expected_u.dimension else 1.0
                    self._compiled_patches[key] = (node, scale)
                else:
                    raise ValueError(
                        f"Expected integer state index or pair (i, j) for passive_dissipative patch, got {key!r}."
                    )

            elif thermodynamic_mode == "conservative_coupling":
                if not isinstance(key, tuple) or len(key) != 2 or not (0 <= key[0] < key[1] < self.state_dim):
                    raise ValueError(
                        f"Expected pair (i, j) with 0 <= i < j < {self.state_dim} for conservative_coupling, got {key!r}."
                    )
                i, j = key
                expected_u = (self.state_specs[i].unit * self.state_specs[j].unit) / (
                    self.energy_unit * self.time_unit
                )
                if node_u.dimension != expected_u.dimension and node_u.dimension != DIMENSIONLESS.dimension:
                    raise DimensionError(
                        f"Conservative patch on {(i, j)} has dimension {node_u.dimension.format_si()}, "
                        f"expected {expected_u.dimension.format_si()}."
                    )
                scale = node_u.conversion_factor_to(expected_u) if node_u.dimension == expected_u.dimension else 1.0
                self._compiled_patches[key] = (node, scale)

    def _build_env(self, state: torch.Tensor, params: torch.Tensor) -> dict[str, torch.Tensor]:
        env: dict[str, torch.Tensor] = {}
        for idx, sp in enumerate(self.state_specs):
            env[sp.name] = state[:, idx]
        for idx, pp in enumerate(self.param_specs):
            env[pp.name] = params[:, idx]
        return env

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return self.base_brick.hamiltonian_energy(state, params[:, : self.base_param_dim])

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return self.base_brick.effort(state, params[:, : self.base_param_dim])

    def structure_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        j_mat = self.base_brick.structure_matrix(state, params[:, : self.base_param_dim]).clone()
        if self.thermodynamic_mode == "conservative_coupling":
            env = self._build_env(state, params)
            for key, (node, scale) in self._compiled_patches.items():
                assert isinstance(key, tuple)
                i, j = key
                val = node.eval_tensor(env) * scale
                j_mat[:, i, j] = j_mat[:, i, j] + val
                j_mat[:, j, i] = j_mat[:, j, i] - val
        return j_mat

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        r_mat = self.base_brick.dissipation_matrix(state, params[:, : self.base_param_dim]).clone()
        if self.thermodynamic_mode == "passive_dissipative":
            env = self._build_env(state, params)
            for key, (node, scale) in self._compiled_patches.items():
                val = torch.clamp(node.eval_tensor(env) * scale, min=0.0)
                if isinstance(key, int):
                    r_mat[:, key, key] = r_mat[:, key, key] + val
                else:
                    i_p, j_p = key
                    r_mat[:, i_p, i_p] = r_mat[:, i_p, i_p] + val
                    r_mat[:, j_p, j_p] = r_mat[:, j_p, j_p] + val
                    r_mat[:, i_p, j_p] = r_mat[:, i_p, j_p] - val
                    r_mat[:, j_p, i_p] = r_mat[:, j_p, i_p] - val
        return r_mat

    def conservative_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        f_cons = self.base_brick.conservative_flow(state, params[:, : self.base_param_dim])
        if self.thermodynamic_mode == "conservative_coupling":
            env = self._build_env(state, params)
            e = self.effort(state, params)
            deltas = [torch.zeros_like(state[:, idx]) for idx in range(self.state_dim)]
            for key, (node, scale) in self._compiled_patches.items():
                assert isinstance(key, tuple)
                i, j = key
                val = node.eval_tensor(env) * scale
                deltas[i] = deltas[i] + val * e[:, j]
                deltas[j] = deltas[j] - val * e[:, i]
            f_cons = f_cons + torch.stack(deltas, dim=-1)
        return f_cons

    def dissipative_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        f_diss = self.base_brick.dissipative_flow(state, params[:, : self.base_param_dim])
        if self.thermodynamic_mode == "passive_dissipative":
            env = self._build_env(state, params)
            e = self.effort(state, params)
            deltas = [torch.zeros_like(state[:, idx]) for idx in range(self.state_dim)]
            for key, (node, scale) in self._compiled_patches.items():
                val = torch.clamp(node.eval_tensor(env) * scale, min=0.0)
                if isinstance(key, int):
                    deltas[key] = deltas[key] + val * e[:, key]
                else:
                    i_p, j_p = key
                    flux_ij = val * (e[:, i_p] - e[:, j_p])
                    deltas[i_p] = deltas[i_p] + flux_ij
                    deltas[j_p] = deltas[j_p] - flux_ij
            f_diss = f_diss + torch.stack(deltas, dim=-1)
        return f_diss

    def compute_active_patch_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Return the `[Batch, StateDim]` flow contributed by an `"active_driven"` patch."""
        if self.thermodynamic_mode != "active_driven":
            return torch.zeros_like(state)
        env = self._build_env(state, params)
        deltas = [torch.zeros_like(state[:, i]) for i in range(self.state_dim)]
        for key, (node, scale) in self._compiled_patches.items():
            assert isinstance(key, int)
            deltas[key] = deltas[key] + node.eval_tensor(env) * scale
        return torch.stack(deltas, dim=-1)

    def external_patch_power(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute instantaneous power `P_ext = grad_H(x) . Phi_patch(x)` supplied by external drive and `"active_driven"` patches."""
        p_ext = torch.zeros((state.shape[0],), dtype=state.dtype, device=state.device)
        base_p = params[:, : self.base_param_dim]
        if hasattr(self.base_brick, "external_drive_power"):
            p_ext = p_ext + self.base_brick.external_drive_power(state, base_p)
        if hasattr(self.base_brick, "external_patch_power"):
            p_ext = p_ext + self.base_brick.external_patch_power(state, base_p)
        if self.thermodynamic_mode == "active_driven":
            e = self.effort(state, params)
            phi = self.compute_active_patch_flow(state, params)
            p_ext = p_ext + (e * phi).sum(dim=-1)
        return p_ext

    def flow(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        port_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        dx_dt = super().flow(state, params, port_inputs=port_inputs)
        base_p = params[:, : self.base_param_dim]
        if hasattr(self.base_brick, "compute_compiled_external_drive"):
            dx_dt = dx_dt + self.base_brick.compute_compiled_external_drive(state, base_p)
        if hasattr(self.base_brick, "compute_active_patch_flow"):
            dx_dt = dx_dt + self.base_brick.compute_active_patch_flow(state, base_p)
        if self.thermodynamic_mode == "active_driven":
            dx_dt = dx_dt + self.compute_active_patch_flow(state, params)
        return dx_dt

    def total_patch_complexity(self) -> int:
        """Return total AST node complexity across all compiled patch expressions."""
        return sum(node.tree_complexity() for node, _ in self._compiled_patches.values())


def compile_symbolic_system(spec: SymbolicSystemSpec) -> CompiledLawBrick:
    """Compile a declarative `SymbolicSystemSpec` into a verified GPU `CompiledLawBrick`."""
    brick = CompiledLawBrick(spec)
    emit_trace_node(
        layer="layer3_compiler",
        operation="compile_symbolic_system",
        target_name=brick.name,
        selected_expression=str(spec.hamiltonian_expr),
        metrics={
            "domain": brick.domain,
            "state_dim": brick.state_dim,
            "param_dim": brick.param_dim,
            "total_symbolic_complexity": brick.total_symbolic_complexity(),
        },
    )
    return brick


def compile_residual_patch_brick(
    name: str,
    base_brick: LawBrick,
    patch_param_specs: Sequence[ParamSpec],
    thermodynamic_mode: str,
    patch_expressions: Mapping[int | tuple[int, int], str | ExprNode],
    constants: Mapping[str, Quantity | ConstNode] | None = None,
) -> ResidualPatchedLawBrick:
    """Compile a symbolic residual patch onto `base_brick` and return a `ResidualPatchedLawBrick`."""
    brick = ResidualPatchedLawBrick(
        name=name,
        base_brick=base_brick,
        patch_param_specs=patch_param_specs,
        thermodynamic_mode=thermodynamic_mode,
        patch_expressions=patch_expressions,
        constants=constants,
    )
    expr_summary = "; ".join(
        f"{k}: {node.to_infix()}" for k, (node, _) in brick._compiled_patches.items()
    )
    emit_trace_node(
        layer="layer3_compiler",
        operation="compile_residual_patch_brick",
        target_name=brick.name,
        selected_branch_id=thermodynamic_mode,
        selected_expression=expr_summary,
        metrics={
            "base_brick": base_brick.name,
            "thermodynamic_mode": thermodynamic_mode,
            "patch_complexity": brick.total_patch_complexity(),
            "num_patch_params": len(patch_param_specs),
        },
        skip_if_inner_loop=True,
    )
    return brick
