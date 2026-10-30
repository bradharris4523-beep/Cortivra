"""
Port-Hamiltonian subsystem contract and Dirac multi-domain interconnection.

Subsystems follow the open dynamical form:
    dx/dt = (J(x) - R(x)) * grad_H(x) + B(x) * u
or the matrix-free O(N) vector form:
    dx/dt = conservative_flow(x) - dissipative_flow(x) + external_flow(x)
where:
    - x in R^[Batch, N] is the batched state tensor.
    - H(x) is stored energy / thermodynamic potential.
    - e = grad_H(x) is the generalized effort.
    - J(x) = -J(x)^T routes energy conservatively (grad_H . conservative_flow == 0).
    - R(x) = R(x)^T >= 0 dissipates free energy (grad_H . dissipative_flow >= 0).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch

from cortivra.layer1_axioms.units import (
    DIM_ENERGY,
    DIM_POWER,
    DIMENSIONLESS,
    DimensionError,
     J,
    Unit,
    s,
)


@dataclass(frozen=True)
class StateVarSpec:
    """Metadata and physical bounds for a single state variable."""

    name: str
    unit: Unit
    default_value: float = 0.0
    hard_min: float = -math.inf
    hard_max: float = math.inf
    homeostatic_min: float = -math.inf
    homeostatic_max: float = math.inf
    valence: int = 0
    molar_mass_g_mol: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("StateVarSpec name cannot be empty.")
        if self.hard_min > self.hard_max:
            raise ValueError(
                f"State '{self.name}': hard_min ({self.hard_min}) > hard_max ({self.hard_max})."
            )
        if self.homeostatic_min > self.homeostatic_max:
            raise ValueError(
                f"State '{self.name}': homeostatic_min ({self.homeostatic_min}) > "
                f"homeostatic_max ({self.homeostatic_max})."
            )
        if not (self.hard_min <= self.default_value <= self.hard_max):
            raise ValueError(
                f"State '{self.name}': default_value ({self.default_value}) outside "
                f"hard bounds [{self.hard_min}, {self.hard_max}]."
            )


@dataclass(frozen=True)
class ParamSpec:
    """Metadata and physical bounds for a model parameter."""

    name: str
    unit: Unit
    default_value: float
    min_value: float = -math.inf
    max_value: float = math.inf
    tunable: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("ParamSpec name cannot be empty.")
        if self.min_value > self.max_value:
            raise ValueError(
                f"Param '{self.name}': min_value ({self.min_value}) > max_value ({self.max_value})."
            )
        if not (self.min_value <= self.default_value <= self.max_value):
            raise ValueError(
                f"Param '{self.name}': default_value ({self.default_value}) outside "
                f"bounds [{self.min_value}, {self.max_value}]."
            )


@dataclass(frozen=True)
class PowerPortSpec:
    """
    Interaction port pairing an effort variable `e` and flow variable `f`.
    When `require_power_conjugate=True`, enforces `Dimension(effort_unit * flow_unit) == DIM_POWER` (Watts).
    """

    name: str
    effort_unit: Unit
    flow_unit: Unit
    domain: str = "universal"
    state_var_name: str | None = None
    require_power_conjugate: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("PowerPortSpec name cannot be empty.")
        if self.require_power_conjugate:
            product_dim = (self.effort_unit * self.flow_unit).dimension
            if product_dim != DIM_POWER:
                raise DimensionError(
                    f"Port '{self.name}' ({self.domain}) is not power-conjugate: "
                    f"'{self.effort_unit.symbol}' * '{self.flow_unit.symbol}' "
                    f"has dimension {product_dim.format_si()}, expected {DIM_POWER.format_si()}."
                )

    @property
    def power_scale_to_watts(self) -> float:
        return (self.effort_unit * self.flow_unit).scale


@dataclass(frozen=True)
class SpatialNode3DSpec:
    """
    Universal 3D spatial node descriptor mapping state variables onto continuous
    `R^3` particle/body coordinates `(x, y, z)` or a 3D compartment center.
    """

    node_id: str
    position_state_names: tuple[str, ...] = ()
    momentum_state_names: tuple[str, ...] = ()
    fixed_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    volume_m3: float = 1.0
    radius_m: float = 1.0
    associated_state_names: tuple[str, ...] = ()


class LawBrick(ABC):
    """Base class for port-Hamiltonian subsystems operating on batched state tensors `[Batch, StateDim]`."""

    def __init__(
        self,
        name: str,
        domain: str,
        state_specs: Sequence[StateVarSpec],
        param_specs: Sequence[ParamSpec] = (),
        port_specs: Sequence[PowerPortSpec] = (),
        time_unit: Unit = s,
        energy_unit: Unit = J,
        matrix_free: bool = False,
        thermal_sink_state_name: str | None = None,
        thermal_capacity_scale: float = 1.0,
        spatial_nodes_3d: Sequence[SpatialNode3DSpec] = (),
    ) -> None:
        if not state_specs:
            raise ValueError(f"LawBrick '{name}' must define at least one StateVarSpec.")
        if energy_unit.dimension != DIM_ENERGY and not energy_unit.dimension.is_dimensionless:
            raise DimensionError(
                f"LawBrick '{name}' energy_unit '{energy_unit.symbol}' has dimension "
                f"{energy_unit.dimension.format_si()}, expected energy ({DIM_ENERGY.format_si()}) or dimensionless."
            )
        self.name = name
        self.domain = domain
        self.state_specs = tuple(state_specs)
        self.param_specs = tuple(param_specs)
        self.port_specs = tuple(port_specs)
        self.time_unit = time_unit
        self.energy_unit = energy_unit
        self.is_matrix_free = matrix_free
        self.spatial_nodes_3d = tuple(spatial_nodes_3d)

        self._state_index = {spec.name: idx for idx, spec in enumerate(self.state_specs)}
        self._param_index = {spec.name: idx for idx, spec in enumerate(self.param_specs)}
        self._port_index = {spec.name: idx for idx, spec in enumerate(self.port_specs)}

        if len(self._state_index) != len(self.state_specs):
            raise ValueError(f"LawBrick '{name}' has duplicate state variable names.")
        if len(self._param_index) != len(self.param_specs):
            raise ValueError(f"LawBrick '{name}' has duplicate parameter names.")

        if thermal_sink_state_name is not None:
            if thermal_sink_state_name not in self._state_index:
                raise ValueError(
                    f"LawBrick '{name}' thermal_sink_state_name '{thermal_sink_state_name}' "
                    f"is not in state_specs."
                )
            if thermal_capacity_scale <= 0.0:
                raise ValueError(
                    f"LawBrick '{name}' thermal_capacity_scale must be positive, got {thermal_capacity_scale}."
                )
            self.thermal_sink_index: int | None = self._state_index[thermal_sink_state_name]
        else:
            self.thermal_sink_index = None
        self.thermal_sink_state_name = thermal_sink_state_name
        self.thermal_capacity_scale = float(thermal_capacity_scale)

    @property
    def state_dim(self) -> int:
        return len(self.state_specs)

    @property
    def param_dim(self) -> int:
        return len(self.param_specs)

    def state_idx(self, name: str) -> int:
        return self._state_index[name]

    def param_idx(self, name: str) -> int:
        return self._param_index[name]

    def default_state(
        self,
        batch_size: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        vals = [spec.default_value for spec in self.state_specs]
        base = torch.tensor(vals, dtype=dtype, device=device).unsqueeze(0)
        return base.expand(batch_size, -1).clone()

    def default_params(
        self,
        batch_size: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if not self.param_specs:
            return torch.empty((batch_size, 0), dtype=dtype, device=device)
        vals = [spec.default_value for spec in self.param_specs]
        base = torch.tensor(vals, dtype=dtype, device=device).unsqueeze(0)
        return base.expand(batch_size, -1).clone()

    def clamp_to_hard_bounds(self, state: torch.Tensor) -> torch.Tensor:
        device, dtype = state.device, state.dtype
        mins = torch.tensor([sp.hard_min for sp in self.state_specs], dtype=dtype, device=device)
        maxs = torch.tensor([sp.hard_max for sp in self.state_specs], dtype=dtype, device=device)
        return torch.clamp(state, min=mins, max=maxs)

    def clamp_params(self, params: torch.Tensor) -> torch.Tensor:
        """Clamp parameter tensor `[Batch, ParamDim]` to `[min_value, max_value]` bounds."""
        if not self.param_specs or params.numel() == 0:
            return params
        device, dtype = params.device, params.dtype
        mins = torch.tensor([pp.min_value for pp in self.param_specs], dtype=dtype, device=device)
        maxs = torch.tensor([pp.max_value for pp in self.param_specs], dtype=dtype, device=device)
        return torch.clamp(params, min=mins, max=maxs)

    def validate_params(self, params: torch.Tensor) -> None:
        """Raise `ValueError` if any parameter in `params` is non-finite or outside `[min_value, max_value]`."""
        if not self.param_specs or params.numel() == 0:
            return
        if not torch.isfinite(params).all():
            raise ValueError(f"LawBrick '{self.name}' received non-finite parameters.")
        for idx, pp in enumerate(self.param_specs):
            col = params[:, idx]
            if bool(torch.any(col < pp.min_value).item()) or bool(torch.any(col > pp.max_value).item()):
                v_min = float(torch.min(col).item())
                v_max = float(torch.max(col).item())
                raise ValueError(
                    f"LawBrick '{self.name}' parameter '{pp.name}' out of bounds "
                    f"[{pp.min_value}, {pp.max_value}]: got range [{v_min}, {v_max}]."
                )

    @abstractmethod
    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute stored energy `H(x)` of shape `[Batch]`."""

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute generalized effort `e(x) = grad_x H(x)` of shape `[Batch, StateDim]`."""
        need_graph = torch.is_grad_enabled() and (state.requires_grad or params.requires_grad)
        with torch.enable_grad():
            x = state if state.requires_grad else state.detach().requires_grad_(True)
            energy = self.hamiltonian_energy(x, params)
            grad = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=x,
                create_graph=need_graph,
            )[0]
        return grad

    def structure_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Skew-symmetric routing matrix `J(x)` of shape `[Batch, StateDim, StateDim]`."""
        batch = state.shape[0]
        n = self.state_dim
        return torch.zeros((batch, n, n), dtype=state.dtype, device=state.device)

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Symmetric positive semi-definite matrix `R(x)` of shape `[Batch, StateDim, StateDim]`."""
        batch = state.shape[0]
        n = self.state_dim
        return torch.zeros((batch, n, n), dtype=state.dtype, device=state.device)

    def conservative_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Energy-conserving flow vector `F_cons(x)` (`[Batch, StateDim]`) satisfying
        `grad_H(x) . F_cons(x) == 0`. Subclasses with `matrix_free=True` override this directly.
        """
        e = self.effort(state, params).unsqueeze(-1)
        j_mat = self.structure_matrix(state, params)
        return torch.bmm(j_mat, e).squeeze(-1)

    def dissipative_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Entropy-dissipating flow vector `F_diss(x)` (`[Batch, StateDim]`) satisfying
        `grad_H(x) . F_diss(x) >= 0`. Subclasses with `matrix_free=True` override this directly.
        """
        e = self.effort(state, params).unsqueeze(-1)
        r_mat = self.dissipation_matrix(state, params)
        return torch.bmm(r_mat, e).squeeze(-1)

    def dissipation_power(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Instantaneous dissipated power `D(x) = grad_H(x) . F_diss(x)` of shape `[Batch]`."""
        e = self.effort(state, params)
        f_diss = self.dissipative_flow(state, params)
        return (e * f_diss).sum(dim=-1)

    def thermal_sink_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        GENERIC 1st-Law internal thermal/entropy production flow vector (`[Batch, StateDim]`).
        Routes 100% of dissipated free-energy power `D(x) = grad_H(x) . F_diss(x) >= 0` into
        the designated `thermal_sink_index` scaled by `thermal_capacity_scale = 1 / C_th`.
        """
        out = torch.zeros_like(state)
        if self.thermal_sink_index is not None:
            diss_p = self.dissipation_power(state, params)
            out[:, self.thermal_sink_index] = diss_p * self.thermal_capacity_scale
        return out

    def total_conserved_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Compute total GENERIC 1st-Law energy `E_total(x) = H_work(x) + E_thermal(x)` (`[Batch]`).
        For an isolated dissipative system (`P_ext == 0`) with a designated thermal sink state,
        `dE_total / dt == 0` even as `H_work(x)` dissipates into `E_thermal(x)`.
        """
        h_work = self.hamiltonian_energy(state, params)
        if self.thermal_sink_index is not None:
            e_th = state[:, self.thermal_sink_index] / self.thermal_capacity_scale
            return h_work + e_th
        return h_work

    def extract_3d_coordinates(self, state: torch.Tensor) -> torch.Tensor:
        """
        Extract universal 3D Euclidean coordinates `(x, y, z)` for all spatial nodes in the system.
        Accepts `[Batch, StateDim]`, `[Time, StateDim]`, or `[Batch, Time, StateDim]` and returns
        `[..., NumNodes, 3]`.
        """
        lead_shape = state.shape[:-1]
        flat = state.reshape(-1, self.state_dim)
        b_flat = flat.shape[0]
        device, dtype = state.device, state.dtype

        if self.spatial_nodes_3d:
            nodes_out: list[torch.Tensor] = []
            for node_spec in self.spatial_nodes_3d:
                coords = [
                    torch.full((b_flat,), float(node_spec.fixed_position_m[axis]), dtype=dtype, device=device)
                    for axis in range(3)
                ]
                for axis, st_name in enumerate(node_spec.position_state_names[:3]):
                    if st_name in self._state_index:
                        coords[axis] = flat[:, self._state_index[st_name]]
                nodes_out.append(torch.stack(coords, dim=-1))
            stacked = torch.stack(nodes_out, dim=1)
            return stacked.reshape(*lead_shape, len(self.spatial_nodes_3d), 3)

        # Automatic 3D coordinate inference when spatial_nodes_3d is not explicitly specified
        names = [sp.name for sp in self.state_specs]
        if {"x", "y", "z"}.issubset(self._state_index):
            xyz = torch.stack(
                [flat[:, self._state_index["x"]], flat[:, self._state_index["y"]], flat[:, self._state_index["z"]]],
                dim=-1,
            ).unsqueeze(1)
            return xyz.reshape(*lead_shape, 1, 3)
        if {"x", "y"}.issubset(self._state_index):
            xyz = torch.stack(
                [
                    flat[:, self._state_index["x"]],
                    flat[:, self._state_index["y"]],
                    torch.zeros(b_flat, dtype=dtype, device=device),
                ],
                dim=-1,
            ).unsqueeze(1)
            return xyz.reshape(*lead_shape, 1, 3)

        # Fallback: embed state phase-space or compartment nodes into R^3
        if self.state_dim >= 3:
            xyz = flat[:, :3].unsqueeze(1)
            return xyz.reshape(*lead_shape, 1, 3)
        if self.state_dim == 2:
            xyz = torch.stack(
                [flat[:, 0], flat[:, 1], torch.zeros(b_flat, dtype=dtype, device=device)],
                dim=-1,
            ).unsqueeze(1)
            return xyz.reshape(*lead_shape, 1, 3)
        xyz = torch.stack(
            [
                flat[:, 0],
                torch.zeros(b_flat, dtype=dtype, device=device),
                torch.zeros(b_flat, dtype=dtype, device=device),
            ],
            dim=-1,
        ).unsqueeze(1)
        return xyz.reshape(*lead_shape, 1, 3)

    def compute_3d_spatial_geometry_summary(
        self,
        states: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """
        Compute universal 3D Euclidean compartment geometry, pairwise inter-node distances,
        geometric Fickian/electrostatic conductance factors `A_ij / d_ij`, and 3D trajectory
        arc-length / bounding-box kinematics.
        """
        nodes_summary: list[dict[str, object]] = []
        for nd in self.spatial_nodes_3d:
            nodes_summary.append(
                {
                    "node_id": nd.node_id,
                    "fixed_position_m": [float(f"{float(c):.6g}") for c in nd.fixed_position_m],
                    "volume_m3": float(f"{float(nd.volume_m3):.6g}"),
                    "radius_m": float(f"{float(nd.radius_m):.6g}"),
                    "position_state_names": list(nd.position_state_names),
                    "associated_state_names": list(nd.associated_state_names),
                }
            )

        pairwise_couplings: list[dict[str, object]] = []
        n_nodes = len(self.spatial_nodes_3d)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                ni = self.spatial_nodes_3d[i]
                nj = self.spatial_nodes_3d[j]
                dx = float(ni.fixed_position_m[0] - nj.fixed_position_m[0])
                dy = float(ni.fixed_position_m[1] - nj.fixed_position_m[1])
                dz = float(ni.fixed_position_m[2] - nj.fixed_position_m[2])
                dist_m = math.sqrt(dx * dx + dy * dy + dz * dz)
                eff_r = min(float(ni.radius_m), float(nj.radius_m))
                cross_area_m2 = math.pi * eff_r * eff_r
                geom_factor_m = cross_area_m2 / max(dist_m, 1e-9) if dist_m > 0.0 else 0.0
                pairwise_couplings.append(
                    {
                        "pair": f"{ni.node_id}<->{nj.node_id}",
                        "euclidean_distance_m": float(f"{dist_m:.6g}"),
                        "effective_cross_section_m2": float(f"{cross_area_m2:.6g}"),
                        "fickian_geometric_conductance_factor_m": float(f"{geom_factor_m:.6g}"),
                    }
                )

        kinematics: dict[str, object] = {}
        if states is not None and states.numel() > 0:
            s_t = states[0] if states.ndim == 3 else states
            coords_3d = self.extract_3d_coordinates(s_t)  # [T, NumNodes, 3]
            if coords_3d.shape[0] >= 2:
                step_disp = torch.linalg.norm(coords_3d[1:] - coords_3d[:-1], dim=-1)  # [T-1, NumNodes]
                arc_lengths = torch.sum(step_disp, dim=0)  # [NumNodes]
                bbox_min = torch.min(coords_3d.reshape(-1, 3), dim=0).values
                bbox_max = torch.max(coords_3d.reshape(-1, 3), dim=0).values
                bbox_span = bbox_max - bbox_min
                kinematics = {
                    "num_3d_nodes_tracked": int(coords_3d.shape[1]),
                    "node_arc_lengths_m": [round(float(v), 6) for v in arc_lengths.tolist()],
                    "bounding_box_span_xyz": [round(float(v), 6) for v in bbox_span.tolist()],
                }
                if coords_3d.shape[1] >= 2:
                    sep_t = torch.linalg.norm(coords_3d[:, 0, :] - coords_3d[:, 1, :], dim=-1)
                    kinematics["min_inter_node_separation_m"] = round(float(torch.min(sep_t).item()), 6)
                    kinematics["max_inter_node_separation_m"] = round(float(torch.max(sep_t).item()), 6)

        return {
            "num_spatial_nodes_3d": n_nodes,
            "spatial_nodes": nodes_summary,
            "pairwise_3d_geometric_couplings": pairwise_couplings,
            "trajectory_3d_kinematics": kinematics,
        }

    def flow(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        port_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute time derivative `dx/dt` (`[Batch, StateDim]`) in units of `state_unit / time_unit`."""
        dx_dt = self.conservative_flow(state, params) - self.dissipative_flow(state, params)
        if self.thermal_sink_index is not None:
            dx_dt = dx_dt + self.thermal_sink_flow(state, params)
        if port_inputs and "external_flow" in port_inputs:
            dx_dt = dx_dt + port_inputs["external_flow"]
        return dx_dt

    def step_rk4(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        dt: float,
        port_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Advance state by one 4th-order Runge-Kutta step of duration `dt`."""
        k1 = self.flow(state, params, port_inputs)
        k2 = self.flow(state + 0.5 * dt * k1, params, port_inputs)
        k3 = self.flow(state + 0.5 * dt * k2, params, port_inputs)
        k4 = self.flow(state + dt * k3, params, port_inputs)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.clamp_to_hard_bounds(next_state)

    def step_symplectic_midpoint(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        dt: float,
        port_inputs: dict[str, torch.Tensor] | None = None,
        fixed_point_iters: int = 8,
        tol: float = 1e-7,
    ) -> torch.Tensor:
        """
        Implicit midpoint integrator (`x_{n+1} = x_n + dt * flow(0.5 * (x_n + x_{n+1}))`).
        Preserves quadratic Hamiltonian invariants across discrete steps.
        """
        x_next = state + dt * self.flow(state, params, port_inputs)
        for _ in range(fixed_point_iters):
            x_mid = 0.5 * (state + x_next)
            x_candidate = state + dt * self.flow(x_mid, params, port_inputs)
            delta = float(torch.max(torch.abs(x_candidate - x_next)).detach().item())
            x_next = x_candidate
            if delta <= tol:
                break
        return self.clamp_to_hard_bounds(x_next)


class CanonicalHamiltonianBrick(LawBrick):
    """
    2-state port-Hamiltonian oscillator (`x = [q, p]`) with Hamiltonian
    `H(q, p) = 0.5 * k * q^2 + 0.5 * (p^2 / m)` and viscous damping `d`.
    """

    def __init__(
        self,
        name: str = "canonical_oscillator",
        domain: str = "mechanics",
        q_unit: Unit = DIMENSIONLESS,
        p_unit: Unit = DIMENSIONLESS,
        k_default: float = 1.0,
        m_default: float = 1.0,
        damping_default: float = 0.0,
        port_specs: Sequence[PowerPortSpec] = (),
    ) -> None:
        state_specs = [
            StateVarSpec(
                name="q",
                unit=q_unit,
                default_value=1.0,
                homeostatic_min=-100.0,
                homeostatic_max=100.0,
                description="Generalized coordinate / charge",
            ),
            StateVarSpec(
                name="p",
                unit=p_unit,
                default_value=0.0,
                homeostatic_min=-100.0,
                homeostatic_max=100.0,
                description="Generalized momentum / flux",
            ),
        ]
        param_specs = [
            ParamSpec("k", DIMENSIONLESS, default_value=k_default, min_value=0.0),
            ParamSpec("m", DIMENSIONLESS, default_value=m_default, min_value=1e-9),
            ParamSpec("damping", DIMENSIONLESS, default_value=damping_default, min_value=0.0),
        ]
        super().__init__(
            name=name,
            domain=domain,
            state_specs=state_specs,
            param_specs=param_specs,
            port_specs=port_specs,
        )

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        q = state[:, 0]
        p = state[:, 1]
        k = params[:, 0]
        m = params[:, 1]
        return 0.5 * k * (q**2) + 0.5 * (p**2) / m

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        q = state[:, 0]
        p = state[:, 1]
        k = params[:, 0]
        m = params[:, 1]
        return torch.stack([k * q, p / m], dim=-1)

    def structure_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        j_single = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=state.dtype, device=state.device)
        return j_single.unsqueeze(0).expand(batch, -1, -1).clone()

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        damping = params[:, 2]
        r_mat = torch.zeros((batch, 2, 2), dtype=state.dtype, device=state.device)
        r_mat[:, 1, 1] = damping
        return r_mat


@dataclass(frozen=True)
class PortConnection:
    """Skew-symmetric Dirac coupling between `source_brick.source_port` and `target_brick.target_port`."""

    source_brick: str
    source_port: PowerPortSpec
    target_brick: str
    target_port: PowerPortSpec
    coupling_gain: float = 1.0

    def __post_init__(self) -> None:
        src_power_dim = (self.source_port.effort_unit * self.source_port.flow_unit).dimension
        dst_power_dim = (self.target_port.effort_unit * self.target_port.flow_unit).dimension
        if src_power_dim != dst_power_dim:
            raise DimensionError(
                f"Cannot connect '{self.source_brick}.{self.source_port.name}' "
                f"({src_power_dim.format_si()}) to '{self.target_brick}.{self.target_port.name}' "
                f"({dst_power_dim.format_si()}): power dimensions do not match."
            )


class CoupledPortHamiltonianSystem(LawBrick):
    """
    Composite port-Hamiltonian system coupling multiple `LawBrick` instances via
    skew-symmetric Dirac interconnections (`J_total = block_diag(tau_b * J_b) + J_Dirac`),
    with automatic timescale conversion `tau_b = self.time_unit.scale / b.time_unit.scale`.
    """

    def __init__(
        self,
        name: str,
        bricks: Sequence[LawBrick],
        connections: Sequence[PortConnection] = (),
        time_unit: Unit = s,
        energy_unit: Unit = J,
        thermal_sink_state_name: str | None = None,
        thermal_capacity_scale: float = 1.0,
    ) -> None:
        if len(bricks) < 1:
            raise ValueError("CoupledPortHamiltonianSystem requires at least one LawBrick.")

        self.bricks = tuple(bricks)
        self.connections = tuple(connections)
        self._brick_map: dict[str, LawBrick] = {}
        self._state_slices: dict[str, slice] = {}
        self._param_slices: dict[str, slice] = {}

        combined_states: list[StateVarSpec] = []
        combined_params: list[ParamSpec] = []
        combined_ports: list[PowerPortSpec] = []
        combined_nodes_3d: list[SpatialNode3DSpec] = []

        s_offset = 0
        p_offset = 0
        for b in self.bricks:
            if b.name in self._brick_map:
                raise ValueError(f"Duplicate brick name '{b.name}' in CoupledPortHamiltonianSystem.")
            self._brick_map[b.name] = b
            self._state_slices[b.name] = slice(s_offset, s_offset + b.state_dim)
            self._param_slices[b.name] = slice(p_offset, p_offset + b.param_dim)
            s_offset += b.state_dim
            p_offset += b.param_dim

            for sp in b.state_specs:
                combined_states.append(
                    StateVarSpec(
                        name=f"{b.name}.{sp.name}",
                        unit=sp.unit,
                        default_value=sp.default_value,
                        hard_min=sp.hard_min,
                        hard_max=sp.hard_max,
                        homeostatic_min=sp.homeostatic_min,
                        homeostatic_max=sp.homeostatic_max,
                        valence=sp.valence,
                        molar_mass_g_mol=sp.molar_mass_g_mol,
                        description=sp.description,
                    )
                )
            for pp in b.param_specs:
                combined_params.append(
                    ParamSpec(
                        name=f"{b.name}.{pp.name}",
                        unit=pp.unit,
                        default_value=pp.default_value,
                        min_value=pp.min_value,
                        max_value=pp.max_value,
                        tunable=pp.tunable,
                        description=pp.description,
                    )
                )
            for port in b.port_specs:
                combined_ports.append(
                    PowerPortSpec(
                        name=f"{b.name}.{port.name}",
                        effort_unit=port.effort_unit,
                        flow_unit=port.flow_unit,
                        domain=port.domain,
                        state_var_name=f"{b.name}.{port.state_var_name}" if port.state_var_name else None,
                        require_power_conjugate=port.require_power_conjugate,
                        description=port.description,
                    )
                )
            for n3d in b.spatial_nodes_3d:
                combined_nodes_3d.append(
                    SpatialNode3DSpec(
                        node_id=f"{b.name}.{n3d.node_id}",
                        position_state_names=tuple(f"{b.name}.{sn}" for sn in n3d.position_state_names),
                        momentum_state_names=tuple(f"{b.name}.{mn}" for mn in n3d.momentum_state_names),
                        fixed_position_m=n3d.fixed_position_m,
                        volume_m3=n3d.volume_m3,
                        radius_m=n3d.radius_m,
                        associated_state_names=tuple(f"{b.name}.{an}" for an in n3d.associated_state_names),
                    )
                )

        super().__init__(
            name=name,
            domain="multi_domain_coupled",
            state_specs=combined_states,
            param_specs=combined_params,
            port_specs=combined_ports,
            time_unit=time_unit,
            energy_unit=energy_unit,
            matrix_free=any(b.is_matrix_free for b in self.bricks),
            thermal_sink_state_name=thermal_sink_state_name,
            thermal_capacity_scale=thermal_capacity_scale,
            spatial_nodes_3d=combined_nodes_3d,
        )

        self._time_scales: dict[str, float] = {
            b.name: (self.time_unit.scale / b.time_unit.scale) for b in self.bricks
        }
        self._energy_scales: dict[str, float] = {
            b.name: (b.energy_unit.scale / self.energy_unit.scale) for b in self.bricks
        }

        self._dirac_pairs: list[tuple[int, int, float]] = []
        for conn in self.connections:
            if conn.source_brick not in self._brick_map:
                raise KeyError(f"Unknown source_brick '{conn.source_brick}' in PortConnection.")
            if conn.target_brick not in self._brick_map:
                raise KeyError(f"Unknown target_brick '{conn.target_brick}' in PortConnection.")

            src_var = conn.source_port.state_var_name or self._brick_map[conn.source_brick].state_specs[-1].name
            dst_var = conn.target_port.state_var_name or self._brick_map[conn.target_brick].state_specs[-1].name

            src_global_idx = self.state_idx(f"{conn.source_brick}.{src_var}")
            dst_global_idx = self.state_idx(f"{conn.target_brick}.{dst_var}")
            self._dirac_pairs.append((src_global_idx, dst_global_idx, conn.coupling_gain))

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        total_h = torch.zeros(state.shape[0], dtype=state.dtype, device=state.device)
        for b in self.bricks:
            s_slice = self._state_slices[b.name]
            p_slice = self._param_slices[b.name]
            alpha_b = self._energy_scales[b.name]
            total_h = total_h + alpha_b * b.hamiltonian_energy(state[:, s_slice], params[:, p_slice])
        return total_h

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        efforts = [
            self._energy_scales[b.name]
            * b.effort(state[:, self._state_slices[b.name]], params[:, self._param_slices[b.name]])
            for b in self.bricks
        ]
        return torch.cat(efforts, dim=-1)

    def conservative_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        f_parts = []
        for b in self.bricks:
            s_slice = self._state_slices[b.name]
            p_slice = self._param_slices[b.name]
            tau_b = self._time_scales[b.name]
            f_parts.append(tau_b * b.conservative_flow(state[:, s_slice], params[:, p_slice]))
        f_cons = torch.cat(f_parts, dim=-1)

        if self._dirac_pairs:
            e = self.effort(state, params)
            dirac_deltas = [torch.zeros_like(f_cons[:, i]) for i in range(self.state_dim)]
            for src_idx, dst_idx, gamma in self._dirac_pairs:
                dirac_deltas[src_idx] = dirac_deltas[src_idx] + gamma * e[:, dst_idx]
                dirac_deltas[dst_idx] = dirac_deltas[dst_idx] - gamma * e[:, src_idx]
            f_cons = f_cons + torch.stack(dirac_deltas, dim=-1)
        return f_cons

    def dissipative_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        f_parts = []
        for b in self.bricks:
            s_slice = self._state_slices[b.name]
            p_slice = self._param_slices[b.name]
            tau_b = self._time_scales[b.name]
            f_parts.append(tau_b * b.dissipative_flow(state[:, s_slice], params[:, p_slice]))
        return torch.cat(f_parts, dim=-1)

    def structure_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        n = self.state_dim
        j_total = torch.zeros((batch, n, n), dtype=state.dtype, device=state.device)

        for b in self.bricks:
            s_slice = self._state_slices[b.name]
            p_slice = self._param_slices[b.name]
            tau_b = self._time_scales[b.name]
            alpha_b = self._energy_scales[b.name]
            j_total[:, s_slice, s_slice] = (tau_b / alpha_b) * b.structure_matrix(
                state[:, s_slice], params[:, p_slice]
            )

        for src_idx, dst_idx, gamma in self._dirac_pairs:
            j_total[:, src_idx, dst_idx] += gamma
            j_total[:, dst_idx, src_idx] -= gamma

        return j_total

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        n = self.state_dim
        r_total = torch.zeros((batch, n, n), dtype=state.dtype, device=state.device)

        for b in self.bricks:
            s_slice = self._state_slices[b.name]
            p_slice = self._param_slices[b.name]
            tau_b = self._time_scales[b.name]
            alpha_b = self._energy_scales[b.name]
            r_total[:, s_slice, s_slice] = (tau_b / alpha_b) * b.dissipation_matrix(
                state[:, s_slice], params[:, p_slice]
            )

        return r_total

    def external_drive_power(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Aggregate external drive power across all child bricks in system energy/time units."""
        e = self.effort(state, params)
        f_ext_parts: list[torch.Tensor] = []
        for b in self.bricks:
            s_slice = self._state_slices[b.name]
            p_slice = self._param_slices[b.name]
            tau_b = self._time_scales[b.name]
            x_b = state[:, s_slice]
            p_b = params[:, p_slice]
            f_b_total = b.flow(x_b, p_b)
            f_b_jr = b.conservative_flow(x_b, p_b) - b.dissipative_flow(x_b, p_b)
            if b.thermal_sink_index is not None:
                f_b_jr = f_b_jr + b.thermal_sink_flow(x_b, p_b)
            f_ext_parts.append(tau_b * (f_b_total - f_b_jr))
        f_ext_vec = torch.cat(f_ext_parts, dim=-1)
        return (e * f_ext_vec).sum(dim=-1)

    def flow(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        port_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Compute composite flow `dx/dt` combining Dirac-coupled conservative flows,
        dissipative flows, child external/thermal flows, and composite thermal sink routing.
        """
        dx_dt = self.conservative_flow(state, params) - self.dissipative_flow(state, params)
        f_extra_parts: list[torch.Tensor] = []
        for b in self.bricks:
            s_slice = self._state_slices[b.name]
            p_slice = self._param_slices[b.name]
            tau_b = self._time_scales[b.name]
            x_b = state[:, s_slice]
            p_b = params[:, p_slice]
            f_b_total = b.flow(x_b, p_b)
            f_b_jr = b.conservative_flow(x_b, p_b) - b.dissipative_flow(x_b, p_b)
            f_extra_parts.append(tau_b * (f_b_total - f_b_jr))
        dx_dt = dx_dt + torch.cat(f_extra_parts, dim=-1)
        if self.thermal_sink_index is not None:
            dx_dt = dx_dt + self.thermal_sink_flow(state, params)
        if port_inputs and "external_flow" in port_inputs:
            dx_dt = dx_dt + port_inputs["external_flow"]
        return dx_dt
