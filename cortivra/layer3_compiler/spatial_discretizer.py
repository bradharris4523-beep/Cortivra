"""
Parametric physical geometry discretizer and spatial compartment graph compiler
(`cortivra.layer3_compiler.spatial_discretizer`).

Computes exact physical volumes, surface areas, capacitances, Faraday flux-to-concentration
conversion factors, and sparse inter-compartment coupling graphs (`edge_index`) from either:
    1. Parametric geometric primitives (`sphere`, `cylinder`, `cuboid`) with Layer 1 `Quantity` dimensions.
    2. Connected multi-compartment graph specifications or 2D/3D integer-labeled spatial masks
       (computing exact voxel polyhedral surface areas, shared interface areas, and centroid distances).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import (
    LawBrick,
    ParamSpec,
    StateVarSpec,
)
from cortivra.layer1_axioms.units import (
    DIM_AREA,
    DIM_CAPACITANCE,
    DIM_DIFFUSIVITY,
    DIM_LENGTH,
    DIMENSIONLESS,
    FARADAY,
    Quantity,
    Unit,
    fL,
    m,
    m2,
    m2_per_s,
    m3,
    mM,
    pF,
    s,
    uF_per_cm2,
    um,
    um2,
)
from cortivra.layer2_fabric.transport_and_fields import SpatialAdvectionDiffusionBrick

DIM_SPECIFIC_CAPACITANCE = DIM_CAPACITANCE / DIM_AREA


@dataclass(frozen=True)
class CompartmentGeometry:
    """
    Physical geometry of a single spatial compartment derived from parametric dimensions
    or segmented voxel regions.

    All fields are exact Layer 1 `Quantity` instances:
        - `volume`: Compartment volume (`m^3` or `fL`).
        - `surface_area`: Boundary surface area (`m^2` or `um^2`).
        - `cross_section_area`: Characteristic cross-sectional area (`m^2` or `um^2`).
        - `characteristic_length`: Characteristic axial length or diameter (`m` or `um`).
    """

    name: str
    shape: str  # "sphere", "cylinder", "cuboid", or "voxel_region"
    volume: Quantity
    surface_area: Quantity
    cross_section_area: Quantity
    characteristic_length: Quantity

    @property
    def surface_to_volume_ratio(self) -> Quantity:
        """Return boundary surface-to-volume ratio `A_surface / Volume` (`1/m` or `1/um`)."""
        return self.surface_area / self.volume

    def total_capacitance(self, specific_capacitance: Quantity = Quantity(1.0, uF_per_cm2)) -> Quantity:
        """Compute total boundary capacitance `C = c_spec * A_surface` in `pF`."""
        specific_capacitance.assert_dimension(DIM_SPECIFIC_CAPACITANCE, context="specific_capacitance")
        return (specific_capacitance * self.surface_area).to(pF)

    def faraday_flux_to_concentration_rate_factor(
        self,
        valence_z: int = 1,
        current_unit: Unit | None = None,
        conc_rate_unit: Unit | None = None,
    ) -> float:
        """
        Compute the exact stoichiometric conversion factor `alpha` from electrical current `I`
        (default `pA`) to concentration rate `dc/dt` (default `mM / s`):
            `dc/dt = -alpha * I` where `alpha = 1 / (|z| * F * Volume)`
        """
        if valence_z == 0:
            raise ValueError("Valence `valence_z` must be non-zero.")
        from cortivra.layer1_axioms.units import pA

        i_u = current_unit if current_unit is not None else pA
        dc_dt_u = conc_rate_unit if conc_rate_unit is not None else (mM / s)

        one_unit_current = Quantity(1.0, i_u)
        molar_flux = one_unit_current / (abs(valence_z) * FARADAY)
        conc_rate = (molar_flux / self.volume).to(dc_dt_u)
        return float(conc_rate.value)


def make_sphere_compartment(name: str, radius: Quantity) -> CompartmentGeometry:
    """Construct a spherical `CompartmentGeometry` from physical `radius` (`DIM_LENGTH`)."""
    radius.assert_dimension(DIM_LENGTH, context=f"sphere '{name}' radius")
    r_m = float(radius.to(m).value)
    if r_m <= 0.0:
        raise ValueError(f"Sphere '{name}' radius must be positive, got {radius}.")

    vol = Quantity((4.0 / 3.0) * math.pi * (r_m**3), m3)
    surf = Quantity(4.0 * math.pi * (r_m**2), m2)
    cross = Quantity(math.pi * (r_m**2), m2)
    length = Quantity(2.0 * r_m, m)

    if radius.unit == um:
        return CompartmentGeometry(
            name=name,
            shape="sphere",
            volume=vol.to(fL),
            surface_area=surf.to(um2),
            cross_section_area=cross.to(um2),
            characteristic_length=length.to(um),
        )
    return CompartmentGeometry(
        name=name,
        shape="sphere",
        volume=vol,
        surface_area=surf,
        cross_section_area=cross,
        characteristic_length=length,
    )


def make_cylinder_compartment(
    name: str,
    radius: Quantity,
    length: Quantity,
    include_end_caps: bool = False,
) -> CompartmentGeometry:
    """Construct a cylindrical `CompartmentGeometry` from physical `radius` and `length` (`DIM_LENGTH`)."""
    radius.assert_dimension(DIM_LENGTH, context=f"cylinder '{name}' radius")
    length.assert_dimension(DIM_LENGTH, context=f"cylinder '{name}' length")
    r_m = float(radius.to(m).value)
    l_m = float(length.to(m).value)
    if r_m <= 0.0 or l_m <= 0.0:
        raise ValueError(f"Cylinder '{name}' radius and length must be positive.")

    cross_m2 = math.pi * (r_m**2)
    vol_m3 = cross_m2 * l_m
    surf_m2 = 2.0 * math.pi * r_m * l_m + (2.0 * cross_m2 if include_end_caps else 0.0)

    vol = Quantity(vol_m3, m3)
    surf = Quantity(surf_m2, m2)
    cross = Quantity(cross_m2, m2)
    char_l = Quantity(l_m, m)

    if radius.unit == um or length.unit == um:
        return CompartmentGeometry(
            name=name,
            shape="cylinder",
            volume=vol.to(fL),
            surface_area=surf.to(um2),
            cross_section_area=cross.to(um2),
            characteristic_length=char_l.to(um),
        )
    return CompartmentGeometry(
        name=name,
        shape="cylinder",
        volume=vol,
        surface_area=surf,
        cross_section_area=cross,
        characteristic_length=char_l,
    )


def make_cuboid_compartment(
    name: str,
    width: Quantity,
    height: Quantity,
    depth: Quantity,
) -> CompartmentGeometry:
    """Construct a rectangular cuboid `CompartmentGeometry` from `width`, `height`, and `depth`."""
    for label, q in (("width", width), ("height", height), ("depth", depth)):
        q.assert_dimension(DIM_LENGTH, context=f"cuboid '{name}' {label}")
    w_m = float(width.to(m).value)
    h_m = float(height.to(m).value)
    d_m = float(depth.to(m).value)
    if w_m <= 0.0 or h_m <= 0.0 or d_m <= 0.0:
        raise ValueError(f"Cuboid '{name}' dimensions must be positive.")

    vol = Quantity(w_m * h_m * d_m, m3)
    surf = Quantity(2.0 * (w_m * h_m + h_m * d_m + w_m * d_m), m2)
    cross = Quantity(w_m * h_m, m2)
    char_l = Quantity(d_m, m)

    if width.unit == um or height.unit == um or depth.unit == um:
        return CompartmentGeometry(
            name=name,
            shape="cuboid",
            volume=vol.to(fL),
            surface_area=surf.to(um2),
            cross_section_area=cross.to(um2),
            characteristic_length=char_l.to(um),
        )
    return CompartmentGeometry(
        name=name,
        shape="cuboid",
        volume=vol,
        surface_area=surf,
        cross_section_area=cross,
        characteristic_length=char_l,
    )


class HeterogeneousCompartmentTransportBrick(LawBrick):
    """
    Finite-volume multi-compartment diffusive transport `LawBrick` across compartments of
    arbitrary heterogeneous volumes `V_i` and interface geometric factors `G_e = A_e / L_e`.

    Formulated in extensive state variables `M_i = V_i * c_i` (scaled so `V_ref = mean(V_i)`):
        - Stored quadratic chemical potential energy: `H(M) = sum_i 0.5 * (V_ref / V_i) * M_i^2`
        - Conjugate intensive effort (concentration): `e_i = dH / dM_i = (V_ref / V_i) * M_i = c_i`
        - Symmetric PSD graph Laplacian dissipation: `R = D * L_G` where `L_G = L_G^T >= 0`
        - Exact total mass conservation: `d/dt sum_i M_i(t) == 0` for any heterogeneous volumes `V_i`
        - Steady-state equilibrium: uniform intensive concentration `c_i = c_j` across all connected nodes.
    """

    def __init__(
        self,
        name: str,
        volumes_m3: torch.Tensor,
        edge_index: torch.Tensor,
        edge_geometric_factors_m: torch.Tensor,
        diffusivity_default: float = 1e-9,
        state_unit: Unit = mM,
        time_unit: Unit = s,
    ) -> None:
        n = int(volumes_m3.numel())
        if n < 1:
            raise ValueError("HeterogeneousCompartmentTransportBrick requires at least 1 compartment.")
        v_mean = float(torch.mean(volumes_m3).item())
        self.relative_volumes = (volumes_m3.to(torch.float64) / max(v_mean, 1e-30)).clone()
        self.edge_index = edge_index.to(torch.long).clone()
        g_mean = (
            float(torch.mean(edge_geometric_factors_m).item())
            if edge_geometric_factors_m.numel() > 0
            else 1.0
        )
        g_norm = edge_geometric_factors_m.to(torch.float64) / max(g_mean, 1e-30)
        self.normalized_edge_conductances = g_norm.clone()

        state_specs = [
            StateVarSpec(
                name=f"M_{i}",
                unit=state_unit,
                default_value=1.0 if i == 0 else 0.1,
                hard_min=0.0,
                homeostatic_min=0.0,
                homeostatic_max=1e5,
            )
            for i in range(n)
        ]
        param_specs = [
            ParamSpec(
                name="D_scale",
                unit=DIMENSIONLESS,
                default_value=diffusivity_default,
                min_value=0.0,
                max_value=1e6,
            )
        ]
        super().__init__(
            name=name,
            domain="transport",
            state_specs=state_specs,
            param_specs=param_specs,
            port_specs=(),
            time_unit=time_unit,
            energy_unit=DIMENSIONLESS,
            matrix_free=False,
        )

    def intensive_concentration(self, state: torch.Tensor) -> torch.Tensor:
        """Convert extensive scaled amounts `M_i` (`[Batch, N]`) to intensive concentrations `c_i = M_i / v_rel_i`."""
        v_rel = self.relative_volumes.to(dtype=state.dtype, device=state.device).unsqueeze(0)
        return state / torch.clamp(v_rel, min=1e-18)

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        v_rel = self.relative_volumes.to(dtype=state.dtype, device=state.device).unsqueeze(0)
        return 0.5 * torch.sum((state**2) / torch.clamp(v_rel, min=1e-18), dim=-1)

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return self.intensive_concentration(state)

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        n = self.state_dim
        r_mat = torch.zeros((batch, n, n), dtype=state.dtype, device=state.device)
        if self.edge_index.numel() == 0:
            return r_mat
        d_scale = torch.clamp(params[:, 0], min=0.0)
        g_vec = self.normalized_edge_conductances.to(dtype=state.dtype, device=state.device)
        for e_idx in range(self.edge_index.shape[1]):
            i = int(self.edge_index[0, e_idx].item())
            j = int(self.edge_index[1, e_idx].item())
            w = d_scale * g_vec[e_idx]
            r_mat[:, i, i] = r_mat[:, i, i] + w
            r_mat[:, j, j] = r_mat[:, j, j] + w
            r_mat[:, i, j] = r_mat[:, i, j] - w
            r_mat[:, j, i] = r_mat[:, j, i] - w
        return r_mat


class MultiFieldReactionDiffusionPDEBrick(LawBrick):
    """
    Multi-species 2D/3D Finite-Volume Reaction-Diffusion PDE `LawBrick` across `S >= 2`
    interacting fields (`species_names`) on an `N`-compartment spatial graph or voxel mesh.

    State vector layout (`[Batch, S * N]`):
        - Species-major ordering: `[s0_node0, ..., s0_node{N-1}, s1_node0, ..., s{S-1}_node{N-1}]`
          where each state `M_{s, i} = v_rel_i * c_{s, i}` is extensive scaled amount.

    Thermodynamic & Port-Hamiltonian structure:
        - Stored chemical free energy:
            `H(M) = sum_{s=0}^{S-1} sum_{i=0}^{N-1} 0.5 * (M_{s, i}^2 / v_rel_i)`
        - Conjugate intensive chemical potentials / concentrations:
            `e_{s, i} = dH / dM_{s, i} = M_{s, i} / v_rel_i = c_{s, i}`
        - Symmetric PSD Onsager dissipation matrix `R(M) = R(M)^T >= 0`:
            1. Per-species differential spatial diffusion `blkdiag(D_0 * L_G, ..., D_{S-1} * L_G)`
               preserving exact per-species mass `sum_i M_{s, i}` under diffusion.
            2. Local reversible inter-species Onsager conversion (`s_0 <-> s_1`) at each node `i`
               with rate `k_exchange >= 0`, preserving total combined mass `sum_{s, i} M_{s, i}`.
        - Optional local symbolic reaction kinetics `reaction_exprs` (e.g. Turing activator-inhibitor
          morphogenesis or nonlinear cellular kinetics) evaluated identically across all `N` nodes.
    """

    def __init__(
        self,
        name: str,
        volumes_m3: torch.Tensor,
        edge_index: torch.Tensor,
        edge_geometric_factors_m: torch.Tensor,
        species_names: Sequence[str] = ("u", "v"),
        diffusivities_default: Sequence[float] = (0.1, 1.0),
        inter_species_exchange_default: float = 0.0,
        reaction_exprs: Sequence[str] | None = None,
        reaction_param_specs: Sequence[ParamSpec] = (),
        state_unit: Unit = mM,
        time_unit: Unit = s,
    ) -> None:
        from cortivra.layer3_compiler.ast_nodes import ExprNode, parse_symbolic_expr

        num_species = len(species_names)
        if num_species < 2:
            raise ValueError("MultiFieldReactionDiffusionPDEBrick requires at least 2 species.")
        if len(diffusivities_default) != num_species:
            raise ValueError(
                f"Expected {num_species} diffusivities_default, got {len(diffusivities_default)}."
            )
        n_nodes = int(volumes_m3.numel())
        if n_nodes < 1:
            raise ValueError("MultiFieldReactionDiffusionPDEBrick requires at least 1 compartment.")

        self.species_names = tuple(str(s_name) for s_name in species_names)
        self.num_species = num_species
        self.num_nodes = n_nodes

        v_mean = float(torch.mean(volumes_m3).item())
        self.relative_volumes = (volumes_m3.to(torch.float64) / max(v_mean, 1e-30)).clone()
        self.edge_index = edge_index.to(torch.long).clone()
        g_mean = (
            float(torch.mean(edge_geometric_factors_m).item())
            if edge_geometric_factors_m.numel() > 0
            else 1.0
        )
        self.normalized_edge_conductances = (
            edge_geometric_factors_m.to(torch.float64) / max(g_mean, 1e-30)
        ).clone()

        state_specs: list[StateVarSpec] = []
        for s_idx, s_name in enumerate(self.species_names):
            for i in range(n_nodes):
                val_def = (1.2 if i == 0 else 0.4) if s_idx == 0 else (0.3 if i == 0 else 0.8)
                state_specs.append(
                    StateVarSpec(
                        name=f"{s_name}_{i}",
                        unit=state_unit,
                        default_value=val_def,
                        hard_min=0.0,
                        homeostatic_min=0.0,
                        homeostatic_max=1e5,
                    )
                )

        param_specs: list[ParamSpec] = [
            ParamSpec(
                name=f"D_{s_name}",
                unit=DIMENSIONLESS,
                default_value=float(diffusivities_default[s_idx]),
                min_value=0.0,
                max_value=1e6,
            )
            for s_idx, s_name in enumerate(self.species_names)
        ]
        param_specs.append(
            ParamSpec(
                name="k_exchange",
                unit=DIMENSIONLESS,
                default_value=float(inter_species_exchange_default),
                min_value=0.0,
                max_value=1e5,
            )
        )
        self._reaction_param_specs = tuple(reaction_param_specs)
        param_specs.extend(self._reaction_param_specs)

        self._reaction_nodes: tuple[ExprNode, ...] = ()
        if reaction_exprs is not None:
            if len(reaction_exprs) != num_species:
                raise ValueError(
                    f"Expected {num_species} reaction_exprs, got {len(reaction_exprs)}."
                )
            self._reaction_nodes = tuple(
                parse_symbolic_expr(expr) if isinstance(expr, str) else expr
                for expr in reaction_exprs
            )

        super().__init__(
            name=name,
            domain="transport",
            state_specs=state_specs,
            param_specs=param_specs,
            port_specs=(),
            time_unit=time_unit,
            energy_unit=DIMENSIONLESS,
            matrix_free=False,
        )

    def intensive_concentrations(self, state: torch.Tensor) -> torch.Tensor:
        """
        Return intensive species concentrations `c_{s, i} = M_{s, i} / v_rel_i` shaped
        `[Batch, NumSpecies, NumNodes]`.
        """
        batch = state.shape[0]
        m_reshaped = state.view(batch, self.num_species, self.num_nodes)
        v_rel = self.relative_volumes.to(dtype=state.dtype, device=state.device).view(1, 1, self.num_nodes)
        return m_reshaped / torch.clamp(v_rel, min=1e-18)

    def species_masses(self, state: torch.Tensor) -> torch.Tensor:
        """Compute total extensive mass `sum_{i=0}^{N-1} M_{s, i}` for each species `s` (`[Batch, NumSpecies]`)."""
        batch = state.shape[0]
        m_reshaped = state.view(batch, self.num_species, self.num_nodes)
        return torch.sum(m_reshaped, dim=-1)

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        m_reshaped = state.view(batch, self.num_species, self.num_nodes)
        v_rel = self.relative_volumes.to(dtype=state.dtype, device=state.device).view(1, 1, self.num_nodes)
        return 0.5 * torch.sum((m_reshaped**2) / torch.clamp(v_rel, min=1e-18), dim=(1, 2))

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return self.intensive_concentrations(state).reshape(state.shape[0], self.state_dim)

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        s_cnt = self.num_species
        n = self.num_nodes
        total_dim = s_cnt * n
        r_mat = torch.zeros((batch, total_dim, total_dim), dtype=state.dtype, device=state.device)

        g_vec = self.normalized_edge_conductances.to(dtype=state.dtype, device=state.device)
        for s_idx in range(s_cnt):
            d_s = torch.clamp(params[:, s_idx], min=0.0)
            offset = s_idx * n
            for e_idx in range(self.edge_index.shape[1]):
                i = offset + int(self.edge_index[0, e_idx].item())
                j = offset + int(self.edge_index[1, e_idx].item())
                w = d_s * g_vec[e_idx]
                r_mat[:, i, i] = r_mat[:, i, i] + w
                r_mat[:, j, j] = r_mat[:, j, j] + w
                r_mat[:, i, j] = r_mat[:, i, j] - w
                r_mat[:, j, i] = r_mat[:, j, i] - w

        k_ex = torch.clamp(params[:, s_cnt], min=0.0)
        v_rel = self.relative_volumes.to(dtype=state.dtype, device=state.device)
        for i in range(n):
            idx_0 = i
            idx_1 = n + i
            w_ex = k_ex * v_rel[i]
            r_mat[:, idx_0, idx_0] = r_mat[:, idx_0, idx_0] + w_ex
            r_mat[:, idx_1, idx_1] = r_mat[:, idx_1, idx_1] + w_ex
            r_mat[:, idx_0, idx_1] = r_mat[:, idx_0, idx_1] - w_ex
            r_mat[:, idx_1, idx_0] = r_mat[:, idx_1, idx_0] - w_ex

        return r_mat

    def _reaction_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if not self._reaction_nodes:
            return torch.zeros_like(state)
        batch = state.shape[0]
        c_fields = self.intensive_concentrations(state)  # [B, S, N]
        v_rel = self.relative_volumes.to(dtype=state.dtype, device=state.device).view(1, self.num_nodes)
        env: dict[str, torch.Tensor] = {
            s_name: c_fields[:, s_idx, :]
            for s_idx, s_name in enumerate(self.species_names)
        }
        base_p_offset = self.num_species + 1
        for p_idx, p_spec in enumerate(self._reaction_param_specs):
            env[p_spec.name] = params[:, base_p_offset + p_idx].unsqueeze(-1).expand(batch, self.num_nodes)

        rxn_blocks: list[torch.Tensor] = []
        for s_idx, node in enumerate(self._reaction_nodes):
            rate_s = node.eval_tensor(env)
            if rate_s.ndim == 0:
                rate_s = rate_s.expand(batch, self.num_nodes)
            rxn_blocks.append(rate_s * v_rel)
        return torch.cat(rxn_blocks, dim=-1)

    def external_drive_power(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if not self._reaction_nodes:
            return torch.zeros(state.shape[0], dtype=state.dtype, device=state.device)
        e = self.effort(state, params)
        rxn = self._reaction_flow(state, params)
        return (e * rxn).sum(dim=-1)

    def flow(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        port_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        base_f = super().flow(state, params, port_inputs=port_inputs)
        if not self._reaction_nodes:
            return base_f
        return base_f + self._reaction_flow(state, params)


@dataclass(frozen=True)
class DiscretizedCompartmentGraph:
    """
    Connected multi-compartment physical geometry with sparse `edge_index` (`[2, NumEdges]`)
    and inter-compartment diffusive coupling weights.
    """

    compartments: tuple[CompartmentGeometry, ...]
    edge_index: torch.Tensor
    edge_geometric_factors_m: torch.Tensor  # A_interface / L_center_to_center in meters (`[NumEdges]`)

    @property
    def num_compartments(self) -> int:
        return len(self.compartments)

    @property
    def volumes_m3(self) -> torch.Tensor:
        return torch.tensor([float(c.volume.to(m3).value) for c in self.compartments], dtype=torch.float64)

    @property
    def surface_areas_m2(self) -> torch.Tensor:
        return torch.tensor([float(c.surface_area.to(m2).value) for c in self.compartments], dtype=torch.float64)

    def compute_diffusive_edge_rates(self, diffusivity: Quantity) -> torch.Tensor:
        """
        Compute inter-compartment diffusive exchange rate constants `g_e = D * (A_ij / (L_ij * V_mean))`
        in `1/s` (`[NumEdges]`).
        """
        diffusivity.assert_dimension(DIM_DIFFUSIVITY, context="diffusivity")
        d_m2_s = float(diffusivity.to(m2_per_s).value)
        if self.edge_index.numel() == 0:
            return torch.zeros((0,), dtype=torch.float64)
        vols = self.volumes_m3
        src = self.edge_index[0]
        dst = self.edge_index[1]
        v_harm = 2.0 * vols[src] * vols[dst] / torch.clamp(vols[src] + vols[dst], min=1e-30)
        return d_m2_s * self.edge_geometric_factors_m / torch.clamp(v_harm, min=1e-30)

    def to_spatial_brick(
        self,
        name: str = "discretized_spatial_graph",
        diffusivity_default: float = 0.5,
        field_unit: Unit = mM,
        time_unit: Unit = s,
    ) -> SpatialAdvectionDiffusionBrick:
        """Compile this `DiscretizedCompartmentGraph` into a Layer 2 `SpatialAdvectionDiffusionBrick`."""
        mean_l = float(
            sum(float(c.characteristic_length.to(m).value) for c in self.compartments)
            / max(len(self.compartments), 1)
        )
        return SpatialAdvectionDiffusionBrick(
            name=name,
            num_nodes=self.num_compartments,
            dx=max(mean_l, 1e-9),
            diffusivity_default=diffusivity_default,
            velocity_default=0.0,
            decay_default=0.0,
            edge_index=self.edge_index.clone(),
            field_unit=field_unit,
            time_unit=time_unit,
        )

    def to_diffusion_brick(
        self,
        name: str = "discretized_spatial_graph",
        diffusivity_default: float = 0.5,
        field_unit: Unit = mM,
        time_unit: Unit = s,
    ) -> SpatialAdvectionDiffusionBrick:
        """Alias for `to_spatial_brick`."""
        return self.to_spatial_brick(
            name=name,
            diffusivity_default=diffusivity_default,
            field_unit=field_unit,
            time_unit=time_unit,
        )

    def to_heterogeneous_transport_brick(
        self,
        name: str = "heterogeneous_compartment_transport",
        diffusivity_default: float = 0.5,
        state_unit: Unit = mM,
        time_unit: Unit = s,
    ) -> HeterogeneousCompartmentTransportBrick:
        """
        Compile this `DiscretizedCompartmentGraph` into a `HeterogeneousCompartmentTransportBrick`
        that preserves exact Port-Hamiltonian symmetry (`R = R^T >= 0`), exact total mass conservation,
        and intensive concentration equilibration across arbitrarily unequal compartment volumes.
        """
        return HeterogeneousCompartmentTransportBrick(
            name=name,
            volumes_m3=self.volumes_m3,
            edge_index=self.edge_index,
            edge_geometric_factors_m=self.edge_geometric_factors_m,
            diffusivity_default=diffusivity_default,
            state_unit=state_unit,
            time_unit=time_unit,
        )

    def to_multifield_reaction_diffusion_brick(
        self,
        name: str = "multifield_reaction_diffusion",
        species_names: Sequence[str] = ("u", "v"),
        diffusivities_default: Sequence[float] = (0.1, 1.0),
        inter_species_exchange_default: float = 0.0,
        reaction_exprs: Sequence[str] | None = None,
        reaction_param_specs: Sequence[ParamSpec] = (),
        state_unit: Unit = mM,
        time_unit: Unit = s,
    ) -> MultiFieldReactionDiffusionPDEBrick:
        """
        Compile this `DiscretizedCompartmentGraph` into a `MultiFieldReactionDiffusionPDEBrick`
        modeling `S >= 2` interacting species across `N` spatial compartments with differential
        diffusivities `D_s`, reversible Onsager exchange, and optional symbolic reaction kinetics.
        """
        return MultiFieldReactionDiffusionPDEBrick(
            name=name,
            volumes_m3=self.volumes_m3,
            edge_index=self.edge_index,
            edge_geometric_factors_m=self.edge_geometric_factors_m,
            species_names=species_names,
            diffusivities_default=diffusivities_default,
            inter_species_exchange_default=inter_species_exchange_default,
            reaction_exprs=reaction_exprs,
            reaction_param_specs=reaction_param_specs,
            state_unit=state_unit,
            time_unit=time_unit,
        )


def build_compartment_graph(
    compartments: Sequence[CompartmentGeometry],
    edges: Sequence[tuple[int, int]],
) -> DiscretizedCompartmentGraph:
    """
    Build a `DiscretizedCompartmentGraph` from a list of `CompartmentGeometry` nodes and
    undirected index pairs `(i, j)`.

    Computes the harmonic interface factor `G_ij = 1 / (L_i / (2 * A_i) + L_j / (2 * A_j))`
    (units: `meters`) between connected compartments `i` and `j`.
    """
    if not compartments:
        raise ValueError("`compartments` cannot be empty.")
    n = len(compartments)
    src_list: list[int] = []
    dst_list: list[int] = []
    geom_factors: list[float] = []

    for i, j in edges:
        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise ValueError(f"Invalid compartment edge ({i}, {j}) for {n} compartments.")
        ci = compartments[i]
        cj = compartments[j]
        ai = float(ci.cross_section_area.to(m2).value)
        aj = float(cj.cross_section_area.to(m2).value)
        li = float(ci.characteristic_length.to(m).value)
        lj = float(cj.characteristic_length.to(m).value)

        r_series = (li / max(2.0 * ai, 1e-30)) + (lj / max(2.0 * aj, 1e-30))
        g_ij = 1.0 / max(r_series, 1e-30)

        src_list.append(i)
        dst_list.append(j)
        geom_factors.append(g_ij)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        factors_t = torch.tensor(geom_factors, dtype=torch.float64)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        factors_t = torch.empty((0,), dtype=torch.float64)

    return DiscretizedCompartmentGraph(
        compartments=tuple(compartments),
        edge_index=edge_index,
        edge_geometric_factors_m=factors_t,
    )


def discretize_labeled_mask(
    labeled_mask: torch.Tensor,
    voxel_pitch: Quantity,
    slice_thickness: Quantity | None = None,
) -> DiscretizedCompartmentGraph:
    """
    Convert a 2D (`[H, W]`) or 3D (`[D, H, W]`) integer-labeled segmentation mask into a
    `DiscretizedCompartmentGraph` where each positive integer label `k > 0` becomes a
    physical compartment with:
        1. Exact voxel volume `V_k = N_k * dx * dy * dz`.
        2. Exact polyhedral voxel boundary surface area `A_surface,k` summed over all exposed
           6-connected voxel faces.
        3. Exact shared interface area `A_interface(i, j)` and physical centroid-to-centroid
           Euclidean distance `L_ij = ||r_i - r_j||_2` so `G_ij = A_interface(i, j) / L_ij`.
    """
    voxel_pitch.assert_dimension(DIM_LENGTH, context="voxel_pitch")
    pitch_m = float(voxel_pitch.to(m).value)
    if pitch_m <= 0.0:
        raise ValueError("voxel_pitch must be positive.")

    if labeled_mask.ndim == 2:
        if slice_thickness is not None:
            slice_thickness.assert_dimension(DIM_LENGTH, context="slice_thickness")
            dz_m = float(slice_thickness.to(m).value)
        else:
            dz_m = pitch_m
        mask_3d = labeled_mask.unsqueeze(0).to(torch.long)
    elif labeled_mask.ndim == 3:
        dz_m = float(slice_thickness.to(m).value) if slice_thickness is not None else pitch_m
        mask_3d = labeled_mask.to(torch.long)
    else:
        raise ValueError(f"Expected 2D or 3D labeled_mask, got shape {tuple(labeled_mask.shape)}.")

    if dz_m <= 0.0:
        raise ValueError("slice_thickness must be positive.")

    unique_labels = sorted(int(v.item()) for v in torch.unique(mask_3d) if int(v.item()) > 0)
    if not unique_labels:
        raise ValueError("labeled_mask contains no positive compartment labels (> 0).")

    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    voxel_vol_m3 = (pitch_m**2) * dz_m

    # Face areas normal to axis 0 (z), axis 1 (y), axis 2 (x)
    face_areas_m2 = (
        pitch_m * pitch_m,  # axis 0 (z-normal face: dx * dy)
        pitch_m * dz_m,     # axis 1 (y-normal face: dx * dz)
        pitch_m * dz_m,     # axis 2 (x-normal face: dy * dz)
    )
    spacings_m = (dz_m, pitch_m, pitch_m)

    # Pad mask with 0 on all 6 sides to count exact exposed faces for each label
    padded = torch.nn.functional.pad(mask_3d, (1, 1, 1, 1, 1, 1), mode="constant", value=0)
    d_dim, h_dim, w_dim = mask_3d.shape

    use_micro = voxel_pitch.unit == um
    compartments: list[CompartmentGeometry] = []
    centroids_m: list[torch.Tensor] = []

    z_coords = (torch.arange(d_dim, dtype=torch.float64, device=mask_3d.device) + 0.5) * spacings_m[0]
    y_coords = (torch.arange(h_dim, dtype=torch.float64, device=mask_3d.device) + 0.5) * spacings_m[1]
    x_coords = (torch.arange(w_dim, dtype=torch.float64, device=mask_3d.device) + 0.5) * spacings_m[2]
    zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing="ij")

    for lbl in unique_labels:
        reg = mask_3d == lbl
        count = int(torch.sum(reg).item())
        vol_m3 = count * voxel_vol_m3

        # Compute exact polyhedral surface area across all 6 face directions
        surf_m2 = 0.0
        # Axis 0 (z) neighbors
        n_z_minus = padded[0:d_dim, 1 : h_dim + 1, 1 : w_dim + 1] != lbl
        n_z_plus = padded[2 : d_dim + 2, 1 : h_dim + 1, 1 : w_dim + 1] != lbl
        surf_m2 += float(torch.sum(reg & n_z_minus).item() + torch.sum(reg & n_z_plus).item()) * face_areas_m2[0]
        # Axis 1 (y) neighbors
        n_y_minus = padded[1 : d_dim + 1, 0:h_dim, 1 : w_dim + 1] != lbl
        n_y_plus = padded[1 : d_dim + 1, 2 : h_dim + 2, 1 : w_dim + 1] != lbl
        surf_m2 += float(torch.sum(reg & n_y_minus).item() + torch.sum(reg & n_y_plus).item()) * face_areas_m2[1]
        # Axis 2 (x) neighbors
        n_x_minus = padded[1 : d_dim + 1, 1 : h_dim + 1, 0:w_dim] != lbl
        n_x_plus = padded[1 : d_dim + 1, 1 : h_dim + 1, 2 : w_dim + 2] != lbl
        surf_m2 += float(torch.sum(reg & n_x_minus).item() + torch.sum(reg & n_x_plus).item()) * face_areas_m2[2]

        # Compute physical centroid in meters
        cz = float(torch.mean(zz[reg]).item())
        cy = float(torch.mean(yy[reg]).item())
        cx = float(torch.mean(xx[reg]).item())
        centroids_m.append(torch.tensor([cz, cy, cx], dtype=torch.float64))

        # Characteristic length and cross-section from bounding extent
        r_eq = ((3.0 * vol_m3) / (4.0 * math.pi)) ** (1.0 / 3.0)
        char_l_m = 2.0 * r_eq
        cross_m2 = vol_m3 / max(char_l_m, 1e-30)

        vol_q = Quantity(vol_m3, m3).to(fL) if use_micro else Quantity(vol_m3, m3)
        surf_q = Quantity(surf_m2, m2).to(um2) if use_micro else Quantity(surf_m2, m2)
        cross_q = Quantity(cross_m2, m2).to(um2) if use_micro else Quantity(cross_m2, m2)
        len_q = Quantity(char_l_m, m).to(um) if use_micro else Quantity(char_l_m, m)

        compartments.append(
            CompartmentGeometry(
                name=f"region_{lbl}",
                shape="voxel_region",
                volume=vol_q,
                surface_area=surf_q,
                cross_section_area=cross_q,
                characteristic_length=len_q,
            )
        )

    # Accumulate exact shared interface area A_interface(i, j) across all 6-connected faces
    interface_areas_m2: dict[tuple[int, int], float] = {}
    for axis in range(3):
        if mask_3d.shape[axis] < 2:
            continue
        a = torch.narrow(mask_3d, axis, 0, mask_3d.shape[axis] - 1)
        b = torch.narrow(mask_3d, axis, 1, mask_3d.shape[axis] - 1)
        diff_mask = (a > 0) & (b > 0) & (a != b)
        if bool(diff_mask.any().item()):
            pairs = torch.stack([a[diff_mask], b[diff_mask]], dim=-1).tolist()
            f_area = face_areas_m2[axis]
            for l1, l2 in pairs:
                i1, i2 = label_to_idx[int(l1)], label_to_idx[int(l2)]
                edge = (min(i1, i2), max(i1, i2))
                interface_areas_m2[edge] = interface_areas_m2.get(edge, 0.0) + f_area

    src_list: list[int] = []
    dst_list: list[int] = []
    geom_factors: list[float] = []
    for (i, j) in sorted(interface_areas_m2.keys()):
        a_ij = interface_areas_m2[(i, j)]
        dist_m = float(torch.linalg.norm(centroids_m[i] - centroids_m[j]).item())
        l_ij = max(dist_m, min(spacings_m))
        src_list.append(i)
        dst_list.append(j)
        geom_factors.append(a_ij / l_ij)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        factors_t = torch.tensor(geom_factors, dtype=torch.float64)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        factors_t = torch.empty((0,), dtype=torch.float64)

    return DiscretizedCompartmentGraph(
        compartments=tuple(compartments),
        edge_index=edge_index,
        edge_geometric_factors_m=factors_t,
    )
