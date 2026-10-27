"""
Spatial continuum PDEs, graph Laplacian diffusion, advection, and event-driven coupling
(`cortivra.layer2_fabric.transport_and_fields`).

Provides:
    - `SpatialAdvectionDiffusionBrick`: O(N) `matrix_free=True` port-Hamiltonian continuum
      field combining conservative skew-symmetric advection (`-v * du/dx`), dissipative
      Neumann Laplacian diffusion (`D * d^2u/dx^2`), and linear decay (`-k * u`).
      Scales to 100,000+ spatial compartments without allocating dense N x N matrices.
    - `EventImpulseCouplingBrick`: Differentiable event-triggered impulse and double-exponential
      relaxation filter for threshold-coupled networks (synaptic transmission, pacemaker resets).
"""

from __future__ import annotations

import torch

from cortivra.layer1_axioms.hamiltonian import (
    LawBrick,
    ParamSpec,
    PowerPortSpec,
    StateVarSpec,
)
from cortivra.layer1_axioms.units import (
    DIMENSIONLESS,
    J,
    Unit,
    m2_per_s,
    mM,
    m_per_s,
    ms,
    mV,
    nS,
    pA,
    s,
)
from cortivra.layer2_fabric.primitives import (
    conservative_advection_1d,
    graph_laplacian_diffusion_1d,
    graph_laplacian_sparse_edges,
    linear_relaxation_rate,
    surrogate_event_impulse,
)


class SpatialAdvectionDiffusionBrick(LawBrick):
    """
    O(N) matrix-free port-Hamiltonian spatial advection-diffusion-decay field:
        `du_i/dt = -v * (u_{i+1} - u_{i-1}) / (2 * dx) + D * L(u)_i - k_decay * u_i`
    Supports both uniform 1D grids (`edge_index=None`) and arbitrary sparse graph/tree/mesh
    topologies (`edge_index` of shape `[2, NumEdges]`).

    Port-Hamiltonian Energy & Effort:
        - `H(u) = 0.5 * sum_i u_i^2`
        - `e_i = dH/du_i = u_i`
        - `F_cons(u) = -v * du/dx` (skew-symmetric periodic finite difference, so `e . F_cons == 0`)
        - `F_diss(u) = -D * L(u) + k_decay * u` (positive semi-definite Laplacian + decay, so `e . F_diss >= 0`)
    """

    def __init__(
        self,
        name: str = "spatial_advection_diffusion_field",
        num_nodes: int = 8,
        dx: float = 1.0,
        diffusivity_default: float = 0.5,
        velocity_default: float = 0.2,
        decay_default: float = 0.0,
        edge_index: torch.Tensor | None = None,
        field_unit: Unit = mM,
        time_unit: Unit = s,
        energy_unit: Unit = J,
    ) -> None:
        if num_nodes < 2:
            raise ValueError("SpatialAdvectionDiffusionBrick requires `num_nodes >= 2`.")
        self.num_nodes = num_nodes
        self.dx = float(dx)
        self.edge_index = edge_index

        state_specs = [
            StateVarSpec(
                name=f"u_{i}",
                unit=field_unit,
                default_value=1.0 if i == num_nodes // 2 else 0.1,
                hard_min=-1e6,
                hard_max=1e6,
                homeostatic_min=-1e4,
                homeostatic_max=1e4,
                description=f"Field value at node {i}",
            )
            for i in range(num_nodes)
        ]

        param_specs = [
            ParamSpec("diffusivity", m2_per_s, default_value=diffusivity_default, min_value=0.0),
            ParamSpec("velocity", m_per_s, default_value=velocity_default, min_value=-1e4, max_value=1e4),
            ParamSpec("decay", DIMENSIONLESS, default_value=decay_default, min_value=0.0),
        ]

        super().__init__(
            name=name,
            domain="transport",
            state_specs=state_specs,
            param_specs=param_specs,
            port_specs=(),
            time_unit=time_unit,
            energy_unit=energy_unit,
            matrix_free=True,
        )

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(state**2, dim=-1)

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return state

    def conservative_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if self.edge_index is not None:
            return torch.zeros_like(state)
        v = params[:, 1].unsqueeze(-1)
        return conservative_advection_1d(state, velocity=v, dx=self.dx)

    def dissipative_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        d_coeff = torch.clamp(params[:, 0], min=0.0).unsqueeze(-1)
        decay = torch.clamp(params[:, 2], min=0.0).unsqueeze(-1)
        # Note: dissipative_flow has opposite sign to du/dt_diss so du/dt = F_cons - F_diss
        if self.edge_index is not None:
            num_edges = int(self.edge_index.shape[1])
            edge_g = (d_coeff / (self.dx * self.dx)).expand(-1, num_edges)
            diffusion_term = graph_laplacian_sparse_edges(state, self.edge_index, edge_g)
        else:
            diffusion_term = graph_laplacian_diffusion_1d(state, diffusivity=d_coeff, dx=self.dx)
        return -diffusion_term + decay * state


class EventImpulseCouplingBrick(LawBrick):
    """
    Differentiable event-triggered impulse and exponential conductance filter:
        `dg_syn/dt = -g_syn / tau_decay + w_max * delta_beta(V_pre - V_th)`

    State vector (`[Batch, 1]`):
        - `g_syn`: Active coupling conductance (`nS`, `>= 0`).

    Parameters (`[Batch, 4]`):
        - `tau_decay`: Exponential relaxation time constant (`ms`, `> 0`).
        - `w_max`: Impulse coupling weight (`nS`, `>= 0`).
        - `v_th`: Event threshold (`mV`, default `-20.0 mV`).
        - `e_rev`: Target reversal potential (`mV`, default `0.0 mV` for excitatory, `-75.0 mV` for inhibitory).
    """

    def __init__(
        self,
        name: str = "event_impulse_coupling",
        tau_decay_default: float = 5.0,
        w_max_default: float = 10.0,
        v_th_default: float = -20.0,
        e_rev_default: float = 0.0,
        sharpness_beta: float = 0.5,
    ) -> None:
        self.sharpness_beta = float(sharpness_beta)
        state_specs = [
            StateVarSpec(
                name="g_syn",
                unit=nS,
                default_value=0.2,
                hard_min=0.0,
                hard_max=1e5,
                homeostatic_min=0.0,
                homeostatic_max=1e4,
                description="Event-triggered coupling conductance",
            )
        ]
        param_specs = [
            ParamSpec("tau_decay", ms, default_value=tau_decay_default, min_value=0.05),
            ParamSpec("w_max", nS, default_value=w_max_default, min_value=0.0),
            ParamSpec("v_th", mV, default_value=v_th_default, min_value=-100.0, max_value=100.0),
            ParamSpec("e_rev", mV, default_value=e_rev_default, min_value=-150.0, max_value=150.0),
        ]
        port_specs = [
            PowerPortSpec(
                name="post_electrical_port",
                effort_unit=mV,
                flow_unit=pA,
                domain="electrodynamics",
                state_var_name="g_syn",
            )
        ]
        super().__init__(
            name=name,
            domain="transport",
            state_specs=state_specs,
            param_specs=param_specs,
            port_specs=port_specs,
            time_unit=ms,
            energy_unit=J,
        )

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        g_syn = state[:, 0]
        return 0.5 * (g_syn**2)

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return state

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        tau = torch.clamp(params[:, 0], min=0.05)
        return (1.0 / tau).view(-1, 1, 1)

    def dissipative_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        tau = torch.clamp(params[:, 0], min=0.05).unsqueeze(-1)
        return -linear_relaxation_rate(state, torch.zeros_like(state), tau)

    def compute_event_drive_and_post_current(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        v_pre_mv: torch.Tensor,
        v_post_mv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute `(dg_syn_dt, i_post_pa)`:
            - `dg_syn_dt`: Total conductance derivative `[-g_syn / tau + w_max * delta_beta(V_pre - V_th)]` (`[Batch, 1]`).
            - `i_post_pa`: Postsynaptic current delivered into target node `-g_syn * (V_post - E_rev)` in `pA` (`[Batch]`).
        """
        g_syn = torch.clamp(state[:, 0], min=0.0)
        tau = torch.clamp(params[:, 0], min=0.05)
        w_max = torch.clamp(params[:, 1], min=0.0)
        v_th = params[:, 2]
        e_rev = params[:, 3]

        impulse = surrogate_event_impulse(v_pre_mv, v_th, sharpness_beta=self.sharpness_beta)
        dg_dt = (-g_syn / tau + w_max * impulse).unsqueeze(-1)
        i_post_pa = -g_syn * (v_post_mv - e_rev)
        return dg_dt, i_post_pa
