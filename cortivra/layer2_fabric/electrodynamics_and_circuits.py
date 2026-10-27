"""
Nonlinear electrodynamics, RC/RLC circuits, and voltage-gated conductance membranes
(`cortivra.layer2_fabric.electrodynamics_and_circuits`).

Provides `NonlinearConductanceCircuitBrick`:
    A port-Hamiltonian capacitive-conductive-gating subsystem that models any voltage-driven
    excitable or dissipative circuit (solid-state nonlinear RC/RLC oscillators, electrochemical
    cells, or multi-channel biological membranes with Na_V, K_V, Kv3.1, and leak pathways).

    Supports both standard 4th-order Runge-Kutta (`step_rk4`) and stiff-stable Rush-Larsen
    exponential-Euler integration (`step_exponential_euler`) for gating variables.
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
    fJ,
    ms,
    mV,
    nS,
    pA,
    pF,
)
from cortivra.layer2_fabric.primitives import (
    boltzmann_partition_switch,
    linear_relaxation_rate,
    ohmic_dissipative_driving_force,
    step_rush_larsen,
)


class NonlinearConductanceCircuitBrick(LawBrick):
    """
    Port-Hamiltonian capacitive circuit with voltage-gated Boltzmann channels and thermodynamic
    reversal batteries.

    State vector (`[Batch, 5]`):
        0. `V`: Capacitive potential (`mV`, default `-65.0 mV`).
        1. `m_fast`: Fast depolarizing activation gate (`1`, `[0, 1]`).
        2. `h_fast`: Fast depolarizing inactivation gate (`1`, `[0, 1]`).
        3. `n_slow`: Delayed rectifying repolarization gate (`1`, `[0, 1]`).
        4. `n_high`: High-threshold fast-repolarizing gate (`1`, `[0, 1]`).

    Parameters (`[Batch, 11]`):
        0. `C_m`: Capacitance (`pF`, default `100.0 pF`).
        1. `g_fast`: Max conductance of fast inward pathway (`nS`, default `1200.0 nS`).
        2. `g_slow`: Max conductance of delayed outward pathway (`nS`, default `360.0 nS`).
        3. `g_high`: Max conductance of high-threshold fast-repolarizing pathway (`nS`, default `200.0 nS`).
        4. `g_leak`: Passive leak conductance (`nS`, default `3.0 nS`).
        5. `E_fast`: Reversal battery of fast inward pathway (`mV`, default `+50.0 mV`).
        6. `E_out`: Reversal battery of outward repolarizing pathways (`mV`, default `-85.0 mV`).
        7. `E_leak`: Reversal battery of passive leak (`mV`, default `-65.0 mV`).
        8. `vh_fast`: Half-activation potential of fast gate (`mV`, default `-38.0 mV`).
        9. `vh_high`: Half-activation potential of high-threshold gate (`mV`, default `-15.0 mV`).
        10. `tau_high`: Relaxation time constant of high-threshold gate (`ms`, default `0.8 ms`).

    Port-Hamiltonian Energy Representation:
        Shifted electrostatic + gating free energy around the passive resting state `(V_eq, m_inf, ...)`:
            `H(x) = 0.5 * C_m * (V - E_leak)^2 + 0.5 * sum_k (g_k - g_{k,inf}(V))^2`
        Effort vector `e = grad_x H(x)` drives positive semi-definite diagonal dissipation `R(x) >= 0`,
        while active battery currents `(E_k - E_leak)` enter via `compute_battery_drive_flow`.
    """

    def __init__(
        self,
        name: str = "nonlinear_conductance_circuit",
        c_m_default: float = 100.0,
        g_fast_default: float = 1200.0,
        g_slow_default: float = 360.0,
        g_high_default: float = 200.0,
        g_leak_default: float = 3.0,
        e_fast_default: float = 50.0,
        e_out_default: float = -85.0,
        e_leak_default: float = -65.0,
        vh_fast_default: float = -38.0,
        vh_high_default: float = -15.0,
        tau_high_default: float = 0.8,
    ) -> None:
        v0 = e_leak_default
        v0_t = torch.tensor(v0, dtype=torch.float64)
        m0 = float(boltzmann_partition_switch(v0_t, vh_fast_default, 7.0).item())
        h0 = float(boltzmann_partition_switch(v0_t, -62.0, -6.0).item())
        n0 = float(boltzmann_partition_switch(v0_t, -50.0, 12.0).item())
        nh0 = float(boltzmann_partition_switch(v0_t, vh_high_default, 9.0).item())

        state_specs = [
            StateVarSpec(
                name="V",
                unit=mV,
                default_value=v0,
                hard_min=-200.0,
                hard_max=200.0,
                homeostatic_min=-120.0,
                homeostatic_max=80.0,
                description="Capacitive potential",
            ),
            StateVarSpec(
                name="m_fast",
                unit=DIMENSIONLESS,
                default_value=m0,
                hard_min=0.0,
                hard_max=1.0,
                homeostatic_min=0.0,
                homeostatic_max=1.0,
                description="Fast activation gate",
            ),
            StateVarSpec(
                name="h_fast",
                unit=DIMENSIONLESS,
                default_value=h0,
                hard_min=0.0,
                hard_max=1.0,
                homeostatic_min=0.0,
                homeostatic_max=1.0,
                description="Fast inactivation gate",
            ),
            StateVarSpec(
                name="n_slow",
                unit=DIMENSIONLESS,
                default_value=n0,
                hard_min=0.0,
                hard_max=1.0,
                homeostatic_min=0.0,
                homeostatic_max=1.0,
                description="Delayed rectifier gate",
            ),
            StateVarSpec(
                name="n_high",
                unit=DIMENSIONLESS,
                default_value=nh0,
                hard_min=0.0,
                hard_max=1.0,
                homeostatic_min=0.0,
                homeostatic_max=1.0,
                description="High-threshold fast recovery gate",
            ),
        ]

        param_specs = [
            ParamSpec("C_m", pF, default_value=c_m_default, min_value=1e-3, max_value=1e5),
            ParamSpec("g_fast", nS, default_value=g_fast_default, min_value=0.0, max_value=1e6),
            ParamSpec("g_slow", nS, default_value=g_slow_default, min_value=0.0, max_value=1e6),
            ParamSpec("g_high", nS, default_value=g_high_default, min_value=0.0, max_value=1e6),
            ParamSpec("g_leak", nS, default_value=g_leak_default, min_value=0.0, max_value=1e5),
            ParamSpec("E_fast", mV, default_value=e_fast_default, min_value=-150.0, max_value=200.0),
            ParamSpec("E_out", mV, default_value=e_out_default, min_value=-150.0, max_value=100.0),
            ParamSpec("E_leak", mV, default_value=e_leak_default, min_value=-150.0, max_value=100.0),
            ParamSpec("vh_fast", mV, default_value=vh_fast_default, min_value=-100.0, max_value=50.0),
            ParamSpec("vh_high", mV, default_value=vh_high_default, min_value=-100.0, max_value=60.0),
            ParamSpec("tau_high", ms, default_value=tau_high_default, min_value=0.05, max_value=500.0),
        ]

        # 1 mV * 1 pA = 1e-15 W = 1 fJ / ms
        port_specs = [
            PowerPortSpec(
                name="electrical_port",
                effort_unit=mV,
                flow_unit=pA,
                domain="electrodynamics",
                state_var_name="V",
            )
        ]

        super().__init__(
            name=name,
            domain="electrodynamics",
            state_specs=state_specs,
            param_specs=param_specs,
            port_specs=port_specs,
            time_unit=ms,
            energy_unit=fJ,
        )

    def steady_state_gates_and_taus(
        self, v_mv: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute `[m_inf, h_inf, n_inf, nh_inf]` and `[tau_m, tau_h, tau_n, tau_nh]` (`[Batch, 4]`)."""
        vh_fast = params[:, 8]
        vh_high = params[:, 9]
        tau_high = torch.clamp(params[:, 10], min=0.05)

        m_inf = boltzmann_partition_switch(v_mv, vh_fast, 7.0)
        h_inf = boltzmann_partition_switch(v_mv, -62.0, -6.0)
        n_inf = boltzmann_partition_switch(v_mv, -50.0, 12.0)
        nh_inf = boltzmann_partition_switch(v_mv, vh_high, 9.0)

        # Bell-shaped voltage-dependent relaxation time constants in ms
        tau_m = 0.10 + 0.40 * torch.exp(-(((v_mv - vh_fast) / 20.0) ** 2))
        tau_h = 1.00 + 6.00 * torch.exp(-(((v_mv + 60.0) / 20.0) ** 2))
        tau_n = 1.20 + 4.00 * torch.exp(-(((v_mv + 50.0) / 25.0) ** 2))

        g_inf = torch.stack([m_inf, h_inf, n_inf, nh_inf], dim=-1)
        taus = torch.stack([tau_m, tau_h, tau_n, tau_high], dim=-1)
        return g_inf, taus

    def effective_conductances(self, state: torch.Tensor, params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return instantaneous path conductances `(G_fast, G_slow, G_high, G_leak)` in `nS`."""
        m_g = torch.clamp(state[:, 1], 0.0, 1.0)
        h_g = torch.clamp(state[:, 2], 0.0, 1.0)
        n_g = torch.clamp(state[:, 3], 0.0, 1.0)
        nh_g = torch.clamp(state[:, 4], 0.0, 1.0)

        g_fast = torch.clamp(params[:, 1], min=0.0) * (m_g**3) * h_g
        g_slow = torch.clamp(params[:, 2], min=0.0) * (n_g**4)
        g_high = torch.clamp(params[:, 3], min=0.0) * (nh_g**2)
        g_leak = torch.clamp(params[:, 4], min=0.0)
        return g_fast, g_slow, g_high, g_leak

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        v_mv = state[:, 0]
        gates = state[:, 1:]
        c_m = torch.clamp(params[:, 0], min=1e-6)
        e_leak = params[:, 7]
        g_inf, _ = self.steady_state_gates_and_taus(v_mv, params)
        elec_energy = 0.5 * c_m * ((v_mv - e_leak) ** 2)
        gate_energy = 0.5 * torch.sum((gates - g_inf) ** 2, dim=-1)
        return elec_energy + gate_energy

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        v_mv = state[:, 0]
        gates = state[:, 1:]
        c_m = torch.clamp(params[:, 0], min=1e-6)
        e_leak = params[:, 7]
        g_inf, _ = self.steady_state_gates_and_taus(v_mv, params)

        # Compute exact d(gate_energy)/dV = -sum_k (g_k - g_{k,inf}(V)) * dg_{k,inf}/dV
        # so effort(x) == torch.autograd.grad(hamiltonian_energy(x).sum(), x) to machine precision
        dm_dv = g_inf[:, 0] * (1.0 - g_inf[:, 0]) / 7.0
        dh_dv = g_inf[:, 1] * (1.0 - g_inf[:, 1]) / (-6.0)
        dn_dv = g_inf[:, 2] * (1.0 - g_inf[:, 2]) / 12.0
        dnh_dv = g_inf[:, 3] * (1.0 - g_inf[:, 3]) / 9.0
        dg_inf_dv = torch.stack([dm_dv, dh_dv, dn_dv, dnh_dv], dim=-1)

        e_gates = gates - g_inf
        dgate_energy_dv = -torch.sum(e_gates * dg_inf_dv, dim=-1)
        e_v = (c_m * (v_mv - e_leak) + dgate_energy_dv).unsqueeze(-1)
        return torch.cat([e_v, e_gates], dim=-1)

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        v_mv = state[:, 0]
        c_m = torch.clamp(params[:, 0], min=1e-6)
        g_fast, g_slow, g_high, g_leak = self.effective_conductances(state, params)
        g_total = g_fast + g_slow + g_high + g_leak
        _, taus = self.steady_state_gates_and_taus(v_mv, params)

        r_mat = torch.zeros((batch, 5, 5), dtype=state.dtype, device=state.device)
        r_mat[:, 0, 0] = g_total / (c_m * c_m)
        r_mat[:, 1, 1] = 1.0 / taus[:, 0]
        r_mat[:, 2, 2] = 1.0 / taus[:, 1]
        r_mat[:, 3, 3] = 1.0 / taus[:, 2]
        r_mat[:, 4, 4] = 1.0 / taus[:, 3]
        return r_mat

    def dissipative_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        e = self.effort(state, params)
        r_mat = self.dissipation_matrix(state, params)
        return torch.bmm(r_mat, e.unsqueeze(-1)).squeeze(-1)

    def channel_currents_pa(self, state: torch.Tensor, params: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute individual pathway currents `(I_fast, I_slow, I_high, I_leak, I_total)` in `pA`."""
        v_mv = state[:, 0]
        g_fast, g_slow, g_high, g_leak = self.effective_conductances(state, params)
        e_fast = params[:, 5]
        e_out = params[:, 6]
        e_leak = params[:, 7]

        # 1 nS * 1 mV = 1 pA
        i_fast = ohmic_dissipative_driving_force(v_mv, e_fast, g_fast)
        i_slow = ohmic_dissipative_driving_force(v_mv, e_out, g_slow)
        i_high = ohmic_dissipative_driving_force(v_mv, e_out, g_high)
        i_leak = ohmic_dissipative_driving_force(v_mv, e_leak, g_leak)
        i_total = i_fast + i_slow + i_high + i_leak
        return {
            "I_fast": i_fast,
            "I_slow": i_slow,
            "I_high": i_high,
            "I_leak": i_leak,
            "I_total": i_total,
        }

    def compute_battery_drive_flow(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        stimulus_current_pa: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        """
        Compute the external port flow `[u_V, 0, 0, 0, 0]` (`[Batch, 5]`) generated by the
        active Nernst reversal batteries `(E_k - E_leak)`, gating cross-potential compensation,
        and injected stimulus current `I_stim` so `self.flow(..., port_inputs={"external_flow": u})`
        equals the exact Kirchhoff membrane equation `(I_stim - I_total) / C_m`.
        """
        v_mv = state[:, 0]
        c_m = torch.clamp(params[:, 0], min=1e-6)
        currents = self.channel_currents_pa(state, params)
        target_dv_dt = (stimulus_current_pa - currents["I_total"]) / c_m
        f_diss_v = self.dissipative_flow(state, params)[:, 0]
        u_v = (target_dv_dt + f_diss_v).unsqueeze(-1)
        zeros_gates = torch.zeros_like(state[:, 1:])
        return torch.cat([u_v, zeros_gates], dim=-1)

    def full_excitable_flow(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        stimulus_current_pa: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        """
        Compute the active driven ODE flow `dx/dt` including the Nernst battery EMF offsets
        (`E_fast != E_leak`, `E_out != E_leak`) and injected stimulus current `I_stim` (`pA`).
        Identical to `self.flow(state, params, port_inputs={"external_flow": self.compute_battery_drive_flow(...)})`.
        """
        u_ext = self.compute_battery_drive_flow(state, params, stimulus_current_pa=stimulus_current_pa)
        return self.flow(state, params, port_inputs={"external_flow": u_ext})

    def step_exponential_euler(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        dt: float,
        stimulus_current_pa: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        """
        Advance one time step `dt` (`ms`) using Rush-Larsen exponential-Euler integration for
        gating states `m, h, n, n_high` and semi-implicit conductance update for voltage `V`.
        """
        v_mv = state[:, 0]
        gates = state[:, 1:]
        c_m = torch.clamp(params[:, 0], min=1e-6)

        g_inf, taus = self.steady_state_gates_and_taus(v_mv, params)
        next_gates = step_rush_larsen(gates, g_inf, taus, dt=dt)

        mid_state = torch.cat([v_mv.unsqueeze(-1), next_gates], dim=-1)
        g_fast, g_slow, g_high, g_leak = self.effective_conductances(mid_state, params)
        e_fast = params[:, 5]
        e_out = params[:, 6]
        e_leak = params[:, 7]

        # Semi-implicit voltage update: C_m * (V_{n+1} - V_n) / dt = I_stim - sum G_k * (V_{n+1} - E_k)
        g_sum = g_fast + g_slow + g_high + g_leak
        weighted_e = g_fast * e_fast + (g_slow + g_high) * e_out + g_leak * e_leak + stimulus_current_pa
        next_v = (v_mv + (dt / c_m) * weighted_e) / (1.0 + (dt / c_m) * g_sum)

        next_state = torch.cat([next_v.unsqueeze(-1), next_gates], dim=-1)
        return self.clamp_to_hard_bounds(next_state)
