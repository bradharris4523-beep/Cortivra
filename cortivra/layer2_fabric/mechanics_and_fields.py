"""
Classical, orbital, central-force, and viscoelastic Hamiltonian mechanics (`cortivra.layer2_fabric.mechanics_and_fields`).

Provides:
    - `CentralForceHamiltonianBrick`: N-body gravitational / Coulomb / power-law central-force
      system `H(q, p) = sum ||p_i||^2 / (2*m_i) + 0.5 * sum_{i!=j} alpha_{ij} / ||q_i - q_j||^p`.
      Supports orbital mechanics, Keplerian conservation, and unmodeled gravitational
      perturbation discovery (e.g., Uranus-to-Neptune residual analysis).
    - `NonlinearPendulumAndViscoelasticBrick`: Trigonometric pendulum / Josephson / Kuramoto
      oscillator combined with Hookean/Kelvin-Voigt viscoelasticity and optional active force input.
"""

from __future__ import annotations

from typing import Sequence

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
    N_force,
    Unit,
    kg,
    m,
    m_per_s,
    s,
)
from cortivra.layer2_fabric.primitives import (
    inverse_distance_potential_and_force,
    ohmic_dissipative_driving_force,
    quadratic_energy_storage,
)


class CentralForceHamiltonianBrick(LawBrick):
    """
    N-body central-force port-Hamiltonian system in `D` spatial dimensions.

    State vector layout (`[Batch, 2 * N * D]`):
        - First `N * D` components: positions `q` (`[q_{0,x}, q_{0,y}, ..., q_{N-1,D-1}]`).
        - Last `N * D` components: momenta `p` (`[p_{0,x}, p_{0,y}, ..., p_{N-1,D-1}]`).

    Parameters (`[Batch, N + 2]`):
        - `m_0, ..., m_{N-1}`: Body masses.
        - `g_coupling`: Pairwise interaction constant `G` (for attractive gravity, `alpha_{ij} = -G * m_i * m_j`).
        - `drag`: Linear viscous damping coefficient (`>= 0`, `0.0` for conservative vacuum orbits).
    """

    def __init__(
        self,
        name: str = "central_force_nbody",
        num_bodies: int = 2,
        spatial_dim: int = 2,
        potential_exponent_p: float = 1.0,
        softening_eps: float = 1e-4,
        default_masses: Sequence[float] | None = None,
        g_default: float = 1.0,
        drag_default: float = 0.0,
        length_unit: Unit = m,
        momentum_unit: Unit = N_force * s,
        mass_unit: Unit = kg,
        time_unit: Unit = s,
        energy_unit: Unit = J,
    ) -> None:
        if num_bodies < 2:
            raise ValueError("CentralForceHamiltonianBrick requires `num_bodies >= 2`.")
        if spatial_dim < 1:
            raise ValueError("CentralForceHamiltonianBrick requires `spatial_dim >= 1`.")

        self.num_bodies = num_bodies
        self.spatial_dim = spatial_dim
        self.potential_exponent_p = float(potential_exponent_p)
        self.softening_eps = float(softening_eps)
        self._nd = num_bodies * spatial_dim

        coord_names = ("x", "y", "z") if spatial_dim <= 3 else tuple(f"d{k}" for k in range(spatial_dim))
        state_specs: list[StateVarSpec] = []
        # Default non-overlapping initial positions along axis 0 so default_state() never starts at r=0
        for i in range(num_bodies):
            for d_idx in range(spatial_dim):
                q_def = float(i * 1.5) if d_idx == 0 else 0.0
                state_specs.append(
                    StateVarSpec(
                        name=f"q_{i}_{coord_names[d_idx]}",
                        unit=length_unit,
                        default_value=q_def,
                        homeostatic_min=-1e6,
                        homeostatic_max=1e6,
                        description=f"Body {i} position ({coord_names[d_idx]})",
                    )
                )
        for i in range(num_bodies):
            for d_idx in range(spatial_dim):
                p_def = 0.5 if (i == 1 and d_idx == min(1, spatial_dim - 1)) else 0.0
                state_specs.append(
                    StateVarSpec(
                        name=f"p_{i}_{coord_names[d_idx]}",
                        unit=momentum_unit,
                        default_value=p_def,
                        homeostatic_min=-1e6,
                        homeostatic_max=1e6,
                        description=f"Body {i} momentum ({coord_names[d_idx]})",
                    )
                )

        masses = list(default_masses) if default_masses is not None else [1.0] * num_bodies
        if len(masses) != num_bodies:
            raise ValueError(f"Expected {num_bodies} default masses, got {len(masses)}.")

        param_specs: list[ParamSpec] = [
            ParamSpec(f"m_{i}", mass_unit, default_value=float(masses[i]), min_value=1e-9)
            for i in range(num_bodies)
        ]
        param_specs.append(ParamSpec("g_coupling", DIMENSIONLESS, default_value=g_default, min_value=0.0))
        param_specs.append(ParamSpec("drag", DIMENSIONLESS, default_value=drag_default, min_value=0.0))

        vel_unit = length_unit / time_unit
        force_unit = momentum_unit / time_unit
        port_specs = [
            PowerPortSpec(
                name="body0_mechanical_port",
                effort_unit=vel_unit,
                flow_unit=force_unit,
                domain="mechanics",
                state_var_name=f"p_0_{coord_names[0]}",
                require_power_conjugate=(vel_unit == m_per_s and force_unit == N_force),
            )
        ]

        super().__init__(
            name=name,
            domain="mechanics",
            state_specs=state_specs,
            param_specs=param_specs,
            port_specs=port_specs,
            time_unit=time_unit,
            energy_unit=energy_unit,
        )

    def _unpack(
        self, state: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        q = state[:, : self._nd].view(batch, self.num_bodies, self.spatial_dim)
        p = state[:, self._nd :].view(batch, self.num_bodies, self.spatial_dim)
        masses = torch.clamp(params[:, : self.num_bodies], min=1e-9)  # [B, N]
        g_val = params[:, self.num_bodies]  # [B]
        drag = torch.clamp(params[:, self.num_bodies + 1], min=0.0)  # [B]
        return q, p, masses, g_val, drag

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        q, p, masses, g_val, _ = self._unpack(state, params)
        # Kinetic energy: sum_i ||p_i||^2 / (2 * m_i)
        p_sq = torch.sum(p**2, dim=-1)  # [B, N]
        kinetic = torch.sum(0.5 * p_sq / masses, dim=-1)

        # Attractive pairwise gravitational coupling: alpha_{ij} = -G * m_i * m_j
        m_prod = masses.unsqueeze(2) * masses.unsqueeze(1)  # [B, N, N]
        couplings = -g_val.view(-1, 1, 1) * m_prod
        potential, _ = inverse_distance_potential_and_force(
            positions=q,
            couplings=couplings,
            exponent_p=self.potential_exponent_p,
            softening_eps=self.softening_eps,
        )
        return kinetic + potential

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        q, p, masses, g_val, _ = self._unpack(state, params)
        m_prod = masses.unsqueeze(2) * masses.unsqueeze(1)
        couplings = -g_val.view(-1, 1, 1) * m_prod
        _, forces = inverse_distance_potential_and_force(
            positions=q,
            couplings=couplings,
            exponent_p=self.potential_exponent_p,
            softening_eps=self.softening_eps,
        )
        # grad_q H = -forces, grad_p H = p / m
        grad_q = (-forces).reshape(state.shape[0], self._nd)
        grad_p = (p / masses.unsqueeze(-1)).reshape(state.shape[0], self._nd)
        return torch.cat([grad_q, grad_p], dim=-1)

    def structure_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        nd = self._nd
        eye = torch.eye(nd, dtype=state.dtype, device=state.device)
        zeros = torch.zeros((nd, nd), dtype=state.dtype, device=state.device)
        top = torch.cat([zeros, eye], dim=-1)
        bottom = torch.cat([-eye, zeros], dim=-1)
        j_single = torch.cat([top, bottom], dim=0)
        return j_single.unsqueeze(0).expand(batch, -1, -1).clone()

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        nd = self._nd
        _, _, _, _, drag = self._unpack(state, params)
        r_mat = torch.zeros((batch, 2 * nd, 2 * nd), dtype=state.dtype, device=state.device)
        idx = torch.arange(nd, 2 * nd, device=state.device)
        r_mat[:, idx, idx] = drag.unsqueeze(-1)
        return r_mat

    def angular_momentum_z(self, state: torch.Tensor) -> torch.Tensor:
        """Compute total z-axis angular momentum `L_z = sum_i (q_{i,x} * p_{i,y} - q_{i,y} * p_{i,x})` (`[Batch]`)."""
        if self.spatial_dim < 2:
            return torch.zeros(state.shape[0], dtype=state.dtype, device=state.device)
        batch = state.shape[0]
        q = state[:, : self._nd].view(batch, self.num_bodies, self.spatial_dim)
        p = state[:, self._nd :].view(batch, self.num_bodies, self.spatial_dim)
        return torch.sum(q[:, :, 0] * p[:, :, 1] - q[:, :, 1] * p[:, :, 0], dim=-1)


class NonlinearPendulumAndViscoelasticBrick(LawBrick):
    """
    Port-Hamiltonian mechanical oscillator combining both:
        - Trigonometric gravity/Josephson restoring potential `m * g * l * (1 - cos(q))`
        - Hookean elastic storage `0.5 * k_spring * q^2`
        - Kelvin-Voigt viscous damping `damping * (p / m)`

    State vector (`[Batch, 2]`):
        - `q`: Displacement or angle (`m` or `1`).
        - `p`: Linear or angular momentum (`N*s`).

    Parameters (`[Batch, 4]`):
        - `m`: Inertia / mass (`> 0`).
        - `k_spring`: Linear elastic stiffness (`>= 0`).
        - `mgl_trig`: Trigonometric potential amplitude (`>= 0`).
        - `damping`: Viscous damping coefficient (`>= 0`).
    """

    def __init__(
        self,
        name: str = "nonlinear_pendulum_viscoelastic",
        m_default: float = 1.0,
        k_spring_default: float = 1.0,
        mgl_trig_default: float = 9.80665,
        damping_default: float = 0.05,
        q_unit: Unit = m,
        p_unit: Unit = N_force * s,
        time_unit: Unit = s,
        energy_unit: Unit = J,
    ) -> None:
        state_specs = [
            StateVarSpec(
                name="q",
                unit=q_unit,
                default_value=0.5,
                homeostatic_min=-100.0,
                homeostatic_max=100.0,
                description="Generalized coordinate / displacement",
            ),
            StateVarSpec(
                name="p",
                unit=p_unit,
                default_value=0.0,
                homeostatic_min=-500.0,
                homeostatic_max=500.0,
                description="Generalized momentum",
            ),
        ]
        param_specs = [
            ParamSpec("m", kg, default_value=m_default, min_value=1e-9),
            ParamSpec("k_spring", DIMENSIONLESS, default_value=k_spring_default, min_value=0.0),
            ParamSpec("mgl_trig", J, default_value=mgl_trig_default, min_value=0.0),
            ParamSpec("damping", DIMENSIONLESS, default_value=damping_default, min_value=0.0),
        ]
        port_specs = [
            PowerPortSpec(
                name="mechanical_force_port",
                effort_unit=m_per_s,
                flow_unit=N_force,
                domain="mechanics",
                state_var_name="p",
            )
        ]
        super().__init__(
            name=name,
            domain="mechanics",
            state_specs=state_specs,
            param_specs=param_specs,
            port_specs=port_specs,
            time_unit=time_unit,
            energy_unit=energy_unit,
        )

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        q = state[:, 0]
        p = state[:, 1]
        m_val = torch.clamp(params[:, 0], min=1e-9)
        k_sp = torch.clamp(params[:, 1], min=0.0)
        mgl = torch.clamp(params[:, 2], min=0.0)
        return 0.5 * (p**2) / m_val + 0.5 * k_sp * (q**2) + mgl * (1.0 - torch.cos(q))

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        q = state[:, 0]
        p = state[:, 1]
        m_val = torch.clamp(params[:, 0], min=1e-9)
        k_sp = torch.clamp(params[:, 1], min=0.0)
        mgl = torch.clamp(params[:, 2], min=0.0)
        grad_q = k_sp * q + mgl * torch.sin(q)
        grad_p = p / m_val
        return torch.stack([grad_q, grad_p], dim=-1)

    def structure_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        j_single = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=state.dtype, device=state.device)
        return j_single.unsqueeze(0).expand(batch, -1, -1).clone()

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        damping = torch.clamp(params[:, 3], min=0.0)
        r_mat = torch.zeros((batch, 2, 2), dtype=state.dtype, device=state.device)
        r_mat[:, 1, 1] = ohmic_dissipative_driving_force(
            effort=torch.ones_like(damping),
            equilibrium_effort=0.0,
            conductance=damping,
        )
        return r_mat


class QuantumSymplecticLindbladBrick(LawBrick):
    """
    Open quantum system mapped onto canonical symplectic phase-space quadratures `(q, p)` in `R^{2N}`
    where the complex wavefunction / coherent-state amplitude is `psi_k = (q_k + i * p_k) / sqrt(2)`.

    In this canonical quadrature representation:
        - The Born probability norm is `C(q, p) = sum_k |psi_k|^2 = 0.5 * sum_k (q_k^2 + p_k^2)`.
        - The rotating-frame Rabi-Kerr quantum Hamiltonian is:
            `H(q, p) = sum_k 0.5 * delta_k * (q_k^2 + p_k^2)`
            `        + sum_k 0.125 * chi_k * (q_k^2 + p_k^2)^2`
            `        + sum_{j < k} omega_{jk} * (q_j * q_k + p_j * p_k)`
        - Unitary Schrodinger-von Neumann evolution corresponds exactly to canonical Hamilton's equations
          `dq/dt = +grad_p H`, `dp/dt = -grad_q H` with `J = [[0, I_N], [-I_N, 0]]`, which identically
          conserves both `H(q, p)` and the Born norm `C(q, p)` by U(1) Noether gauge symmetry.
        - Open-system Lindblad relaxation / dephasing (`T_1 / T_2`) enters via the symmetric positive
          semi-definite Onsager dissipation matrix `R(q, p) = diag(gamma_0, ..., gamma_{N-1}, gamma_0, ..., gamma_{N-1}) >= 0`.

    State vector (`[Batch, 2 * N]`):
        - `q_0, ..., q_{N-1}`: In-phase real quadratures (`1`).
        - `p_0, ..., p_{N-1}`: Out-of-phase imaginary quadratures (`1`).

    Parameters (`[Batch, 3 * N + N*(N-1)/2]`):
        - `delta_0, ..., delta_{N-1}`: Level detunings / transition frequencies (`> 0`).
        - `chi_0, ..., chi_{N-1}`: Kerr anharmonicity coefficients (`>= 0`).
        - `omega_{j}_{k}` (`0 <= j < k < N`): Symmetric Rabi coupling amplitudes.
        - `gamma_0, ..., gamma_{N-1}`: Lindblad relaxation / dephasing rates (`>= 0`).
    """

    def __init__(
        self,
        name: str = "quantum_symplectic_lindblad",
        num_levels: int = 2,
        detunings_default: Sequence[float] | None = None,
        kerr_default: Sequence[float] | None = None,
        rabi_coupling_default: float = 0.25,
        lindblad_gamma_default: float = 0.0,
        time_unit: Unit = s,
        energy_unit: Unit = J,
    ) -> None:
        if num_levels < 2:
            raise ValueError("QuantumSymplecticLindbladBrick requires `num_levels >= 2`.")
        self.num_levels = int(num_levels)
        n = self.num_levels

        state_specs: list[StateVarSpec] = []
        for k in range(n):
            q_def = 1.0 if k == 0 else (0.4 if k == 1 else 0.0)
            state_specs.append(
                StateVarSpec(
                    name=f"q_{k}",
                    unit=DIMENSIONLESS,
                    default_value=q_def,
                    homeostatic_min=-10.0,
                    homeostatic_max=10.0,
                    description=f"Level {k} real symplectic quadrature Re(psi_{k}) * sqrt(2)",
                )
            )
        for k in range(n):
            p_def = 0.2 if k == 0 else (-0.3 if k == 1 else 0.0)
            state_specs.append(
                StateVarSpec(
                    name=f"p_{k}",
                    unit=DIMENSIONLESS,
                    default_value=p_def,
                    homeostatic_min=-10.0,
                    homeostatic_max=10.0,
                    description=f"Level {k} imaginary symplectic quadrature Im(psi_{k}) * sqrt(2)",
                )
            )

        detunings = (
            list(detunings_default)
            if detunings_default is not None
            else [1.5 + 0.5 * float(k) for k in range(n)]
        )
        kerrs = (
            list(kerr_default)
            if kerr_default is not None
            else [0.1 for _ in range(n)]
        )
        if len(detunings) != n or len(kerrs) != n:
            raise ValueError(f"Expected {n} detunings and Kerr coefficients.")

        param_specs: list[ParamSpec] = []
        for k in range(n):
            param_specs.append(
                ParamSpec(f"delta_{k}", energy_unit, default_value=float(detunings[k]), min_value=0.1)
            )
        for k in range(n):
            param_specs.append(
                ParamSpec(f"chi_{k}", energy_unit, default_value=float(kerrs[k]), min_value=0.0)
            )
        self._rabi_pairs: list[tuple[int, int]] = []
        for j in range(n):
            for k in range(j + 1, n):
                self._rabi_pairs.append((j, k))
                param_specs.append(
                    ParamSpec(
                        f"omega_{j}_{k}",
                        energy_unit,
                        default_value=float(rabi_coupling_default),
                        min_value=-0.45,
                        max_value=0.45,
                    )
                )
        for k in range(n):
            param_specs.append(
                ParamSpec(
                    f"gamma_{k}",
                    DIMENSIONLESS,
                    default_value=float(lindblad_gamma_default),
                    min_value=0.0,
                )
            )

        super().__init__(
            name=name,
            domain="quantum",
            state_specs=state_specs,
            param_specs=param_specs,
            port_specs=(),
            time_unit=time_unit,
            energy_unit=energy_unit,
        )

    def _unpack(
        self, state: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.num_levels
        num_pairs = len(self._rabi_pairs)
        q = state[:, :n]
        p = state[:, n : 2 * n]
        deltas = torch.clamp(params[:, :n], min=1e-4)
        chis = torch.clamp(params[:, n : 2 * n], min=0.0)
        omegas = params[:, 2 * n : 2 * n + num_pairs]
        gammas = torch.clamp(params[:, 2 * n + num_pairs : 3 * n + num_pairs], min=0.0)
        return q, p, deltas, chis, omegas, gammas

    def born_norm(self, state: torch.Tensor) -> torch.Tensor:
        """Compute total Born probability norm `sum_k |psi_k|^2 = 0.5 * sum_k (q_k^2 + p_k^2)` (`[Batch]`)."""
        return 0.5 * torch.sum(state**2, dim=-1)

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        q, p, deltas, chis, omegas, _ = self._unpack(state, params)
        amp_sq = q**2 + p**2  # 2 * |psi_k|^2
        h_diag = torch.sum(0.5 * deltas * amp_sq + 0.125 * chis * (amp_sq**2), dim=-1)
        h_rabi = torch.zeros_like(h_diag)
        for pair_idx, (j, k) in enumerate(self._rabi_pairs):
            omega_jk = omegas[:, pair_idx]
            h_rabi = h_rabi + omega_jk * (q[:, j] * q[:, k] + p[:, j] * p[:, k])
        return h_diag + h_rabi

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        q, p, deltas, chis, omegas, _ = self._unpack(state, params)
        amp_sq = q**2 + p**2
        coeff = deltas + 0.5 * chis * amp_sq
        grad_q = coeff * q
        grad_p = coeff * p
        for pair_idx, (j, k) in enumerate(self._rabi_pairs):
            omega_jk = omegas[:, pair_idx]
            grad_q = grad_q.clone()
            grad_p = grad_p.clone()
            grad_q[:, j] = grad_q[:, j] + omega_jk * q[:, k]
            grad_q[:, k] = grad_q[:, k] + omega_jk * q[:, j]
            grad_p[:, j] = grad_p[:, j] + omega_jk * p[:, k]
            grad_p[:, k] = grad_p[:, k] + omega_jk * p[:, j]
        return torch.cat([grad_q, grad_p], dim=-1)

    def structure_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        n = self.num_levels
        eye = torch.eye(n, dtype=state.dtype, device=state.device)
        zeros = torch.zeros((n, n), dtype=state.dtype, device=state.device)
        top = torch.cat([zeros, eye], dim=-1)
        bottom = torch.cat([-eye, zeros], dim=-1)
        j_single = torch.cat([top, bottom], dim=0)
        return j_single.unsqueeze(0).expand(batch, -1, -1).clone()

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        n = self.num_levels
        _, _, _, _, _, gammas = self._unpack(state, params)
        r_mat = torch.zeros((batch, 2 * n, 2 * n), dtype=state.dtype, device=state.device)
        idx_q = torch.arange(0, n, device=state.device)
        idx_p = torch.arange(n, 2 * n, device=state.device)
        r_mat[:, idx_q, idx_q] = gammas
        r_mat[:, idx_p, idx_p] = gammas
        return r_mat
