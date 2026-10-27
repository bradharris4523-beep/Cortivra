"""
Differentiable mathematical primitives and operator registry (`cortivra.layer2_fabric.primitives`).

Implements 16 differential and algebraic operators across four families:
    I.   Conservative and geometric field operators (quadratic energy storage, inverse-distance
         potentials, trigonometric phase coupling, skew-symmetric Poisson routing).
    II.  Statistical mechanics and thermodynamic barrier operators (Boltzmann partition switch,
         Arrhenius/Eyring barrier rate, Nernst logarithmic potential, GHK/Bernoulli flux).
    III. Chemical kinetics, saturation, and feedback operators (reversible mass-action flux,
         cooperative Hill/Michaelis-Menten saturation, ohmic/nonlinear dissipative driving force,
         cubic bistable/restoring polynomial).
    IV.  Spatial PDEs, relaxation, and event operators (first-order relaxation / Rush-Larsen step,
         graph Laplacian diffusion, conservative spatial advection, smooth event impulse).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from cortivra.layer1_axioms.units import (
    BODY_TEMPERATURE_K,
    DIM_DIMENSIONLESS,
    FARADAY,
    R_GAS,
    Dimension,
)


def _clamp_straight_through(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Clamp tensor values in forward pass while preserving identity gradients in backward pass."""
    clamped = torch.clamp(x, min=min_val, max=max_val)
    return x + (clamped - x).detach()


# ---------------------------------------------------------------------------
# Family I: Conservative & Geometric Field Operators
# ---------------------------------------------------------------------------


def quadratic_energy_storage(state: torch.Tensor, capacity_or_inertia: torch.Tensor | float) -> torch.Tensor:
    """
    Primitive #1: Quadratic energy storage `E(x) = 0.5 * (x^2 / C)` or `0.5 * k * x^2`.
    Here `capacity_or_inertia` is the generalized capacitance/mass `M` in `E(x) = 0.5 * x^2 / M`
    (for stiffness `k`, pass `1 / k`).
    """
    denom = torch.clamp(torch.as_tensor(capacity_or_inertia, dtype=state.dtype, device=state.device), min=1e-12)
    return 0.5 * (state**2) / denom


def inverse_distance_potential_and_force(
    positions: torch.Tensor,
    couplings: torch.Tensor,
    exponent_p: float = 1.0,
    softening_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Primitive #2: Pairwise power-law / inverse-distance interaction field (Gravitation, Coulomb, Lennard-Jones).

    Args:
        positions: Tensor of shape `[Batch, NumBodies, SpatialDim]`.
        couplings: Symmetric pairwise coupling matrix `alpha_{ij}` of shape `[Batch, NumBodies, NumBodies]`
                   (e.g., `-G * m_i * m_j` for gravity where `V(r) = alpha / r^p` is negative).
        exponent_p: Potential power `p` in `V(r) = alpha / r^p` (`p = 1.0` gives `1/r` potential and `1/r^2` force).
        softening_eps: Regularization length to prevent zero-distance division on collision.

    Returns:
        `(potential_energy, forces)` where `potential_energy` has shape `[Batch]` and
        `forces = -grad_q V` has shape `[Batch, NumBodies, SpatialDim]`.
    """
    # diff[b, i, j, d] = q_i - q_j
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, N, N, D]
    dist_sq = torch.sum(diff**2, dim=-1) + (softening_eps**2)  # [B, N, N]
    dist = torch.sqrt(dist_sq)

    n = positions.shape[1]
    off_diag = 1.0 - torch.eye(n, dtype=positions.dtype, device=positions.device).unsqueeze(0)

    pair_pot = couplings * off_diag / (dist**exponent_p)
    potential_energy = 0.5 * torch.sum(pair_pot, dim=(1, 2))

    # F_i = -dV/dq_i = sum_{j != i} p * alpha_{ij} * (q_i - q_j) / r^{p + 2}
    force_mag = (exponent_p * couplings * off_diag / (dist ** (exponent_p + 2.0))).unsqueeze(-1)
    forces = torch.sum(force_mag * diff, dim=2)  # [B, N, D]
    return potential_energy, forces


def trigonometric_phase_interaction(
    phases: torch.Tensor,
    adjacency_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Primitive #3: Periodic trigonometric phase potential and restoring torque (Pendulum, Josephson, Kuramoto).

    For phase vector `theta` (`[Batch, N]`) and symmetric coupling matrix `K_{ij}` (`[Batch, N, N]`):
        `V(theta) = 0.5 * sum_{i,j} K_{ij} * (1 - cos(theta_i - theta_j))`
        `-dV/dtheta_i = sum_j K_{ij} * sin(theta_j - theta_i)`
    """
    # diff[b, i, j] = theta_j - theta_i
    diff = phases.unsqueeze(1) - phases.unsqueeze(2)  # [B, N, N]
    potential = 0.5 * torch.sum(adjacency_weights * (1.0 - torch.cos(diff)), dim=(1, 2))
    torque = torch.sum(adjacency_weights * torch.sin(diff), dim=2)  # [B, N]
    return potential, torque


def skew_symmetric_poisson_flow(
    upper_params: torch.Tensor,
    effort: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Primitive #4: Skew-symmetric Poisson routing `F_cons = J * e` where `J = A - A^T` (`J = -J^T`).
    Guarantees `e . F_cons == 0` for any effort vector `e` (`[Batch, N]`) and matrix `upper_params` (`[Batch, N, N]`).
    """
    j_mat = upper_params - upper_params.transpose(1, 2)
    flow = torch.bmm(j_mat, effort.unsqueeze(-1)).squeeze(-1)
    return flow, j_mat


# ---------------------------------------------------------------------------
# Family II: Statistical Mechanics & Thermodynamic Barrier Operators
# ---------------------------------------------------------------------------


def boltzmann_partition_switch(
    x: torch.Tensor,
    x_half: torch.Tensor | float,
    slope_k: torch.Tensor | float,
) -> torch.Tensor:
    """
    Primitive #5: Thermodynamic two-state Boltzmann partition function:
        `sigma(x) = 1 / (1 + exp((x_half - x) / k))`
    Evaluated via `torch.special.expit` for overflow-free gradients across all float ranges.
    """
    k_t = torch.as_tensor(slope_k, dtype=x.dtype, device=x.device)
    sign = torch.where(k_t >= 0, torch.ones_like(k_t), -torch.ones_like(k_t))
    k_safe = sign * torch.clamp(torch.abs(k_t), min=1e-9)
    z = _clamp_straight_through((x - x_half) / k_safe, -80.0, 80.0)
    return torch.special.expit(z)


def eyring_barrier_rate(
    driving_energy_j_per_mol: torch.Tensor,
    base_rate: torch.Tensor | float,
    asymmetry_delta: torch.Tensor | float = 0.5,
    temperature_k: torch.Tensor | float = float(BODY_TEMPERATURE_K.value),
) -> torch.Tensor:
    """
    Primitive #6: Eyring / Arrhenius / Bell-Evans transition-state barrier rate:
        `k(DeltaG) = k_0 * exp((delta * DeltaG) / (R * T))`
    """
    rt = float(R_GAS.value) * torch.clamp(
        torch.as_tensor(temperature_k, dtype=driving_energy_j_per_mol.dtype, device=driving_energy_j_per_mol.device),
        min=1e-3,
    )
    exponent = _clamp_straight_through((asymmetry_delta * driving_energy_j_per_mol) / rt, -80.0, 80.0)
    return base_rate * torch.exp(exponent)


def nernst_equilibrium_potential_mv(
    c_out: torch.Tensor,
    c_in: torch.Tensor,
    valence_z: int | float = 1,
    temperature_k: torch.Tensor | float = float(BODY_TEMPERATURE_K.value),
) -> torch.Tensor:
    """
    Primitive #7: Nernst chemical equilibrium potential in millivolts (`mV`):
        `E_eq = (1000 * R * T / (z * F)) * ln(c_out / c_in)`
    """
    if valence_z == 0:
        raise ValueError("Valence `valence_z` in Nernst potential must be non-zero.")
    c_out_safe = torch.clamp(c_out, min=1e-12)
    c_in_safe = torch.clamp(c_in, min=1e-12)
    rt_over_zf_mv = (1000.0 * float(R_GAS.value) * temperature_k) / (float(valence_z) * float(FARADAY.value))
    return rt_over_zf_mv * torch.log(c_out_safe / c_in_safe)


def smooth_bernoulli_quotient(u: torch.Tensor) -> torch.Tensor:
    """
    C^infinity smooth evaluation of the Bernoulli function `B(u) = u / (exp(u) - 1)`.
    Uses the exact 4th-order Taylor series `1 - u/2 + u^2/12 - u^4/720` inside `|u| < 1e-4`
    to eliminate the 0/0 removable singularity at `u = 0` while preserving 1st and 2nd derivatives.
    """
    u_clamped = _clamp_straight_through(u, -80.0, 80.0)
    small_mask = torch.abs(u_clamped) < 1e-4
    # Replace small values in denominator branch to avoid 0/0 in unused branch of torch.where backward
    u_safe = torch.where(small_mask, torch.ones_like(u_clamped), u_clamped)
    exact_branch = u_safe / torch.expm1(u_safe)
    u2 = u_clamped * u_clamped
    taylor_branch = 1.0 - 0.5 * u_clamped + (u2 / 12.0) - (u2 * u2) / 720.0
    return torch.where(small_mask, taylor_branch, exact_branch)


def ghk_electrodiffusion_flux(
    voltage_mv: torch.Tensor,
    c_in: torch.Tensor,
    c_out: torch.Tensor,
    permeability: torch.Tensor | float,
    valence_z: int | float = 1,
    temperature_k: torch.Tensor | float = float(BODY_TEMPERATURE_K.value),
) -> torch.Tensor:
    """
    Primitive #8: Goldman-Hodgkin-Katz (GHK) nonlinear electrodiffusion current/flux.
    Rewritten in terms of `smooth_bernoulli_quotient(u)` where `u = z * F * V / (R * T)`:
        `I_GHK = P * z * F * (c_in * B(-u) - c_out * B(u))`
    Completely free of 0/0 division singularities at `V = 0 mV`.
    """
    v_th_mv = (1000.0 * float(R_GAS.value) * temperature_k) / (float(valence_z) * float(FARADAY.value))
    u = voltage_mv / v_th_mv
    zf = float(valence_z) * float(FARADAY.value)
    return permeability * zf * (c_in * smooth_bernoulli_quotient(-u) - c_out * smooth_bernoulli_quotient(u))


# ---------------------------------------------------------------------------
# Family III: Chemical Kinetics, Saturation, & Feedback Operators
# ---------------------------------------------------------------------------


def reversible_mass_action_rates(
    concentrations: torch.Tensor,
    reactant_stoich: torch.Tensor,
    product_stoich: torch.Tensor,
    k_forward: torch.Tensor,
    k_reverse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Primitive #9: Reversible mass-action reaction rates and species time derivatives.

    Args:
        concentrations: Species concentrations `[Batch, NumSpecies]` (`c >= 0`).
        reactant_stoich: Forward stoichiometric coefficients `nu^+` (`[NumSpecies, NumReactions]`).
        product_stoich: Reverse stoichiometric coefficients `nu^-` (`[NumSpecies, NumReactions]`).
        k_forward: Forward rate constants `[Batch, NumReactions]` or `[NumReactions]`.
        k_reverse: Reverse rate constants `[Batch, NumReactions]` or `[NumReactions]`.

    Returns:
        `(dc_dt, net_reaction_rates)` of shapes `([Batch, NumSpecies], [Batch, NumReactions])`.
    """
    c_safe = torch.clamp(concentrations, min=1e-15)
    log_c = torch.log(c_safe)  # [B, S]
    # prod_i c_i^{nu_{ir}} = exp(sum_i nu_{ir} * ln(c_i))
    fwd_activity = torch.exp(_clamp_straight_through(torch.matmul(log_c, reactant_stoich), -80.0, 80.0))
    rev_activity = torch.exp(_clamp_straight_through(torch.matmul(log_c, product_stoich), -80.0, 80.0))

    net_rates = k_forward * fwd_activity - k_reverse * rev_activity  # [B, R]
    net_stoich = product_stoich - reactant_stoich  # [S, R]
    dc_dt = torch.matmul(net_rates, net_stoich.T)  # [B, S]
    return dc_dt, net_rates


def cooperative_hill_saturation(
    concentration: torch.Tensor,
    half_maximal_k: torch.Tensor | float,
    hill_n: torch.Tensor | float = 1.0,
    inhibitory: bool = False,
) -> torch.Tensor:
    """
    Primitive #10: Cooperative Hill / Langmuir / Michaelis-Menten saturation in log-space:
        Activation (`inhibitory=False`): `c^n / (K^n + c^n) = expit(n * (ln(c) - ln(K)))`
        Inhibition (`inhibitory=True`):  `K^n / (K^n + c^n) = expit(n * (ln(K) - ln(c)))`
    """
    c_safe = torch.clamp(concentration, min=1e-15)
    k_safe = torch.clamp(
        torch.as_tensor(half_maximal_k, dtype=concentration.dtype, device=concentration.device),
        min=1e-15,
    )
    log_ratio = torch.log(k_safe) - torch.log(c_safe) if inhibitory else torch.log(c_safe) - torch.log(k_safe)
    z = _clamp_straight_through(hill_n * log_ratio, -80.0, 80.0)
    return torch.special.expit(z)


def ohmic_dissipative_driving_force(
    effort: torch.Tensor,
    equilibrium_effort: torch.Tensor | float,
    conductance: torch.Tensor | float,
    power_exponent: float = 1.0,
) -> torch.Tensor:
    """
    Primitive #11: Dissipative flux driven by generalized effort difference:
        `J = G * (e - e_eq) * |e - e_eq|^{p - 1}`
    When `G >= 0` and `p >= 1`, entropy production `(e - e_eq) * J >= 0` is guaranteed everywhere.
    """
    g_safe = torch.clamp(
        torch.as_tensor(conductance, dtype=effort.dtype, device=effort.device),
        min=0.0,
    )
    delta_e = effort - equilibrium_effort
    if power_exponent == 1.0:
        return g_safe * delta_e
    return g_safe * delta_e * ((torch.abs(delta_e) + 1e-12) ** (power_exponent - 1.0))


def cubic_bistable_feedback(
    x: torch.Tensor,
    linear_gain: torch.Tensor | float = 1.0,
    cubic_damping: torch.Tensor | float = 1.0,
    bias: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """
    Primitive #12: Polynomial bistable / Ginzburg-Landau / FitzHugh-Nagumo feedback:
        `f(x) = a * x - b * x^3 + bias`
    """
    return linear_gain * x - cubic_damping * (x**3) + bias


# ---------------------------------------------------------------------------
# Family IV: Spatial PDEs, Relaxation, & Event Operators
# ---------------------------------------------------------------------------


def linear_relaxation_rate(
    state: torch.Tensor,
    steady_state: torch.Tensor,
    tau: torch.Tensor | float,
) -> torch.Tensor:
    """
    Primitive #13a: Continuous first-order relaxation derivative `dx/dt = (x_inf - x) / tau`.
    """
    tau_safe = torch.clamp(
        torch.as_tensor(tau, dtype=state.dtype, device=state.device),
        min=1e-9,
    )
    return (steady_state - state) / tau_safe


def step_rush_larsen(
    state: torch.Tensor,
    steady_state: torch.Tensor,
    tau: torch.Tensor | float,
    dt: float,
) -> torch.Tensor:
    """
    Primitive #13b: Exact exponential-Euler (Rush-Larsen) step for stiff relaxation equations:
        `x(t + dt) = x_inf + (x(t) - x_inf) * exp(-dt / tau)`
    Unconditionally stable for any `dt > 0` and preserves gating bounds `x in [0, 1]` when `x_inf in [0, 1]`.
    """
    tau_safe = torch.clamp(
        torch.as_tensor(tau, dtype=state.dtype, device=state.device),
        min=1e-9,
    )
    decay = torch.exp(_clamp_straight_through(-dt / tau_safe, -80.0, 0.0))
    return steady_state + (state - steady_state) * decay


def graph_laplacian_diffusion_1d(
    field: torch.Tensor,
    diffusivity: torch.Tensor | float,
    dx: float = 1.0,
) -> torch.Tensor:
    """
    Primitive #14a: O(N) matrix-free 1D spatial Laplacian diffusion with zero-flux (Neumann) boundaries:
        `du_i/dt = (D / dx^2) * (u_{i-1} - 2*u_i + u_{i+1})`
    Conserves total integral `sum_i u_i` to machine precision (`1^T L = 0`).
    """
    if field.shape[-1] < 2:
        return torch.zeros_like(field)
    d_safe = torch.clamp(
        torch.as_tensor(diffusivity, dtype=field.dtype, device=field.device),
        min=0.0,
    )
    left = torch.cat([field[..., :1], field[..., :-1]], dim=-1)
    right = torch.cat([field[..., 1:], field[..., -1:]], dim=-1)
    laplacian = left - 2.0 * field + right
    return (d_safe / (dx * dx)) * laplacian


def graph_laplacian_sparse_edges(
    node_values: torch.Tensor,
    edge_index: torch.Tensor,
    edge_conductances: torch.Tensor,
) -> torch.Tensor:
    """
    Primitive #14b: O(|E|) matrix-free diffusion over an arbitrary undirected graph topology
    (branched trees, 2D/3D irregular meshes, network graphs):
        `du_i/dt = sum_{e=(i,j)} g_e * (u_j - u_i)`

    Args:
        node_values: Field values at graph nodes `[Batch, NumNodes]`.
        edge_index: Integer tensor `[2, NumEdges]` of undirected edge endpoints `(src, dst)`.
        edge_conductances: Non-negative edge weights `[NumEdges]` or `[Batch, NumEdges]`.

    Returns:
        `du_dt` of shape `[Batch, NumNodes]`, satisfying `sum_i du_i/dt == 0` (exact conservation)
        and `-sum_i u_i * du_i/dt == sum_{e=(i,j)} g_e * (u_i - u_j)^2 >= 0` (2nd Law PSD).
    """
    if edge_index.numel() == 0:
        return torch.zeros_like(node_values)
    src = edge_index[0].to(dtype=torch.long, device=node_values.device)
    dst = edge_index[1].to(dtype=torch.long, device=node_values.device)
    g_safe = torch.clamp(
        torch.as_tensor(edge_conductances, dtype=node_values.dtype, device=node_values.device),
        min=0.0,
    )
    # Flux from src to dst: J_{src->dst} = g_e * (u_src - u_dst)
    u_src = node_values[:, src]  # [B, E]
    u_dst = node_values[:, dst]  # [B, E]
    edge_flux = g_safe * (u_src - u_dst)  # [B, E]

    du_dt = torch.zeros_like(node_values)
    batch = node_values.shape[0]
    src_exp = src.unsqueeze(0).expand(batch, -1)
    dst_exp = dst.unsqueeze(0).expand(batch, -1)
    du_dt = du_dt.scatter_add(dim=1, index=src_exp, src=-edge_flux)
    du_dt = du_dt.scatter_add(dim=1, index=dst_exp, src=edge_flux)
    return du_dt


def conservative_advection_1d(
    field: torch.Tensor,
    velocity: torch.Tensor | float,
    dx: float = 1.0,
) -> torch.Tensor:
    """
    Primitive #15: O(N) matrix-free skew-symmetric periodic advection `-v * du/dx`:
        `du_i/dt = -v * (u_{(i+1) mod N} - u_{(i-1) mod N}) / (2 * dx)`
    Skew-symmetry guarantees `sum_i u_i * (du_i/dt) == 0` (exact L2 energy conservation).
    """
    right = torch.roll(field, shifts=-1, dims=-1)
    left = torch.roll(field, shifts=1, dims=-1)
    return -velocity * (right - left) / (2.0 * dx)


def surrogate_event_impulse(
    state: torch.Tensor,
    threshold: torch.Tensor | float,
    sharpness_beta: float = 1.0,
) -> torch.Tensor:
    """
    Primitive #16: Smooth event-triggered impulse kernel `delta_beta(x - x_th)`:
        `delta_beta(u) = beta * sigmoid(beta * u) * (1 - sigmoid(beta * u))`
    Integrates to 1 over `(-inf, +inf)` and provides non-zero gradients across threshold crossings.
    """
    u = _clamp_straight_through(sharpness_beta * (state - threshold), -80.0, 80.0)
    sig = torch.special.expit(u)
    return sharpness_beta * sig * (1.0 - sig)


# ---------------------------------------------------------------------------
# Formal Primitive Registry for Symbolic Compilation & Discovery (Layers 3-4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrimitiveSpec:
    """Metadata and dimensional signature for one of the 16 universal mathematical primitives."""

    primitive_id: str
    family: str
    thermodynamic_role: str  # "conservative", "dissipative", "equilibrium_partition", or "state_transition"
    symbolic_template: str
    output_dimension: Dimension | None
    fn: Callable
    description: str


PRIMITIVE_REGISTRY: dict[str, PrimitiveSpec] = {
    "quadratic_energy_storage": PrimitiveSpec(
        primitive_id="quadratic_energy_storage",
        family="I_conservative_geometric",
        thermodynamic_role="conservative",
        symbolic_template="0.5 * x^2 / C",
        output_dimension=None,
        fn=quadratic_energy_storage,
        description="Quadratic energy storage in mass, capacitance, inductance, or Hookean elasticity.",
    ),
    "inverse_distance_potential": PrimitiveSpec(
        primitive_id="inverse_distance_potential",
        family="I_conservative_geometric",
        thermodynamic_role="conservative",
        symbolic_template="alpha / r^p",
        output_dimension=None,
        fn=inverse_distance_potential_and_force,
        description="Pairwise inverse-distance potential and central force (Gravitation, Coulomb, Lennard-Jones).",
    ),
    "trigonometric_phase_interaction": PrimitiveSpec(
        primitive_id="trigonometric_phase_interaction",
        family="I_conservative_geometric",
        thermodynamic_role="conservative",
        symbolic_template="K * sin(theta_j - theta_i)",
        output_dimension=None,
        fn=trigonometric_phase_interaction,
        description="Periodic trigonometric phase potential and restoring torque (Pendulum, Josephson, Kuramoto).",
    ),
    "skew_symmetric_poisson_flow": PrimitiveSpec(
        primitive_id="skew_symmetric_poisson_flow",
        family="I_conservative_geometric",
        thermodynamic_role="conservative",
        symbolic_template="(A - A^T) * grad_H",
        output_dimension=None,
        fn=skew_symmetric_poisson_flow,
        description="Energy-conserving skew-symmetric Poisson routing.",
    ),
    "boltzmann_partition_switch": PrimitiveSpec(
        primitive_id="boltzmann_partition_switch",
        family="II_statistical_thermodynamic",
        thermodynamic_role="equilibrium_partition",
        symbolic_template="1 / (1 + exp((x_half - x) / k))",
        output_dimension=DIM_DIMENSIONLESS,
        fn=boltzmann_partition_switch,
        description="Two-state Boltzmann thermal occupancy switch.",
    ),
    "eyring_barrier_rate": PrimitiveSpec(
        primitive_id="eyring_barrier_rate",
        family="II_statistical_thermodynamic",
        thermodynamic_role="state_transition",
        symbolic_template="k0 * exp(delta * dG / (R * T))",
        output_dimension=None,
        fn=eyring_barrier_rate,
        description="Eyring/Arrhenius/Bell-Evans exponential barrier transition rate.",
    ),
    "nernst_equilibrium_potential": PrimitiveSpec(
        primitive_id="nernst_equilibrium_potential",
        family="II_statistical_thermodynamic",
        thermodynamic_role="equilibrium_partition",
        symbolic_template="(R * T / (z * F)) * ln(c_out / c_in)",
        output_dimension=None,
        fn=nernst_equilibrium_potential_mv,
        description="Nernst chemical equilibrium potential across a semipermeable boundary.",
    ),
    "ghk_electrodiffusion_flux": PrimitiveSpec(
        primitive_id="ghk_electrodiffusion_flux",
        family="II_statistical_thermodynamic",
        thermodynamic_role="dissipative",
        symbolic_template="P * z * F * (c_in * B(-u) - c_out * B(u))",
        output_dimension=None,
        fn=ghk_electrodiffusion_flux,
        description="Singularity-free Goldman-Hodgkin-Katz electrodiffusion flux.",
    ),
    "reversible_mass_action_rates": PrimitiveSpec(
        primitive_id="reversible_mass_action_rates",
        family="III_kinetics_feedback",
        thermodynamic_role="dissipative",
        symbolic_template="k_f * prod(c^nu_f) - k_r * prod(c^nu_r)",
        output_dimension=None,
        fn=reversible_mass_action_rates,
        description="Reversible stoichiometric mass-action reaction kinetics.",
    ),
    "cooperative_hill_saturation": PrimitiveSpec(
        primitive_id="cooperative_hill_saturation",
        family="III_kinetics_feedback",
        thermodynamic_role="equilibrium_partition",
        symbolic_template="c^n / (K^n + c^n)",
        output_dimension=DIM_DIMENSIONLESS,
        fn=cooperative_hill_saturation,
        description="Log-space cooperative Hill / Langmuir / Michaelis-Menten saturation.",
    ),
    "ohmic_dissipative_driving_force": PrimitiveSpec(
        primitive_id="ohmic_dissipative_driving_force",
        family="III_kinetics_feedback",
        thermodynamic_role="dissipative",
        symbolic_template="G * (e - e_eq) * |e - e_eq|^(p - 1)",
        output_dimension=None,
        fn=ohmic_dissipative_driving_force,
        description="Linear or power-law dissipative driving force (Ohm, Fourier, Stokes drag).",
    ),
    "cubic_bistable_feedback": PrimitiveSpec(
        primitive_id="cubic_bistable_feedback",
        family="III_kinetics_feedback",
        thermodynamic_role="state_transition",
        symbolic_template="a * x - b * x^3 + bias",
        output_dimension=None,
        fn=cubic_bistable_feedback,
        description="Cubic bistable / Ginzburg-Landau / FitzHugh-Nagumo feedback.",
    ),
    "linear_relaxation_rate": PrimitiveSpec(
        primitive_id="linear_relaxation_rate",
        family="IV_spatial_relaxation_events",
        thermodynamic_role="dissipative",
        symbolic_template="(x_inf - x) / tau",
        output_dimension=None,
        fn=linear_relaxation_rate,
        description="First-order exponential relaxation toward steady state.",
    ),
    "graph_laplacian_diffusion_1d": PrimitiveSpec(
        primitive_id="graph_laplacian_diffusion_1d",
        family="IV_spatial_relaxation_events",
        thermodynamic_role="dissipative",
        symbolic_template="D * laplacian(u)",
        output_dimension=None,
        fn=graph_laplacian_diffusion_1d,
        description="O(N) mass-conserving spatial Laplacian diffusion.",
    ),
    "conservative_advection_1d": PrimitiveSpec(
        primitive_id="conservative_advection_1d",
        family="IV_spatial_relaxation_events",
        thermodynamic_role="conservative",
        symbolic_template="-v * du/dx",
        output_dimension=None,
        fn=conservative_advection_1d,
        description="O(N) skew-symmetric energy-conserving spatial advection.",
    ),
    "surrogate_event_impulse": PrimitiveSpec(
        primitive_id="surrogate_event_impulse",
        family="IV_spatial_relaxation_events",
        thermodynamic_role="state_transition",
        symbolic_template="beta * sigmoid(beta*(x - x_th)) * (1 - sigmoid(beta*(x - x_th)))",
        output_dimension=DIM_DIMENSIONLESS,
        fn=surrogate_event_impulse,
        description="Differentiable surrogate Dirac impulse across threshold events.",
    ),
}


def get_primitives_by_role(thermodynamic_role: str) -> list[PrimitiveSpec]:
    """Return all registered primitives matching `thermodynamic_role`."""
    return [spec for spec in PRIMITIVE_REGISTRY.values() if spec.thermodynamic_role == thermodynamic_role]
