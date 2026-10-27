"""
Closed-loop symbolic law compilation, Hamiltonian potential patching, non-Markovian
latent state augmentation, and cross-domain archetype classification
(`cortivra.layer5_discovery.latent_discovery`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick, ParamSpec, StateVarSpec
from cortivra.layer1_axioms.shields import AuditCertificate, audit_law_brick
from cortivra.layer1_axioms.units import (
    DIMENSIONLESS,
    DimensionError,
    DimensionlessGroup,
    J,
    Quantity,
    Unit,
    discover_dimensionless_groups,
    s,
)
from cortivra.layer3_compiler.ast_nodes import (
    BinaryOpNode,
    ConstNode,
    ExprNode,
    PrimitiveCallNode,
    SymbolNode,
    UnaryOpNode,
    parse_symbolic_expr,
)
from cortivra.layer3_compiler.symbolic_compiler import (
    CompiledLawBrick,
    ResidualPatchedLawBrick,
    SymbolicSystemSpec,
    build_symbol_table,
    compile_residual_patch_brick,
    compile_symbolic_system,
)
from cortivra.layer4_engine.elimination_funnel import (
    HypothesisRankingEntry,
    rank_hypothesis_portfolio,
    run_elimination_funnel,
)
from cortivra.layer4_engine.inverse_solver import (
    CalibrationAndResidualReport,
    calibrate_and_audit_residual,
)
from cortivra.layer4_engine.rollout_arena import (
    estimate_trajectory_derivatives,
    rollout_trajectories,
)
from cortivra.layer5_discovery.basis_library import generate_symbolic_basis_library
from cortivra.layer5_discovery.sparse_regression import (
    InvariantDiscoveryReport,
    ParetoDiscoveryReport,
    SymbolicCandidate,
    discover_conservation_invariants,
    discover_rational_pade_candidates,
    solve_sparse_pareto_front,
)


class HamiltonianPatchedLawBrick(LawBrick):
    """
    Composite `LawBrick` formed by augmenting `base_brick` with a symbolic potential energy
    addition `Delta_H(x, params)` in `base_brick.energy_unit`.

    The effort vector `e(x) = grad_x(H_base(x) + Delta_H(x))` is differentiated symbolically
    and routed through `base_brick`'s `(J(x) - R(x))` operators, preserving exact conservative
    orthogonality (`grad_H . F_cons == 0`) and 2nd Law non-negativity (`grad_H . F_diss >= 0`).
    """

    def __init__(
        self,
        name: str,
        base_brick: LawBrick,
        patch_param_specs: Sequence[ParamSpec],
        delta_hamiltonian_expr: str | ExprNode,
        constants: Mapping[str, Quantity | ConstNode] | None = None,
    ) -> None:
        self.base_brick = base_brick
        self.base_param_dim = base_brick.param_dim
        self.patch_param_specs = tuple(patch_param_specs)

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
        )

        self.symbol_table = build_symbol_table(self.state_specs, self.param_specs, constants)
        if isinstance(delta_hamiltonian_expr, ExprNode):
            node = delta_hamiltonian_expr.simplify()
            node.infer_unit()
        else:
            node = parse_symbolic_expr(delta_hamiltonian_expr, symbol_table=self.symbol_table).simplify()

        h_u = node.infer_unit()
        if h_u.dimension != self.energy_unit.dimension and h_u.dimension != DIMENSIONLESS.dimension:
            raise DimensionError(
                f"HamiltonianPatchedLawBrick '{name}': Delta_H has dimension "
                f"{h_u.dimension.format_si()}, expected {self.energy_unit.dimension.format_si()}."
            )
        self._h_scale = (
            h_u.conversion_factor_to(self.energy_unit)
            if h_u.dimension == self.energy_unit.dimension
            else 1.0
        )
        self.delta_hamiltonian_ast = node
        self._delta_effort_asts = [
            self.delta_hamiltonian_ast.diff(sp.name).simplify() for sp in self.state_specs
        ]

    def _build_env(self, state: torch.Tensor, params: torch.Tensor) -> dict[str, torch.Tensor]:
        env: dict[str, torch.Tensor] = {}
        for idx, sp in enumerate(self.state_specs):
            env[sp.name] = state[:, idx]
        for idx, pp in enumerate(self.param_specs):
            env[pp.name] = params[:, idx]
        return env

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        h_base = self.base_brick.hamiltonian_energy(state, params[:, : self.base_param_dim])
        env = self._build_env(state, params)
        h_delta = self.delta_hamiltonian_ast.eval_tensor(env) * self._h_scale
        return h_base + h_delta

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        e_base = self.base_brick.effort(state, params[:, : self.base_param_dim])
        env = self._build_env(state, params)
        e_delta = torch.stack(
            [node.eval_tensor(env) * self._h_scale for node in self._delta_effort_asts],
            dim=-1,
        )
        return e_base + e_delta

    def structure_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return self.base_brick.structure_matrix(state, params[:, : self.base_param_dim])

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return self.base_brick.dissipation_matrix(state, params[:, : self.base_param_dim])

    def total_patch_complexity(self) -> int:
        return self.delta_hamiltonian_ast.tree_complexity()


class LatentAugmentedLawBrick(LawBrick):
    """
    Composite `(D + 1)`-state `LawBrick` formed by augmenting `base_brick` with an unobserved
    internal state variable `z(t)` that couples to `target_state_index` and evolves according
    to a relaxation / adaptation ODE:
        `dx_{target}/dt = dx_{target, base}/dt + k_couple * z`
        `dz/dt = -k_decay * z + drive_expr(x, params)`
    """

    def __init__(
        self,
        name: str,
        base_brick: LawBrick,
        target_state_index: int,
        latent_state_spec: StateVarSpec,
        latent_param_specs: Sequence[ParamSpec],
        coupling_expr: str | ExprNode,
        latent_drive_expr: str | ExprNode,
        decay_param_name: str,
    ) -> None:
        if not (0 <= target_state_index < base_brick.state_dim):
            raise ValueError(f"Invalid target_state_index {target_state_index}.")

        self.base_brick = base_brick
        self.base_state_dim = base_brick.state_dim
        self.base_param_dim = base_brick.param_dim
        self.target_state_index = target_state_index
        self.latent_state_spec = latent_state_spec
        self.latent_param_specs = tuple(latent_param_specs)
        self.decay_param_name = decay_param_name

        aug_states = list(base_brick.state_specs) + [latent_state_spec]
        aug_params = list(base_brick.param_specs) + list(self.latent_param_specs)

        super().__init__(
            name=name,
            domain=base_brick.domain,
            state_specs=aug_states,
            param_specs=aug_params,
            port_specs=base_brick.port_specs,
            time_unit=base_brick.time_unit,
            energy_unit=base_brick.energy_unit,
            matrix_free=base_brick.is_matrix_free,
        )

        self.symbol_table = build_symbol_table(self.state_specs, self.param_specs)
        self.coupling_ast = (
            coupling_expr.simplify()
            if isinstance(coupling_expr, ExprNode)
            else parse_symbolic_expr(coupling_expr, symbol_table=self.symbol_table).simplify()
        )
        self.latent_drive_ast = (
            latent_drive_expr.simplify()
            if isinstance(latent_drive_expr, ExprNode)
            else parse_symbolic_expr(latent_drive_expr, symbol_table=self.symbol_table).simplify()
        )
        self.decay_param_index = next(
            idx for idx, p in enumerate(self.param_specs) if p.name == decay_param_name
        )

    def _build_env(self, state: torch.Tensor, params: torch.Tensor) -> dict[str, torch.Tensor]:
        env: dict[str, torch.Tensor] = {}
        for idx, sp in enumerate(self.state_specs):
            env[sp.name] = state[:, idx]
        for idx, pp in enumerate(self.param_specs):
            env[pp.name] = params[:, idx]
        return env

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x_base = state[:, : self.base_state_dim]
        p_base = params[:, : self.base_param_dim]
        z = state[:, self.base_state_dim]
        return self.base_brick.hamiltonian_energy(x_base, p_base) + 0.5 * (z * z)

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x_base = state[:, : self.base_state_dim]
        p_base = params[:, : self.base_param_dim]
        z = state[:, self.base_state_dim : self.base_state_dim + 1]
        e_base = self.base_brick.effort(x_base, p_base)
        return torch.cat([e_base, z], dim=-1)

    def structure_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        n = self.state_dim
        j_full = torch.zeros((batch, n, n), dtype=state.dtype, device=state.device)
        if not self.base_brick.is_matrix_free:
            j_base = self.base_brick.structure_matrix(
                state[:, : self.base_state_dim], params[:, : self.base_param_dim]
            )
            j_full[:, : self.base_state_dim, : self.base_state_dim] = j_base
        return j_full

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        n = self.state_dim
        r_full = torch.zeros((batch, n, n), dtype=state.dtype, device=state.device)
        if not self.base_brick.is_matrix_free:
            r_base = self.base_brick.dissipation_matrix(
                state[:, : self.base_state_dim], params[:, : self.base_param_dim]
            )
            r_full[:, : self.base_state_dim, : self.base_state_dim] = r_base
        k_dec = torch.clamp(params[:, self.decay_param_index], min=0.0)
        r_full[:, self.base_state_dim, self.base_state_dim] = k_dec
        return r_full

    def compute_active_patch_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        env = self._build_env(state, params)
        u = torch.zeros_like(state)
        u[:, self.target_state_index] = self.coupling_ast.eval_tensor(env)
        u[:, self.base_state_dim] = self.latent_drive_ast.eval_tensor(env)
        return u

    def external_patch_power(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        e = self.effort(state, params)
        u = self.compute_active_patch_flow(state, params)
        return (e * u).sum(dim=-1)

    def flow(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        port_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        dx_dt = super().flow(state, params, port_inputs=port_inputs)
        return dx_dt + self.compute_active_patch_flow(state, params)


class LatentOscillatorAugmentedLawBrick(LawBrick):
    """
    Composite `(D + 2)`-state `LawBrick` formed by augmenting `base_brick` with an unobserved
    conjugate harmonic oscillator mode `(z_q(t), z_p(t))` (e.g., an unseen orbital perturber,
    hidden electromagnetic resonator, or intracellular clock) coupled to `target_state_index`:
        `dx_{target}/dt = dx_{target, base}/dt + k_latent_couple * z_q`
        `dz_q/dt = omega_latent * z_p`
        `dz_p/dt = -omega_latent * z_q - k_latent_damp * z_p`
    """

    def __init__(
        self,
        name: str,
        base_brick: LawBrick,
        target_state_index: int,
        zq_spec: StateVarSpec,
        zp_spec: StateVarSpec,
        latent_param_specs: Sequence[ParamSpec],
    ) -> None:
        if not (0 <= target_state_index < base_brick.state_dim):
            raise ValueError(f"Invalid target_state_index {target_state_index}.")

        self.base_brick = base_brick
        self.base_state_dim = base_brick.state_dim
        self.base_param_dim = base_brick.param_dim
        self.target_state_index = target_state_index
        self.latent_param_specs = tuple(latent_param_specs)

        aug_states = list(base_brick.state_specs) + [zq_spec, zp_spec]
        aug_params = list(base_brick.param_specs) + list(self.latent_param_specs)

        super().__init__(
            name=name,
            domain=base_brick.domain,
            state_specs=aug_states,
            param_specs=aug_params,
            port_specs=base_brick.port_specs,
            time_unit=base_brick.time_unit,
            energy_unit=base_brick.energy_unit,
            matrix_free=base_brick.is_matrix_free,
        )

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x_base = state[:, : self.base_state_dim]
        p_base = params[:, : self.base_param_dim]
        zq = state[:, self.base_state_dim]
        zp = state[:, self.base_state_dim + 1]
        return self.base_brick.hamiltonian_energy(x_base, p_base) + 0.5 * (zq * zq + zp * zp)

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x_base = state[:, : self.base_state_dim]
        p_base = params[:, : self.base_param_dim]
        z_pair = state[:, self.base_state_dim : self.base_state_dim + 2]
        e_base = self.base_brick.effort(x_base, p_base)
        return torch.cat([e_base, z_pair], dim=-1)

    def structure_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        n = self.state_dim
        j_full = torch.zeros((batch, n, n), dtype=state.dtype, device=state.device)
        if not self.base_brick.is_matrix_free:
            j_base = self.base_brick.structure_matrix(
                state[:, : self.base_state_dim], params[:, : self.base_param_dim]
            )
            j_full[:, : self.base_state_dim, : self.base_state_dim] = j_base
        omega = params[:, self.base_param_dim + 1]
        idx_q = self.base_state_dim
        idx_p = self.base_state_dim + 1
        j_full[:, idx_q, idx_p] = omega
        j_full[:, idx_p, idx_q] = -omega
        return j_full

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        n = self.state_dim
        r_full = torch.zeros((batch, n, n), dtype=state.dtype, device=state.device)
        if not self.base_brick.is_matrix_free:
            r_base = self.base_brick.dissipation_matrix(
                state[:, : self.base_state_dim], params[:, : self.base_param_dim]
            )
            r_full[:, : self.base_state_dim, : self.base_state_dim] = r_base
        k_damp = torch.clamp(params[:, self.base_param_dim + 2], min=0.0)
        r_full[:, self.base_state_dim + 1, self.base_state_dim + 1] = k_damp
        return r_full

    def compute_active_patch_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        k_couple = params[:, self.base_param_dim]
        zq = state[:, self.base_state_dim]
        u = torch.zeros_like(state)
        u[:, self.target_state_index] = k_couple * zq
        return u

    def external_patch_power(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        e = self.effort(state, params)
        u = self.compute_active_patch_flow(state, params)
        return (e * u).sum(dim=-1)

    def flow(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        port_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        dx_dt = super().flow(state, params, port_inputs=port_inputs)
        return dx_dt + self.compute_active_patch_flow(state, params)


@dataclass(frozen=True)
class NonMarkovianMemoryReport:
    """
    Diagnostic report testing whether an unexplained residual `r(t)` can be written as a
    pure Markovian function `g(x(t))` of the observed states or requires an unobserved
    internal state variable `z(t)`.

    Attributes:
        is_non_markovian: True when hysteresis or autocorrelated static-fit failure indicates
                          an unmeasured state variable `z(t)`.
        hysteresis_ratio: Ratio of residual difference `||r(t_a) - r(t_b)||` at revisited states
                          `||x(t_a) - x(t_b)|| <= tol` to overall residual standard deviation.
        static_basis_r2: Best `R^2` achievable by a cubic polynomial basis of observed states `x(t)` alone.
        residual_lag1_autocorr: Lag-1 temporal autocorrelation of `r(t) - g_static(x(t))`.
        primary_affected_state: State index exhibiting the strongest non-Markovian residual.
    """

    is_non_markovian: bool
    hysteresis_ratio: float
    static_basis_r2: float
    residual_lag1_autocorr: float
    primary_affected_state: int


@dataclass(frozen=True)
class LatentAugmentationResult:
    """
    Result of synthesizing and calibrating a `(D + 1)`-state `LatentAugmentedLawBrick`
    or `(D + 2)`-state `LatentOscillatorAugmentedLawBrick`.

    Attributes:
        augmented_brick: Compiled `LatentAugmentedLawBrick` or `LatentOscillatorAugmentedLawBrick`.
        calibrated_params: `[1, P_aug]` calibrated parameter vector.
        reconstructed_initial_state: `[1, D_aug]` full initial state including latent coordinate(s).
        trajectory_r2: `R^2` on the observed state trajectory `[T, D]`.
        latent_ode_r2: `R^2` of the discovered latent ODE.
        audit_certificate: Layer 1 `AuditCertificate` for `augmented_brick`.
        latent_mode_type: `"relaxation_1d"` or `"oscillatory_2d"`.
    """

    augmented_brick: LatentAugmentedLawBrick | LatentOscillatorAugmentedLawBrick
    calibrated_params: torch.Tensor
    reconstructed_initial_state: torch.Tensor
    trajectory_r2: float
    latent_ode_r2: float
    audit_certificate: AuditCertificate
    latent_mode_type: str = "relaxation_1d"


@dataclass(frozen=True)
class ArchetypeFingerprint:
    """
    Universal mathematical archetype classification, differential operator invariants,
    and dimensionless Buckingham-Pi invariants.

    Attributes:
        archetype_name: Canonical operator class identifier.
        thermodynamic_slot: Thermodynamic operator slot of the discovered law.
        parity_symmetry: `"even"`, `"odd"`, or `"asymmetric"` under state inversion `x -> -x`.
        is_saturating: True if the discovered operator saturates at large state amplitude.
        dimensionless_groups: Tuple of Layer 1 `DimensionlessGroup` invariants.
        cross_domain_analogues: Tuple of mathematically isomorphic mechanisms across physics,
                                chemistry, electrodynamics, and biology.
        mean_phase_space_divergence: Mean trace of the flow Jacobian `<Tr(dF/dx)>` (Liouville rate).
        volume_behavior: `"volume_contracting_attractor"`, `"volume_preserving_liouville"`,
                         or `"volume_expanding_instability"`.
        max_polynomial_degree: Highest polynomial degree detected in the symbolic AST.
        coupled_state_pairs: Tuple of interacting state variable name pairs `(x_i, x_j)`.
    """

    archetype_name: str
    thermodynamic_slot: str
    parity_symmetry: str
    is_saturating: bool
    dimensionless_groups: tuple[DimensionlessGroup, ...]
    cross_domain_analogues: tuple[str, ...]
    mean_phase_space_divergence: float = 0.0
    volume_behavior: str = "volume_preserving_liouville"
    max_polynomial_degree: int = 1
    coupled_state_pairs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ClosedLoopDiscoveryReport:
    """
    Complete end-to-end report from Layer 5 (`discover_and_compile_law`).

    Attributes:
        discovery_triggered: True if an unmodeled mechanism was required and synthesized.
        winning_brick: Best compiled `LawBrick` (`ResidualPatchedLawBrick`, `HamiltonianPatchedLawBrick`,
                       `LatentAugmentedLawBrick`, or `base_brick` if no latent mechanism was needed).
        winning_params: `[1, P_total]` calibrated and Levenberg-Marquardt polished parameter vector.
        winning_candidate: Winning `SymbolicCandidate` (or None if latent-augmented or unneeded).
        pareto_report: `ParetoDiscoveryReport` over the symbolic candidate library.
        portfolio_rankings: Tuple of `HypothesisRankingEntry` from Layer 4 trajectory competition.
        memory_report: `NonMarkovianMemoryReport` testing for hidden variables.
        latent_augmentation: Optional `LatentAugmentationResult` when non-Markovian memory was discovered.
        archetype: `ArchetypeFingerprint` classifying the universal mathematical structure.
        audit_certificate: Layer 1 `AuditCertificate` for `winning_brick`.
        out_of_sample_r2: Horizon extrapolation `R^2` evaluated on the second half of the trajectory.
    """

    discovery_triggered: bool
    winning_brick: LawBrick
    winning_params: torch.Tensor
    winning_candidate: SymbolicCandidate | None
    pareto_report: ParetoDiscoveryReport | None
    portfolio_rankings: tuple[HypothesisRankingEntry, ...]
    memory_report: NonMarkovianMemoryReport
    latent_augmentation: LatentAugmentationResult | None
    archetype: ArchetypeFingerprint
    audit_certificate: AuditCertificate
    out_of_sample_r2: float = 1.0


def detect_non_markovian_memory(
    states: torch.Tensor,
    residual_flow: torch.Tensor,
    dt: float = 0.01,
    state_proximity_quantile: float = 0.08,
    min_time_separation_steps: int = 3,
) -> NonMarkovianMemoryReport:
    """
    Test whether `residual_flow` (`[T, D]`) is a single-valued function of the observed
    `states` (`[T, D]`) or exhibits non-Markovian hysteresis / hidden-variable memory.
    """
    x = states.detach().to(torch.float64)
    r = residual_flow.detach().to(torch.float64)
    t_steps, state_dim = x.shape

    rms_per_state = torch.sqrt(torch.mean(r * r, dim=0))
    primary_idx = int(torch.argmax(rms_per_state).item()) if state_dim > 0 else 0
    r_prim = r[:, primary_idx]
    r_std = float(torch.std(r_prim).item())
    if r_std <= 1e-10 or t_steps < 8:
        return NonMarkovianMemoryReport(
            is_non_markovian=False,
            hysteresis_ratio=0.0,
            static_basis_r2=1.0,
            residual_lag1_autocorr=0.0,
            primary_affected_state=primary_idx,
        )

    # 1. Compute pairwise state distance and residual difference for non-adjacent time steps
    x_scale = torch.clamp(torch.max(x, dim=0).values - torch.min(x, dim=0).values, min=1e-6)
    x_norm = x / x_scale.unsqueeze(0)
    dist_x = torch.cdist(x_norm, x_norm, p=2)  # [T, T]
    diff_r = torch.abs(r_prim.unsqueeze(1) - r_prim.unsqueeze(0))  # [T, T]

    idx_grid = torch.arange(t_steps, device=x.device)
    time_sep = torch.abs(idx_grid.unsqueeze(1) - idx_grid.unsqueeze(0)) >= min_time_separation_steps

    hysteresis_ratio = 0.0
    if bool(time_sep.any().item()):
        sep_dists = dist_x[time_sep]
        thresh_x = max(float(torch.quantile(sep_dists, state_proximity_quantile).item()), 0.05)
        close_mask = time_sep & (dist_x <= thresh_x)
        if bool(close_mask.any().item()):
            max_close_r_diff = float(torch.quantile(diff_r[close_mask], 0.90).item())
            hysteresis_ratio = max_close_r_diff / max(r_std, 1e-12)

    # 2. Fit a 3rd-order static polynomial of observed states x(t) and check remaining residual
    ones = torch.ones((t_steps, 1), dtype=torch.float64, device=x.device)
    cols = [ones, x, x ** 2, x ** 3]
    for i in range(state_dim):
        for j in range(i + 1, state_dim):
            cols.append((x[:, i] * x[:, j]).unsqueeze(-1))
    design = torch.cat(cols, dim=-1)
    sol = torch.linalg.lstsq(design, r_prim.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)
    r_hat = design @ sol
    err = r_prim - r_hat
    ss_res = float(torch.sum(err * err).item())
    ss_tot = float(torch.sum((r_prim - torch.mean(r_prim)) ** 2).item())
    static_r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-18))

    # Lag-1 autocorrelation of the unexplained static error e(t)
    e0 = err[:-1] - torch.mean(err[:-1])
    e1 = err[1:] - torch.mean(err[1:])
    denom = float(torch.sqrt(torch.sum(e0 * e0) * torch.sum(e1 * e1)).item())
    lag1_autocorr = float(torch.sum(e0 * e1).item()) / max(denom, 1e-12) if denom > 1e-12 else 0.0

    # Non-Markovian memory is present when even a cubic static basis fails to explain the residual
    # (static_r2 < 0.78) while the unexplained error is smooth in time (lag1_autocorr > 0.70)
    # and exhibits multi-valued hysteresis (hysteresis_ratio > 0.65)
    is_non_markov = bool(
        static_r2 < 0.78 and lag1_autocorr > 0.70 and hysteresis_ratio > 0.65
    )

    return NonMarkovianMemoryReport(
        is_non_markovian=is_non_markov,
        hysteresis_ratio=hysteresis_ratio,
        static_basis_r2=static_r2,
        residual_lag1_autocorr=lag1_autocorr,
        primary_affected_state=primary_idx,
    )


def discover_latent_state_augmentation(
    base_brick: LawBrick,
    observed_states: torch.Tensor,
    base_params: torch.Tensor,
    residual_flow: torch.Tensor,
    dt: float,
    target_state_index: int = 0,
    latent_name: str = "z_latent",
) -> LatentAugmentationResult:
    """
    Synthesize and calibrate a `(D + 1)`-state `LatentAugmentedLawBrick` when a residual
    `r_target(t)` is driven by an unobserved relaxation/adaptation variable `z(t)`:
        `dx_{target}/dt = dx_{target, base}/dt + k_couple * z`
        `dz/dt = -k_decay * z + sum_j c_j * x_j + c_bias`
    """
    x = observed_states.detach().to(torch.float64)
    r = residual_flow.detach().to(torch.float64)
    t_steps, state_dim = x.shape

    r_target = r[:, target_state_index]
    r_scale = max(float(torch.max(torch.abs(r_target)).item()), 1e-3)
    # Reconstruct canonical normalized latent trajectory z(t) = r_target(t) / r_scale
    z_traj = (r_target / r_scale).unsqueeze(-1)  # [T, 1]
    dz_dt = estimate_trajectory_derivatives(z_traj, dt=dt).squeeze(-1)  # [T]

    # Fit linear relaxation model: dz/dt = -k_decay * z + sum_j c_j * x_j + c_bias
    ones = torch.ones((t_steps, 1), dtype=torch.float64, device=x.device)
    design = torch.cat([-z_traj, x, ones], dim=-1)  # [T, 1 + D + 1]
    sol = torch.linalg.lstsq(design, dz_dt.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)

    k_decay_est = max(float(sol[0].item()), 0.05)
    state_coeffs = [float(sol[1 + j].item()) for j in range(state_dim)]
    bias_coeff = float(sol[1 + state_dim].item())

    dz_pred = design @ sol
    ss_res = float(torch.sum((dz_dt - dz_pred) ** 2).item())
    ss_tot = float(torch.sum((dz_dt - torch.mean(dz_dt)) ** 2).item())
    latent_ode_r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-18))

    # Test 2nd-order harmonic oscillator hypothesis: d2z/dt2 = -omega^2 * z - k_damp * dz/dt
    d2z_dt2 = estimate_trajectory_derivatives(dz_dt.unsqueeze(-1), dt=dt).squeeze(-1)
    osc_design = torch.stack([-z_traj.squeeze(-1), -dz_dt], dim=-1)
    osc_sol = torch.linalg.lstsq(osc_design, d2z_dt2.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)
    omega_sq_est = float(osc_sol[0].item())
    k_damp_est = max(float(osc_sol[1].item()), 0.0)
    d2z_pred = osc_design @ osc_sol
    osc_ss_res = float(torch.sum((d2z_dt2 - d2z_pred) ** 2).item())
    osc_ss_tot = float(torch.sum((d2z_dt2 - torch.mean(d2z_dt2)) ** 2).item())
    osc_r2 = max(0.0, 1.0 - osc_ss_res / max(osc_ss_tot, 1e-18))

    sp_target = base_brick.state_specs[target_state_index]
    flow_unit = sp_target.unit / base_brick.time_unit
    rate_unit = DIMENSIONLESS / base_brick.time_unit

    p_base_row = base_params[0] if base_params.ndim == 2 else base_params
    fixed_bounds: dict[str, tuple[float, float]] = {}
    for idx_p, bp in enumerate(base_brick.param_specs):
        val = float(p_base_row[idx_p].item())
        fixed_bounds[bp.name] = (val, val)

    # Check if the hidden state is a 2D conjugate harmonic oscillator (sign changes in z + high osc_r2)
    z_sign_changes = int(torch.sum((z_traj[:-1, 0] * z_traj[1:, 0]) < 0.0).item())
    if omega_sq_est > 0.05 and z_sign_changes >= 2 and osc_r2 > max(latent_ode_r2 + 0.10, 0.85):
        omega_est = math.sqrt(omega_sq_est)
        zp_traj = dz_dt / max(omega_est, 1e-6)
        zq_spec = StateVarSpec(
            name=f"{latent_name}_q",
            unit=DIMENSIONLESS,
            default_value=float(z_traj[0, 0].item()),
            homeostatic_min=-50.0,
            homeostatic_max=50.0,
            hard_min=-500.0,
            hard_max=500.0,
        )
        zp_spec = StateVarSpec(
            name=f"{latent_name}_p",
            unit=DIMENSIONLESS,
            default_value=float(zp_traj[0].item()),
            homeostatic_min=-50.0,
            homeostatic_max=50.0,
            hard_min=-500.0,
            hard_max=500.0,
        )
        p_couple_osc = ParamSpec(
            name="k_latent_couple",
            unit=flow_unit,
            default_value=r_scale,
            min_value=-r_scale * 10.0 - 10.0,
            max_value=r_scale * 10.0 + 10.0,
        )
        p_omega_osc = ParamSpec(
            name="omega_latent",
            unit=rate_unit,
            default_value=omega_est,
            min_value=max(omega_est * 0.1, 1e-3),
            max_value=max(omega_est * 10.0, 50.0),
        )
        p_damp_osc = ParamSpec(
            name="k_latent_damp",
            unit=rate_unit,
            default_value=k_damp_est,
            min_value=0.0,
            max_value=max(k_damp_est * 10.0 + 5.0, 20.0),
        )
        osc_brick = LatentOscillatorAugmentedLawBrick(
            name=f"{base_brick.name}_latent_osc_{latent_name}",
            base_brick=base_brick,
            target_state_index=target_state_index,
            zq_spec=zq_spec,
            zp_spec=zp_spec,
            latent_param_specs=(p_couple_osc, p_omega_osc, p_damp_osc),
        )
        init_osc = torch.cat([x[0:1], z_traj[0:1], zp_traj[0:1].unsqueeze(-1)], dim=-1)
        funnel_osc = run_elimination_funnel(
            brick=osc_brick,
            observed_states=x,
            dt=dt,
            observed_state_indices=list(range(state_dim)),
            initial_full_state=init_osc,
            custom_bounds=fixed_bounds,
            num_initial_samples=48,
            num_collocation_survivors=12,
            num_gradient_starts=4,
            gradient_steps=45,
            learning_rate=0.03,
        )
        return LatentAugmentationResult(
            augmented_brick=osc_brick,
            calibrated_params=funnel_osc.best_params,
            reconstructed_initial_state=init_osc,
            trajectory_r2=funnel_osc.trajectory_r2,
            latent_ode_r2=osc_r2,
            audit_certificate=funnel_osc.audit_certificate,
            latent_mode_type="oscillatory_2d",
        )

    z_spec = StateVarSpec(
        name=latent_name,
        unit=DIMENSIONLESS,
        default_value=float(z_traj[0, 0].item()),
        homeostatic_min=-50.0,
        homeostatic_max=50.0,
        hard_min=-500.0,
        hard_max=500.0,
    )

    p_couple = ParamSpec(
        name="k_latent_couple",
        unit=flow_unit,
        default_value=r_scale,
        min_value=-r_scale * 10.0 - 10.0,
        max_value=r_scale * 10.0 + 10.0,
    )
    p_decay = ParamSpec(
        name="k_latent_decay",
        unit=rate_unit,
        default_value=k_decay_est,
        min_value=0.0,
        max_value=max(k_decay_est * 20.0, 50.0),
    )

    latent_params: list[ParamSpec] = [p_couple, p_decay]
    drive_terms: list[ExprNode] = []

    # Include state-driven terms with non-trivial magnitude
    best_j = int(torch.argmax(torch.tensor([abs(c) for c in state_coeffs])).item())
    for j, c_val in enumerate(state_coeffs):
        if abs(c_val) > 1e-3 or j == best_j:
            sp_j = base_brick.state_specs[j]
            p_drv = ParamSpec(
                name=f"c_latent_{sp_j.name}",
                unit=rate_unit / sp_j.unit,
                default_value=c_val,
                min_value=-abs(c_val) * 15.0 - 10.0,
                max_value=abs(c_val) * 15.0 + 10.0,
            )
            latent_params.append(p_drv)
            drive_terms.append(
                BinaryOpNode(
                    "*",
                    SymbolNode(p_drv.name, p_drv.unit, role="param"),
                    SymbolNode(sp_j.name, sp_j.unit, role="state"),
                )
            )

    if abs(bias_coeff) > 1e-3 or not drive_terms:
        p_bias = ParamSpec(
            name="c_latent_bias",
            unit=rate_unit,
            default_value=bias_coeff,
            min_value=-abs(bias_coeff) * 15.0 - 10.0,
            max_value=abs(bias_coeff) * 15.0 + 10.0,
        )
        latent_params.append(p_bias)
        drive_terms.append(SymbolNode(p_bias.name, p_bias.unit, role="param"))

    drive_expr = drive_terms[0]
    for nxt in drive_terms[1:]:
        drive_expr = BinaryOpNode("+", drive_expr, nxt)

    coupling_expr = BinaryOpNode(
        "*",
        SymbolNode(p_couple.name, p_couple.unit, role="param"),
        SymbolNode(latent_name, DIMENSIONLESS, role="state"),
    )

    aug_brick = LatentAugmentedLawBrick(
        name=f"{base_brick.name}_latent_{latent_name}",
        base_brick=base_brick,
        target_state_index=target_state_index,
        latent_state_spec=z_spec,
        latent_param_specs=latent_params,
        coupling_expr=coupling_expr,
        latent_drive_expr=drive_expr,
        decay_param_name=p_decay.name,
    )

    # Freeze base parameters to their already-calibrated values and polish latent parameters + z(0)
    p_base_row = base_params[0] if base_params.ndim == 2 else base_params
    fixed_bounds: dict[str, tuple[float, float]] = {}
    for idx_p, bp in enumerate(base_brick.param_specs):
        val = float(p_base_row[idx_p].item())
        fixed_bounds[bp.name] = (val, val)

    init_full = torch.cat([x[0:1], z_traj[0:1]], dim=-1)
    funnel = run_elimination_funnel(
        brick=aug_brick,
        observed_states=x,
        dt=dt,
        observed_state_indices=list(range(state_dim)),
        initial_full_state=init_full,
        custom_bounds=fixed_bounds,
        num_initial_samples=48,
        num_collocation_survivors=12,
        num_gradient_starts=4,
        gradient_steps=45,
        learning_rate=0.03,
    )

    return LatentAugmentationResult(
        augmented_brick=aug_brick,
        calibrated_params=funnel.best_params,
        reconstructed_initial_state=init_full,
        trajectory_r2=funnel.trajectory_r2,
        latent_ode_r2=latent_ode_r2,
        audit_certificate=funnel.audit_certificate,
    )


def compile_symbolic_candidate(
    base_brick: LawBrick,
    candidate: SymbolicCandidate,
    name_prefix: str = "discovered",
) -> LawBrick:
    """
    Compile a `SymbolicCandidate` onto `base_brick` as either a `HamiltonianPatchedLawBrick`
    (when `thermodynamic_slot == "hamiltonian_potential"`) or a `ResidualPatchedLawBrick`.
    """
    brick_name = f"{base_brick.name}_{name_prefix}_{candidate.candidate_id}"
    if candidate.thermodynamic_slot == "hamiltonian_potential":
        h_expr = candidate.patch_expressions["hamiltonian"]
        return HamiltonianPatchedLawBrick(
            name=brick_name,
            base_brick=base_brick,
            patch_param_specs=candidate.param_specs,
            delta_hamiltonian_expr=h_expr,
        )

    patch_exprs_typed: dict[int | tuple[int, int], ExprNode] = {}
    for k, v in candidate.patch_expressions.items():
        assert isinstance(k, (int, tuple))
        patch_exprs_typed[k] = v

    return compile_residual_patch_brick(
        name=brick_name,
        base_brick=base_brick,
        patch_param_specs=candidate.param_specs,
        thermodynamic_mode=candidate.thermodynamic_slot,
        patch_expressions=patch_exprs_typed,
    )


def _infer_ast_polynomial_degree(node: ExprNode, state_names: set[str]) -> int:
    """Recursively compute the polynomial degree of `node` with respect to `state_names`."""
    if isinstance(node, ConstNode):
        return 0
    if isinstance(node, SymbolNode):
        return 1 if node.name in state_names else 0
    if isinstance(node, UnaryOpNode):
        return _infer_ast_polynomial_degree(node.operand, state_names)
    if isinstance(node, BinaryOpNode):
        dl = _infer_ast_polynomial_degree(node.left, state_names)
        dr = _infer_ast_polynomial_degree(node.right, state_names)
        if node.op in ("+", "-"):
            return max(dl, dr)
        if node.op == "*":
            return dl + dr
        if node.op == "**" and isinstance(node.right, ConstNode):
            return int(round(dl * abs(node.right.value)))
        return max(dl, dr)
    if isinstance(node, PrimitiveCallNode):
        return max((_infer_ast_polynomial_degree(a, state_names) for a in node.args), default=0)
    return 1


def _ast_has_rational_or_saturation(node: ExprNode, state_names: set[str]) -> bool:
    """Return True if `node` contains a saturating primitive or state-dependent denominator."""
    if isinstance(node, PrimitiveCallNode):
        if node.primitive_name in ("hill", "hill_inhib", "michaelis_menten", "boltzmann", "saturating_drive"):
            return True
        return any(_ast_has_rational_or_saturation(a, state_names) for a in node.args)
    if isinstance(node, UnaryOpNode):
        if node.op in ("sin", "cos", "tanh"):
            return True
        return _ast_has_rational_or_saturation(node.operand, state_names)
    if isinstance(node, BinaryOpNode):
        if node.op == "/" and bool(node.right.free_symbols() & state_names):
            return True
        return _ast_has_rational_or_saturation(node.left, state_names) or _ast_has_rational_or_saturation(
            node.right, state_names
        )
    return False


def _compute_mean_phase_space_divergence(brick: LawBrick) -> tuple[float, str]:
    """
    Compute the mean phase-space divergence `<Tr(dF/dx)>` across probe states via exact autograd
    to classify Liouville volume preservation vs. dissipative contraction vs. active expansion.
    """
    state_dim = brick.state_dim
    probes = brick.clamp_to_hard_bounds(
        torch.stack(
            [
                brick.default_state(batch_size=1, dtype=torch.float64)[0] + offset
                for offset in (-0.45, 0.15, 0.65)
            ],
            dim=0,
        )
    )
    p_batch = brick.default_params(batch_size=3, dtype=torch.float64)
    traces: list[float] = []
    for b in range(3):
        xb = probes[b : b + 1].detach().requires_grad_(True)
        pb = p_batch[b : b + 1]
        fb = brick.flow(xb, pb)
        tr_val = 0.0
        for d in range(state_dim):
            grad_d = torch.autograd.grad(
                fb[0, d], xb, retain_graph=(d < state_dim - 1), create_graph=False
            )[0]
            tr_val += float(grad_d[0, d].item())
        traces.append(tr_val)
    mean_div = sum(traces) / len(traces)
    if abs(mean_div) <= 1e-5:
        behavior = "volume_preserving_liouville"
    elif mean_div < -1e-5:
        behavior = "volume_contracting_attractor"
    else:
        behavior = "volume_expanding_instability"
    return mean_div, behavior


def classify_mathematical_archetype(
    brick: LawBrick,
    candidate: SymbolicCandidate | None = None,
    is_latent_augmented: bool = False,
) -> ArchetypeFingerprint:
    """
    Classify the universal mathematical operator archetype of a discovered symbolic law,
    compute its phase-space divergence `<Tr(dF/dx)>`, AST polynomial degree, coupling topology,
    and dimensionless Buckingham-Pi invariants.
    """
    variables: list[tuple[str, object]] = []
    seen_names: set[str] = set()
    state_names: set[str] = set()
    for sp in brick.state_specs:
        variables.append((sp.name, sp.unit))
        seen_names.add(sp.name)
        state_names.add(sp.name)
    for pp in brick.param_specs:
        if pp.name not in seen_names:
            variables.append((pp.name, pp.unit))
            seen_names.add(pp.name)
    variables.append(("t_char", brick.time_unit))

    pi_groups = tuple(discover_dimensionless_groups(variables))  # type: ignore[arg-type]
    mean_div, vol_behavior = _compute_mean_phase_space_divergence(brick)

    if is_latent_augmented:
        return ArchetypeFingerprint(
            archetype_name="non_markovian_latent_adaptation_loop",
            thermodynamic_slot="latent_state_augmentation",
            parity_symmetry="asymmetric",
            is_saturating=False,
            dimensionless_groups=pi_groups,
            cross_domain_analogues=(
                "Photoreceptor / GPCR intracellular calcium adaptation loop",
                "Maxwell-Wiechert viscoelastic polymer stress relaxation",
                "Integral-feedback control & thermal reservoir coupling",
            ),
            mean_phase_space_divergence=mean_div,
            volume_behavior=vol_behavior,
            max_polynomial_degree=1,
            coupled_state_pairs=((brick.state_specs[0].name, brick.state_specs[-1].name),),
        )

    if candidate is None:
        return ArchetypeFingerprint(
            archetype_name="canonical_baseline_law",
            thermodynamic_slot="baseline",
            parity_symmetry="odd",
            is_saturating=False,
            dimensionless_groups=pi_groups,
            cross_domain_analogues=(
                "Harmonic mechanical oscillator",
                "Linear LC/RC electrodynamic resonator",
            ),
            mean_phase_space_divergence=mean_div,
            volume_behavior=vol_behavior,
            max_polynomial_degree=2,
            coupled_state_pairs=(),
        )

    infix_all = " + ".join(node.to_infix() for node in candidate.patch_expressions.values())
    slot = candidate.thermodynamic_slot

    max_deg = max(
        (_infer_ast_polynomial_degree(node, state_names) for node in candidate.patch_expressions.values()),
        default=1,
    )
    is_saturating = any(
        _ast_has_rational_or_saturation(node, state_names) for node in candidate.patch_expressions.values()
    )

    # Extract coupled state variable pairs from AST symbol co-occurrence and target indices
    coupled_pairs_set: set[tuple[str, str]] = set()
    for key, node in candidate.patch_expressions.items():
        syms = sorted(node.free_symbols() & state_names)
        for i_s in range(len(syms)):
            for j_s in range(i_s + 1, len(syms)):
                coupled_pairs_set.add((syms[i_s], syms[j_s]))
        if isinstance(key, int) and 0 <= key < brick.state_dim:
            target_name = brick.state_specs[key].name
            for s_name in syms:
                if s_name != target_name:
                    pair = (min(target_name, s_name), max(target_name, s_name))
                    coupled_pairs_set.add(pair)

    is_hill_or_mm = (
        ("hill(" in infix_all)
        or ("hill_inhib(" in infix_all)
        or ("michaelis_menten(" in infix_all)
        or (is_saturating and "/" in infix_all)
    )
    is_trig = ("sin(" in infix_all) or ("cos(" in infix_all)

    # Probe numerical parity under state inversion x -> -x
    x_probe = torch.full((1, brick.state_dim), 0.73, dtype=torch.float64)
    p_probe = brick.default_params(batch_size=1, dtype=torch.float64)
    if isinstance(brick, ResidualPatchedLawBrick) and slot == "passive_dissipative":
        r_pos = brick.dissipation_matrix(x_probe, p_probe) - brick.base_brick.dissipation_matrix(
            x_probe, p_probe[:, : brick.base_param_dim]
        )
        r_neg = brick.dissipation_matrix(-x_probe, p_probe) - brick.base_brick.dissipation_matrix(
            -x_probe, p_probe[:, : brick.base_param_dim]
        )
        diff_even = float(torch.max(torch.abs(r_pos - r_neg)).item())
        parity = "even" if diff_even < 1e-7 else "asymmetric"
    else:
        f_pos = brick.flow(x_probe, p_probe)
        f_neg = brick.flow(-x_probe, p_probe)
        if float(torch.max(torch.abs(f_pos + f_neg)).item()) < 1e-6:
            parity = "odd"
        elif float(torch.max(torch.abs(f_pos - f_neg)).item()) < 1e-6:
            parity = "even"
        else:
            parity = "asymmetric"

    if is_hill_or_mm:
        archetype = "cooperative_hill_michaelis_saturation"
        analogues = (
            "Michaelis-Menten / Monod-Wyman-Changeux enzyme & receptor kinetics",
            "Langmuir monolayer surface adsorption isotherm",
            "Saturating transconductance differential amplifier stage",
        )
    elif is_trig:
        archetype = "trigonometric_phase_restoring"
        analogues = (
            "Nonlinear gravitational pendulum restoring torque",
            "Josephson junction superconducting phase-current relation",
            "Kuramoto coupled phase oscillator synchronization",
        )
    elif slot == "hamiltonian_potential":
        archetype = "conservative_anharmonic_potential"
        analogues = (
            "Duffing anharmonic elastic beam / Landau quartic free-energy potential",
            "Optical Kerr nonlinear refractive index dispersion",
            "Anharmonic interatomic bond potential",
        )
    elif slot == "passive_dissipative":
        archetype = "nonlinear_state_dependent_dissipation"
        analogues = (
            "Rayleigh / Van der Pol velocity-squared nonlinear damping",
            "Turbulent aerodynamic & fluid drag",
            "Voltage-dependent nonlinear membrane chord conductance",
        )
    elif slot == "conservative_coupling":
        archetype = "conservative_gyroscopic_poisson_coupling"
        analogues = (
            "Lorentz magnetic / Coriolis gyroscopic cross-force",
            "Ideal lossless transformer / gyrator network coupling",
            "Lotka-Volterra conservative predator-prey exchange",
        )
    else:
        archetype = "nonlinear_polynomial_state_feedback"
        analogues = (
            "Autocatalytic reaction rate feedback",
            "FitzHugh-Nagumo / bistable excitable membrane current",
            "Nonlinear mechanical restoring & forcing field",
        )

    return ArchetypeFingerprint(
        archetype_name=archetype,
        thermodynamic_slot=slot,
        parity_symmetry=parity,
        is_saturating=is_saturating,
        dimensionless_groups=pi_groups,
        cross_domain_analogues=analogues,
        mean_phase_space_divergence=mean_div,
        volume_behavior=vol_behavior,
        max_polynomial_degree=max_deg,
        coupled_state_pairs=tuple(sorted(coupled_pairs_set)),
    )


def discover_and_compile_law(
    base_brick: LawBrick,
    observed_states: torch.Tensor,
    dt: float,
    observed_dx_dt: torch.Tensor | None = None,
    calibration_report: CalibrationAndResidualReport | None = None,
    noise_floor_std: float = 1e-3,
    max_polynomial_degree: int = 4,
    max_active_terms: int = 2,
    top_k_to_compile: int = 3,
    include_rational_primitives: bool = True,
) -> ClosedLoopDiscoveryReport:
    """
    Execute the complete closed-loop scientific discovery workflow:
        1. Calibrate `base_brick` parameters and run Layer 1's `UnmodeledResidualGate`
           (if `calibration_report` is not already supplied).
        2. If `not calibration_report.latent_mechanism_required`, return `base_brick` immediately.
        3. Check for non-Markovian hidden-state hysteresis via `detect_non_markovian_memory`;
           if detected, synthesize a `(D + 1)`-state `LatentAugmentedLawBrick`.
        4. Construct dimensionally verified `SymbolicBasisLibrary` instances across the
           thermodynamic slots indicated by `residual_analysis.thermodynamic_class`, plus
           Sanathanan-Koerner rational Padé candidates.
        5. Solve the sparse Pareto front (`solve_sparse_pareto_front`) using dual pointwise
           and weak-integral collocation.
        6. Compile the top Pareto candidates into executable `LawBrick`s, audit them with
           `audit_law_brick`, and rank/polish them via Layer 4's `rank_hypothesis_portfolio`.
        7. Classify the winning law's universal mathematical archetype (`classify_mathematical_archetype`).
    """
    x_in = observed_states.detach().to(torch.float64)
    if x_in.ndim == 3:
        num_exps, t_steps, state_dim = x_in.shape
        x_flat = x_in.reshape(num_exps * t_steps, state_dim)
        x_first = x_in[0]
    else:
        num_exps = 1
        t_steps, state_dim = x_in.shape
        x_flat = x_in
        x_first = x_in

    if observed_dx_dt is not None:
        dx_in = observed_dx_dt.detach().to(torch.float64)
        dx_flat = dx_in.reshape(num_exps * t_steps, state_dim) if dx_in.ndim == 3 else dx_in
        dx_first = dx_in[0] if dx_in.ndim == 3 else dx_in
    else:
        if x_in.ndim == 3:
            dx_list = [estimate_trajectory_derivatives(x_in[e], dt=dt) for e in range(num_exps)]
            dx_in = torch.stack(dx_list, dim=0)
            dx_flat = dx_in.reshape(num_exps * t_steps, state_dim)
            dx_first = dx_in[0]
        else:
            dx_in = estimate_trajectory_derivatives(x_in, dt=dt)
            dx_flat = dx_in
            dx_first = dx_in

    if calibration_report is None:
        calibration_report = calibrate_and_audit_residual(
            brick=base_brick,
            observed_states=x_flat,
            dt=dt,
            observed_dx_dt=dx_flat,
            noise_floor_std=noise_floor_std,
        )

    p_base = calibration_report.best_params.detach().to(torch.float64)
    res_analysis = calibration_report.residual_analysis
    residual_flow = res_analysis.residual_flow.detach().to(torch.float64)
    x = x_flat
    dx = dx_flat

    memory_report = detect_non_markovian_memory(x_first, residual_flow[:t_steps], dt=dt)
    needs_discovery = bool(
        calibration_report.latent_mechanism_required
        or (res_analysis.max_snr >= 3.0 and memory_report.is_non_markovian)
    )

    if not needs_discovery:
        cert = audit_law_brick(base_brick)
        arch = classify_mathematical_archetype(base_brick, candidate=None, is_latent_augmented=False)
        return ClosedLoopDiscoveryReport(
            discovery_triggered=False,
            winning_brick=base_brick,
            winning_params=p_base,
            winning_candidate=None,
            pareto_report=None,
            portfolio_rankings=(),
            memory_report=memory_report,
            latent_augmentation=None,
            archetype=arch,
            audit_certificate=cert,
        )

    # Check if non-Markovian memory requires a latent state variable z(t)
    if memory_report.is_non_markovian:
        latent_res = discover_latent_state_augmentation(
            base_brick=base_brick,
            observed_states=x,
            base_params=p_base,
            residual_flow=residual_flow,
            dt=dt,
            target_state_index=memory_report.primary_affected_state,
        )
        arch = classify_mathematical_archetype(
            latent_res.augmented_brick, candidate=None, is_latent_augmented=True
        )
        return ClosedLoopDiscoveryReport(
            discovery_triggered=True,
            winning_brick=latent_res.augmented_brick,
            winning_params=latent_res.calibrated_params,
            winning_candidate=None,
            pareto_report=None,
            portfolio_rankings=(),
            memory_report=memory_report,
            latent_augmentation=latent_res,
            archetype=arch,
            audit_certificate=latent_res.audit_certificate,
        )

    # Determine thermodynamic slots to search based on Layer 1 residual signature
    thermo_class = res_analysis.thermodynamic_class
    affected = list(res_analysis.affected_state_indices)
    if thermo_class == "passive_dissipative":
        slots_to_search = ["passive_dissipative", "active_driven"]
    elif thermo_class == "reactive_storage":
        slots_to_search = ["hamiltonian_potential", "passive_dissipative", "active_driven"]
    elif thermo_class == "conservative_coupling":
        slots_to_search = ["conservative_coupling", "hamiltonian_potential", "active_driven"]
    else:
        slots_to_search = ["passive_dissipative", "hamiltonian_potential", "active_driven"]

    all_pareto_candidates: list[SymbolicCandidate] = []
    primary_pareto_report: ParetoDiscoveryReport | None = None

    for slot in slots_to_search:
        if slot == "hamiltonian_potential" and base_brick.is_matrix_free:
            continue
        if slot == "conservative_coupling" and base_brick.state_dim < 2:
            continue

        lib = generate_symbolic_basis_library(
            brick=base_brick,
            states=x,
            base_params=p_base,
            thermodynamic_slot=slot,
            affected_state_indices=affected,
            residual_flow=residual_flow,
            max_polynomial_degree=max_polynomial_degree,
            include_rational_primitives=include_rational_primitives,
        )
        extra_cands: list[SymbolicCandidate] = []
        if slot == "active_driven" and include_rational_primitives:
            extra_cands = discover_rational_pade_candidates(
                brick=base_brick,
                states=x,
                target_residual=residual_flow,
                affected_state_indices=affected,
            )

        if len(lib.terms) == 0 and not extra_cands:
            continue

        rep = solve_sparse_pareto_front(
            library=lib,
            target_residual=residual_flow,
            dt=dt,
            max_active_terms=max_active_terms,
            extra_candidates=extra_cands,
        )
        if primary_pareto_report is None:
            primary_pareto_report = rep
        all_pareto_candidates.extend(rep.pareto_front)

    # Select the best candidate from each searched slot first, followed by remaining Pareto candidates
    best_per_slot: list[SymbolicCandidate] = []
    seen_cand_ids: set[str] = set()
    for slot in slots_to_search:
        slot_cands = [c for c in all_pareto_candidates if c.thermodynamic_slot == slot]
        if slot_cands:
            slot_cands.sort(key=lambda c: c.bic_score)
            best_s = slot_cands[0]
            best_per_slot.append(best_s)
            seen_cand_ids.add(best_s.candidate_id)

    slot_priority = {
        "passive_dissipative": 0,
        "hamiltonian_potential": 1,
        "conservative_coupling": 2,
        "active_driven": 3,
    }
    remaining = [c for c in all_pareto_candidates if c.candidate_id not in seen_cand_ids]
    remaining.sort(key=lambda c: (c.bic_score, slot_priority.get(c.thermodynamic_slot, 9)))
    ordered_candidates = best_per_slot + remaining

    # Compile top candidates and run closed-loop trajectory + Levenberg-Marquardt ranking in Layer 4
    compiled_bricks: list[LawBrick] = []
    cand_by_brick_name: dict[str, SymbolicCandidate] = {}
    max_to_compile = max(top_k_to_compile, len(best_per_slot), 1)
    for cand in ordered_candidates[:max_to_compile]:
        try:
            c_brick = compile_symbolic_candidate(base_brick, cand)
            cert = audit_law_brick(c_brick)
            if cert.passed_all:
                compiled_bricks.append(c_brick)
                cand_by_brick_name[c_brick.name] = cand
        except Exception:
            continue

    if not compiled_bricks:
        best_c = ordered_candidates[0]
        c_brick = compile_symbolic_candidate(base_brick, best_c)
        compiled_bricks.append(c_brick)
        cand_by_brick_name[c_brick.name] = best_c

    # Jointly calibrate and Levenberg-Marquardt polish both base parameters and newly discovered
    # patch parameters so any bias introduced when fitting the incomplete base model is removed
    rankings: list[HypothesisRankingEntry] = []

    for c_brick in compiled_bricks:
        funnel = run_elimination_funnel(
            brick=c_brick,
            observed_states=x_in,
            dt=dt,
            observed_dx_dt=dx_in,
            num_initial_samples=48,
            num_collocation_survivors=12,
            num_gradient_starts=3,
            gradient_steps=35,
        )
        n_obs = max(x.shape[0] * x.shape[1], 1)
        mse = max(funnel.vector_field_mse, 1e-16)
        cand_obj = cand_by_brick_name[c_brick.name]
        # Prefer closed-system thermodynamic laws (P_ext == 0) over open-system active_driven patches
        # when both explain the observed vector field to numerical precision
        structure_bonus = (
            -15.0
            if cand_obj.thermodynamic_slot in ("passive_dissipative", "hamiltonian_potential", "conservative_coupling")
            else 0.0
        )
        k_eff = float(cand_obj.num_parameters) + 0.25 * float(cand_obj.total_complexity)
        bic = float(n_obs) * math.log(mse) + k_eff * math.log(max(n_obs, 2)) + structure_bonus

        rankings.append(
            HypothesisRankingEntry(
                brick_name=c_brick.name,
                domain=c_brick.domain,
                funnel_report=funnel,
                tunable_param_count=cand_obj.num_parameters,
                symbolic_complexity=cand_obj.total_complexity,
                bic_score=bic,
                passed_thermodynamic_audit=funnel.audit_certificate.passed_all,
            )
        )

    rankings.sort(key=lambda r: (not r.passed_thermodynamic_audit, r.bic_score))
    winner_entry = rankings[0]
    winning_brick = next(b for b in compiled_bricks if b.name == winner_entry.brick_name)
    winning_cand = cand_by_brick_name[winning_brick.name]
    arch = classify_mathematical_archetype(winning_brick, candidate=winning_cand, is_latent_augmented=False)

    # Evaluate second-half horizon extrapolation R^2 on the primary trajectory
    win_params = winner_entry.funnel_report.best_params
    out_of_sample_r2 = winner_entry.funnel_report.trajectory_r2
    if t_steps >= 8:
        mid_idx = t_steps // 2
        rem_steps = t_steps - 1 - mid_idx
        extrap_roll = rollout_trajectories(
            winning_brick,
            x_first[mid_idx : mid_idx + 1],
            win_params,
            dt=dt,
            num_steps=rem_steps,
            integrator="rk4",
        )
        gt_half = x_first[mid_idx:]
        pred_half = extrap_roll.states[0]
        ss_res = float(torch.sum((gt_half - pred_half) ** 2).item())
        ss_tot = float(torch.sum((gt_half - torch.mean(gt_half, dim=0, keepdim=True)) ** 2).item())
        if ss_tot > 1e-15:
            out_of_sample_r2 = 1.0 - ss_res / ss_tot

    return ClosedLoopDiscoveryReport(
        discovery_triggered=True,
        winning_brick=winning_brick,
        winning_params=win_params,
        winning_candidate=winning_cand,
        pareto_report=primary_pareto_report,
        portfolio_rankings=tuple(rankings),
        memory_report=memory_report,
        latent_augmentation=None,
        archetype=arch,
        audit_certificate=winner_entry.funnel_report.audit_certificate,
        out_of_sample_r2=out_of_sample_r2,
    )


@dataclass(frozen=True)
class AutonomousDiscoveryReport:
    """
    Result of discovering a complete coupled multi-variable differential equation system
    from raw trajectory data (`discover_autonomous_symbolic_system`) with zero prior `base_brick`.

    Attributes:
        compiled_brick: Standalone `CompiledLawBrick` compiled from the discovered symbolic equations.
        calibrated_params: `[1, P]` calibrated and Levenberg-Marquardt polished parameters.
        per_state_expressions: Mapping from state variable name to discovered symbolic `dx_i/dt` string.
        invariant_report: `InvariantDiscoveryReport` of conserved quantities discovered along the trajectory.
        trajectory_r2: Full trajectory reconstruction `R^2`.
        vector_field_r2: Vector-field `R^2` across all state variables.
        archetype: `ArchetypeFingerprint` of the discovered system.
        audit_certificate: Layer 1 `AuditCertificate` for `compiled_brick`.
    """

    compiled_brick: CompiledLawBrick
    calibrated_params: torch.Tensor
    per_state_expressions: dict[str, str]
    invariant_report: InvariantDiscoveryReport
    trajectory_r2: float
    vector_field_r2: float
    archetype: ArchetypeFingerprint
    audit_certificate: AuditCertificate


def discover_autonomous_symbolic_system(
    name: str,
    domain: str,
    state_specs: Sequence[StateVarSpec],
    observed_states: torch.Tensor,
    dt: float,
    observed_dx_dt: torch.Tensor | None = None,
    time_unit: Unit = s,
    energy_unit: Unit = DIMENSIONLESS,
    max_polynomial_degree: int = 3,
    max_terms_per_state: int = 2,
    include_rational_primitives: bool = True,
    include_trigonometric: bool = True,
) -> AutonomousDiscoveryReport:
    """
    Discover an entire coupled multi-variable system of differential equations directly from
    raw trajectory data `observed_states` (`[T, D]` or `[E, T, D]`) without requiring any
    prior baseline model (`base_brick`), compile it into a standalone `CompiledLawBrick`,
    extract its conservation invariants, and polish its parameters via Layer 4.
    """
    x_in = observed_states.detach().to(torch.float64)
    state_dim = len(state_specs)
    if x_in.ndim == 3:
        num_exps, t_steps, _ = x_in.shape
        x_flat = x_in.reshape(num_exps * t_steps, state_dim)
    else:
        num_exps = 1
        t_steps, _ = x_in.shape
        x_flat = x_in

    if observed_dx_dt is not None:
        dx_in = observed_dx_dt.detach().to(torch.float64)
        dx_flat = dx_in.reshape(num_exps * t_steps, state_dim) if dx_in.ndim == 3 else dx_in
    else:
        if x_in.ndim == 3:
            dx_list = [estimate_trajectory_derivatives(x_in[e], dt=dt) for e in range(num_exps)]
            dx_in = torch.stack(dx_list, dim=0)
            dx_flat = dx_in.reshape(num_exps * t_steps, state_dim)
        else:
            dx_in = estimate_trajectory_derivatives(x_in, dt=dt)
            dx_flat = dx_in

    # Construct a zero-flow blank-slate LawBrick to host symbol tables and unit inference
    blank_spec = SymbolicSystemSpec(
        name=f"{name}_blank_carrier",
        domain=domain,
        state_specs=state_specs,
        param_specs=(),
        time_unit=time_unit,
        energy_unit=energy_unit,
        hamiltonian_expr="0.0",
    )
    blank_brick = compile_symbolic_system(blank_spec)
    empty_params = torch.zeros((1, 0), dtype=torch.float64, device=x_flat.device)

    # 1. Discover any exact algebraic/trigonometric/logarithmic conservation invariants
    inv_report = discover_conservation_invariants(
        brick=blank_brick,
        states=x_flat,
        flows=dx_flat,
        max_degree=max_polynomial_degree + 1,
        include_cross_terms=True,
        include_trigonometric=include_trigonometric,
        include_logarithmic=True,
    )

    # 2. Discover the sparse symbolic equation for each state derivative dx_i/dt
    discovered_drive_exprs: list[ExprNode] = []
    all_param_specs: list[ParamSpec] = []
    seen_p_names: set[str] = set()
    per_state_str: dict[str, str] = {}
    last_cand: SymbolicCandidate | None = None

    for idx in range(state_dim):
        r_single = torch.zeros_like(dx_flat)
        r_single[:, idx] = dx_flat[:, idx]

        lib_i = generate_symbolic_basis_library(
            brick=blank_brick,
            states=x_flat,
            base_params=empty_params,
            thermodynamic_slot="active_driven",
            affected_state_indices=[idx],
            residual_flow=r_single,
            max_polynomial_degree=max_polynomial_degree,
            include_rational_primitives=include_rational_primitives,
            include_trigonometric=include_trigonometric,
        )
        extra_i: list[SymbolicCandidate] = []
        if include_rational_primitives:
            extra_i = discover_rational_pade_candidates(
                brick=blank_brick,
                states=x_flat,
                target_residual=r_single,
                affected_state_indices=[idx],
            )
        pareto_i = solve_sparse_pareto_front(
            library=lib_i,
            target_residual=r_single,
            dt=dt,
            max_active_terms=max_terms_per_state,
            extra_candidates=extra_i,
        )
        best_i = pareto_i.best_candidate
        if best_i is not None and idx in best_i.patch_expressions:
            node_i = best_i.patch_expressions[idx]
            discovered_drive_exprs.append(node_i)
            per_state_str[state_specs[idx].name] = node_i.to_infix()
            for ps in best_i.param_specs:
                if ps.name not in seen_p_names:
                    all_param_specs.append(ps)
                    seen_p_names.add(ps.name)
            last_cand = best_i
        else:
            zero_node = ConstNode(0.0, state_specs[idx].unit / time_unit)
            discovered_drive_exprs.append(zero_node)
            per_state_str[state_specs[idx].name] = "0"

    # 3. Compile standalone CompiledLawBrick and polish all parameters via Layer 4
    synthesized_spec = SymbolicSystemSpec(
        name=name,
        domain=domain,
        state_specs=state_specs,
        param_specs=all_param_specs,
        time_unit=time_unit,
        energy_unit=energy_unit,
        hamiltonian_expr="0.0",
        external_drive_exprs=discovered_drive_exprs,
    )
    compiled_brick = compile_symbolic_system(synthesized_spec)

    funnel = run_elimination_funnel(
        brick=compiled_brick,
        observed_states=x_in,
        dt=dt,
        observed_dx_dt=dx_in,
        num_initial_samples=48,
        num_collocation_survivors=12,
        num_gradient_starts=4,
        gradient_steps=40,
    )
    arch = classify_mathematical_archetype(compiled_brick, candidate=last_cand, is_latent_augmented=False)

    return AutonomousDiscoveryReport(
        compiled_brick=compiled_brick,
        calibrated_params=funnel.best_params,
        per_state_expressions=per_state_str,
        invariant_report=inv_report,
        trajectory_r2=funnel.trajectory_r2,
        vector_field_r2=funnel.vector_field_r2,
        archetype=arch,
        audit_certificate=funnel.audit_certificate,
    )
