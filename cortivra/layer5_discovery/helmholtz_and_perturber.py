"""
Non-Markovian hidden-state hysteresis detection, differential residual trajectory
inversion for unseen perturbers, and `(D + 1)` / `(D + 2)` latent state augmentation
(`cortivra.layer5_discovery.helmholtz_and_perturber`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick, ParamSpec, StateVarSpec
from cortivra.layer1_axioms.shields import AuditCertificate
from cortivra.layer1_axioms.units import DIMENSIONLESS
from cortivra.layer3_compiler.ast_nodes import (
    BinaryOpNode,
    ExprNode,
    SymbolNode,
    parse_symbolic_expr,
)
from cortivra.layer3_compiler.symbolic_compiler import build_symbol_table
from cortivra.layer4_engine.elimination_funnel import run_elimination_funnel
from cortivra.layer4_engine.rollout_arena import estimate_trajectory_derivatives
from cortivra.observability.flight_recorder import (
    AlternativeBranchRecord,
    emit_trace_node,
    summarize_singular_spectrum,
)


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
    conjugate harmonic oscillator mode `(z_q(t), z_p(t))` coupled to `target_state_index`:
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
    """

    is_non_markovian: bool
    hysteresis_ratio: float
    static_basis_r2: float
    residual_lag1_autocorr: float
    primary_affected_state: int


@dataclass(frozen=True)
class DifferentialResidualInversionReport:
    """
    Universal Differential Residual Trajectory Inversion report reconstructing the hidden
    state trajectory `z_hat(t)` and unseen perturber dynamics from `Phi(t) = dx_obs/dt - F_known(x)`.
    """

    reconstructed_latent_trajectory: torch.Tensor
    reconstructed_latent_derivative: torch.Tensor
    reconstructed_latent_acceleration: torch.Tensor
    dominant_mode_type: str
    discovered_latent_ode_expr: str
    latent_ode_r2: float
    estimated_3d_line_of_action: torch.Tensor | None = None
    estimated_inverse_square_distance: torch.Tensor | None = None


@dataclass(frozen=True)
class LatentAugmentationResult:
    """
    Result of synthesizing and calibrating a `(D + 1)`-state `LatentAugmentedLawBrick`
    or `(D + 2)`-state `LatentOscillatorAugmentedLawBrick`.
    """

    augmented_brick: LatentAugmentedLawBrick | LatentOscillatorAugmentedLawBrick
    calibrated_params: torch.Tensor
    reconstructed_initial_state: torch.Tensor
    trajectory_r2: float
    latent_ode_r2: float
    audit_certificate: AuditCertificate
    latent_mode_type: str = "relaxation_1d"
    residual_inversion_report: DifferentialResidualInversionReport | None = None


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

    e0 = err[:-1] - torch.mean(err[:-1])
    e1 = err[1:] - torch.mean(err[1:])
    denom = float(torch.sqrt(torch.sum(e0 * e0) * torch.sum(e1 * e1)).item())
    lag1_autocorr = float(torch.sum(e0 * e1).item()) / max(denom, 1e-12) if denom > 1e-12 else 0.0

    is_non_markov = bool(
        static_r2 < 0.78 and lag1_autocorr > 0.70 and hysteresis_ratio > 0.65
    )

    static_spec = summarize_singular_spectrum(
        design,
        operator_name=f"MarkovianStaticCubicBasis(state={primary_idx})",
    )
    emit_trace_node(
        layer="layer5_discovery",
        operation="detect_non_markovian_memory",
        target_name=f"state_{primary_idx}",
        local_r2=static_r2,
        decision_margin=abs(hysteresis_ratio - 0.65),
        selected_branch_id="non_markovian_latent_state" if is_non_markov else "markovian_state_function",
        selected_expression=(
            f"is_non_markovian={is_non_markov}, hysteresis={hysteresis_ratio:.3f}, "
            f"static_r2={static_r2:.4f}, lag1_autocorr={lag1_autocorr:.3f}"
        ),
        operator_spectrum=static_spec,
        metrics={
            "is_non_markovian": is_non_markov,
            "hysteresis_ratio": round(hysteresis_ratio, 4),
            "static_basis_r2": round(static_r2, 6),
            "residual_lag1_autocorr": round(lag1_autocorr, 4),
            "primary_affected_state": primary_idx,
        },
        skip_if_inner_loop=True,
    )

    return NonMarkovianMemoryReport(
        is_non_markovian=is_non_markov,
        hysteresis_ratio=hysteresis_ratio,
        static_basis_r2=static_r2,
        residual_lag1_autocorr=lag1_autocorr,
        primary_affected_state=primary_idx,
    )


def invert_residual_for_unseen_perturber(
    base_brick: LawBrick,
    observed_states: torch.Tensor,
    residual_flow: torch.Tensor,
    dt: float,
    target_state_index: int = 0,
    latent_name: str = "z_latent",
    spatial_residual_indices: tuple[int, int, int] | None = None,
) -> DifferentialResidualInversionReport:
    """
    Universal Differential Residual Trajectory Inversion:
    Directly invert the unexplained vector-field residual `Phi(t) = dx_obs/dt - F_known(x)`
    to reconstruct the hidden variable trajectory `z_hat(t)`, its time derivatives
    `(dz_hat/dt, d2z_hat/dt2)`, its governing differential equation `dz_hat/dt = g(x, z_hat)`,
    and—when 3D spatial force/velocity residuals are present—its 3D line-of-action direction
    `u_hat(t)` and inverse-square distance profile `r_hat(t) ~ ||Phi_3d(t)||^(-1/2)`.
    """
    x = observed_states.detach().to(torch.float64)
    r = residual_flow.detach().to(torch.float64)
    t_steps, state_dim = x.shape

    r_target = r[:, target_state_index]
    r_scale = max(float(torch.max(torch.abs(r_target)).item()), 1e-3)
    z_traj = (r_target / r_scale).unsqueeze(-1)  # [T, 1]
    dz_dt = estimate_trajectory_derivatives(z_traj, dt=dt).squeeze(-1)  # [T]
    d2z_dt2 = estimate_trajectory_derivatives(dz_dt.unsqueeze(-1), dt=dt).squeeze(-1)  # [T]

    ones = torch.ones((t_steps, 1), dtype=torch.float64, device=x.device)
    lin_design = torch.cat([-z_traj, x, ones], dim=-1)
    lin_sol = torch.linalg.lstsq(lin_design, dz_dt.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)
    dz_lin_pred = lin_design @ lin_sol
    ss_res_lin = float(torch.sum((dz_dt - dz_lin_pred) ** 2).item())
    ss_tot_dz = float(torch.sum((dz_dt - torch.mean(dz_dt)) ** 2).item())
    lin_r2 = max(0.0, 1.0 - ss_res_lin / max(ss_tot_dz, 1e-18))

    osc_design = torch.stack([-z_traj.squeeze(-1), -dz_dt], dim=-1)
    osc_sol = torch.linalg.lstsq(osc_design, d2z_dt2.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)
    omega_sq_est = float(osc_sol[0].item())
    k_damp_est = max(float(osc_sol[1].item()), 0.0)
    d2z_pred = osc_design @ osc_sol
    ss_res_osc = float(torch.sum((d2z_dt2 - d2z_pred) ** 2).item())
    ss_tot_d2z = float(torch.sum((d2z_dt2 - torch.mean(d2z_dt2)) ** 2).item())
    osc_r2 = max(0.0, 1.0 - ss_res_osc / max(ss_tot_d2z, 1e-18))

    z_sign_changes = int(torch.sum((z_traj[:-1, 0] * z_traj[1:, 0]) < 0.0).item())
    if (
        omega_sq_est > 0.05
        and z_sign_changes >= 2
        and osc_r2 > 0.85
        and (osc_r2 > lin_r2 + 0.10 or (osc_r2 > 0.95 and osc_r2 >= lin_r2))
    ):
        mode_type = "oscillatory_2d"
        best_r2 = osc_r2
        omega_est = math.sqrt(omega_sq_est)
        ode_str = (
            f"d2({latent_name})/dt2 = -{omega_sq_est:.4f}*{latent_name} "
            f"- {k_damp_est:.4f}*d({latent_name})/dt (omega={omega_est:.4f})"
        )
    else:
        mode_type = "relaxation_1d"
        best_r2 = lin_r2
        k_dec = max(float(lin_sol[0].item()), 0.05)
        terms_str = [f"-{k_dec:.4f}*{latent_name}"]
        for j in range(state_dim):
            c_j = float(lin_sol[1 + j].item())
            if abs(c_j) > 1e-3:
                terms_str.append(f"{c_j:+.4f}*{base_brick.state_specs[j].name}")
        b_c = float(lin_sol[1 + state_dim].item())
        if abs(b_c) > 1e-3:
            terms_str.append(f"{b_c:+.4f}")
        ode_str = f"d({latent_name})/dt = " + " ".join(terms_str)

    line_of_action: torch.Tensor | None = None
    inv_sq_dist: torch.Tensor | None = None
    indices_3d = spatial_residual_indices
    if indices_3d is None and len(base_brick.spatial_nodes_3d) >= 1:
        node0 = base_brick.spatial_nodes_3d[0]
        name_to_idx = {sp.name: i for i, sp in enumerate(base_brick.state_specs)}
        mom_idx = (
            (
                name_to_idx[node0.momentum_state_names[0]],
                name_to_idx[node0.momentum_state_names[1]],
                name_to_idx[node0.momentum_state_names[2]],
            )
            if len(node0.momentum_state_names) == 3 and all(nm in name_to_idx for nm in node0.momentum_state_names)
            else None
        )
        pos_idx = (
            (
                name_to_idx[node0.position_state_names[0]],
                name_to_idx[node0.position_state_names[1]],
                name_to_idx[node0.position_state_names[2]],
            )
            if len(node0.position_state_names) == 3 and all(nm in name_to_idx for nm in node0.position_state_names)
            else None
        )
        if mom_idx is not None and float(torch.linalg.norm(r[:, list(mom_idx)]).item()) > 1e-9:
            indices_3d = mom_idx
        elif pos_idx is not None and float(torch.linalg.norm(r[:, list(pos_idx)]).item()) > 1e-9:
            indices_3d = pos_idx
        else:
            indices_3d = mom_idx if mom_idx is not None else pos_idx

    if indices_3d is not None and all(0 <= idx < state_dim for idx in indices_3d):
        r_3d = torch.stack([r[:, indices_3d[0]], r[:, indices_3d[1]], r[:, indices_3d[2]]], dim=-1)
        mag_3d = torch.sqrt(torch.sum(r_3d * r_3d, dim=-1, keepdim=True) + 1e-18)
        line_of_action = r_3d / mag_3d
        inv_sq_dist = 1.0 / torch.sqrt(mag_3d.squeeze(-1) + 1e-12)

    return DifferentialResidualInversionReport(
        reconstructed_latent_trajectory=z_traj,
        reconstructed_latent_derivative=dz_dt.unsqueeze(-1),
        reconstructed_latent_acceleration=d2z_dt2.unsqueeze(-1),
        dominant_mode_type=mode_type,
        discovered_latent_ode_expr=ode_str,
        latent_ode_r2=best_r2,
        estimated_3d_line_of_action=line_of_action,
        estimated_inverse_square_distance=inv_sq_dist,
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
    Synthesize and calibrate a `(D + 1)`-state `LatentAugmentedLawBrick` or `(D + 2)`-state
    `LatentOscillatorAugmentedLawBrick` by first running `invert_residual_for_unseen_perturber`
    on the unexplained residual `r_target(t)` and then compiling + polishing the augmented system.
    """
    inv_rep = invert_residual_for_unseen_perturber(
        base_brick=base_brick,
        observed_states=observed_states,
        residual_flow=residual_flow,
        dt=dt,
        target_state_index=target_state_index,
        latent_name=latent_name,
    )
    x = observed_states.detach().to(torch.float64)
    r = residual_flow.detach().to(torch.float64)
    t_steps, state_dim = x.shape

    r_target = r[:, target_state_index]
    r_scale = max(float(torch.max(torch.abs(r_target)).item()), 1e-3)
    z_traj = inv_rep.reconstructed_latent_trajectory
    dz_dt = inv_rep.reconstructed_latent_derivative.squeeze(-1)

    ones = torch.ones((t_steps, 1), dtype=torch.float64, device=x.device)
    design = torch.cat([-z_traj, x, ones], dim=-1)
    sol = torch.linalg.lstsq(design, dz_dt.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)

    k_decay_est = max(float(sol[0].item()), 0.05)
    state_coeffs = [float(sol[1 + j].item()) for j in range(state_dim)]
    bias_coeff = float(sol[1 + state_dim].item())

    dz_pred = design @ sol
    ss_res = float(torch.sum((dz_dt - dz_pred) ** 2).item())
    ss_tot = float(torch.sum((dz_dt - torch.mean(dz_dt)) ** 2).item())
    latent_ode_r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-18))

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
        emit_trace_node(
            layer="layer5_discovery",
            operation="discover_latent_state_augmentation",
            target_name=osc_brick.name,
            local_r2=osc_r2,
            global_r2=funnel_osc.trajectory_r2,
            decision_margin=abs(osc_r2 - latent_ode_r2),
            selected_branch_id="oscillatory_2d",
            selected_expression=f"oscillatory_2d (omega={omega_est:.4f}, R^2={funnel_osc.trajectory_r2:.5f})",
            alternatives=(
                AlternativeBranchRecord(
                    branch_id="relaxation_1d",
                    expression_or_summary=f"1D relaxation (k_decay={k_decay_est:.4f})",
                    objective_score=round(1.0 - latent_ode_r2, 6),
                    local_r2=round(latent_ode_r2, 6),
                    complexity=2,
                ),
            ),
            metrics={
                "latent_mode_type": "oscillatory_2d",
                "latent_ode_r2": round(osc_r2, 6),
                "trajectory_r2": round(funnel_osc.trajectory_r2, 6),
            },
            skip_if_inner_loop=True,
        )
        return LatentAugmentationResult(
            augmented_brick=osc_brick,
            calibrated_params=funnel_osc.best_params,
            reconstructed_initial_state=init_osc,
            trajectory_r2=funnel_osc.trajectory_r2,
            latent_ode_r2=osc_r2,
            audit_certificate=funnel_osc.audit_certificate,
            latent_mode_type="oscillatory_2d",
            residual_inversion_report=inv_rep,
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
    emit_trace_node(
        layer="layer5_discovery",
        operation="discover_latent_state_augmentation",
        target_name=aug_brick.name,
        local_r2=latent_ode_r2,
        global_r2=funnel.trajectory_r2,
        decision_margin=abs(latent_ode_r2 - osc_r2),
        selected_branch_id="relaxation_1d",
        selected_expression=f"relaxation_1d (dz/dt = -{k_decay_est:.3f}*{latent_name} + {drive_expr.to_infix()})",
        alternatives=(
            AlternativeBranchRecord(
                branch_id="oscillatory_2d",
                expression_or_summary=f"2D harmonic oscillator (omega_sq={omega_sq_est:.4f})",
                objective_score=round(1.0 - osc_r2, 6),
                local_r2=round(osc_r2, 6),
                complexity=3,
            ),
        ),
        metrics={
            "latent_mode_type": "relaxation_1d",
            "latent_ode_r2": round(latent_ode_r2, 6),
            "trajectory_r2": round(funnel.trajectory_r2, 6),
        },
        skip_if_inner_loop=True,
    )

    return LatentAugmentationResult(
        augmented_brick=aug_brick,
        calibrated_params=funnel.best_params,
        reconstructed_initial_state=init_full,
        trajectory_r2=funnel.trajectory_r2,
        latent_ode_r2=latent_ode_r2,
        audit_certificate=funnel.audit_certificate,
        residual_inversion_report=inv_rep,
    )
