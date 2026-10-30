"""
Trajectory integration, time-reversal retrodiction, and equilibrium stability solvers
(`cortivra.layer4_engine.rollout_arena`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick
from cortivra.layer1_axioms.shields import compute_viable_universe_mask
from cortivra.observability.flight_recorder import (
    OperatorSpectrumSummary,
    emit_trace_node,
    record_simulation_activity,
    summarize_singular_spectrum,
    suppress_inner_trace,
)


@dataclass(frozen=True)
class RolloutResult:
    """Batched trajectory state history, flows, energies, and viability masks."""

    states: torch.Tensor
    flows: torch.Tensor
    energies: torch.Tensor
    step_viable_mask: torch.Tensor
    all_time_viable_mask: torch.Tensor
    dt: float
    integrator: str

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.states.shape[1] - 1)

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[2])

    @property
    def viable_fraction(self) -> float:
        if self.all_time_viable_mask.numel() == 0:
            return 0.0
        return float(self.all_time_viable_mask.to(torch.float64).mean().item())

    @property
    def max_energy_drift(self) -> float:
        """Maximum absolute change `|H(t) - H(0)|` across the batch and trajectory."""
        if self.energies.numel() == 0:
            return 0.0
        drift = torch.abs(self.energies - self.energies[:, :1])
        return float(torch.max(drift).item())


@dataclass(frozen=True)
class EquilibriumAnalysis:
    """Steady-state (`dx/dt = 0`) solution and local Jacobian eigenvalue stability report."""

    equilibrium_state: torch.Tensor
    residual_norm: torch.Tensor
    converged_mask: torch.Tensor
    jacobian: torch.Tensor
    eigenvalues: torch.Tensor
    max_real_eigenvalue: torch.Tensor
    max_imag_eigenvalue: torch.Tensor
    num_conserved_modes: torch.Tensor
    stability_classes: tuple[str, ...]


def select_optimal_integrator(
    brick: LawBrick,
    sample_state: torch.Tensor,
    sample_params: torch.Tensor,
    dt: float,
    port_inputs: dict[str, torch.Tensor] | None = None,
) -> str:
    """
    Automatically select the best structure-preserving or adaptive numerical integrator for `brick`:
        1. If `brick` implements `step_exponential_euler` (stiff Rush-Larsen gating) and no external
           port inputs are passed, returns `"exponential_euler"`.
        2. If dissipation power is identically zero (`D(x) == 0`) and no external drive is active,
           returns `"symplectic_midpoint"` to preserve Hamiltonian energy invariants.
        3. If local Lipschitz stiffness `||f(x + eps) - f(x)|| / eps * dt > 0.4`, returns `"adaptive_rk"`.
        4. Otherwise returns `"rk4"`.
    """
    if hasattr(brick, "step_exponential_euler") and not port_inputs:
        return "exponential_euler"

    s0 = sample_state[:1].detach().to(torch.float64)
    p0 = sample_params[:1].detach().to(torch.float64)
    with torch.no_grad():
        s_probe = brick.clamp_to_hard_bounds(s0 + 0.137)
        diss = max(
            float(torch.max(torch.abs(brick.dissipation_power(s0, p0))).item()),
            float(torch.max(torch.abs(brick.dissipation_power(s_probe, p0))).item()),
        )
        has_ext = bool(port_inputs)
        if hasattr(brick, "external_drive_power"):
            has_ext = has_ext or (
                float(torch.max(torch.abs(brick.external_drive_power(s_probe, p0))).item()) > 1e-12
            )
        if hasattr(brick, "external_patch_power"):
            has_ext = has_ext or (
                float(torch.max(torch.abs(brick.external_patch_power(s_probe, p0))).item()) > 1e-12
            )

        if diss <= 1e-12 and not has_ext:
            return "symplectic_midpoint"

        f0 = brick.flow(s0, p0)
        f_probe = brick.flow(s_probe, p0)
        eps = 1e-5
        local_lip = 0.0
        for d_i in range(brick.state_dim):
            for base_s, base_f in ((s0, f0), (s_probe, f_probe)):
                s_pert = base_s.clone()
                s_pert[:, d_i] = s_pert[:, d_i] + eps
                s_pert = brick.clamp_to_hard_bounds(s_pert)
                f1 = brick.flow(s_pert, p0)
                col_lip = float(torch.linalg.norm(f1 - base_f).item()) / max(eps, 1e-12)
                if col_lip > local_lip:
                    local_lip = col_lip
        if local_lip * dt > 8.0:
            return "rosenbrock2"
        if local_lip * dt > 0.4:
            return "adaptive_rk"

    return "rk4"


def _compute_flow_jacobian_batched(
    brick: LawBrick,
    state: torch.Tensor,
    params: torch.Tensor,
    port_inputs: dict[str, torch.Tensor] | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the batched state Jacobian `J_F(x) = dF/dx` of shape `[Batch, StateDim, StateDim]`
    via centered finite differences (preserving outer autograd graph wrt `state` and `params`).
    """
    batch, state_dim = state.shape
    cols: list[torch.Tensor] = []
    for j in range(state_dim):
        e_j = torch.zeros_like(state)
        e_j[:, j] = eps
        f_plus = brick.flow(state + e_j, params, port_inputs=port_inputs)
        f_minus = brick.flow(state - e_j, params, port_inputs=port_inputs)
        cols.append((f_plus - f_minus) / (2.0 * eps))
    return torch.stack(cols, dim=-1)


def _step_rosenbrock2(
    brick: LawBrick,
    state: torch.Tensor,
    params: torch.Tensor,
    dt: float,
    port_inputs: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    2-stage, 2nd-order L-stable Rosenbrock-Wanner (`ROS2`) stiff multi-timescale integrator:
        gamma = 1 - 1 / sqrt(2)
        (I - gamma * dt * J_F(x_n)) k_1 = F(x_n)
        (I - gamma * dt * J_F(x_n)) k_2 = F(x_n + 0.5 * dt * k_1) - gamma * dt * J_F(x_n) k_1
        x_{n+1} = x_n + dt * k_2
    Unconditionally L-stable (`R(infinity) = 0`) for arbitrarily stiff decaying modes (`Re(lambda)*dt -> -inf`)
    and 100% differentiable via `torch.linalg.solve` without nonlinear Newton iteration loops.
    """
    gamma = 1.0 - 0.7071067811865475  # 1 - 1/sqrt(2)
    batch, state_dim = state.shape
    f0 = brick.flow(state, params, port_inputs=port_inputs)
    j_mat = _compute_flow_jacobian_batched(brick, state, params, port_inputs=port_inputs)
    eye = torch.eye(state_dim, dtype=state.dtype, device=state.device).unsqueeze(0).expand(batch, -1, -1)
    w_mat = eye - (gamma * dt) * j_mat

    try:
        k1 = torch.linalg.solve(w_mat, f0.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        w_reg = w_mat + 1e-8 * eye
        k1 = torch.linalg.lstsq(w_reg, f0.unsqueeze(-1)).solution.squeeze(-1)

    x_mid = brick.clamp_to_hard_bounds(state + 0.5 * dt * k1)
    f_mid = brick.flow(x_mid, params, port_inputs=port_inputs)
    jk1 = torch.bmm(j_mat, k1.unsqueeze(-1)).squeeze(-1)
    rhs2 = f_mid - (gamma * dt) * jk1

    try:
        k2 = torch.linalg.solve(w_mat, rhs2.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        w_reg = w_mat + 1e-8 * eye
        k2 = torch.linalg.lstsq(w_reg, rhs2.unsqueeze(-1)).solution.squeeze(-1)

    return brick.clamp_to_hard_bounds(state + dt * k2)


def _step_implicit_midpoint(
    brick: LawBrick,
    state: torch.Tensor,
    params: torch.Tensor,
    dt: float,
    port_inputs: dict[str, torch.Tensor] | None = None,
    max_newton_iters: int = 8,
    tol: float = 1e-9,
) -> torch.Tensor:
    """
    A-stable, symmetric, symplectic 2nd-order Implicit Midpoint integrator:
        x_{n+1} = x_n + dt * F(0.5 * (x_n + x_{n+1}))
    Solved via damped Newton-Raphson with exact Jacobian factorization `(I - 0.5 * dt * J_F)`.
    """
    batch, state_dim = state.shape
    f0 = brick.flow(state, params, port_inputs=port_inputs)
    j_mat = _compute_flow_jacobian_batched(brick, state, params, port_inputs=port_inputs)
    eye = torch.eye(state_dim, dtype=state.dtype, device=state.device).unsqueeze(0).expand(batch, -1, -1)
    w_mat = eye - (0.5 * dt) * j_mat

    try:
        dx_init = torch.linalg.solve(w_mat, (dt * f0).unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        dx_init = dt * f0
    x_next = brick.clamp_to_hard_bounds(state + dx_init)

    for _ in range(max_newton_iters):
        x_mid = 0.5 * (state + x_next)
        f_mid = brick.flow(x_mid, params, port_inputs=port_inputs)
        res = x_next - state - dt * f_mid
        max_r = float(torch.max(torch.abs(res)).detach().item())
        if max_r <= tol:
            break
        try:
            delta = torch.linalg.solve(w_mat, (-res).unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            delta = -res
        x_next = brick.clamp_to_hard_bounds(x_next + delta)

    return x_next


def _step_adaptive_rk4(
    brick: LawBrick,
    state: torch.Tensor,
    params: torch.Tensor,
    dt: float,
    port_inputs: dict[str, torch.Tensor] | None = None,
    adaptive_tol: float = 1e-5,
    max_substeps: int = 16,
) -> torch.Tensor:
    """
    Dyadic error-controlled 4th-order Runge-Kutta with 5th-order Richardson extrapolation.
    Doubles the substep count `n_sub = 1, 2, 4, ..., max_substeps` until the maximum local
    Richardson truncation error across the interval `dt` is `<= adaptive_tol`.
    Automatically falls back to `_step_rosenbrock2` if explicit substeps encounter stiffness.
    """
    n_sub = 1
    limit = max(int(max_substeps), 1)
    best_extrap = state

    while True:
        sub_dt = dt / float(n_sub)
        curr = state
        max_err = 0.0
        stiff_overflow = False
        for _ in range(n_sub):
            x_full = brick.step_rk4(curr, params, dt=sub_dt, port_inputs=port_inputs)
            x_h1 = brick.step_rk4(curr, params, dt=0.5 * sub_dt, port_inputs=port_inputs)
            x_h2 = brick.step_rk4(x_h1, params, dt=0.5 * sub_dt, port_inputs=port_inputs)
            if not (bool(torch.isfinite(x_full).all().item()) and bool(torch.isfinite(x_h2).all().item())):
                stiff_overflow = True
                break
            step_err = float(torch.max(torch.abs(x_h2 - x_full)).detach().item()) / 15.0
            if step_err > max_err:
                max_err = step_err
            curr = brick.clamp_to_hard_bounds(x_h2 + (x_h2 - x_full) / 15.0)

        if stiff_overflow:
            if n_sub >= limit:
                return _step_rosenbrock2(brick, state, params, dt=dt, port_inputs=port_inputs)
            n_sub = min(n_sub * 2, limit)
            continue

        best_extrap = curr
        if max_err <= adaptive_tol:
            break
        if n_sub >= limit:
            if max_err > 100.0 * adaptive_tol:
                return _step_rosenbrock2(brick, state, params, dt=dt, port_inputs=port_inputs)
            break
        n_sub = min(n_sub * 2, limit)

    return best_extrap


def _slice_port_inputs(
    port_inputs: dict[str, torch.Tensor] | None,
    step_idx: int,
    num_steps: int,
) -> dict[str, torch.Tensor] | None:
    if not port_inputs:
        return None
    sliced: dict[str, torch.Tensor] = {}
    for k, v in port_inputs.items():
        if v.ndim == 3 and v.shape[1] >= num_steps:
            idx = min(step_idx, v.shape[1] - 1)
            sliced[k] = v[:, idx, :]
        else:
            sliced[k] = v
    return sliced


def rollout_trajectories(
    brick: LawBrick,
    initial_state: torch.Tensor,
    params: torch.Tensor,
    dt: float,
    num_steps: int,
    integrator: str = "rk4",
    port_inputs: dict[str, torch.Tensor] | None = None,
    adaptive_tol: float = 1e-5,
    record_flows_and_energies: bool = True,
) -> RolloutResult:
    """
    Roll out batched trajectories `[Batch, NumSteps + 1, StateDim]` through `brick`
    across `"auto"`, `"rk4"`, `"adaptive_rk"`, `"symplectic_midpoint"`, `"exponential_euler"`,
    or `"euler"` while tracking energy and homeostatic viability.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive for forward rollout, got {dt} (use retrodict_trajectory for backward).")
    valid_integrators = (
        "auto",
        "rk4",
        "adaptive_rk",
        "rosenbrock2",
        "implicit_midpoint",
        "symplectic_midpoint",
        "exponential_euler",
        "euler",
    )
    if integrator not in valid_integrators:
        raise ValueError(f"Unsupported integrator '{integrator}'. Expected one of {valid_integrators}.")

    resolved_integrator = (
        select_optimal_integrator(brick, initial_state, params, dt=dt, port_inputs=port_inputs)
        if integrator == "auto"
        else integrator
    )

    x = brick.clamp_to_hard_bounds(initial_state)
    states_list: list[torch.Tensor] = [x]
    viable_list: list[torch.Tensor] = [compute_viable_universe_mask(brick, x)]

    flows_list: list[torch.Tensor] = []
    energies_list: list[torch.Tensor] = []
    if record_flows_and_energies:
        p0 = _slice_port_inputs(port_inputs, 0, num_steps)
        flows_list.append(brick.flow(x, params, port_inputs=p0))
        energies_list.append(brick.hamiltonian_energy(x, params))

    for step_idx in range(num_steps):
        p_step = _slice_port_inputs(port_inputs, step_idx, num_steps)
        if resolved_integrator == "rk4":
            x_next = brick.step_rk4(x, params, dt=dt, port_inputs=p_step)
            if integrator == "auto" and not bool(torch.isfinite(x_next).all().item()):
                x_next = _step_adaptive_rk4(brick, x, params, dt=dt, port_inputs=p_step, adaptive_tol=adaptive_tol)
            x = x_next
        elif resolved_integrator == "adaptive_rk":
            x = _step_adaptive_rk4(brick, x, params, dt=dt, port_inputs=p_step, adaptive_tol=adaptive_tol)
        elif resolved_integrator == "rosenbrock2":
            x = _step_rosenbrock2(brick, x, params, dt=dt, port_inputs=p_step)
        elif resolved_integrator == "implicit_midpoint":
            x = _step_implicit_midpoint(brick, x, params, dt=dt, port_inputs=p_step)
        elif resolved_integrator == "symplectic_midpoint":
            x = brick.step_symplectic_midpoint(x, params, dt=dt, port_inputs=p_step)
        elif resolved_integrator == "exponential_euler":
            if hasattr(brick, "step_exponential_euler") and not p_step:
                x = brick.step_exponential_euler(x, params, dt=dt)
            else:
                x = brick.step_rk4(x, params, dt=dt, port_inputs=p_step)
        elif resolved_integrator == "euler":
            dx = brick.flow(x, params, port_inputs=p_step)
            x = brick.clamp_to_hard_bounds(x + dt * dx)

        states_list.append(x)
        viable_list.append(compute_viable_universe_mask(brick, x))
        if record_flows_and_energies:
            p_next = _slice_port_inputs(port_inputs, min(step_idx + 1, num_steps - 1), num_steps)
            flows_list.append(brick.flow(x, params, port_inputs=p_next))
            energies_list.append(brick.hamiltonian_energy(x, params))

    states_t = torch.stack(states_list, dim=1)
    step_viable_t = torch.stack(viable_list, dim=1)
    all_viable_t = step_viable_t.all(dim=1)

    if record_flows_and_energies:
        flows_t = torch.stack(flows_list, dim=1)
        energies_t = torch.stack(energies_list, dim=1)
    else:
        flows_t = torch.zeros_like(states_t)
        energies_t = torch.zeros((states_t.shape[0], states_t.shape[1]), dtype=states_t.dtype, device=states_t.device)

    batch_cnt = int(states_t.shape[0])
    record_simulation_activity(
        ode_rollout_calls=1,
        trajectories_integrated=batch_cnt,
        integration_steps_executed=batch_cnt * int(num_steps),
        integrator=resolved_integrator,
    )
    return RolloutResult(
        states=states_t,
        flows=flows_t,
        energies=energies_t,
        step_viable_mask=step_viable_t,
        all_time_viable_mask=all_viable_t,
        dt=dt,
        integrator=resolved_integrator,
    )


def retrodict_trajectory(
    brick: LawBrick,
    final_state: torch.Tensor,
    params: torch.Tensor,
    dt: float,
    num_steps: int,
    mode: str = "conservative_exact",
    max_dissipative_reverse_rate: float = 5.0,
) -> RolloutResult:
    """
    Integrate backwards in time from `final_state` `x(t_T)` to reconstruct `x(t_0), ..., x(t_T)`.
    Returned `states` tensor is ordered chronologically from `t_0` (index `0`) to `t_T` (`num_steps`).
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}.")
    if dt <= 0.0:
        raise ValueError(f"Pass positive step magnitude dt > 0 to retrodict_trajectory, got {dt}.")
    if mode not in ("conservative_exact", "regularized_full"):
        raise ValueError(f"Unsupported retrodiction mode '{mode}'.")

    def _backward_velocity(state_in: torch.Tensor) -> torch.Tensor:
        f_cons = brick.conservative_flow(state_in, params)
        if mode == "conservative_exact":
            return -f_cons
        f_diss = brick.dissipative_flow(state_in, params)
        cap = max(max_dissipative_reverse_rate, 1e-6)
        f_diss_reg = cap * torch.tanh(f_diss / cap)
        return -f_cons + f_diss_reg

    x = brick.clamp_to_hard_bounds(final_state)
    rev_states: list[torch.Tensor] = [x]

    for _ in range(num_steps):
        x_next = x + dt * _backward_velocity(x)
        for _ in range(8):
            x_mid = 0.5 * (x + x_next)
            x_cand = x + dt * _backward_velocity(x_mid)
            delta = float(torch.max(torch.abs(x_cand - x_next)).detach().item())
            x_next = x_cand
            if delta <= 1e-9:
                break
        x = brick.clamp_to_hard_bounds(x_next)
        rev_states.append(x)

    chron_states = list(reversed(rev_states))
    states_t = torch.stack(chron_states, dim=1)
    flows_t = torch.stack([brick.flow(s, params) for s in chron_states], dim=1)
    energies_t = torch.stack([brick.hamiltonian_energy(s, params) for s in chron_states], dim=1)
    step_viable_t = torch.stack([compute_viable_universe_mask(brick, s) for s in chron_states], dim=1)
    all_viable_t = step_viable_t.all(dim=1)

    batch_retro = int(states_t.shape[0])
    record_simulation_activity(
        ode_rollout_calls=1,
        trajectories_integrated=batch_retro,
        integration_steps_executed=batch_retro * int(num_steps),
        integrator=f"retrodict_{mode}",
    )
    res_obj = RolloutResult(
        states=states_t,
        flows=flows_t,
        energies=energies_t,
        step_viable_mask=step_viable_t,
        all_time_viable_mask=all_viable_t,
        dt=dt,
        integrator=f"retrodict_{mode}",
    )
    emit_trace_node(
        layer="layer4_engine",
        operation="retrodict_trajectory",
        target_name=brick.name,
        global_r2=res_obj.viable_fraction,
        selected_branch_id=f"retrodict_{mode}",
        selected_expression=f"x(0)={[round(float(v), 5) for v in states_t[0, 0].tolist()]}",
        metrics={
            "mode": mode,
            "num_steps": num_steps,
            "viable_fraction": round(res_obj.viable_fraction, 4),
            "max_energy_drift": round(res_obj.max_energy_drift, 6),
        },
        skip_if_inner_loop=True,
    )
    return res_obj


def _compute_batch_state_jacobian(
    brick: LawBrick,
    state: torch.Tensor,
    params: torch.Tensor,
    port_inputs: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute flow `f(x)` (`[B, N]`) and exact state Jacobian `df/dx` (`[B, N, N]`)."""
    batch_size, state_dim = state.shape
    x = state.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        f_val = brick.flow(x, params, port_inputs=port_inputs)
        cols: list[torch.Tensor] = []
        for i in range(state_dim):
            grad_i = torch.autograd.grad(
                outputs=f_val[:, i].sum(),
                inputs=x,
                retain_graph=(i < state_dim - 1),
                create_graph=False,
            )[0]
            cols.append(grad_i)
        jac = torch.stack(cols, dim=1)
    return f_val.detach(), jac.detach()


def find_steady_state_and_stability(
    brick: LawBrick,
    initial_guess: torch.Tensor,
    params: torch.Tensor,
    port_inputs: dict[str, torch.Tensor] | None = None,
    max_newton_iters: int = 35,
    damping_lambda: float = 1e-4,
    tol: float = 1e-7,
    eigen_zero_tol: float = 1e-5,
) -> EquilibriumAnalysis:
    """
    Solve `f(x^*, params) = 0` via invariant-subspace-preserving Drazin/Range(J)
    Levenberg-Marquardt Newton steps (`J^2 z = -f, delta_x = J z`), preserving linear
    conservation laws (`m^T f(x) == 0` => `m^T delta_x == 0`).
    """
    x = brick.clamp_to_hard_bounds(initial_guess.detach().clone().to(torch.float64))
    p = params.detach().to(torch.float64)
    batch_size, n = x.shape
    eye = torch.eye(n, dtype=torch.float64, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
    lam_vec = torch.full((batch_size, 1, 1), float(damping_lambda), dtype=torch.float64, device=x.device)

    for _ in range(max_newton_iters):
        f_val, jac = _compute_batch_state_jacobian(brick, x, p, port_inputs=port_inputs)
        res_norm = torch.linalg.norm(f_val, dim=-1)  # [B]
        if float(torch.max(res_norm).item()) <= tol:
            break

        # Subspace-preserving step: solve J^2 * z = -f, then set delta_x = J * z in Range(J)
        j2 = torch.bmm(jac, jac)
        j2_t = j2.transpose(1, 2)
        lhs_sub = torch.bmm(j2_t, j2) + lam_vec * eye
        rhs_sub = -torch.bmm(j2_t, f_val.unsqueeze(-1))
        z_vec = torch.linalg.lstsq(lhs_sub, rhs_sub, driver="gelsd").solution
        delta_x_sub = torch.bmm(jac, z_vec).squeeze(-1)

        # Standard LM step as fallback when J is nilpotent
        jt = jac.transpose(1, 2)
        lhs_std = torch.bmm(jt, jac) + lam_vec * eye
        rhs_std = -torch.bmm(jt, f_val.unsqueeze(-1))
        delta_x_std = torch.linalg.lstsq(lhs_std, rhs_std, driver="gelsd").solution.squeeze(-1)

        # Per-batch-element line search: each item b independently accepts its first improving step
        improved_mask = torch.zeros((batch_size,), dtype=torch.bool, device=x.device)
        best_x = x.clone()
        best_norm = res_norm.clone()

        for delta_cand in (delta_x_sub, delta_x_std):
            for alpha in (1.0, 0.5, 0.2, 0.05):
                if bool(improved_mask.all().item()):
                    break
                x_cand = brick.clamp_to_hard_bounds(x + alpha * delta_cand)
                with torch.no_grad():
                    f_cand = brick.flow(x_cand, p, port_inputs=port_inputs)
                    cand_norm = torch.linalg.norm(f_cand, dim=-1)
                accept_b = (~improved_mask) & torch.isfinite(cand_norm) & (cand_norm < best_norm * (1.0 - 1e-6))
                if bool(accept_b.any().item()):
                    best_x = torch.where(accept_b.unsqueeze(-1), x_cand, best_x)
                    best_norm = torch.where(accept_b, cand_norm, best_norm)
                    improved_mask = improved_mask | accept_b

        x = best_x
        lam_flat = lam_vec.squeeze(-1).squeeze(-1)
        lam_flat = torch.where(
            improved_mask,
            torch.clamp(lam_flat * 0.5, min=1e-10),
            torch.clamp(lam_flat * 5.0, max=1e4),
        )
        lam_vec = lam_flat.unsqueeze(-1).unsqueeze(-1)

    f_final, jac_final = _compute_batch_state_jacobian(brick, x, p, port_inputs=port_inputs)
    record_simulation_activity(newton_jacobian_solves=batch_size)
    res_norms = torch.linalg.norm(f_final, dim=-1)
    converged = res_norms <= max(tol * 10.0, 1e-5)

    eigvals = torch.linalg.eigvals(jac_final)  # [B, N] complex128
    real_parts = torch.real(eigvals)
    imag_parts = torch.abs(torch.imag(eigvals))
    abs_eigs = torch.abs(eigvals)

    zero_mask = abs_eigs <= eigen_zero_tol
    num_conserved = torch.sum(zero_mask, dim=-1)

    max_real_list: list[float] = []
    max_imag = torch.max(imag_parts, dim=-1).values
    classes: list[str] = []

    for b in range(batch_size):
        non_zero_reals = real_parts[b][~zero_mask[b]]
        if non_zero_reals.numel() > 0:
            mr = float(torch.max(non_zero_reals).item())
        else:
            mr = 0.0
        max_real_list.append(mr)
        mi = float(max_imag[b].item())

        if non_zero_reals.numel() == 0:
            classes.append("marginal_neutral")
        elif mr < -eigen_zero_tol:
            classes.append("stable_spiral" if mi > eigen_zero_tol else "stable_node")
        elif abs(mr) <= eigen_zero_tol:
            classes.append("conservative_center" if mi > eigen_zero_tol else "marginal_neutral")
        else:
            classes.append("unstable_spiral_or_hopf" if mi > eigen_zero_tol else "unstable_saddle_or_node")

    max_real_t = torch.tensor(max_real_list, dtype=torch.float64, device=x.device)

    jac_spec = (
        summarize_singular_spectrum(
            jac_final[0],
            operator_name=f"StateJacobian({brick.name})",
        )
        if batch_size >= 1
        else None
    )
    emit_trace_node(
        layer="layer4_engine",
        operation="find_steady_state_and_stability",
        target_name=brick.name,
        local_error_metric=float(res_norms[0].item()) if batch_size >= 1 else 0.0,
        selected_branch_id=classes[0] if classes else "none",
        selected_expression=f"stability={classes[0] if classes else 'none'}, max_real_eig={max_real_list[0]:.5f}",
        operator_spectrum=jac_spec,
        metrics={
            "converged_fraction": round(float(converged.to(torch.float64).mean().item()), 4),
            "max_real_eigenvalue": round(max_real_list[0], 6) if max_real_list else 0.0,
            "stability_classes": classes[:4],
        },
        skip_if_inner_loop=True,
    )

    return EquilibriumAnalysis(
        equilibrium_state=x,
        residual_norm=res_norms,
        converged_mask=converged,
        jacobian=jac_final,
        eigenvalues=eigvals,
        max_real_eigenvalue=max_real_t,
        max_imag_eigenvalue=max_imag,
        num_conserved_modes=num_conserved,
        stability_classes=tuple(classes),
    )


@dataclass(frozen=True)
class EquilibriumLandscapeReport:
    """Multi-root phase-space equilibrium catalog discovered by `discover_equilibrium_landscape`."""

    distinct_equilibria: torch.Tensor
    stability_classes: tuple[str, ...]
    max_real_eigenvalues: torch.Tensor
    max_imag_eigenvalues: torch.Tensor
    basin_counts: tuple[int, ...]
    stable_attractor_indices: tuple[int, ...]
    unstable_fixed_point_indices: tuple[int, ...]


@dataclass(frozen=True)
class BifurcationScanReport:
    """1D parameter continuation and eigenvalue bifurcation scan report."""

    param_name: str
    param_values: torch.Tensor
    equilibria: torch.Tensor
    max_real_eigenvalues: torch.Tensor
    max_imag_eigenvalues: torch.Tensor
    stability_classes: tuple[str, ...]
    bifurcation_indices: tuple[int, ...]
    critical_param_estimates: tuple[float, ...]


def discover_equilibrium_landscape(
    brick: LawBrick,
    params: torch.Tensor,
    num_probes: int = 24,
    probe_span: float = 3.0,
    max_newton_iters: int = 40,
    tol: float = 1e-7,
    cluster_tol: float = 1e-3,
    seed: int = 42,
) -> EquilibriumLandscapeReport:
    """
    Discover all distinct fixed points `f(x^*, params) = 0` (stable attractors, unstable saddles,
    and conservative centers) across the phase space of `brick` by launching `num_probes`
    quasi-random Sobol initial state guesses in parallel and deduplicating converged roots.
    """
    n_probes = max(int(num_probes), 1)
    p_single = params[:1].detach().to(torch.float64)
    device = p_single.device
    state_dim = brick.state_dim

    default_x = brick.default_state(batch_size=1, dtype=torch.float64, device=device)
    probes = default_x.expand(n_probes, -1).clone()

    if n_probes > 1 and state_dim > 0:
        record_simulation_activity(sobol_qmc_universes_sampled=n_probes)
        sobol = torch.quasirandom.SobolEngine(dimension=state_dim, scramble=True, seed=seed)
        u = sobol.draw(n_probes).to(dtype=torch.float64, device=device)
        for s_i, sp in enumerate(brick.state_specs):
            d_val = float(sp.default_value)
            span = max(abs(d_val), float( probe_span ))
            lo = max(float(sp.homeostatic_min), d_val - span)
            hi = min(float(sp.homeostatic_max), d_val + span)
            probes[:, s_i] = lo + u[:, s_i] * (hi - lo)
        probes[0] = default_x[0]
        if n_probes > 2:
            probes[1] = brick.clamp_to_hard_bounds(torch.zeros_like(default_x))[0]

    p_batch = p_single.expand(n_probes, -1)
    with suppress_inner_trace():
        eq_batch = find_steady_state_and_stability(
            brick=brick,
            initial_guess=probes,
            params=p_batch,
            max_newton_iters=max_newton_iters,
            tol=tol,
        )

    distinct_roots: list[torch.Tensor] = []
    distinct_classes: list[str] = []
    distinct_mr: list[float] = []
    distinct_mi: list[float] = []
    counts: list[int] = []

    for b in range(n_probes):
        if not bool(eq_batch.converged_mask[b].item()):
            continue
        cand = eq_batch.equilibrium_state[b]
        matched_idx = -1
        for k, existing in enumerate(distinct_roots):
            if float(torch.linalg.norm(cand - existing).item()) <= cluster_tol:
                matched_idx = k
                break
        if matched_idx >= 0:
            counts[matched_idx] += 1
        else:
            distinct_roots.append(cand)
            distinct_classes.append(eq_batch.stability_classes[b])
            distinct_mr.append(float(eq_batch.max_real_eigenvalue[b].item()))
            distinct_mi.append(float(eq_batch.max_imag_eigenvalue[b].item()))
            counts.append(1)

    if distinct_roots:
        # Sort distinct roots lexicographically by first state coordinate for deterministic ordering
        order = sorted(range(len(distinct_roots)), key=lambda i: float(distinct_roots[i][0].item()))
        distinct_t = torch.stack([distinct_roots[i] for i in order], dim=0)
        classes_t = tuple(distinct_classes[i] for i in order)
        mr_t = torch.tensor([distinct_mr[i] for i in order], dtype=torch.float64, device=device)
        mi_t = torch.tensor([distinct_mi[i] for i in order], dtype=torch.float64, device=device)
        counts_t = tuple(counts[i] for i in order)
    else:
        distinct_t = torch.empty((0, state_dim), dtype=torch.float64, device=device)
        classes_t = ()
        mr_t = torch.empty((0,), dtype=torch.float64, device=device)
        mi_t = torch.empty((0,), dtype=torch.float64, device=device)
        counts_t = ()

    stable_idx = tuple(
        i for i, c in enumerate(classes_t) if c in ("stable_node", "stable_spiral")
    )
    unstable_idx = tuple(
        i for i, c in enumerate(classes_t) if c in ("unstable_saddle_or_node", "unstable_spiral_or_hopf")
    )

    jac_spec_land: OperatorSpectrumSummary | None = None
    if distinct_t.shape[0] >= 1:
        _, j_root0 = _compute_batch_state_jacobian(brick, distinct_t[:1], p_single)
        jac_spec_land = summarize_singular_spectrum(
            j_root0[0],
            operator_name=f"EquilibriumJacobian({brick.name})",
        )

    emit_trace_node(
        layer="layer4_engine",
        operation="discover_equilibrium_landscape",
        target_name=brick.name,
        selected_expression=f"{len(classes_t)} equilibria: {list(classes_t)}",
        operator_spectrum=jac_spec_land,
        metrics={
            "num_distinct_equilibria": len(classes_t),
            "stability_classes": list(classes_t),
            "num_stable_attractors": len(stable_idx),
            "num_unstable_fixed_points": len(unstable_idx),
        },
        skip_if_inner_loop=True,
    )

    return EquilibriumLandscapeReport(
        distinct_equilibria=distinct_t,
        stability_classes=classes_t,
        max_real_eigenvalues=mr_t,
        max_imag_eigenvalues=mi_t,
        basin_counts=counts_t,
        stable_attractor_indices=stable_idx,
        unstable_fixed_point_indices=unstable_idx,
    )


def scan_parameter_bifurcation(
    brick: LawBrick,
    param_name: str,
    param_values: torch.Tensor,
    initial_guess: torch.Tensor,
    base_params: torch.Tensor | None = None,
    max_newton_iters: int = 35,
    tol: float = 1e-7,
    eigen_zero_tol: float = 1e-5,
) -> BifurcationScanReport:
    """
    Sweep parameter `param_name` across `param_values` (`[NumGrid]`), tracking the equilibrium
    branch `x^*(theta)`, local Jacobian eigenvalues, stability regime transitions, and critical
    bifurcation points where `max_i Re(lambda_i(theta))` crosses zero.
    """
    p_idx = -1
    for idx, pp in enumerate(brick.param_specs):
        if pp.name == param_name:
            p_idx = idx
            break
    if p_idx < 0:
        raise ValueError(f"Unknown parameter '{param_name}' on brick '{brick.name}'.")

    grid = param_values.reshape(-1).to(torch.float64)
    n_grid = int(grid.numel())
    if n_grid < 2:
        raise ValueError("param_values must contain at least 2 grid points for bifurcation scanning.")

    device = grid.device
    p_base = (
        base_params[:1].detach().to(dtype=torch.float64, device=device)
        if base_params is not None
        else brick.default_params(batch_size=1, dtype=torch.float64, device=device)
    )
    p_sweep = p_base.expand(n_grid, -1).clone()
    p_sweep[:, p_idx] = grid

    x_guesses = initial_guess[:1].detach().to(dtype=torch.float64, device=device).expand(n_grid, -1).clone()
    with suppress_inner_trace():
        eq_res = find_steady_state_and_stability(
            brick=brick,
            initial_guess=x_guesses,
            params=p_sweep,
            max_newton_iters=max_newton_iters,
            tol=tol,
            eigen_zero_tol=eigen_zero_tol,
        )

    bif_indices: list[int] = []
    crit_params: list[float] = []
    mr = eq_res.max_real_eigenvalue
    classes = eq_res.stability_classes

    for k in range(n_grid - 1):
        r0 = float(mr[k].item())
        r1 = float(mr[k + 1].item())
        c0 = classes[k]
        c1 = classes[k + 1]
        sign_cross = (r0 < -eigen_zero_tol and r1 > eigen_zero_tol) or (
            r0 > eigen_zero_tol and r1 < -eigen_zero_tol
        )
        if sign_cross or (c0 != c1):
            bif_indices.append(k)
            p0 = float(grid[k].item())
            p1 = float(grid[k + 1].item())
            if abs(r1 - r0) > 1e-12 and (r0 * r1 <= 0.0):
                theta_c = p0 + (-r0 / (r1 - r0)) * (p1 - p0)
            else:
                theta_c = 0.5 * (p0 + p1)
            crit_params.append(theta_c)

    emit_trace_node(
        layer="layer4_engine",
        operation="scan_parameter_bifurcation",
        target_name=f"{brick.name}.{param_name}",
        selected_expression=f"bifurcations={len(crit_params)}, crit_params={[round(c, 5) for c in crit_params]}",
        metrics={
            "param_name": param_name,
            "num_grid_points": n_grid,
            "bifurcation_indices": list(bif_indices),
            "critical_param_estimates": [round(c, 6) for c in crit_params],
        },
        skip_if_inner_loop=True,
    )

    return BifurcationScanReport(
        param_name=param_name,
        param_values=grid,
        equilibria=eq_res.equilibrium_state,
        max_real_eigenvalues=mr,
        max_imag_eigenvalues=eq_res.max_imag_eigenvalue,
        stability_classes=classes,
        bifurcation_indices=tuple(bif_indices),
        critical_param_estimates=tuple(crit_params),
    )


def estimate_trajectory_noise_std(states: torch.Tensor) -> float:
    """
    Estimate additive measurement noise standard deviation `sigma_noise` on a trajectory
    `states` (`[T, D]` or `[B, T, D]`) using the 4th-order finite-difference annihilator
    `Delta^4 x_k = x_{k+2} - 4 x_{k+1} + 6 x_k - 4 x_{k-1} + x_{k-2}`, which annihilates
    smooth cubic polynomials while scaling additive white noise by `sqrt(70) * sigma_noise`.
    """
    s = states.detach().to(torch.float64)
    if s.ndim == 2:
        s = s.unsqueeze(0)
    if s.shape[1] < 6:
        return 0.0
    d4 = s[:, 4:, :] - 4.0 * s[:, 3:-1, :] + 6.0 * s[:, 2:-2, :] - 4.0 * s[:, 1:-3, :] + s[:, :-4, :]
    # Median Absolute Deviation (MAD) estimator: sigma = median(|d4|) / (0.67449 * sqrt(70))
    mad = float(torch.median(torch.abs(d4)).item())
    return mad / (0.67448975 * (70.0**0.5))


def savitzky_golay_denoise_and_differentiate(
    states: torch.Tensor,
    dt: float,
    window_length: int | None = None,
    poly_order: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Denoise `states` (`[T, D]` or `[B, T, D]`) and compute regularized analytical first derivatives
    `dx/dt` using exact local polynomial Savitzky-Golay least-squares projection across both
    interior and asymmetric boundary windows (eliminating boundary padding artifacts).
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    squeeze_batch = False
    s = states.to(torch.float64)
    if s.ndim == 2:
        s = s.unsqueeze(0)
        squeeze_batch = True
    if s.ndim != 3:
        raise ValueError(f"Expected 2D or 3D states tensor, got shape {tuple(states.shape)}.")

    batch, t_len, state_dim = s.shape
    if window_length is None:
        target_w = max(9, min(19, int(round(0.35 / max(dt, 1e-3)))))
        if target_w % 2 == 0:
            target_w += 1
        w = min(target_w, t_len)
    else:
        w = min(int(window_length), t_len)
    if w % 2 == 0:
        w = max(w - 1, 1)
    p_ord = min(int(poly_order), w - 1)
    if w < 5 or p_ord < 1:
        dx = estimate_trajectory_derivatives(states, dt=dt, denoise=False)
        return states, dx

    half_w = w // 2
    s_smooth = torch.zeros_like(s)
    dx_smooth = torch.zeros_like(s)
    exponents = torch.arange(p_ord + 1, dtype=torch.float64, device=s.device).unsqueeze(0)  # [1, p+1]

    for t_idx in range(t_len):
        start = max(0, min(t_idx - half_w, t_len - w))
        end = start + w
        # Local time offsets tau = (k - t_idx) * dt around evaluation index t_idx
        tau = (torch.arange(start, end, dtype=torch.float64, device=s.device) - float(t_idx)) * dt  # [w]
        v_mat = tau.unsqueeze(-1) ** exponents  # [w, p+1]
        # Solve least-squares polynomial coefficients c = pinv(V) @ y_win
        pinv_v = torch.linalg.pinv(v_mat)  # [p+1, w]
        w_val = pinv_v[0, :]  # row 0 evaluates c_0 = x_hat(t_idx)
        w_der = pinv_v[1, :]  # row 1 evaluates c_1 = dx_hat/dt(t_idx)
        y_win = s[:, start:end, :]  # [B, w, D]
        s_smooth[:, t_idx, :] = torch.einsum("w,bwd->bd", w_val, y_win)
        dx_smooth[:, t_idx, :] = torch.einsum("w,bwd->bd", w_der, y_win)

    if squeeze_batch:
        return s_smooth.squeeze(0), dx_smooth.squeeze(0)
    return s_smooth, dx_smooth


def estimate_trajectory_derivatives(
    states: torch.Tensor,
    dt: float,
    denoise: bool | str = "auto",
) -> torch.Tensor:
    """
    Estimate instantaneous velocities `dx/dt` from a state trajectory tensor of shape
    `[Batch, NumSteps, StateDim]` or `[NumSteps, StateDim]` using 4th-order central finite
    differences in the interior (`O(dt^4)`) and 2nd-order stencils at the boundaries.
    When `denoise="auto"` (or `denoise=True`), automatically switches to local polynomial
    Savitzky-Golay derivative projection when high-frequency sensor noise is detected.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    squeeze_batch = False
    if states.ndim == 2:
        states = states.unsqueeze(0)
        squeeze_batch = True
    if states.ndim != 3:
        raise ValueError(f"Expected 2D or 3D states tensor, got shape {tuple(states.shape)}.")

    t_len = states.shape[1]
    if t_len < 2:
        raise ValueError("Need at least 2 time points to estimate derivatives.")

    if denoise is True and t_len >= 7:
        _, dx_sg = savitzky_golay_denoise_and_differentiate(states, dt=dt)
        return dx_sg.squeeze(0) if squeeze_batch else dx_sg
    elif denoise == "auto" and t_len >= 9:
        noise_std = estimate_trajectory_noise_std(states)
        sig_std = max(float(torch.std(states.detach().to(torch.float64)).item()), 1e-12)
        if noise_std > 0.008 * sig_std and noise_std > 25.0 * (dt**3) * sig_std:
            _, dx_sg = savitzky_golay_denoise_and_differentiate(states, dt=dt)
            return dx_sg.squeeze(0) if squeeze_batch else dx_sg

    dx_dt = torch.zeros_like(states)
    if t_len == 2:
        diff = (states[:, 1] - states[:, 0]) / dt
        dx_dt[:, 0] = diff
        dx_dt[:, 1] = diff
    elif t_len < 5:
        dx_dt[:, 0] = (-3.0 * states[:, 0] + 4.0 * states[:, 1] - states[:, 2]) / (2.0 * dt)
        dx_dt[:, 1:-1] = (states[:, 2:] - states[:, :-2]) / (2.0 * dt)
        dx_dt[:, -1] = (3.0 * states[:, -1] - 4.0 * states[:, -2] + states[:, -3]) / (2.0 * dt)
    else:
        dx_dt[:, 0] = (-3.0 * states[:, 0] + 4.0 * states[:, 1] - states[:, 2]) / (2.0 * dt)
        dx_dt[:, 1] = (states[:, 2] - states[:, 0]) / (2.0 * dt)
        dx_dt[:, 2:-2] = (
            -states[:, 4:]
            + 8.0 * states[:, 3:-1]
            - 8.0 * states[:, 1:-3]
            + states[:, :-4]
        ) / (12.0 * dt)
        dx_dt[:, -2] = (states[:, -1] - states[:, -3]) / (2.0 * dt)
        dx_dt[:, -1] = (3.0 * states[:, -1] - 4.0 * states[:, -2] + states[:, -3]) / (2.0 * dt)

    return dx_dt.squeeze(0) if squeeze_batch else dx_dt


@dataclass(frozen=True)
class StochasticLangevinRolloutResult:
    """
    Fluctuation-Dissipation Theorem (FDT) Stochastic Port-Hamiltonian Langevin ensemble result.
    """

    ensemble_states: torch.Tensor
    mean_trajectory: torch.Tensor
    std_trajectory: torch.Tensor
    ensemble_energies: torch.Tensor
    viable_fraction: float
    thermal_energy_scale: float
    max_conservative_noise_leakage: float


@dataclass(frozen=True)
class KramersBarrierReport:
    """
    Free-energy barrier `Delta_H^dagger` and Eyring-Kramers transition rate report between
    metastable attractors and saddle points.
    """

    num_wells: int
    num_saddles: int
    well_states: torch.Tensor
    saddle_states: torch.Tensor
    well_energies: torch.Tensor
    saddle_energies: torch.Tensor
    energy_barrier_delta_h: float
    eyring_kramers_escape_rate: float
    mean_first_passage_time: float


def _symmetric_psd_matrix_sqrt(r_mat: torch.Tensor) -> torch.Tensor:
    """
    Compute the exact spectral matrix square root `Sigma = V * diag(sqrt(max(lambda_i, 0))) * V^T`
    of a batch of symmetric positive semi-definite Onsager dissipation matrices `R(x)` (`[B, D, D]`).
    """
    r_sym = 0.5 * (r_mat + r_mat.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(r_sym)
    sqrt_eigs = torch.sqrt(torch.clamp(eigvals, min=0.0))
    return torch.bmm(eigvecs * sqrt_eigs.unsqueeze(-2), eigvecs.transpose(-1, -2))


def rollout_stochastic_langevin_ensemble(
    brick: LawBrick,
    initial_state: torch.Tensor,
    params: torch.Tensor,
    dt: float,
    num_steps: int,
    thermal_energy_scale: float = 0.02,
    num_trajectories: int = 16,
    seed: int = 42,
) -> StochasticLangevinRolloutResult:
    """
    Integrate the Fluctuation-Dissipation Stochastic Port-Hamiltonian Langevin equation:
        `dx = [(J(x) - R(x)) * grad_H(x) + u(x)] dt + sqrt(2 * eps_th * R(x)) dW_t`
    across `num_trajectories` parallel realizations.

    Because the noise covariance `2 * eps_th * R(x)` is constructed from the exact spectral
    square root of the Onsager dissipation matrix `R(x) >= 0`:
        1. Purely conservative coordinates (where `R_{ii} = 0`) receive identically zero
           spurious thermal noise (`max_conservative_noise_leakage == 0`).
        2. Any linear conservation law `m^T R(x) = 0` is preserved pathwise by the stochastic noise.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if num_trajectories < 1:
        raise ValueError(f"num_trajectories must be >= 1, got {num_trajectories}.")

    x0 = initial_state.detach().to(torch.float64)
    if x0.ndim == 1:
        x0 = x0.unsqueeze(0)
    x = brick.clamp_to_hard_bounds(x0[:1].expand(num_trajectories, -1).clone())

    p0 = params.detach().to(torch.float64)
    if p0.ndim == 1:
        p0 = p0.unsqueeze(0)
    p = p0[:1].expand(num_trajectories, -1).clone()

    eps_th = max(float(thermal_energy_scale), 0.0)
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))

    states_hist: list[torch.Tensor] = [x]
    energies_hist: list[torch.Tensor] = [brick.hamiltonian_energy(x, p)]
    viable_hist: list[torch.Tensor] = [compute_viable_universe_mask(brick, x)]
    max_cons_leak = 0.0

    for _ in range(num_steps):
        with torch.no_grad():
            x_det = brick.step_rk4(x, p, dt=dt)
            if eps_th > 0.0 and not getattr(brick, "is_matrix_free", False):
                r_mat = brick.dissipation_matrix(x, p)
                sigma = _symmetric_psd_matrix_sqrt(r_mat)
                dw = torch.randn(x.shape, dtype=torch.float64, device=x.device, generator=gen) * (
                    (2.0 * eps_th * dt) ** 0.5
                )
                noise_step = torch.bmm(sigma, dw.unsqueeze(-1)).squeeze(-1)
                diag_r = torch.diagonal(r_mat, dim1=-2, dim2=-1)
                cons_mask = diag_r <= 1e-14
                if bool(cons_mask.any().item()):
                    leak = float(torch.max(torch.abs(noise_step[cons_mask])).item())
                    max_cons_leak = max(max_cons_leak, leak)
                x = brick.clamp_to_hard_bounds(x_det + noise_step)
            else:
                x = x_det

            states_hist.append(x)
            energies_hist.append(brick.hamiltonian_energy(x, p))
            viable_hist.append(compute_viable_universe_mask(brick, x))

    ens_states = torch.stack(states_hist, dim=1)  # [N_traj, T+1, D]
    ens_energies = torch.stack(energies_hist, dim=1)  # [N_traj, T+1]
    step_viable = torch.stack(viable_hist, dim=1)
    all_viable = step_viable.all(dim=1)
    viable_frac = float(all_viable.to(torch.float64).mean().item())

    mean_traj = torch.mean(ens_states, dim=0)
    std_traj = torch.std(ens_states, dim=0, unbiased=False)
    record_simulation_activity(
        ode_rollout_calls=1,
        trajectories_integrated=int(num_trajectories),
        integration_steps_executed=int(num_trajectories) * int(num_steps),
        stochastic_langevin_trajectories=int(num_trajectories),
        integrator="stochastic_langevin_rk4",
    )

    emit_trace_node(
        layer="layer4_engine",
        operation="rollout_stochastic_langevin_ensemble",
        target_name=brick.name,
        global_r2=viable_frac,
        selected_expression=(
            f"trajectories={num_trajectories}, eps_th={eps_th:.4f}, "
            f"viable={viable_frac:.3f}, cons_leak={max_cons_leak:.2e}"
        ),
        metrics={
            "num_trajectories": num_trajectories,
            "thermal_energy_scale": eps_th,
            "viable_fraction": round(viable_frac, 4),
            "max_conservative_noise_leakage": max_cons_leak,
        },
        skip_if_inner_loop=True,
    )

    return StochasticLangevinRolloutResult(
        ensemble_states=ens_states,
        mean_trajectory=mean_traj,
        std_trajectory=std_traj,
        ensemble_energies=ens_energies,
        viable_fraction=viable_frac,
        thermal_energy_scale=eps_th,
        max_conservative_noise_leakage=max_cons_leak,
    )


def compute_kramers_barrier_and_escape_rate(
    brick: LawBrick,
    params: torch.Tensor,
    landscape_report: EquilibriumLandscapeReport | None = None,
    thermal_energy_scale: float = 0.1,
    num_probes: int = 24,
) -> KramersBarrierReport:
    """
    Compute the Hamiltonian free-energy activation barrier `Delta_H^dagger = H(x_saddle) - H(x_well)`
    and the Eyring-Kramers thermally activated escape rate:
        `k_escape = (sqrt(|lambda_unstable| * lambda_stable) / (2 * pi)) * exp(-Delta_H^dagger / eps_th)`
    between metastable attractor wells and index-1 saddle points.
    """
    import math

    p0 = params.detach().to(torch.float64)
    if p0.ndim == 1:
        p0 = p0.unsqueeze(0)
    p0 = p0[:1]

    landscape = (
        landscape_report
        if landscape_report is not None
        else discover_equilibrium_landscape(brick, p0, num_probes=num_probes)
    )

    eq_states = landscape.distinct_equilibria
    n_eq = int(eq_states.shape[0])
    if n_eq == 0:
        empty = torch.zeros((0, brick.state_dim), dtype=torch.float64, device=p0.device)
        empty_e = torch.zeros((0,), dtype=torch.float64, device=p0.device)
        return KramersBarrierReport(
            num_wells=0,
            num_saddles=0,
            well_states=empty,
            saddle_states=empty,
            well_energies=empty_e,
            saddle_energies=empty_e,
            energy_barrier_delta_h=0.0,
            eyring_kramers_escape_rate=0.0,
            mean_first_passage_time=float("inf"),
        )

    p_batch = p0.expand(n_eq, -1)
    with torch.no_grad():
        energies = brick.hamiltonian_energy(eq_states, p_batch)

    well_indices: list[int] = list(landscape.stable_attractor_indices)
    saddle_indices: list[int] = list(landscape.unstable_fixed_point_indices)
    if not well_indices and not saddle_indices:
        for idx, cls_name in enumerate(landscape.stability_classes):
            if "unstable" in cls_name:
                saddle_indices.append(idx)
            else:
                well_indices.append(idx)

    # If all equilibria are conservative centers (e.g., undamped pendulum where saddle has positive real eig
    # and well has purely imaginary eigs), well_indices collects centers/stable nodes and saddle_indices collects saddles.
    if not well_indices and n_eq > 0:
        min_idx = int(torch.argmin(energies).item())
        well_indices.append(min_idx)
    if not saddle_indices and n_eq >= 2:
        max_idx = int(torch.argmax(energies).item())
        if max_idx not in well_indices:
            saddle_indices.append(max_idx)

    well_states = eq_states[well_indices] if well_indices else eq_states[:0]
    well_energies = energies[well_indices] if well_indices else energies[:0]
    saddle_states = eq_states[saddle_indices] if saddle_indices else eq_states[:0]
    saddle_energies = energies[saddle_indices] if saddle_indices else energies[:0]

    if well_states.shape[0] > 0 and saddle_states.shape[0] > 0:
        best_well_local = int(torch.argmin(well_energies).item())
        best_saddle_local = int(torch.argmin(saddle_energies).item())
        h_well = float(well_energies[best_well_local].item())
        h_saddle = float(saddle_energies[best_saddle_local].item())
        delta_h = max(0.0, h_saddle - h_well)

        w_global = well_indices[best_well_local]
        s_global = saddle_indices[best_saddle_local]
        lam_well = max(abs(float(landscape.max_real_eigenvalues[w_global].item())), 0.5)
        lam_saddle = max(abs(float(landscape.max_real_eigenvalues[s_global].item())), 0.5)
        eps_th = max(float(thermal_energy_scale), 1e-9)
        prefactor = math.sqrt(lam_well * lam_saddle) / (2.0 * math.pi)
        k_esc = float(prefactor * math.exp(-min(delta_h / eps_th, 100.0)))
        mfpt = 1.0 / max(k_esc, 1e-15)
    else:
        delta_h = 0.0
        k_esc = 0.0
        mfpt = float("inf")

    emit_trace_node(
        layer="layer4_engine",
        operation="compute_kramers_barrier_and_escape_rate",
        target_name=brick.name,
        selected_expression=(
            f"wells={len(well_indices)}, saddles={len(saddle_indices)}, "
            f"Delta_H={delta_h:.4f}, k_escape={k_esc:.4e}"
        ),
        metrics={
            "num_wells": len(well_indices),
            "num_saddles": len(saddle_indices),
            "energy_barrier_delta_h": round(delta_h, 6),
            "eyring_kramers_escape_rate": k_esc,
        },
        skip_if_inner_loop=True,
    )

    return KramersBarrierReport(
        num_wells=len(well_indices),
        num_saddles=len(saddle_indices),
        well_states=well_states,
        saddle_states=saddle_states,
        well_energies=well_energies,
        saddle_energies=saddle_energies,
        energy_barrier_delta_h=delta_h,
        eyring_kramers_escape_rate=k_esc,
        mean_first_passage_time=mfpt,
    )
