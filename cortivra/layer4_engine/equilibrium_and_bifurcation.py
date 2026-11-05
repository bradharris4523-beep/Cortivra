"""
Steady-state equilibrium solvers, multi-root phase-space landscape discovery,
1D parameter bifurcation continuation, and Eyring-Kramers barrier calculation
(`cortivra.layer4_engine.equilibrium_and_bifurcation`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick
from cortivra.observability.flight_recorder import (
    OperatorSpectrumSummary,
    emit_trace_node,
    record_simulation_activity,
    summarize_singular_spectrum,
    suppress_inner_trace,
)


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
        if not f_val.requires_grad:
            return f_val.detach(), torch.zeros((batch_size, state_dim, state_dim), dtype=x.dtype, device=x.device)
        cols: list[torch.Tensor] = []
        for i in range(state_dim):
            if not f_val[:, i].requires_grad:
                cols.append(torch.zeros_like(x))
                continue
            grad_i = torch.autograd.grad(
                outputs=f_val[:, i].sum(),
                inputs=x,
                retain_graph=(i < state_dim - 1),
                create_graph=False,
                allow_unused=True,
            )[0]
            cols.append(grad_i if grad_i is not None else torch.zeros_like(x))
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
        jac_clean = torch.nan_to_num(jac, nan=0.0, posinf=1e6, neginf=-1e6)
        f_clean = torch.nan_to_num(f_val, nan=0.0, posinf=1e6, neginf=-1e6)
        j2 = torch.clamp(torch.bmm(jac_clean, jac_clean), min=-1e12, max=1e12)
        j2_t = j2.transpose(1, 2)
        lhs_sub = torch.bmm(j2_t, j2) + lam_vec * eye
        rhs_sub = -torch.bmm(j2_t, f_clean.unsqueeze(-1))
        lhs_sub = torch.where(torch.isfinite(lhs_sub), lhs_sub, eye)
        rhs_sub = torch.nan_to_num(rhs_sub, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            z_vec = torch.linalg.lstsq(lhs_sub, rhs_sub, driver="gelsd").solution
            delta_x_sub = torch.bmm(jac_clean, z_vec).squeeze(-1)
        except RuntimeError:
            delta_x_sub = torch.zeros_like(x)

        # Standard LM step as fallback when J is nilpotent
        jt = jac_clean.transpose(1, 2)
        lhs_std = torch.bmm(jt, jac_clean) + lam_vec * eye
        rhs_std = -torch.bmm(jt, f_clean.unsqueeze(-1))
        lhs_std = torch.where(torch.isfinite(lhs_std), lhs_std, eye)
        rhs_std = torch.nan_to_num(rhs_std, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            delta_x_std = torch.linalg.lstsq(lhs_std, rhs_std, driver="gelsd").solution.squeeze(-1)
        except RuntimeError:
            delta_x_std = torch.zeros_like(x)

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
    converged = torch.isfinite(res_norms) & (res_norms <= max(tol * 10.0, 1e-5))

    jac_eig_safe = torch.nan_to_num(jac_final, nan=0.0, posinf=1e6, neginf=-1e6)
    try:
        eigvals = torch.linalg.eigvals(jac_eig_safe)  # [B, N] complex128
    except RuntimeError:
        eigvals = torch.zeros((batch_size, n), dtype=torch.complex128, device=x.device)
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
            span = max(abs(d_val), float(probe_span))
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
