"""
Conservation, thermodynamic, viability, and residual-analysis shields.

Provides:
    - Dimensional consistency and power-conjugacy checks.
    - Scale-invariant stoichiometric mass conservation and ionic electroneutrality checks.
    - Kirchhoff capacitive current balance checks.
    - 1st and 2nd law thermodynamic verification (both matrix-based eigenvalue PSD and O(N) matrix-free).
    - Wegscheider microscopic detailed balance checks around closed cycles.
    - Vectorized GPU state viability masking.
    - Gated residual analysis (`analyze_unexplained_residual`) that isolates unmodeled dynamics
      only when known parameter fits are exhausted, signal-to-noise exceeds threshold, and the
      residual is a coherent function of state rather than random noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick
from cortivra.layer1_axioms.units import (
    BODY_TEMPERATURE_K,
    DIM_POWER,
    R_GAS,
    DimensionError,
    Quantity,
    Unit,
)


class ConservationViolationError(RuntimeError):
    """Raised when a law or trajectory violates a conservation or thermodynamic invariant."""


@dataclass(frozen=True)
class ShieldResult:
    """Result of a single invariant check."""

    shield_name: str
    passed: bool
    max_violation: float
    tolerance: float
    message: str


@dataclass(frozen=True)
class AuditCertificate:
    """Multi-shield verification certificate for a `LawBrick` or trajectory."""

    target_name: str
    passed_all: bool
    results: tuple[ShieldResult, ...] = field(default_factory=tuple)
    viable_fraction: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "passed_all": self.passed_all,
            "viable_fraction": self.viable_fraction,
            "results": [asdict(r) for r in self.results],
        }

    def raise_if_failed(self) -> None:
        if not self.results:
            raise ConservationViolationError(
                f"Audit failed for '{self.target_name}': certificate contains no shield results."
            )
        if not self.passed_all or any(not r.passed for r in self.results):
            failed_msgs = [r.message for r in self.results if not r.passed] or ["passed_all is False"]
            joined = "; ".join(failed_msgs)
            raise ConservationViolationError(f"Audit failed for '{self.target_name}': {joined}")


@dataclass(frozen=True)
class LatentResidualAnalysis:
    """
    Diagnostic report evaluating whether a discrepancy between observed dynamics
    `dx/dt_obs` and model predictions `F_known(x, params)` justifies fitting an
    unmodeled physical term.

    `latent_term_justified` is True only if all three conditions hold:
        1. `parameter_fit_exhausted` is True (known parameters were already optimized).
        2. `max_snr >= min_snr_threshold` (residual exceeds measurement/thermal noise floor).
        3. `state_coherence_r2 >= min_coherence_r2` (residual is a deterministic function
           of state `x` rather than uncorrelated random noise).
    """

    latent_term_justified: bool
    rejection_reason: str | None
    max_snr: float
    state_coherence_r2: float
    affected_state_indices: tuple[int, ...]
    required_derivative_units: tuple[Unit, ...]
    thermodynamic_class: str  # "none", "passive_dissipative", "conservative_coupling", or "active_driven"
    residual_flow: torch.Tensor
    missing_power: torch.Tensor


def verify_dimensional_consistency(
    brick: LawBrick,
    derivative_units: Sequence[Unit] | None = None,
) -> ShieldResult:
    """Verify state time-derivative dimensions (`Dim(x_i) / Dim(t)`) and port power conjugacy."""
    try:
        if derivative_units is not None:
            if len(derivative_units) != brick.state_dim:
                return ShieldResult(
                    shield_name="dimensional_consistency",
                    passed=False,
                    max_violation=float(abs(len(derivative_units) - brick.state_dim)),
                    tolerance=0.0,
                    message=f"Expected {brick.state_dim} derivative units, got {len(derivative_units)}.",
                )
            for idx, (spec, d_unit) in enumerate(zip(brick.state_specs, derivative_units)):
                expected_dim = spec.unit.dimension / brick.time_unit.dimension
                if d_unit.dimension != expected_dim:
                    return ShieldResult(
                        shield_name="dimensional_consistency",
                        passed=False,
                        max_violation=1.0,
                        tolerance=0.0,
                        message=(
                            f"State #{idx} ('{spec.name}'): d/dt unit '{d_unit.symbol}' has "
                            f"dimension {d_unit.dimension.format_si()}, expected {expected_dim.format_si()}."
                        ),
                    )

        for port in brick.port_specs:
            if port.require_power_conjugate:
                prod_dim = (port.effort_unit * port.flow_unit).dimension
                if prod_dim != DIM_POWER:
                    return ShieldResult(
                        shield_name="dimensional_consistency",
                        passed=False,
                        max_violation=1.0,
                        tolerance=0.0,
                        message=(
                            f"Port '{port.name}' violates power conjugacy: "
                            f"{prod_dim.format_si()} != {DIM_POWER.format_si()}."
                        ),
                    )
    except DimensionError as exc:
        return ShieldResult(
            shield_name="dimensional_consistency",
            passed=False,
            max_violation=1.0,
            tolerance=0.0,
            message=str(exc),
        )

    return ShieldResult(
        shield_name="dimensional_consistency",
        passed=True,
        max_violation=0.0,
        tolerance=0.0,
        message=f"All {brick.state_dim} states and {len(brick.port_specs)} ports are dimensionally consistent.",
    )


def verify_equation_dimensions(lhs: Quantity, rhs: Quantity, context: str = "equation") -> ShieldResult:
    """Verify that left-hand and right-hand Quantities share identical SI dimensions."""
    if lhs.dimension != rhs.dimension:
        return ShieldResult(
            shield_name="equation_dimensional_homogeneity",
            passed=False,
            max_violation=1.0,
            tolerance=0.0,
            message=(
                f"[{context}] LHS '{lhs.unit.symbol}' ({lhs.dimension.format_si()}) != "
                f"RHS '{rhs.unit.symbol}' ({rhs.dimension.format_si()})."
            ),
        )
    return ShieldResult(
        shield_name="equation_dimensional_homogeneity",
        passed=True,
        max_violation=0.0,
        tolerance=0.0,
        message=f"[{context}] Dimensional homogeneity verified ({lhs.dimension.format_si()}).",
    )


def verify_mass_conservation(
    stoichiometric_matrix: torch.Tensor,
    molar_masses: torch.Tensor,
    reaction_rates: torch.Tensor | None = None,
    tol: float = 1e-5,
) -> ShieldResult:
    """
    Verify scale-invariant mass conservation (`m^T * S == 0` and `m^T * S * v^T == 0`).
    Residuals are normalized by `max(|m|)` and `max(1, |v|)` to remain invariant across mass scales.
    """
    m = molar_masses.to(dtype=torch.float64)
    s_mat = stoichiometric_matrix.to(dtype=torch.float64)
    if m.ndim != 1 or s_mat.ndim != 2 or m.shape[0] != s_mat.shape[0]:
        raise ValueError(
            f"Shape mismatch: molar_masses {tuple(m.shape)} vs stoichiometric_matrix {tuple(s_mat.shape)}."
        )
    if not torch.isfinite(m).all() or not torch.isfinite(s_mat).all():
        return ShieldResult(
            shield_name="mass_conservation",
            passed=False,
            max_violation=float("inf"),
            tolerance=tol,
            message="Non-finite values in molar_masses or stoichiometric_matrix.",
        )

    m_scale = float(torch.max(torch.abs(m)).item()) if m.numel() > 0 else 1.0
    m_scale = max(m_scale, 1e-12)

    reaction_imbalance = torch.matmul(m, s_mat)
    max_struct_err = (
        float(torch.max(torch.abs(reaction_imbalance)).item()) / m_scale
        if reaction_imbalance.numel() > 0
        else 0.0
    )

    max_flux_err = 0.0
    if reaction_rates is not None:
        v = reaction_rates.to(dtype=torch.float64)
        v_scale = max(float(torch.max(torch.abs(v)).item()), 1.0) if v.numel() > 0 else 1.0
        dn_dt = torch.matmul(v, s_mat.T)
        mass_rate = torch.matmul(dn_dt, m)
        max_flux_err = float(torch.max(torch.abs(mass_rate)).item()) / (m_scale * v_scale)

    worst = max(max_struct_err, max_flux_err)
    passed = worst <= tol
    msg = (
        f"Mass conservation verified (max relative residual = {worst:.2e} <= {tol:.2e})."
        if passed
        else f"Mass conservation violated: max relative residual = {worst:.2e} > {tol:.2e}."
    )
    return ShieldResult(
        shield_name="mass_conservation",
        passed=passed,
        max_violation=worst,
        tolerance=tol,
        message=msg,
    )


def verify_charge_conservation(
    capacitance_f: torch.Tensor | float,
    dv_dt: torch.Tensor,
    ionic_currents: torch.Tensor,
    external_current: torch.Tensor | float = 0.0,
    tol: float = 1e-5,
) -> ShieldResult:
    """Verify Kirchhoff capacitive current balance: `C_m * (dV_m / dt) + sum_k(I_ion,k) - I_ext == 0`."""
    total_ionic = ionic_currents.sum(dim=-1) if ionic_currents.ndim > 1 else ionic_currents
    residual = capacitance_f * dv_dt + total_ionic - external_current
    max_err = float(torch.max(torch.abs(residual)).item())
    passed = max_err <= tol
    msg = (
        f"Charge balance verified (max Kirchhoff residual = {max_err:.2e} <= {tol:.2e})."
        if passed
        else f"Charge conservation violated: Kirchhoff residual = {max_err:.2e} > {tol:.2e}."
    )
    return ShieldResult(
        shield_name="charge_conservation",
        passed=passed,
        max_violation=max_err,
        tolerance=tol,
        message=msg,
    )


def verify_reaction_electroneutrality(
    stoichiometric_matrix: torch.Tensor,
    valences: torch.Tensor,
    tol: float = 1e-6,
) -> ShieldResult:
    """Verify reaction electroneutrality (`z^T * S == 0`)."""
    z = valences.to(dtype=torch.float64)
    s_mat = stoichiometric_matrix.to(dtype=torch.float64)
    charge_imbalance = torch.matmul(z, s_mat)
    max_err = float(torch.max(torch.abs(charge_imbalance)).item()) if charge_imbalance.numel() > 0 else 0.0
    passed = max_err <= tol
    msg = (
        f"Reaction electroneutrality verified (max charge residual = {max_err:.2e})."
        if passed
        else f"Reaction electroneutrality violated: net charge residual = {max_err:.2e} > {tol:.2e}."
    )
    return ShieldResult(
        shield_name="reaction_electroneutrality",
        passed=passed,
        max_violation=max_err,
        tolerance=tol,
        message=msg,
    )


def verify_thermodynamic_laws(
    brick: LawBrick,
    state: torch.Tensor,
    params: torch.Tensor,
    port_inputs: dict[str, torch.Tensor] | None = None,
    external_power: torch.Tensor | None = None,
    tol: float = 1e-4,
) -> ShieldResult:
    """
    Verify 1st and 2nd Laws of Thermodynamics on `brick`:
        - Conservative orthogonality: `|grad_H(x) . F_cons(x)| <= tol`
        - Non-negative entropy dissipation: `D(x) = grad_H(x) . F_diss(x) >= -tol`
        - First Law power balance: `dH/dt = grad_H(x) . dx/dt == P_ext - D(x)`
        - When `not brick.is_matrix_free`: also checks `J = -J^T`, `R = R^T`, and `eig_min(R) >= -tol`.
    """
    e = brick.effort(state, params)
    f_cons = brick.conservative_flow(state, params)
    dissipated = brick.dissipation_power(state, params)
    dx_dt = brick.flow(state, params, port_inputs=port_inputs)

    cons_power_err = float(torch.max(torch.abs((e * f_cons).sum(dim=-1))).item())
    second_law_violation = float(torch.max(torch.clamp(-dissipated, min=0.0)).item())

    dh_dt = (e * dx_dt).sum(dim=-1)
    if external_power is not None:
        p_ext = external_power
    elif port_inputs and "external_flow" in port_inputs:
        p_ext = (e * port_inputs["external_flow"]).sum(dim=-1)
    elif hasattr(brick, "external_drive_power"):
        p_ext = brick.external_drive_power(state, params)
    elif hasattr(brick, "external_patch_power"):
        p_ext = brick.external_patch_power(state, params)
    else:
        p_ext = torch.zeros_like(dh_dt)
    power_balance_err = float(torch.max(torch.abs(dh_dt - (p_ext - dissipated))).item())

    skew_err = 0.0
    sym_err = 0.0
    psd_violation = 0.0
    if not brick.is_matrix_free:
        j_mat = brick.structure_matrix(state, params)
        r_mat = brick.dissipation_matrix(state, params)
        skew_err = float(torch.max(torch.abs(j_mat + j_mat.transpose(1, 2))).item())
        sym_err = float(torch.max(torch.abs(r_mat - r_mat.transpose(1, 2))).item())
        # Symmetrize before calling eigvalsh to isolate PSD check from asymmetry check
        r_sym = 0.5 * (r_mat + r_mat.transpose(1, 2))
        eigvals = torch.linalg.eigvalsh(r_sym)
        psd_violation = float(torch.max(torch.clamp(-eigvals, min=0.0)).item())

    worst = max(
        cons_power_err,
        skew_err,
        sym_err,
        psd_violation,
        second_law_violation,
        power_balance_err,
    )
    passed = worst <= tol

    if not passed:
        details = (
            f"Thermodynamic violation (max={worst:.2e} > {tol:.2e}): "
            f"cons_power_err={cons_power_err:.2e}, skew_err={skew_err:.2e}, "
            f"sym_err={sym_err:.2e}, psd_neg_eig={psd_violation:.2e}, "
            f"2nd_law_neg_dissipation={second_law_violation:.2e}, "
            f"1st_law_power_balance_err={power_balance_err:.2e}."
        )
    else:
        details = (
            f"1st & 2nd Laws of Thermodynamics verified across batch={state.shape[0]} "
            f"(max residual={worst:.2e} <= {tol:.2e})."
        )

    return ShieldResult(
        shield_name="thermodynamic_1st_and_2nd_law",
        passed=passed,
        max_violation=worst,
        tolerance=tol,
        message=details,
    )


def verify_detailed_balance_cycle(
    forward_rates: torch.Tensor,
    backward_rates: torch.Tensor,
    external_free_energy_j_per_mol: torch.Tensor | float = 0.0,
    temperature_k: float = float(BODY_TEMPERATURE_K.value),
    tol_j_per_mol: float = 10.0,
) -> ShieldResult:
    """
    Verify Wegscheider microscopic detailed balance around a closed cycle:
        R * T * ln( prod_i(k_i^+) / prod_i(k_i^-) ) == -DeltaG_ext
    """
    if forward_rates.shape != backward_rates.shape:
        raise ValueError(
            f"Shape mismatch: forward_rates {tuple(forward_rates.shape)} vs "
            f"backward_rates {tuple(backward_rates.shape)}."
        )
    if torch.any(forward_rates <= 0.0) or torch.any(backward_rates <= 0.0):
        return ShieldResult(
            shield_name="detailed_balance_cycle",
            passed=False,
            max_violation=float("inf"),
            tolerance=tol_j_per_mol,
            message="Transition rates around a reversible cycle must be strictly positive.",
        )

    rt = float(R_GAS.value) * temperature_k
    log_ratio = torch.sum(
        torch.log(forward_rates.to(torch.float64)) - torch.log(backward_rates.to(torch.float64)),
        dim=-1,
    )
    cycle_affinity = rt * log_ratio
    expected_drive = -external_free_energy_j_per_mol
    residual = torch.abs(cycle_affinity - expected_drive)
    max_err = float(torch.max(residual).item())
    passed = max_err <= tol_j_per_mol

    msg = (
        f"Detailed balance verified (max cycle affinity error = {max_err:.2f} J/mol <= {tol_j_per_mol:.2f} J/mol)."
        if passed
        else (
            f"Detailed balance violated (unbalanced cycle): cycle affinity mismatch = "
            f"{max_err:.2f} J/mol > {tol_j_per_mol:.2f} J/mol."
        )
    )
    return ShieldResult(
        shield_name="detailed_balance_cycle",
        passed=passed,
        max_violation=max_err,
        tolerance=tol_j_per_mol,
        message=msg,
    )


def compute_viable_universe_mask(
    brick: LawBrick,
    trajectory_or_state: torch.Tensor,
    enforce_homeostatic: bool = True,
) -> torch.Tensor:
    """Return boolean mask `[Batch]` indicating trajectories that stay finite and within bounds."""
    x = trajectory_or_state
    if x.ndim == 2:
        x = x.unsqueeze(1)
    elif x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D state tensor, got shape {tuple(x.shape)}.")

    device, dtype = x.device, x.dtype
    finite_mask = torch.isfinite(x).all(dim=(1, 2))

    hard_mins = torch.tensor(
        [sp.hard_min for sp in brick.state_specs], dtype=dtype, device=device
    ).view(1, 1, -1)
    hard_maxs = torch.tensor(
        [sp.hard_max for sp in brick.state_specs], dtype=dtype, device=device
    ).view(1, 1, -1)
    hard_ok = ((x >= hard_mins) & (x <= hard_maxs)).all(dim=(1, 2))

    if not enforce_homeostatic:
        return finite_mask & hard_ok

    homeo_mins = torch.tensor(
        [sp.homeostatic_min for sp in brick.state_specs], dtype=dtype, device=device
    ).view(1, 1, -1)
    homeo_maxs = torch.tensor(
        [sp.homeostatic_max for sp in brick.state_specs], dtype=dtype, device=device
    ).view(1, 1, -1)
    homeo_ok = ((x >= homeo_mins) & (x <= homeo_maxs)).all(dim=(1, 2))

    return finite_mask & hard_ok & homeo_ok


def verify_homeostatic_viability(
    brick: LawBrick,
    trajectory_or_state: torch.Tensor,
    min_viable_fraction: float = 1.0,
) -> tuple[ShieldResult, torch.Tensor]:
    """Evaluate state viability across a batch and return `(ShieldResult, viable_mask)`."""
    viable_mask = compute_viable_universe_mask(brick, trajectory_or_state, enforce_homeostatic=True)
    viable_fraction = float(viable_mask.to(torch.float32).mean().item()) if viable_mask.numel() > 0 else 0.0
    shortfall = max(0.0, min_viable_fraction - viable_fraction)
    passed = viable_fraction >= min_viable_fraction

    msg = (
        f"Homeostatic viability passed: {viable_fraction * 100:.1f}% viable "
        f"(required >= {min_viable_fraction * 100:.1f}%)."
        if passed
        else (
            f"Homeostatic viability failed: {viable_fraction * 100:.1f}% viable "
            f"(required >= {min_viable_fraction * 100:.1f}%)."
        )
    )
    result = ShieldResult(
        shield_name="homeostatic_viability",
        passed=passed,
        max_violation=shortfall,
        tolerance=1.0 - min_viable_fraction,
        message=msg,
    )
    return result, viable_mask


def analyze_unexplained_residual(
    brick: LawBrick,
    state: torch.Tensor,
    params: torch.Tensor,
    observed_dx_dt: torch.Tensor,
    noise_floor_std: torch.Tensor | float = 1e-2,
    parameter_fit_exhausted: bool = False,
    min_snr_threshold: float = 3.0,
    min_coherence_r2: float = 0.50,
    power_tol: float = 1e-4,
) -> LatentResidualAnalysis:
    """
    Evaluate whether a discrepancy `Phi(x) = observed_dx_dt - brick.flow(state, params)`
    justifies fitting an unmodeled physical mechanism.

    Conditions required before `latent_term_justified` is set to True:
        1. `parameter_fit_exhausted` must be True (known parameters already tuned).
        2. `max_snr >= min_snr_threshold` (RMS residual exceeds noise floor).
        3. `state_coherence_r2 >= min_coherence_r2` (residual projects coherently onto
           a 2nd-order polynomial basis `[1, x, x^2]` rather than random noise).
    """
    if observed_dx_dt.shape != state.shape:
        raise ValueError(
            f"Shape mismatch: observed_dx_dt {tuple(observed_dx_dt.shape)} vs state {tuple(state.shape)}."
        )

    known_flow = brick.flow(state, params)
    residual_flow = observed_dx_dt - known_flow  # [Batch, StateDim]
    e = brick.effort(state, params)  # [Batch, StateDim]
    missing_power = (e * residual_flow).sum(dim=-1)  # [Batch]

    req_units = tuple(sp.unit / brick.time_unit for sp in brick.state_specs)

    # Compute per-state RMS residual and SNR against noise_floor_std
    rms_per_state = torch.sqrt(torch.mean(residual_flow.to(torch.float64) ** 2, dim=0))
    if isinstance(noise_floor_std, torch.Tensor):
        noise_vec = torch.clamp(noise_floor_std.to(torch.float64), min=1e-12)
    else:
        noise_vec = torch.full_like(rms_per_state, max(float(noise_floor_std), 1e-12))

    snr_per_state = rms_per_state / noise_vec
    max_snr = float(torch.max(snr_per_state).item()) if snr_per_state.numel() > 0 else 0.0
    affected_indices = tuple(
        int(i) for i in torch.nonzero(snr_per_state >= min_snr_threshold, as_tuple=False).flatten().tolist()
    )

    # Compute state-space coherence R^2 across affected state components using SVD-stabilized
    # polynomial basis [1, x, x^2, x_i * x_j] to distinguish deterministic laws from random noise
    coherence_r2 = 0.0
    batch_size, state_dim = state.shape[0], state.shape[1]
    target_indices = affected_indices if affected_indices else (
        (int(torch.argmax(snr_per_state).item()),) if snr_per_state.numel() > 0 else ()
    )
    if batch_size >= 4 and max_snr > 1e-9 and target_indices:
        x_d = state.to(torch.float64)
        ones = torch.ones((batch_size, 1), dtype=torch.float64, device=state.device)
        basis_cols = [ones, x_d, x_d**2]
        if 1 < state_dim <= 8 and batch_size > (1 + 2 * state_dim + (state_dim * (state_dim - 1)) // 2 + 2):
            cross_terms = [
                (x_d[:, i] * x_d[:, j]).unsqueeze(-1)
                for i in range(state_dim)
                for j in range(i + 1, state_dim)
            ]
            basis_cols.extend(cross_terms)
        design = torch.cat(basis_cols, dim=-1)
        r2_scores: list[float] = []
        for idx in target_indices:
            y = residual_flow[:, idx].to(torch.float64)
            ss_tot = torch.sum((y - torch.mean(y)) ** 2)
            if float(ss_tot.item()) <= 1e-18:
                continue
            sol = torch.linalg.lstsq(design, y.unsqueeze(-1), driver="gelsd").solution
            y_hat = (design @ sol).squeeze(-1)
            ss_res = torch.sum((y - y_hat) ** 2)
            r2 = max(0.0, float((1.0 - ss_res / torch.clamp(ss_tot, min=1e-18)).item()))
            r2_scores.append(r2)
        if r2_scores:
            coherence_r2 = float(sum(r2_scores) / len(r2_scores))

    # Classify thermodynamic signature of the residual power DeltaP = grad_H . Phi
    max_p = float(torch.max(missing_power).item()) if missing_power.numel() > 0 else 0.0
    min_p = float(torch.min(missing_power).item()) if missing_power.numel() > 0 else 0.0
    mean_p = float(torch.mean(missing_power).item()) if missing_power.numel() > 0 else 0.0
    if abs(max_p) <= power_tol and abs(min_p) <= power_tol:
        thermo_class = "conservative_coupling"
    elif max_p <= power_tol and min_p < -power_tol:
        thermo_class = "passive_dissipative"
    elif max_p > power_tol and min_p < -power_tol and mean_p <= power_tol:
        thermo_class = "reactive_storage"
    else:
        thermo_class = "active_driven"

    # Evaluate the 3-Gate Safeguard
    if not parameter_fit_exhausted:
        return LatentResidualAnalysis(
            latent_term_justified=False,
            rejection_reason="known_parameters_not_yet_exhausted",
            max_snr=max_snr,
            state_coherence_r2=coherence_r2,
            affected_state_indices=affected_indices,
            required_derivative_units=req_units,
            thermodynamic_class="none",
            residual_flow=residual_flow,
            missing_power=missing_power,
        )

    if max_snr < min_snr_threshold:
        return LatentResidualAnalysis(
            latent_term_justified=False,
            rejection_reason="residual_within_noise_floor",
            max_snr=max_snr,
            state_coherence_r2=coherence_r2,
            affected_state_indices=affected_indices,
            required_derivative_units=req_units,
            thermodynamic_class="none",
            residual_flow=residual_flow,
            missing_power=missing_power,
        )

    if coherence_r2 < min_coherence_r2:
        return LatentResidualAnalysis(
            latent_term_justified=False,
            rejection_reason="residual_is_uncorrelated_noise",
            max_snr=max_snr,
            state_coherence_r2=coherence_r2,
            affected_state_indices=affected_indices,
            required_derivative_units=req_units,
            thermodynamic_class="none",
            residual_flow=residual_flow,
            missing_power=missing_power,
        )

    return LatentResidualAnalysis(
        latent_term_justified=True,
        rejection_reason=None,
        max_snr=max_snr,
        state_coherence_r2=coherence_r2,
        affected_state_indices=affected_indices,
        required_derivative_units=req_units,
        thermodynamic_class=thermo_class,
        residual_flow=residual_flow,
        missing_power=missing_power,
    )


def audit_law_brick(
    brick: LawBrick,
    sample_state: torch.Tensor | None = None,
    sample_params: torch.Tensor | None = None,
    derivative_units: Sequence[Unit] | None = None,
    tol: float = 1e-4,
) -> AuditCertificate:
    """Run Dimensional, Thermodynamic (1st & 2nd Law), and Homeostatic Viability checks on `brick`."""
    state = sample_state if sample_state is not None else brick.default_state(batch_size=8)
    params = sample_params if sample_params is not None else brick.default_params(batch_size=state.shape[0])

    dim_res = verify_dimensional_consistency(brick, derivative_units=derivative_units)
    thermo_res = verify_thermodynamic_laws(brick, state, params, tol=tol)
    viability_res, viable_mask = verify_homeostatic_viability(brick, state, min_viable_fraction=1.0)
    viable_frac = float(viable_mask.to(torch.float32).mean().item())

    results = (dim_res, thermo_res, viability_res)
    passed_all = all(r.passed for r in results)

    return AuditCertificate(
        target_name=brick.name,
        passed_all=passed_all,
        results=results,
        viable_fraction=viable_frac,
    )
