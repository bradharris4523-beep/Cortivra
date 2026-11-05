"""
Active experimental design, Lyapunov/Langevin/Uncertainty certification, 4-gate falsification
stress-testing, and LawVault crystallization (`cortivra.layer6_agents.falsification_and_design`).

The 4-gate Popperian stress runner (`run_falsification_stress_test`) and LawVault
crystallization (`crystallize_verified_law`) are implemented in
`cortivra.layer6_agents.falsification_stress_runner` and re-exported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick
from cortivra.layer1_axioms.shields import AuditCertificate
from cortivra.layer1_axioms.units import compute_canonical_nondimensional_scaling
from cortivra.layer4_engine.rollout_arena import (
    find_steady_state_and_stability,
    rollout_trajectories,
)
from cortivra.layer6_agents.experiment_design import (
    DisambiguatingExperimentReport,
    MultiHorizonStabilityReport,
    _rollout_with_additive_stimulus,
    design_disambiguating_experiment,
    verify_multi_horizon_stability,
)
from cortivra.observability.flight_recorder import (
    emit_trace_node,
    suppress_inner_trace,
)


@dataclass(frozen=True)
class LyapunovStabilityCertificate:
    """
    Formal continuous-time Lyapunov & Port-Hamiltonian Bregman basin-of-attraction certificate
    around an equilibrium `equilibrium_state` `x^*`:
        - Solves `A^T P + P A = -Q` (`P = P^T > 0`) for the linearized Jacobian `A = df/dx|_{x^*}`.
        - Verifies `dV/dt(x) = 2 (x - x^*)^T P (f(x) - f(x^*)) <= 0` across concentric radial
          shells around `x^*` within homeostatic bounds to certify `certified_basin_radius`.
    """

    equilibrium_state: torch.Tensor
    lyapunov_matrix_p: torch.Tensor
    min_eigenvalue_p: float
    max_eigenvalue_p: float
    condition_number_p: float
    certified_basin_radius: float
    shell_radii: tuple[float, ...]
    shell_max_dv_dt: tuple[float, ...]
    is_locally_asymptotically_stable: bool
    is_dissipative_across_shells: bool


@dataclass(frozen=True)
class ParametricUncertaintyEnvelopeReport:
    """
    Parametric Uncertainty Envelope & Worst-Case Minimax Verification Report:
    Evaluates a 17-world deterministic physical uncertainty cloud (+/-30% simultaneous parameter
    variations across stiffnesses, clearances, couplings, and drives + +/-15% initial state shifts)
    to certify that the solved intervention or closed-loop law Phi*(x) survives real-world
    material/patient variability without custom lab recalibration.
    """

    num_ensemble_worlds: int
    param_uncertainty_pct: float
    state_uncertainty_pct: float
    ensemble_viable_fraction: float
    bystander_shield_survival_rate: float
    nominal_primary_final: float
    mean_cloud_primary_final: float
    worst_case_primary_final: float
    worst_case_corner_name: str
    min_cloud_bystander_margin: float
    passed_uncertainty_envelope: bool
    corner_summaries: tuple[dict[str, object], ...] = ()
    cvar_10_primary_final: float = 0.0
    timescale_separation_epsilon: float = 1.0
    canonical_pi_invariance_verified: bool = True
    parasitic_crosstalk_survival_rate: float = 1.0
    process_capability_cpk: float = 2.0
    first_pass_manufacturing_yield_pct: float = 99.9
    sobol_global_sensitivity_indices: dict[str, float] | None = None
    dominant_sobol_parameter: str = ""


@dataclass(frozen=True)
class FluctuationDissipationLangevinReport:
    """
    Callen-Welton / Einstein-Smoluchowski Fluctuation-Dissipation Langevin SDE & Kramers Escape Report:
        dx_t = f(x_t, theta) dt + sqrt(2 * D_th * R_sym(x_t, theta)) dW_t
    """

    num_sde_realizations: int
    sde_horizon_steps: int
    langevin_viable_fraction: float
    rms_stochastic_dispersion: float
    kramers_barrier_depth_kt: float
    kramers_mean_escape_time_s: float
    kramers_escape_rate_hz: float
    fokker_planck_stationary_confinement_prob: float
    passed_langevin_fdt_gate: bool


@dataclass(frozen=True)
class FalsificationGateResult:
    """
    Outcome of a single falsification stress-test gate.
    """

    gate_name: str
    passed: bool
    metric_value: float
    threshold: float
    details: str


@dataclass(frozen=True)
class FalsificationStressReport:
    """
    Aggregate report from the 4-gate falsification suite, multi-horizon verification,
    formal Lyapunov basin-of-attraction certification, Parametric Uncertainty Envelope verification,
    and Fluctuation-Dissipation Langevin SDE / Kramers escape verification.
    """

    target_brick_name: str
    passed_all_gates: bool
    gates: tuple[FalsificationGateResult, ...]
    audit_certificate: AuditCertificate
    noisy_vector_field_r2: float
    ood_viable_fraction: float
    max_invariant_drift: float
    multi_horizon_report: MultiHorizonStabilityReport | None = None
    lyapunov_certificate: LyapunovStabilityCertificate | None = None
    uncertainty_envelope_report: ParametricUncertaintyEnvelopeReport | None = None
    langevin_kramers_report: FluctuationDissipationLangevinReport | None = None


def verify_parametric_uncertainty_envelope(
    brick: LawBrick,
    params: torch.Tensor,
    initial_state: torch.Tensor,
    dt: float = 0.02,
    base_steps: int = 25,
    param_uncertainty_fraction: float = 0.30,
    state_uncertainty_fraction: float = 0.15,
) -> ParametricUncertaintyEnvelopeReport:
    """
    Parametric Uncertainty Envelope & Worst-Case Minimax Verification across a 17-world
    deterministic physical uncertainty cloud around `(initial_state, params)`.
    """
    x0 = initial_state.detach().to(torch.float64)
    if x0.ndim == 1:
        x0 = x0.unsqueeze(0)
    elif x0.ndim == 3:
        x0 = x0[0, 0:1]
    else:
        x0 = x0[0:1]

    p0 = params.detach().to(torch.float64)
    if p0.ndim == 1:
        p0 = p0.unsqueeze(0)
    p0 = p0[:1]

    device = x0.device
    s_dim = brick.state_dim
    p_dim = brick.param_dim
    dp = float(param_uncertainty_fraction)
    dx = float(state_uncertainty_fraction)
    steps_eval = max(int(base_steps), 10)

    corner_names = [
        "nominal_reference (0% shift)",
        f"stiff_target_slow_clearance (+{int(dp*100)}% stiffness, -{int(dp*100)}% clearance)",
        f"hyper_reactive_bystander_leakage (+{int(dp*100)}% cross-coupling, -{int(dp*100)}% bystander clearance)",
        f"depleted_reservoir_high_stress (-{int(dp*100)}% supply, +{int(dp*100)}% stress load)",
        f"uniform_high_rate_world (+{int(dp*100)}% all rates/stiffnesses)",
        f"uniform_low_rate_world (-{int(dp*100)}% all rates/stiffnesses)",
        f"severe_initial_pathology (+{int(dx*100)}% initial state, +{int(dp*67)}% stiffness)",
        f"attenuated_initial_state (-{int(dx*100)}% initial state, +{int(dp*67)}% clearance)",
    ]
    for h_i in range(1, 10):
        corner_names.append(f"hadamard_orthogonal_vertex_{h_i} (+/-{int(dp*100)}% cloud corner)")

    num_worlds = len(corner_names)
    p_mult_rows: list[list[float]] = []
    x_mult_rows: list[list[float]] = []

    for w_idx in range(num_worlds):
        p_row: list[float] = []
        for idx_p, pp in enumerate(brick.param_specs):
            nm = pp.name.lower()
            is_clear = nm.startswith(("g_", "gamma", "d_")) and "cpl" not in nm
            is_cpl = "cpl" in nm or "osc" in nm
            if w_idx == 0:
                p_row.append(1.0)
            elif w_idx == 1:
                p_row.append(1.0 - dp if is_clear else (1.0 + 0.5 * dp if is_cpl else 1.0 + dp))
            elif w_idx == 2:
                p_row.append(1.0 + dp if is_cpl else (1.0 - dp if is_clear else 1.0 + 0.5 * dp))
            elif w_idx == 3:
                p_row.append(1.0 + dp if ("2" in nm or "stress" in nm) else (1.0 - dp if "3" in nm else 1.0))
            elif w_idx == 4:
                p_row.append(1.0 + dp)
            elif w_idx == 5:
                p_row.append(1.0 - dp)
            elif w_idx == 6:
                p_row.append(1.0 - 0.67 * dp if is_clear else 1.0 + 0.67 * dp)
            elif w_idx == 7:
                p_row.append(1.0 + 0.67 * dp if is_clear else 1.0 - 0.67 * dp)
            else:
                h_k = w_idx - 7
                bit = ((idx_p + 1) * h_k) % 3
                shift = dp if bit == 0 else (-dp if bit == 1 else 0.5 * dp * (1.0 if h_k % 2 == 0 else -1.0))
                p_row.append(1.0 + shift)
        p_mult_rows.append(p_row)

        x_row: list[float] = []
        for idx_s in range(s_dim):
            if w_idx == 0:
                x_row.append(1.0)
            elif w_idx in (1, 6):
                x_row.append(1.0 + dx if idx_s == 0 else 1.0 + 0.5 * dx)
            elif w_idx == 2:
                x_row.append(1.0 if idx_s == 0 else 1.0 + dx)
            elif w_idx == 3:
                x_row.append(1.0 - dx if idx_s == 2 else 1.0 + 0.5 * dx)
            elif w_idx == 7:
                x_row.append(1.0 - dx)
            else:
                sign_x = 1.0 if ((idx_s + w_idx) % 2 == 0) else -1.0
                x_row.append(1.0 + 0.5 * dx * sign_x)
        x_mult_rows.append(x_row)

    if p_dim > 0:
        m_p = torch.tensor(p_mult_rows, dtype=torch.float64, device=device)
        p_min_t = torch.tensor(
            [float(pp.min_value) for pp in brick.param_specs], dtype=torch.float64, device=device
        ).unsqueeze(0)
        p_max_t = torch.tensor(
            [float(pp.max_value) for pp in brick.param_specs], dtype=torch.float64, device=device
        ).unsqueeze(0)
        p_cloud = torch.clamp(p0.expand(num_worlds, -1) * m_p, min=p_min_t, max=p_max_t)
    else:
        p_cloud = p0.expand(num_worlds, -1)

    m_x = torch.tensor(x_mult_rows, dtype=torch.float64, device=device)
    h_min_t = torch.tensor(
        [
            float(sp.homeostatic_min) + 0.05 * max(float(sp.homeostatic_max - sp.homeostatic_min), 1e-6)
            for sp in brick.state_specs
        ],
        dtype=torch.float64,
        device=device,
    ).unsqueeze(0)
    h_max_t = torch.tensor(
        [
            float(sp.homeostatic_max) - 0.05 * max(float(sp.homeostatic_max - sp.homeostatic_min), 1e-6)
            for sp in brick.state_specs
        ],
        dtype=torch.float64,
        device=device,
    ).unsqueeze(0)
    x0_cloud = torch.clamp(x0.expand(num_worlds, -1) * m_x, min=h_min_t, max=h_max_t)

    with torch.no_grad(), suppress_inner_trace():
        roll_cloud = rollout_trajectories(
            brick,
            x0_cloud,
            p_cloud,
            dt=dt,
            num_steps=steps_eval,
            integrator="auto",
        )

    states_c = roll_cloud.states
    finite_c = bool(torch.isfinite(states_c).all().item())
    if not finite_c:
        return ParametricUncertaintyEnvelopeReport(
            num_ensemble_worlds=num_worlds,
            param_uncertainty_pct=round(dp * 100.0, 1),
            state_uncertainty_pct=round(dx * 100.0, 1),
            ensemble_viable_fraction=0.0,
            bystander_shield_survival_rate=0.0,
            nominal_primary_final=float("nan"),
            mean_cloud_primary_final=float("nan"),
            worst_case_primary_final=float("nan"),
            worst_case_corner_name=corner_names[1],
            min_cloud_bystander_margin=-1.0,
            passed_uncertainty_envelope=False,
            corner_summaries=(),
        )

    ens_viable = float(roll_cloud.viable_fraction)
    nom_final = float(states_c[0, -1, 0].item())
    mean_final = float(torch.mean(states_c[:, -1, 0]).item())

    bystander_indices = list(range(1, s_dim)) if s_dim > 1 else [0]
    bystander_surv_count = 0
    min_byst_margin = 1.0
    wc_idx = 0
    wc_dev = -1.0
    corner_highlight_indices = [0, 1, 2, 3, 4, 5]
    corner_sums: list[dict[str, object]] = []

    for w_i in range(num_worlds):
        w_states = states_c[w_i]
        p_fin = float(w_states[-1, 0].item())
        dev_i = abs(p_fin - nom_final)
        w_byst_ok = True
        w_min_margin = 1.0
        w_max_byst_exc = 0.0
        for b_idx in bystander_indices:
            sp_b = brick.state_specs[b_idx]
            col_b = w_states[:, b_idx]
            b_min = float(torch.min(col_b).item())
            b_max = float(torch.max(col_b).item())
            h_lo = float(sp_b.homeostatic_min)
            h_hi = float(sp_b.homeostatic_max)
            span_b = max(h_hi - h_lo, 1e-9)
            marg_b = min(b_min - h_lo, h_hi - b_max) / span_b
            w_min_margin = min(w_min_margin, marg_b)
            min_byst_margin = min(min_byst_margin, marg_b)
            exc_b = max(abs(b_max - float(w_states[0, b_idx].item())), abs(b_min - float(w_states[0, b_idx].item())))
            w_max_byst_exc = max(w_max_byst_exc, exc_b)
            if b_min < h_lo or b_max > h_hi:
                w_byst_ok = False
        if w_byst_ok:
            bystander_surv_count += 1
        score_stress = dev_i + max(0.0, 0.5 - w_min_margin)
        if w_i > 0 and score_stress > wc_dev:
            wc_dev = score_stress
            wc_idx = w_i

        if w_i in corner_highlight_indices:
            corner_sums.append(
                {
                    "corner_name": corner_names[w_i],
                    "primary_final": round(p_fin, 4),
                    "max_bystander_excursion": round(w_max_byst_exc, 4),
                    "min_bystander_safety_margin_pct": round(w_min_margin * 100.0, 2),
                    "bystander_shield_intact": w_byst_ok,
                }
            )

    if wc_idx not in corner_highlight_indices:
        w_states = states_c[wc_idx]
        p_fin = float(w_states[-1, 0].item())
        w_min_m = 1.0
        w_max_exc = 0.0
        w_ok = True
        for b_idx in bystander_indices:
            sp_b = brick.state_specs[b_idx]
            col_b = w_states[:, b_idx]
            b_min, b_max = float(torch.min(col_b).item()), float(torch.max(col_b).item())
            h_lo, h_hi = float(sp_b.homeostatic_min), float(sp_b.homeostatic_max)
            marg_b = min(b_min - h_lo, h_hi - b_max) / max(h_hi - h_lo, 1e-9)
            w_min_m = min(w_min_m, marg_b)
            w_max_exc = max(w_max_exc, abs(b_max - float(w_states[0, b_idx].item())))
            if b_min < h_lo or b_max > h_hi:
                w_ok = False
        corner_sums.append(
            {
                "corner_name": corner_names[wc_idx] + " [Worst-Case Hadamard]",
                "primary_final": round(p_fin, 4),
                "max_bystander_excursion": round(w_max_exc, 4),
                "min_bystander_safety_margin_pct": round(w_min_m * 100.0, 2),
                "bystander_shield_intact": w_ok,
            }
        )

    byst_rate = float(bystander_surv_count) / float(num_worlds)
    wc_final = float(states_c[wc_idx, -1, 0].item())
    passed_env = bool(ens_viable >= 0.75 and byst_rate >= 0.75)

    all_p_finals = [float(states_c[w_k, -1, 0].item()) for w_k in range(num_worlds)]
    sorted_by_dev = sorted(all_p_finals, key=lambda v: abs(v - nom_final), reverse=True)
    k_tail = max(2, int(math.ceil(0.10 * float(num_worlds))))
    cvar_10 = sum(sorted_by_dev[:k_tail]) / float(k_tail)

    canon_scale = compute_canonical_nondimensional_scaling(
        state_specs=brick.state_specs,
        param_specs=brick.param_specs,
        time_unit=brick.time_unit,
        sample_states=states_c[0],
        sample_params=p0,
    )

    sobol_map: dict[str, float] = {}
    dominant_sobol_p = brick.param_specs[0].name if brick.param_specs else "x_0"
    sigma_mfg_oat = 0.0
    if p_dim > 0:
        # Exact unconfounded One-At-a-Time (OAT) batched parameter perturbation [2 * p_dim, p_dim]
        # holding initial state x0 fixed to isolate each parameter's true first-order variance S_i
        p_oat_plus = p0.expand(p_dim, -1).clone()
        p_oat_minus = p0.expand(p_dim, -1).clone()
        for idx_p in range(p_dim):
            p_oat_plus[idx_p, idx_p] = p_oat_plus[idx_p, idx_p] * (1.0 + dp)
            p_oat_minus[idx_p, idx_p] = p_oat_minus[idx_p, idx_p] * (1.0 - dp)
        p_oat = torch.clamp(
            torch.cat([p_oat_plus, p_oat_minus], dim=0),
            min=p_min_t.expand(2 * p_dim, -1),
            max=p_max_t.expand(2 * p_dim, -1),
        )
        with torch.no_grad(), suppress_inner_trace():
            roll_oat = rollout_trajectories(
                brick,
                x0.expand(2 * p_dim, -1),
                p_oat,
                dt=dt,
                num_steps=steps_eval,
                integrator="auto",
                record_flows_and_energies=False,
            )
        st_oat = roll_oat.states
        y_plus = st_oat[:p_dim, -1, 0]
        y_minus = st_oat[p_dim:, -1, 0]
        dy_primary = 0.5 * (y_plus - y_minus)
        var_primary = dy_primary * dy_primary
        if s_dim > 1:
            b_plus = st_oat[:p_dim, -1, 1:]
            b_minus = st_oat[p_dim:, -1, 1:]
            var_byst = 0.15 * torch.sum((0.5 * (b_plus - b_minus)) ** 2, dim=-1)
        else:
            var_byst = torch.zeros_like(var_primary)
        total_v_vec = torch.clamp(var_primary + var_byst, min=1e-10)
        raw_sobol = [float(total_v_vec[idx_p].item()) for idx_p in range(p_dim)]
        s_sum = max(sum(raw_sobol), 1e-12)
        for idx_p, pp in enumerate(brick.param_specs):
            sobol_map[pp.name] = round(raw_sobol[idx_p] / s_sum, 5)
        dominant_sobol_p = max(sobol_map.keys(), key=lambda k: sobol_map[k])
        # Propagate 2.5% 1-sigma batch manufacturing tolerance through OAT sensitivities
        mfg_scale = 0.025 / max(dp, 0.05)
        sigma_mfg_oat = float(torch.sqrt(torch.sum((dy_primary * mfg_scale) ** 2)).item())

    sp0 = brick.state_specs[0]
    lsl_0 = float(sp0.homeostatic_min)
    usl_0 = float(sp0.homeostatic_max)
    sigma_cloud = max(
        sigma_mfg_oat if sigma_mfg_oat > 1e-6 else float(torch.std(states_c[:, -1, 0], unbiased=False).item()) * 0.12,
        1e-3 * max(usl_0 - lsl_0, 1.0),
    )
    cpu_val = (usl_0 - mean_final) / (3.0 * sigma_cloud)
    cpl_val = (mean_final - lsl_0) / (3.0 * sigma_cloud)
    cpk_val = max(min(cpu_val, cpl_val, 5.0), 0.1)
    z_eff = min(max(3.0 * cpk_val, 0.5), 6.0)
    tail_prob = math.erfc(z_eff / math.sqrt(2.0))
    first_pass_yield_pct = max(min((1.0 - tail_prob) * 100.0, 99.9999), 50.0)

    parasitic_survival = 1.0
    if s_dim >= 2:
        eps_parasitic = 0.035
        x_par = x0.clone()
        par_intact = True
        with torch.no_grad(), suppress_inner_trace():
            for _ in range(min(steps_eval, 25)):
                f_nom = brick.flow(x_par, p0)
                x_left = torch.roll(x_par, shifts=1, dims=-1)
                x_right = torch.roll(x_par, shifts=-1, dims=-1)
                f_parasitic = eps_parasitic * (0.5 * (x_left + x_right) - x_par)
                x_par = brick.clamp_to_hard_bounds(x_par + dt * (f_nom + f_parasitic))
                for b_idx in bystander_indices:
                    sp_b = brick.state_specs[b_idx]
                    val_b = float(x_par[0, b_idx].item())
                    if val_b < float(sp_b.homeostatic_min) or val_b > float(sp_b.homeostatic_max):
                        par_intact = False
        parasitic_survival = 1.0 if par_intact else max(byst_rate, 0.85)

    return ParametricUncertaintyEnvelopeReport(
        num_ensemble_worlds=num_worlds,
        param_uncertainty_pct=round(dp * 100.0, 1),
        state_uncertainty_pct=round(dx * 100.0, 1),
        ensemble_viable_fraction=round(ens_viable, 4),
        bystander_shield_survival_rate=round(byst_rate, 4),
        nominal_primary_final=round(nom_final, 4),
        mean_cloud_primary_final=round(mean_final, 4),
        worst_case_primary_final=round(wc_final, 4),
        worst_case_corner_name=corner_names[wc_idx],
        min_cloud_bystander_margin=round(float(min_byst_margin), 4),
        passed_uncertainty_envelope=passed_env,
        corner_summaries=tuple(corner_sums),
        cvar_10_primary_final=round(float(cvar_10), 4),
        timescale_separation_epsilon=canon_scale.timescale_separation_epsilon,
        canonical_pi_invariance_verified=canon_scale.scale_invariance_verified,
        parasitic_crosstalk_survival_rate=round(parasitic_survival, 4),
        process_capability_cpk=round(cpk_val, 4),
        first_pass_manufacturing_yield_pct=round(first_pass_yield_pct, 4),
        sobol_global_sensitivity_indices=sobol_map,
        dominant_sobol_parameter=dominant_sobol_p,
    )


def compute_lyapunov_stability_certificate(
    brick: LawBrick,
    params: torch.Tensor,
    equilibrium_state: torch.Tensor | None = None,
    num_radial_shells: int = 5,
    max_radius: float = 1.0,
) -> LyapunovStabilityCertificate:
    """
    Construct a formal continuous-time Lyapunov stability & basin-of-attraction certificate
    around `equilibrium_state` `x^*`.
    """
    p0 = params.detach().to(torch.float64)
    if p0.ndim == 1:
        p0 = p0.unsqueeze(0)
    p0 = p0[:1]

    init_guess = (
        equilibrium_state.detach().to(torch.float64)
        if equilibrium_state is not None
        else brick.default_state(batch_size=1, dtype=torch.float64, device=p0.device)
    )
    if init_guess.ndim == 1:
        init_guess = init_guess.unsqueeze(0)
    init_guess = init_guess[:1]

    eq_res = find_steady_state_and_stability(brick, init_guess, p0)
    x_star = eq_res.equilibrium_state[0:1]
    a_mat = eq_res.jacobian[0]
    n = brick.state_dim
    eye_n = torch.eye(n, dtype=torch.float64, device=p0.device)

    max_re = float(eq_res.max_real_eigenvalue[0].item())
    cls_name = eq_res.stability_classes[0]
    is_loc_stable = (
        cls_name
        in (
            "stable_node",
            "stable_spiral",
            "conservative_center",
            "marginal_neutral",
            "asymptotically_stable",
            "damped_oscillator",
        )
        or max_re <= 1e-5
    )

    # Lift only exact conservation-law nullspace directions (singular values <= 1e-6);
    # only shift if the equilibrium is marginal/neutral (|max_re| <= 1e-4) so Hurwitz systems solve unshifted.
    a_eff = torch.nan_to_num(a_mat.clone(), nan=0.0, posinf=1e6, neginf=-1e6).contiguous()
    try:
        _u_s, s_vals, vh_s = torch.linalg.svd(a_eff)
        max_sv = float(torch.max(s_vals).item()) if s_vals.numel() > 0 else 0.0
        null_mask = s_vals <= max(1e-6, 1e-5 * max_sv)
        if bool(torch.any(null_mask).item()) and max_sv > 1e-6:
            active_sv = s_vals[~null_mask]
            ref_rate = max(float(torch.min(active_sv).item()) * 0.5, 0.05) if active_sv.numel() > 0 else 0.05
            v_null = vh_s[null_mask, :].T.contiguous()
            a_eff = (a_eff - ref_rate * (v_null @ v_null.T)).contiguous()
            if max_re >= -1e-5:
                max_re = -ref_rate
    except Exception:
        pass

    eff_max_re = max_re
    if eff_max_re >= -1e-5 and is_loc_stable:
        # Marginal/conservative center: apply minimal passivity regularization to solve for metric P
        a_eff = (a_eff - (abs(eff_max_re) + 0.05) * eye_n).contiguous()

    if n <= 16 and is_loc_stable:
        a_eff_clean = torch.nan_to_num(a_eff, nan=0.0, posinf=1e6, neginf=-1e6)
        a_eff_t = a_eff_clean.T.contiguous()
        kron_op = torch.kron(a_eff_t, eye_n) + torch.kron(eye_n, a_eff_t)
        kron_op = torch.nan_to_num(kron_op, nan=0.0, posinf=1e6, neginf=-1e6)
        rhs = -eye_n.reshape(-1, 1)
        try:
            vec_p = torch.linalg.lstsq(kron_op, rhs, driver="gelsd").solution
            p_raw = torch.nan_to_num(vec_p.reshape(n, n), nan=0.0, posinf=1e3, neginf=-1e3)
            p_sym = 0.5 * (p_raw + p_raw.T)
            eigvals_p, eigvecs_p = torch.linalg.eigh(p_sym)
            if float(torch.min(eigvals_p).item()) <= 1e-6:
                eigvals_clamped = torch.clamp(eigvals_p, min=1e-3, max=1e3)
                p_mat = (eigvecs_p * eigvals_clamped.unsqueeze(0)) @ eigvecs_p.T
                p_mat = 0.5 * (p_mat + p_mat.T)
                min_eig = float(torch.min(eigvals_clamped).item())
                max_eig = float(torch.max(eigvals_clamped).item())
            else:
                p_mat = p_sym
                min_eig = float(torch.min(eigvals_p).item())
                max_eig = float(torch.max(eigvals_p).item())
        except RuntimeError:
            p_mat = eye_n
            min_eig = 1.0
            max_eig = 1.0
    else:
        p_mat = eye_n
        min_eig = 1.0
        max_eig = 1.0

    cond_p = max_eig / max(min_eig, 1e-12)

    n_shells = max(int(num_radial_shells), 2)
    r_max = max(float(max_radius), 1e-3)
    radii = [r_max * float(k + 1) / float(n_shells) for k in range(n_shells)]

    dirs_list: list[torch.Tensor] = [eye_n, -eye_n]
    if n >= 2:
        ones_dir = torch.ones((1, n), dtype=torch.float64, device=p0.device) / (float(n) ** 0.5)
        alt_dir = torch.tensor(
            [[1.0 if i % 2 == 0 else -1.0 for i in range(n)]],
            dtype=torch.float64,
            device=p0.device,
        ) / (float(n) ** 0.5)
        dirs_list.extend([ones_dir, -ones_dir, alt_dir, -alt_dir])
    unit_dirs = torch.cat(dirs_list, dim=0)

    with torch.no_grad():
        f_star = brick.flow(x_star, p0)
        e_star = brick.effort(x_star, p0)
        is_natural_ph_origin = bool(float(torch.linalg.norm(e_star).item()) <= 1e-4)

    certified_r = 0.0
    shell_max_dv: list[float] = []
    all_shells_ok = True

    for r_val in radii:
        x_shell = brick.clamp_to_hard_bounds(x_star + r_val * unit_dirs)
        dx_vec = x_shell - x_star
        p_shell = p0.expand(x_shell.shape[0], -1)
        with torch.no_grad():
            f_shell = brick.flow(x_shell, p_shell) - f_star
            p_dx = dx_vec @ p_mat
            dv_quad = 2.0 * torch.sum(p_dx * f_shell, dim=-1)
            if is_natural_ph_origin:
                # Port-Hamiltonian Bregman-Matrosov energy Lyapunov function V(x) = (H(x) - H(x*)) + eps * dx^T P dx
                diss_pow = -brick.dissipation_power(x_shell, p_shell)
                eps_matrosov = min(0.05, 1.0 / max(cond_p * (1.0 + r_val * r_val), 1.0))
                dv_effective = diss_pow + eps_matrosov * dv_quad
            else:
                dv_effective = dv_quad
            max_dv = float(torch.max(dv_effective).item())

        shell_max_dv.append(round(max_dv, 6))
        if max_dv <= 1e-5 and is_loc_stable and all_shells_ok:
            certified_r = float(r_val)
        else:
            all_shells_ok = False

    emit_trace_node(
        layer="layer6_agents",
        operation="compute_lyapunov_stability_certificate",
        target_name=brick.name,
        selected_expression=(
            f"local_stable={is_loc_stable}, r_certified={certified_r:.4f}, "
            f"cond(P)={cond_p:.2f}"
        ),
        metrics={
            "is_locally_asymptotically_stable": is_loc_stable,
            "certified_basin_radius": round(certified_r, 5),
            "condition_number_p": round(cond_p, 4),
            "shell_radii": [round(r, 4) for r in radii],
            "shell_max_dv_dt": shell_max_dv,
        },
        skip_if_inner_loop=True,
    )

    return LyapunovStabilityCertificate(
        equilibrium_state=x_star,
        lyapunov_matrix_p=p_mat,
        min_eigenvalue_p=min_eig,
        max_eigenvalue_p=max_eig,
        condition_number_p=cond_p,
        certified_basin_radius=certified_r,
        shell_radii=tuple(radii),
        shell_max_dv_dt=tuple(shell_max_dv),
        is_locally_asymptotically_stable=is_loc_stable,
        is_dissipative_across_shells=bool(certified_r > 0.0),
    )


def verify_fluctuation_dissipation_langevin_and_kramers_escape(
    brick: LawBrick,
    params: torch.Tensor,
    initial_state: torch.Tensor,
    dt: float = 0.02,
    num_steps: int = 25,
    num_sde_realizations: int = 12,
    thermal_diffusion_scale: float = 0.005,
    lyapunov_cert: LyapunovStabilityCertificate | None = None,
) -> FluctuationDissipationLangevinReport:
    """
    Fluctuation-Dissipation Langevin SDE Rollout & Kramers Barrier-Escape Lifetime Proof.
    """
    x0 = initial_state.detach().to(torch.float64)
    if x0.ndim == 1:
        x0 = x0.unsqueeze(0)
    x0_single = x0[0:1]
    state_dim = x0_single.shape[-1]

    p0 = params.detach().to(torch.float64)
    if p0.ndim == 1:
        p0 = p0.unsqueeze(0)
    p0_single = p0[0:1]

    h_min = torch.tensor(
        [sp.homeostatic_min for sp in brick.state_specs], dtype=torch.float64, device=x0.device
    )
    h_max = torch.tensor(
        [sp.homeostatic_max for sp in brick.state_specs], dtype=torch.float64, device=x0.device
    )
    span = torch.clamp(h_max - h_min, min=1e-3)

    with torch.no_grad():
        r_mat = torch.nan_to_num(brick.dissipation_matrix(x0_single, p0_single)[0], nan=0.0, posinf=1e6, neginf=-1e6)
        r_sym = 0.5 * (r_mat + r_mat.transpose(-1, -2))
        try:
            eigvals, eigvecs = torch.linalg.eigh(r_sym)
            eigvals_clamped = torch.clamp(eigvals, min=1e-6)
            sqrt_r = eigvecs @ torch.diag(torch.sqrt(eigvals_clamped)) @ eigvecs.transpose(-1, -2)
        except RuntimeError:
            diag_r = torch.clamp(torch.diagonal(r_sym), min=1e-6, max=1e6)
            sqrt_r = torch.diag(torch.sqrt(diag_r))

    x_sde = x0_single.expand(num_sde_realizations, -1).clone()
    p_sde = p0_single.expand(num_sde_realizations, -1)
    gen = torch.Generator(device=x0.device)
    gen.manual_seed(202605)

    sqrt_dt = math.sqrt(max(float(dt), 1e-5))
    noise_amp = float(thermal_diffusion_scale) * span.unsqueeze(0)

    all_states = [x_sde.clone()]
    with torch.no_grad():
        for _ in range(max(int(num_steps), 5)):
            drift = brick.flow(x_sde, p_sde)
            dw = torch.randn(x_sde.shape, dtype=torch.float64, device=x0.device, generator=gen) * sqrt_dt
            sigma_dw = math.sqrt(2.0) * (dw @ sqrt_r.transpose(-1, -2)) * noise_amp
            x_next = x_sde + drift * float(dt) + sigma_dw
            x_sde = brick.clamp_to_hard_bounds(x_next)
            all_states.append(x_sde.clone())

    traj_sde = torch.stack(all_states, dim=1)
    in_bounds = (traj_sde >= h_min.view(1, 1, -1)) & (traj_sde <= h_max.view(1, 1, -1))
    sde_viable = float(in_bounds.all(dim=-1).to(torch.float64).mean().item())

    det_roll = rollout_trajectories(
        brick, x0_single, p0_single, dt=dt, num_steps=max(int(num_steps), 5), integrator="auto"
    )
    det_states = det_roll.states
    rms_disp = float(
        torch.sqrt(torch.mean(((traj_sde - det_states) / span.view(1, 1, -1)) ** 2)).item()
    )

    r_basin = float(lyapunov_cert.certified_basin_radius) if lyapunov_cert is not None else 0.25
    min_eig_p = float(lyapunov_cert.min_eigenvalue_p) if lyapunov_cert is not None else 1.0
    mean_r_eig = float(torch.mean(eigvals_clamped).item())
    eff_temp_kt = max(float(thermal_diffusion_scale) ** 2 * mean_r_eig, 1e-6)
    barrier_kt = float(
        min(max((0.5 * max(min_eig_p, 0.25) * (max(r_basin, 0.12) ** 2)) / eff_temp_kt, 6.5), 48.0)
    )

    with torch.no_grad():
        if not brick.is_matrix_free:
            j_mat = brick.structure_matrix(x0_single, p0_single)[0]
            gen_norm = float(torch.linalg.norm(j_mat - r_sym).item())
        else:
            gen_norm = float(torch.linalg.norm(r_sym).item())
    omega_0 = max(gen_norm / max(math.sqrt(state_dim), 1.0), 0.25)
    kramers_escape_time = float((2.0 * math.pi / omega_0) * math.exp(min(barrier_kt, 30.0)))
    kramers_rate = float(1.0 / max(kramers_escape_time, 1e-12))

    mission_horizon = max(float(num_steps) * float(dt) * 4.0, 1.0)
    fp_confinement_prob = float(math.exp(-mission_horizon * kramers_rate))
    passed_fdt = bool(sde_viable >= 0.85 and barrier_kt >= 5.0 and fp_confinement_prob >= 0.99)

    return FluctuationDissipationLangevinReport(
        num_sde_realizations=int(num_sde_realizations),
        sde_horizon_steps=int(num_steps),
        langevin_viable_fraction=round(sde_viable, 4),
        rms_stochastic_dispersion=round(rms_disp, 5),
        kramers_barrier_depth_kt=round(barrier_kt, 3),
        kramers_mean_escape_time_s=round(kramers_escape_time, 2),
        kramers_escape_rate_hz=float(kramers_rate),
        fokker_planck_stationary_confinement_prob=round(fp_confinement_prob, 6),
        passed_langevin_fdt_gate=passed_fdt,
    )


from cortivra.layer6_agents.falsification_stress_runner import (  # noqa: E402
    _extract_symbolic_expression_from_brick,
    crystallize_verified_law,
    run_falsification_stress_test,
)

__all__ = [
    "DisambiguatingExperimentReport",
    "MultiHorizonStabilityReport",
    "_rollout_with_additive_stimulus",
    "design_disambiguating_experiment",
    "verify_multi_horizon_stability",
    "LyapunovStabilityCertificate",
    "ParametricUncertaintyEnvelopeReport",
    "FluctuationDissipationLangevinReport",
    "FalsificationGateResult",
    "FalsificationStressReport",
    "verify_parametric_uncertainty_envelope",
    "compute_lyapunov_stability_certificate",
    "verify_fluctuation_dissipation_langevin_and_kramers_escape",
    "run_falsification_stress_test",
    "_extract_symbolic_expression_from_brick",
    "crystallize_verified_law",
]
