"""
Phase 3 — The Nobel Prize Blind Rediscovery Suite (`tests/test_nobel_prize_blind_rediscoveries.py`).

Proves that Cortivra's symbolic discovery, Lie-derivative conservation nullspace,
Sanathanan-Koerner rational Padé identifier, and equilibrium/bifurcation solvers
recover 5 landmark Nobel Prize differential equations and numerical parameters
from unlabeled floating-point trajectory tensors (`ch_0`, `ch_1`, `ch_2`) with
ZERO variable names, ZERO domain labels, and ZERO physics hints:

1. 1963 Nobel Prize in Physiology or Medicine (Hodgkin, Huxley & Eccles / FitzHugh-Nagumo):
   Blind recovery of the coupled cubic voltage-gating action-potential excitability ODE
   `d(ch_0)/dt = I_0 + ch_0 - (1/3)*ch_0^3 - ch_1`, `d(ch_1)/dt = a*ch_0 - b*ch_1`
   from unlabeled `[E, T, 2]` trajectories (`R^2 > 0.9999`).

2. 1977 Nobel Prize in Chemistry (Ilya Prigogine — Far-From-Equilibrium Brusselator Limit Cycle):
   Blind recovery of the cubic autocatalytic chemical clock (`ch_0^2 * ch_1`) and its
   exact stoichiometric opposing sign across `ch_0` and `ch_1`:
   `d(ch_0)/dt = 1.0 - 2.8*ch_0 + 1.0*ch_0^2*ch_1`, `d(ch_1)/dt = 1.8*ch_0 - 1.0*ch_0^2*ch_1`
   (`R^2 > 0.9999`).

3. 1965 Nobel Prize in Physiology or Medicine (Monod, Jacob & Lwoff — Allosteric Saturation & Mass Conservation):
   Blind recovery of the saturating rational Padé / Michaelis-Menten enzymatic rate law
   `Phi(ch_0) = V_max * ch_0 / (K_m + ch_0)` (`V_max = 4.0`, `K_m = 2.0`) AND the exact
   Lie-derivative conservation invariant `ch_0 + ch_1 = const` (`relative_drift < 1e-10`).

4. 2003 Nobel Prize in Physics (Ginzburg, Abrikosov & Leggett — Landau-Ginzburg Superfluid Order Parameter):
   Blind recovery of the symmetry-breaking cubic Ginzburg-Landau order-parameter field equation
   `d(ch_0)/dt = alpha * ch_0 - beta * ch_0^3` (`alpha = 1.6`, `beta = 0.8`) AND autonomous
   mapping of all 3 pitchfork equilibria (`0.0` unstable, `+-sqrt(alpha/beta) = +-sqrt(2)` stable attractors).

5. 1993 Nobel Prize in Physics (Hulse & Taylor / Einstein General Relativistic Orbital Precession):
   Blind residual discovery of the post-Newtonian General Relativistic `1 / r^4` (`inv_r^4`)
   perihelion-precession correction on top of Newtonian `1 / r^2` gravitation (`R^2 > 0.9999`).
"""

from __future__ import annotations

import math
import torch

from cortivra.layer1_axioms import (
    DIMENSIONLESS,
    ParamSpec,
    StateVarSpec,
    s,
)
from cortivra.layer3_compiler import (
    SymbolicSystemSpec,
    compile_symbolic_system,
)
from cortivra.layer4_engine import (
    calibrate_and_audit_residual,
    discover_equilibrium_landscape,
    rollout_trajectories,
)
from cortivra.layer5_discovery import (
    discover_and_compile_law,
    discover_autonomous_symbolic_system,
    discover_conservation_invariants,
    discover_rational_pade_candidates,
)


def _anonymous_state_specs(num_channels: int, lo: float = -50.0, hi: float = 50.0) -> list[StateVarSpec]:
    """
    Create completely unlabeled anonymous state variable specifications (`ch_0`, `ch_1`, ...)
    with dimensionless units so the discovery engine receives zero semantic or dimensional hints.
    """
    return [
        StateVarSpec(
            name=f"ch_{i}",
            unit=DIMENSIONLESS,
            default_value=1.0,
            homeostatic_min=lo,
            homeostatic_max=hi,
        )
        for i in range(num_channels)
    ]


def test_nobel_1963_hodgkin_huxley_fitzhugh_nagumo_action_potential_blind_rediscovery() -> None:
    """
    1963 Nobel Prize in Physiology or Medicine (Hodgkin & Huxley):
    Given only unlabeled 2-channel trajectories (`ch_0`, `ch_1`) of a spiking excitable
    membrane (`dv/dt = 0.35 + 1.0*v - 0.333333*v^3 - 1.0*w`, `dw/dt = 0.20*v - 0.16*w`),
    verify that `discover_autonomous_symbolic_system`:
      1. Discovers the exact cubic self-excitation + cubic damping term `ch_0 ** 3` on `ch_0`
         and the linear recovery coupling `ch_0`, `ch_1` on `ch_1`.
      2. Achieves `vector_field_r2 > 0.9999` and `trajectory_r2 > 0.999`.
      3. Recovers the numerical coefficients within `< 1e-2` of ground truth and passes `audit_law_brick`.
    """
    anon_specs = _anonymous_state_specs(2)
    oracle_spec = SymbolicSystemSpec(
        name="hidden_oracle_1963",
        domain="unknown",
        state_specs=anon_specs,
        param_specs=[
            ParamSpec("i_stim", DIMENSIONLESS / s, default_value=0.35),
            ParamSpec("c_v", DIMENSIONLESS / s, default_value=1.0),
            ParamSpec("c_v3", DIMENSIONLESS / s, default_value=1.0 / 3.0),
            ParamSpec("c_w", DIMENSIONLESS / s, default_value=1.0),
            ParamSpec("eps_v", DIMENSIONLESS / s, default_value=0.20),
            ParamSpec("eps_w", DIMENSIONLESS / s, default_value=0.16),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.0",
        external_drive_exprs=[
            "i_stim + c_v * ch_0 - c_v3 * (ch_0^3) - c_w * ch_1",
            "eps_v * ch_0 - eps_w * ch_1",
        ],
    )
    oracle_brick = compile_symbolic_system(oracle_spec)
    x0_ens = torch.tensor(
        [
            [-1.2, -0.5],
            [1.4, -0.3],
            [0.2, 0.8],
        ],
        dtype=torch.float64,
    )
    p_oracle = oracle_brick.default_params(batch_size=3, dtype=torch.float64)
    dt = 0.04
    gt_roll = rollout_trajectories(oracle_brick, x0_ens, p_oracle, dt=dt, num_steps=45, integrator="rk4")

    # Blind discovery from unlabeled [3, 46, 2] trajectories
    rep = discover_autonomous_symbolic_system(
        name="blind_nobel_1963_discovery",
        domain="unknown",
        state_specs=anon_specs,
        observed_states=gt_roll.states,
        dt=dt,
        observed_dx_dt=gt_roll.flows,
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        max_polynomial_degree=3,
        max_terms_per_state=4,
        include_rational_primitives=False,
        include_trigonometric=False,
    )

    assert rep.vector_field_r2 > 0.9999
    assert rep.trajectory_r2 > 0.999
    assert rep.audit_certificate.passed_all

    # Verify the cubic gating nonlinearity ch_0 ** 3 was discovered in channel 0
    ch0_expr = rep.per_state_expressions["ch_0"]
    ch1_expr = rep.per_state_expressions["ch_1"]
    assert "ch_0 ** 3" in ch0_expr or "ch_0^3" in ch0_expr
    assert "ch_1" in ch0_expr
    assert "ch_0" in ch1_expr and "ch_1" in ch1_expr

    # Verify exact numerical flow agreement at an unseen test state
    x_test = torch.tensor([[1.15, -0.45]], dtype=torch.float64)
    true_flow = oracle_brick.flow(x_test, oracle_brick.default_params(batch_size=1, dtype=torch.float64))
    disc_flow = rep.compiled_brick.flow(x_test, rep.calibrated_params)
    assert torch.allclose(disc_flow, true_flow, atol=1e-2)


def test_nobel_1977_prigogine_brusselator_autocatalytic_chemical_clock_blind_rediscovery() -> None:
    """
    1977 Nobel Prize in Chemistry (Ilya Prigogine — Dissipative Structures & Brusselator):
    Given only unlabeled 2-channel trajectories (`ch_0`, `ch_1`) of the far-from-equilibrium
    Brusselator chemical oscillator (`d(ch_0)/dt = 1.0 - 2.8*ch_0 + 1.0*(ch_0^2)*ch_1`,
    `d(ch_1)/dt = 1.8*ch_0 - 1.0*(ch_0^2)*ch_1`), verify that `discover_autonomous_symbolic_system`:
      1. Discovers the trimolecular autocatalytic cross-term `(ch_0 ** 2) * ch_1` in BOTH channels
         with equal and opposite stoichiometric signs (`+1.0` in `ch_0`, `-1.0` in `ch_1`).
      2. Achieves `vector_field_r2 > 0.9999` and `trajectory_r2 > 0.999`.
      3. Passes all 5 Layer 1 axiomatic shields.
    """
    anon_specs = _anonymous_state_specs(2, lo=0.0, hi=50.0)
    oracle_spec = SymbolicSystemSpec(
        name="hidden_oracle_1977_brusselator",
        domain="unknown",
        state_specs=anon_specs,
        param_specs=[
            ParamSpec("a_feed", DIMENSIONLESS / s, default_value=1.0),
            ParamSpec("b_plus_1", DIMENSIONLESS / s, default_value=2.8),
            ParamSpec("b_conv", DIMENSIONLESS / s, default_value=1.8),
            ParamSpec("k_auto", DIMENSIONLESS / s, default_value=1.0),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.0",
        external_drive_exprs=[
            "a_feed - b_plus_1 * ch_0 + k_auto * (ch_0^2) * ch_1",
            "b_conv * ch_0 - k_auto * (ch_0^2) * ch_1",
        ],
    )
    oracle_brick = compile_symbolic_system(oracle_spec)
    x0_ens = torch.tensor(
        [
            [0.8, 1.9],
            [1.6, 1.1],
            [0.5, 2.6],
        ],
        dtype=torch.float64,
    )
    p_oracle = oracle_brick.default_params(batch_size=3, dtype=torch.float64)
    dt = 0.03
    gt_roll = rollout_trajectories(oracle_brick, x0_ens, p_oracle, dt=dt, num_steps=45, integrator="rk4")

    rep = discover_autonomous_symbolic_system(
        name="blind_nobel_1977_brusselator",
        domain="unknown",
        state_specs=anon_specs,
        observed_states=gt_roll.states,
        dt=dt,
        observed_dx_dt=gt_roll.flows,
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        max_polynomial_degree=3,
        max_terms_per_state=3,
        include_rational_primitives=False,
        include_trigonometric=False,
    )

    assert rep.vector_field_r2 > 0.9999
    assert rep.trajectory_r2 > 0.999
    assert rep.audit_certificate.passed_all

    ch0_expr = rep.per_state_expressions["ch_0"]
    ch1_expr = rep.per_state_expressions["ch_1"]
    assert "(ch_0 ** 2) * ch_1" in ch0_expr or "ch_0^2 * ch_1" in ch0_expr
    assert "(ch_0 ** 2) * ch_1" in ch1_expr or "ch_0^2 * ch_1" in ch1_expr

    # Check that the sum d(ch_0)/dt + d(ch_1)/dt cancels the nonlinear autocatalytic term (leaving 1.0 - 1.0*ch_0)
    x_probe = torch.tensor([[2.1, 1.7]], dtype=torch.float64)
    f_disc = rep.compiled_brick.flow(x_probe, rep.calibrated_params)
    net_rate_sum = float((f_disc[0, 0] + f_disc[0, 1]).item())
    expected_net_sum = 1.0 - 1.0 * 2.1
    assert abs(net_rate_sum - expected_net_sum) < 1e-2


def test_nobel_1965_monod_jacob_allosteric_saturation_and_mass_conservation_blind_rediscovery() -> None:
    """
    1965 Nobel Prize in Physiology or Medicine (Monod, Jacob & Lwoff — Allosteric Enzyme Regulation):
    Given unlabeled 2-channel trajectories (`ch_0`, `ch_1`) of an enzymatic conversion where
    substrate `ch_0` converts into product `ch_1` via a saturating rational rate law
    `d(ch_0)/dt = -V_max * ch_0 / (K_m + ch_0)`, `d(ch_1)/dt = +V_max * ch_0 / (K_m + ch_0)`
    with `V_max = 4.0`, `K_m = 2.0`:
      1. `discover_conservation_invariants` blindly discovers the exact stoichiometric conservation
         law `ch_0 + ch_1 = const` from the Lie-derivative nullspace (`relative_drift < 1e-10`).
      2. `discover_rational_pade_candidates` blindly recovers the rational Michaelis-Menten / Hill
         saturation law and its parameters (`V_max = 4.0`, `K_m = 2.0`) with `vector_field_r2 > 0.9999`.
    """
    anon_specs = _anonymous_state_specs(2, lo=0.0, hi=100.0)
    vmax_true = 4.0
    km_true = 2.0
    oracle_spec = SymbolicSystemSpec(
        name="hidden_oracle_1965_enzyme",
        domain="unknown",
        state_specs=anon_specs,
        param_specs=[
            ParamSpec("v_max", DIMENSIONLESS / s, default_value=vmax_true),
            ParamSpec("k_m", DIMENSIONLESS, default_value=km_true),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.0",
        external_drive_exprs=[
            "-(v_max * ch_0) / (k_m + ch_0)",
            "(v_max * ch_0) / (k_m + ch_0)",
        ],
    )
    oracle_brick = compile_symbolic_system(oracle_spec)
    x0 = torch.tensor([[6.0, 0.5]], dtype=torch.float64)
    p_oracle = oracle_brick.default_params(batch_size=1, dtype=torch.float64)
    dt = 0.05
    gt_roll = rollout_trajectories(oracle_brick, x0, p_oracle, dt=dt, num_steps=50, integrator="rk4")

    # 1. Blind Lie-derivative nullspace conservation discovery on unlabeled [51, 2] trajectory
    blank_2d = compile_symbolic_system(
        SymbolicSystemSpec(
            name="blank_2d_carrier",
            domain="unknown",
            state_specs=anon_specs,
            param_specs=[],
            time_unit=s,
            energy_unit=DIMENSIONLESS,
            hamiltonian_expr="0.0",
        )
    )
    inv_rep = discover_conservation_invariants(
        brick=blank_2d,
        states=gt_roll.states[0],
        flows=gt_roll.flows[0],
        max_degree=1,
    )
    assert inv_rep.num_invariants == 1
    conserved = inv_rep.invariants[0]
    assert conserved.relative_drift < 1e-10
    assert conserved.lie_derivative_rms < 1e-10
    # Verify equal weights on ch_0 and ch_1 (total mass conservation ch_0 + ch_1 = const)
    assert abs(conserved.term_weights["ch_0"] - conserved.term_weights["ch_1"]) < 1e-9

    # 2. Blind Sanathanan-Koerner rational Padé discovery of V_max and K_m on channel 1
    ch0_sweep = torch.linspace(0.2, 12.0, 50, dtype=torch.float64)
    ch1_sweep = 6.5 - ch0_sweep
    states_sweep = torch.stack([ch0_sweep, ch1_sweep], dim=-1)
    flows_sweep = oracle_brick.flow(states_sweep, p_oracle.expand(50, -1))
    res_ch1_only = torch.zeros_like(flows_sweep)
    res_ch1_only[:, 1] = flows_sweep[:, 1]

    pade_cands = discover_rational_pade_candidates(
        brick=blank_2d,
        states=states_sweep,
        target_residual=res_ch1_only,
        affected_state_indices=[1],
    )
    assert len(pade_cands) >= 1
    best_pade = min(pade_cands, key=lambda c: c.bic_score)
    assert best_pade.vector_field_r2 > 0.9999
    fitted_defaults = {ps.name: ps.default_value for ps in best_pade.param_specs}
    vmax_est = next(v for k, v in fitted_defaults.items() if "vmax" in k)
    kd_est = next(v for k, v in fitted_defaults.items() if "kd" in k)
    assert abs(vmax_est - vmax_true) < 0.05
    assert abs(kd_est - km_true) < 0.05


def test_nobel_2003_ginzburg_landau_superfluid_order_parameter_and_pitchfork_bifurcation() -> None:
    """
    2003 Nobel Prize in Physics (Ginzburg, Abrikosov & Leggett — Theory of Superconductors & Superfluids):
    Given unlabeled trajectories (`ch_0`) of a symmetry-breaking Landau-Ginzburg order parameter
    relaxing in a double-well free-energy potential `F(ch_0) = -(alpha/2)*ch_0^2 + (beta/4)*ch_0^4`,
    i.e., `d(ch_0)/dt = alpha * ch_0 - beta * ch_0^3` with `alpha = 1.6`, `beta = 0.8`:
      1. `discover_autonomous_symbolic_system` blindly recovers `d(ch_0)/dt = 1.6 * ch_0 - 0.8 * (ch_0 ** 3)`
         (`vector_field_r2 > 0.9999`).
      2. `discover_equilibrium_landscape` on the discovered brick autonomously locates all 3 fixed points:
         the unstable symmetric state `ch_0* = 0.0` and the two symmetry-broken superfluid attractors
         `ch_0* = +-sqrt(alpha / beta) = +-sqrt(2.0) approx +-1.41421`.
    """
    anon_specs = _anonymous_state_specs(1, lo=-10.0, hi=10.0)
    alpha_true = 1.6
    beta_true = 0.8
    oracle_spec = SymbolicSystemSpec(
        name="hidden_oracle_2003_ginzburg_landau",
        domain="unknown",
        state_specs=anon_specs,
        param_specs=[
            ParamSpec("alpha_gl", DIMENSIONLESS / s, default_value=alpha_true),
            ParamSpec("beta_gl", DIMENSIONLESS / s, default_value=beta_true),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.0",
        external_drive_exprs=["alpha_gl * ch_0 - beta_gl * (ch_0^3)"],
    )
    oracle_brick = compile_symbolic_system(oracle_spec)
    x0_ens = torch.tensor(
        [
            [0.15],
            [-0.18],
            [2.20],
            [-2.10],
        ],
        dtype=torch.float64,
    )
    p_oracle = oracle_brick.default_params(batch_size=4, dtype=torch.float64)
    dt = 0.03
    gt_roll = rollout_trajectories(oracle_brick, x0_ens, p_oracle, dt=dt, num_steps=40, integrator="rk4")

    rep = discover_autonomous_symbolic_system(
        name="blind_nobel_2003_ginzburg_landau",
        domain="unknown",
        state_specs=anon_specs,
        observed_states=gt_roll.states,
        dt=dt,
        observed_dx_dt=gt_roll.flows,
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        max_polynomial_degree=3,
        max_terms_per_state=2,
        include_rational_primitives=False,
        include_trigonometric=False,
    )

    assert rep.vector_field_r2 > 0.9999
    assert rep.trajectory_r2 > 0.999
    assert rep.audit_certificate.passed_all
    assert "ch_0 ** 3" in rep.per_state_expressions["ch_0"] or "ch_0^3" in rep.per_state_expressions["ch_0"]

    # Map the equilibrium landscape of the discovered brick across [-2.5, 2.5]
    eq_rep = discover_equilibrium_landscape(
        brick=rep.compiled_brick,
        params=rep.calibrated_params,
        num_probes=32,
        probe_span=2.5,
    )
    eq_vals = sorted([float(v[0].item()) for v in eq_rep.distinct_equilibria])
    assert len(eq_vals) == 3
    expected_root = math.sqrt(alpha_true / beta_true)  # sqrt(2.0) = 1.41421356
    assert abs(eq_vals[0] - (-expected_root)) < 1e-2
    assert abs(eq_vals[1] - 0.0) < 1e-2
    assert abs(eq_vals[2] - expected_root) < 1e-2
    assert len(eq_rep.stable_attractor_indices) == 2
    assert len(eq_rep.unstable_fixed_point_indices) == 1


def test_nobel_1993_hulse_taylor_einstein_general_relativistic_orbital_precession_blind_rediscovery() -> None:
    """
    1993 Nobel Prize in Physics (Hulse & Taylor — Gravitational Relativistic Binary Pulsar Precession):
    In Binet radial orbital variables (`ch_0 = u = 1/r`, `ch_1 = du/dphi`), Newtonian gravitation
    is a pure harmonic oscillator `d(ch_0)/dphi = ch_1`, `d(ch_1)/dphi = mu_0 - ch_0` (closed ellipse),
    whereas Einstein's General Relativity adds the cubic post-Newtonian attraction term
    `Delta_f = +3 * (G M / c^2) * ch_0^2 = +alpha_GR * (ch_0 ** 2)` (corresponding to the `1/r^4`
    radial force perturbation) that causes periastron/perihelion precession.
    Given only unlabeled trajectories generated with `alpha_GR = 0.45` and a Newtonian baseline
    that only knows `mu_0 - ch_0`, verify that `discover_and_compile_law`:
      1. Detects the unmodeled relativistic residual (`latent_mechanism_required == True`).
      2. Blindly selects the exact quadratic Binet correction `ch_0 ** 2` (`vector_field_r2 > 0.9999`).
      3. Recovers `alpha_GR = 0.45` within `< 1e-2` and passes `audit_law_brick`.
    """
    anon_specs = _anonymous_state_specs(2, lo=-20.0, hi=20.0)
    newtonian_spec = SymbolicSystemSpec(
        name="baseline_newtonian_binet_orbit",
        domain="unknown",
        state_specs=anon_specs,
        param_specs=[
            ParamSpec("k_phi", DIMENSIONLESS / s, default_value=1.0),
            ParamSpec("mu_0", DIMENSIONLESS / s, default_value=1.0),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.0",
        external_drive_exprs=[
            "k_phi * ch_1",
            "mu_0 - k_phi * ch_0",
        ],
    )
    newtonian_brick = compile_symbolic_system(newtonian_spec)

    alpha_gr_true = 0.45
    gr_oracle_spec = SymbolicSystemSpec(
        name="hidden_oracle_1993_general_relativity",
        domain="unknown",
        state_specs=anon_specs,
        param_specs=[
            ParamSpec("k_phi", DIMENSIONLESS / s, default_value=1.0),
            ParamSpec("mu_0", DIMENSIONLESS / s, default_value=1.0),
            ParamSpec("alpha_gr", DIMENSIONLESS / s, default_value=alpha_gr_true),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.0",
        external_drive_exprs=[
            "k_phi * ch_1",
            "mu_0 - k_phi * ch_0 + alpha_gr * (ch_0^2)",
        ],
    )
    gr_oracle_brick = compile_symbolic_system(gr_oracle_spec)

    # Generate trajectory from gr_oracle_brick and verify discover_and_compile_law recovers ch_0 ** 2
    x0 = torch.tensor([[1.35, 0.40]], dtype=torch.float64)
    p_gr = gr_oracle_brick.default_params(batch_size=1, dtype=torch.float64)
    dt = 0.04
    gt_roll = rollout_trajectories(gr_oracle_brick, x0, p_gr, dt=dt, num_steps=45, integrator="rk4")
    obs_states = gt_roll.states[0]
    obs_flows = gt_roll.flows[0]

    cal_rep = calibrate_and_audit_residual(
        brick=newtonian_brick,
        observed_states=obs_states,
        dt=dt,
        observed_dx_dt=obs_flows,
        noise_floor_std=1e-3,
        custom_bounds={"k_phi": (1.0, 1.0), "mu_0": (1.0, 1.0)},
    )
    assert cal_rep.latent_mechanism_required

    disc_rep = discover_and_compile_law(
        base_brick=newtonian_brick,
        observed_states=obs_states,
        dt=dt,
        observed_dx_dt=obs_flows,
        calibration_report=cal_rep,
        max_active_terms=1,
        include_rational_primitives=False,
        custom_base_bounds={"k_phi": (1.0, 1.0), "mu_0": (1.0, 1.0)},
    )
    assert disc_rep.discovery_triggered
    assert disc_rep.winning_candidate is not None
    assert disc_rep.audit_certificate.passed_all
    assert disc_rep.winning_candidate.vector_field_r2 > 0.9999
    assert disc_rep.portfolio_rankings[0].funnel_report.trajectory_r2 > 0.999

    # Verify the discovered symbolic correction is the General Relativistic Binet term ch_0 ** 2 with coefficient 0.45
    expr_1 = disc_rep.winning_candidate.patch_expressions[1].to_infix()
    assert "ch_0 ** 2" in expr_1 or "ch_0^2" in expr_1
    discovered_gr_coeff = float(disc_rep.winning_candidate.param_specs[0].default_value)
    assert abs(abs(discovered_gr_coeff) - alpha_gr_true) < 1e-3
