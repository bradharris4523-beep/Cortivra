"""
Verification and edge-case test suite for Layer 4 (`cortivra.layer4_engine`).
"""

import math
import torch

from cortivra.layer1_axioms import (
    CanonicalHamiltonianBrick,
    DIMENSIONLESS,
    ParamSpec,
    Quantity,
    StateVarSpec,
    mV,
    ms,
    nS,
    pA,
    pF,
    s,
    um,
)
from cortivra.layer2_fabric import (
    NonlinearConductanceCircuitBrick,
    NonlinearPendulumAndViscoelasticBrick,
    StoichiometricReactionBrick,
)
from cortivra.layer3_compiler import (
    SymbolicSystemSpec,
    build_compartment_graph,
    compile_residual_patch_brick,
    compile_symbolic_system,
    make_cylinder_compartment,
)
from cortivra.layer4_engine import (
    calibrate_and_audit_residual,
    compute_parameter_leverage_and_fisher,
    discover_equilibrium_landscape,
    estimate_trajectory_derivatives,
    find_steady_state_and_stability,
    rank_hypothesis_portfolio,
    retrodict_trajectory,
    rollout_trajectories,
    run_elimination_funnel,
    scan_parameter_bifurcation,
    select_optimal_integrator,
    solve_boundary_value_shooting,
    solve_optimal_control_waveform,
    solve_target_intervention,
)


def test_forward_integrators_and_exact_time_reversal_retrodiction() -> None:
    """
    Verify forward rollout across `rk4`, `adaptive_rk`, `symplectic_midpoint`,
    and `exponential_euler`, and `retrodict_trajectory` reconstructing `x(t_0)`
    from `x(t_T)` on conservative systems.
    """
    pendulum = NonlinearPendulumAndViscoelasticBrick(damping_default=0.0)
    x0 = torch.tensor([[1.1, 0.4]], dtype=torch.float64)
    params = pendulum.default_params(batch_size=1, dtype=torch.float64)

    for integ in ("rk4", "adaptive_rk", "symplectic_midpoint"):
        roll = rollout_trajectories(pendulum, x0, params, dt=0.02, num_steps=40, integrator=integ)
        assert roll.states.shape == (1, 41, 2)
        assert bool(roll.all_time_viable_mask.all().item())
        assert roll.max_energy_drift < 1e-3

    # Verify quadratic Hamiltonian symplectic_midpoint conserves energy within fixed-point tolerance (< 1e-7)
    quad_osc = CanonicalHamiltonianBrick(damping_default=0.0)
    q_roll = rollout_trajectories(
        quad_osc,
        x0,
        quad_osc.default_params(batch_size=1, dtype=torch.float64),
        dt=0.05,
        num_steps=40,
        integrator="symplectic_midpoint",
    )
    assert q_roll.max_energy_drift < 1e-7

    #Symplectic midpoint forward followed by conservative_exact retrodiction must recover x0 to < 1e-8
    fwd_symp = rollout_trajectories(
        pendulum, x0, params, dt=0.02, num_steps=40, integrator="symplectic_midpoint"
    )
    x_final = fwd_symp.states[:, -1, :]
    retro = retrodict_trajectory(
        pendulum, x_final, params, dt=0.02, num_steps=40, mode="conservative_exact"
    )
    reconstructed_x0 = retro.states[:, 0, :]
    assert torch.allclose(reconstructed_x0, x0, atol=1e-8)

    # Also verify regularized_full retrodiction remains finite and viable on a damped system
    damped_pend = NonlinearPendulumAndViscoelasticBrick(damping_default=0.15)
    d_params = damped_pend.default_params(batch_size=1, dtype=torch.float64)
    d_fwd = rollout_trajectories(damped_pend, x0, d_params, dt=0.02, num_steps=20, integrator="rk4")
    d_retro = retrodict_trajectory(
        damped_pend, d_fwd.states[:, -1, :], d_params, dt=0.02, num_steps=20, mode="regularized_full"
    )
    assert torch.isfinite(d_retro.states).all()
    assert bool(d_retro.all_time_viable_mask.all().item())

    # Verify exponential_euler dispatch on NonlinearConductanceCircuitBrick
    circuit = NonlinearConductanceCircuitBrick()
    c_x0 = circuit.default_state(batch_size=2, dtype=torch.float64)
    c_params = circuit.default_params(batch_size=2, dtype=torch.float64)
    c_roll = rollout_trajectories(
        circuit, c_x0, c_params, dt=0.5, num_steps=20, integrator="exponential_euler"
    )
    assert c_roll.states.shape == (2, 21, 5)
    assert bool(c_roll.all_time_viable_mask.all().item())


def test_infinite_horizon_steady_state_and_eigenvalue_bifurcation_classes() -> None:
    """
    Verify `find_steady_state_and_stability` finds `f(x^*) == 0` and classifies
    local Jacobian eigenvalues into `stable_spiral`, `conservative_center`, and `stable_node`.
    """
    # 1. Underdamped harmonic oscillator (damping = 0.4 < 2 * sqrt(k*m) = 2.0) -> stable_spiral
    underdamped = CanonicalHamiltonianBrick(damping_default=0.4)
    guess = torch.tensor([[2.5, -1.8]], dtype=torch.float64)
    p_ud = underdamped.default_params(batch_size=1, dtype=torch.float64)
    eq_ud = find_steady_state_and_stability(underdamped, guess, p_ud)
    assert bool(eq_ud.converged_mask.all().item())
    assert torch.allclose(eq_ud.equilibrium_state, torch.zeros_like(guess), atol=1e-7)
    assert eq_ud.stability_classes[0] == "stable_spiral"
    assert float(eq_ud.max_real_eigenvalue[0].item()) < -0.1
    assert float(eq_ud.max_imag_eigenvalue[0].item()) > 0.5

    # 2. Undamped Hamiltonian oscillator (damping = 0.0) -> conservative_center
    undamped = CanonicalHamiltonianBrick(damping_default=0.0)
    p_con = undamped.default_params(batch_size=1, dtype=torch.float64)
    eq_con = find_steady_state_and_stability(undamped, guess, p_con)
    assert bool(eq_con.converged_mask.all().item())
    assert eq_con.stability_classes[0] == "conservative_center"
    assert abs(float(eq_con.max_real_eigenvalue[0].item())) < 1e-9

    # 3. Overdamped oscillator (damping = 3.0 > 2.0) -> stable_node
    overdamped = CanonicalHamiltonianBrick(damping_default=3.0)
    p_od = overdamped.default_params(batch_size=1, dtype=torch.float64)
    eq_od = find_steady_state_and_stability(overdamped, guess, p_od)
    assert bool(eq_od.converged_mask.all().item())
    assert eq_od.stability_classes[0] == "stable_node"
    assert float(eq_od.max_imag_eigenvalue[0].item()) < 1e-7


def test_4th_order_derivative_estimator_accuracy() -> None:
    """Verify `estimate_trajectory_derivatives` achieves O(dt^4) interior accuracy on smooth waves."""
    dt = 0.02
    t = torch.arange(0, 60, dtype=torch.float64) * dt
    omega = 2.5
    traj = torch.stack([torch.sin(omega * t), torch.cos(omega * t)], dim=-1)  # [60, 2]
    true_deriv = torch.stack([omega * torch.cos(omega * t), -omega * torch.sin(omega * t)], dim=-1)

    est_deriv = estimate_trajectory_derivatives(traj, dt=dt)
    # Interior points [2:-2] use 4th-order stencil -> error < 1e-5
    interior_err = float(torch.max(torch.abs(est_deriv[2:-2] - true_deriv[2:-2])).item())
    assert interior_err < 1e-5


def test_4_stage_elimination_funnel_and_partial_state_observation_recovery() -> None:
    """
    Verify that `run_elimination_funnel`:
        1. Recovers unknown parameters `(k, m, damping)` from a full state trajectory (`R^2 > 0.999`).
        2. Recovers the trajectory dynamics even when ONLY position `q(t)` is observed
           (`observed_state_indices=[0]`, momentum `p(t)` hidden!).
    """
    brick = CanonicalHamiltonianBrick(k_default=1.0, m_default=1.0, damping_default=0.1)
    true_params = torch.tensor([[2.8, 1.0, 0.35]], dtype=torch.float64)
    x0 = torch.tensor([[1.5, 0.0]], dtype=torch.float64)
    dt = 0.04

    gt_roll = rollout_trajectories(brick, x0, true_params, dt=dt, num_steps=35, integrator="rk4")
    obs_full = gt_roll.states[0]  # [36, 2]
    obs_dx = gt_roll.flows[0]     # [36, 2]

    # Case 1: Full-state observation
    rep_full = run_elimination_funnel(
        brick=brick,
        observed_states=obs_full,
        dt=dt,
        observed_dx_dt=obs_dx,
        num_initial_samples=128,
        num_collocation_survivors=24,
        num_gradient_starts=4,
        gradient_steps=70,
    )
    assert rep_full.vector_field_r2 > 0.999
    assert rep_full.trajectory_r2 > 0.995
    assert rep_full.audit_certificate.passed_all

    # Case 2: Partial-state observation (only q(t) at index 0 is measured; p(t) is hidden)
    obs_q_only = obs_full[:, :1]  # [36, 1]
    rep_partial = run_elimination_funnel(
        brick=brick,
        observed_states=obs_q_only,
        dt=dt,
        observed_state_indices=[0],
        initial_full_state=x0,
        num_initial_samples=96,
        num_collocation_survivors=16,
        num_gradient_starts=4,
        gradient_steps=50,
    )
    assert rep_partial.trajectory_r2 > 0.98


def test_two_point_boundary_value_shooting() -> None:
    """
    Verify `solve_boundary_value_shooting`: given initial position `q(0) = -1.2`
    and target terminal position `q(T) = +0.9` after 25 steps, solve for the unknown initial
    momentum `p(0)` that bridges the two boundary states.
    """
    brick = CanonicalHamiltonianBrick(k_default=1.5, m_default=1.0, damping_default=0.1)
    params = brick.default_params(batch_size=1, dtype=torch.float64)

    res = solve_boundary_value_shooting(
        brick=brick,
        known_initial_indices=[0],
        known_initial_values=torch.tensor([-1.2], dtype=torch.float64),
        target_final_indices=[0],
        target_final_values=torch.tensor([0.9], dtype=torch.float64),
        params=params,
        dt=0.04,
        num_steps=25,
        max_iters=90,
        learning_rate=0.15,
        tol=1e-3,
    )
    assert res.converged
    assert res.terminal_error_norm < 1e-2
    assert res.all_time_viable
    assert abs(float(res.solved_initial_state[0, 0].item()) - (-1.2)) < 1e-12


def test_calibrate_and_audit_residual_and_hypothesis_portfolio_ranking() -> None:
    """
    Verify `calibrate_and_audit_residual` and `rank_hypothesis_portfolio`:
        1. When observed data comes from a linear oscillator with shifted parameters (`k = 3.5`),
           `calibrate_and_audit_residual` fits `k = 3.5` and concludes `latent_mechanism_required == False`.
        2. When observed data contains an unmodeled nonlinear cubic damping `-0.5 * p^3`,
           `calibrate_and_audit_residual` exhausts linear parameter fitting and flags
           `latent_mechanism_required == True` (`passive_dissipative` on state `#1`).
        3. `rank_hypothesis_portfolio` compares the baseline model, a wrong linear patch, and
           the true cubic damping patch, ranking the true patch `#1` with lowest BIC.
    """
    base = CanonicalHamiltonianBrick(k_default=1.0, m_default=1.0, damping_default=0.1)
    x0 = torch.tensor([[1.6, 0.8]], dtype=torch.float64)
    dt = 0.03

    # Part 1: Shifted parameter only (no missing law)
    shifted_p = torch.tensor([[3.5, 1.0, 0.2]], dtype=torch.float64)
    shifted_roll = rollout_trajectories(base, x0, shifted_p, dt=dt, num_steps=30, integrator="rk4")
    cal_clean = calibrate_and_audit_residual(
        brick=base,
        observed_states=shifted_roll.states[0],
        dt=dt,
        observed_dx_dt=shifted_roll.flows[0],
        noise_floor_std=1e-2,
    )
    assert not cal_clean.latent_mechanism_required
    assert cal_clean.funnel_report.vector_field_r2 > 0.999

    # Part 2: Unmodeled nonlinear cubic damping on momentum p (state #1)
    true_patched = compile_residual_patch_brick(
        name="cubic_damped_oscillator",
        base_brick=base,
        patch_param_specs=[ParamSpec("gamma_c", base.state_specs[1].unit, default_value=0.6, min_value=0.0)],
        thermodynamic_mode="passive_dissipative",
        patch_expressions={1: "gamma_c * p^2"},
    )
    p_true_patch = torch.tensor([[1.5, 1.0, 0.05, 0.65]], dtype=torch.float64)
    nl_roll = rollout_trajectories(true_patched, x0, p_true_patch, dt=dt, num_steps=35, integrator="rk4")

    cal_missing = calibrate_and_audit_residual(
        brick=base,
        observed_states=nl_roll.states[0],
        dt=dt,
        observed_dx_dt=nl_roll.flows[0],
        noise_floor_std=1e-3,
        custom_bounds={"k": (1.5, 1.5), "m": (1.0, 1.0), "damping": (0.05, 0.05)},
    )
    assert cal_missing.latent_mechanism_required
    assert cal_missing.residual_analysis.thermodynamic_class == "passive_dissipative"
    assert 1 in cal_missing.residual_analysis.affected_state_indices

    # Part 3: Rank competing hypotheses in a portfolio
    wrong_patched = compile_residual_patch_brick(
        name="quartic_q_damped_oscillator",
        base_brick=base,
        patch_param_specs=[ParamSpec("gamma_q", base.state_specs[0].unit, default_value=0.2, min_value=0.0)],
        thermodynamic_mode="passive_dissipative",
        patch_expressions={1: "gamma_q * q^2"},
    )
    rankings = rank_hypothesis_portfolio(
        bricks=[base, wrong_patched, true_patched],
        observed_states=nl_roll.states[0],
        dt=dt,
        observed_dx_dt=nl_roll.flows[0],
        num_initial_samples=96,
        gradient_steps=60,
    )
    assert rankings[0].brick_name == "cubic_damped_oscillator"
    assert rankings[0].passed_thermodynamic_audit
    assert rankings[0].funnel_report.vector_field_r2 > 0.995


def test_fisher_information_leverage_and_counterfactual_restorative_intervention() -> None:
    """
    Verify `compute_parameter_leverage_and_fisher` and `solve_target_intervention`
    on a compiled sensory receptor membrane model:
        1. Identify the highest-leverage restorative parameter via Fisher & susceptibility analysis.
        2. Synthesize a counterfactual parameter intervention that restores the target voltage
           response while staying inside homeostatic bounds and passing `audit_law_brick`.
    """
    sensory_spec = SymbolicSystemSpec(
        name="sensory_receptor_membrane",
        domain="electrodynamics",
        state_specs=[
            StateVarSpec("V_m", mV, default_value=-60.0, homeostatic_min=-95.0, homeostatic_max=40.0),
        ],
        param_specs=[
            ParamSpec("C_m", pF, default_value=50.0, min_value=10.0, max_value=200.0, tunable=False),
            ParamSpec("g_trans", nS, default_value=2.0, min_value=0.1, max_value=40.0, tunable=True),
            ParamSpec("E_rev", mV, default_value=-10.0, min_value=-80.0, max_value=50.0, tunable=True),
            ParamSpec("g_leak", nS, default_value=8.0, min_value=1.0, max_value=30.0, tunable=True),
        ],
        time_unit=ms,
        energy_unit=pF * (mV**2),
        hamiltonian_expr="0.5 * C_m * (V_m^2)",
        dissipation_diag_exprs=["(g_leak + g_trans) / (C_m^2)"],
        external_drive_exprs=["(g_trans * E_rev) / C_m"],
    )
    receptor = compile_symbolic_system(sensory_spec)
    x0 = receptor.default_state(batch_size=1, dtype=torch.float64)
    degraded_params = torch.tensor([[50.0, 1.0, -50.0, 12.0]], dtype=torch.float64)

    # Target: restore steady-state depolarized sensory response to V_target = -20.0 mV
    v_target = -20.0

    def _sensory_loss(roll: object) -> torch.Tensor:
        from cortivra.layer4_engine.rollout_arena import RolloutResult
        assert isinstance(roll, RolloutResult)
        return torch.mean((roll.states[:, -5:, 0] - v_target) ** 2)

    fisher_rep = compute_parameter_leverage_and_fisher(
        brick=receptor,
        initial_state=x0,
        params=degraded_params,
        dt=0.5,
        num_steps=25,
        objective_fn=_sensory_loss,
    )
    assert fisher_rep.fisher_matrix.shape == (4, 4)
    assert bool((fisher_rep.fisher_eigenvalues >= 0.0).all().item())
    # Non-tunable C_m must have 0 restorative leverage; tunable E_rev / g_trans must rank at the top
    assert fisher_rep.ranked_param_names[0] in ("E_rev", "g_trans", "g_leak")
    assert float(fisher_rep.restorative_leverage_scores[0].item()) == 0.0

    intervention = solve_target_intervention(
        brick=receptor,
        initial_state=x0,
        baseline_params=degraded_params,
        dt=0.5,
        num_steps=25,
        target_loss_fn=_sensory_loss,
        max_iters=90,
        learning_rate=0.5,
    )
    assert intervention.intervened_loss < intervention.baseline_loss * 0.01
    assert intervention.intervened_loss < 0.5
    assert bool(intervention.restored_rollout.all_time_viable_mask.all().item())
    assert intervention.audit_certificate.passed_all


def test_auto_integrator_selection_across_physical_regimes() -> None:
    """
    Verify `select_optimal_integrator` and `integrator='auto'` automatically inspect the
    mathematical structure of `brick` without user hints:
        - Pure conservative system (`R(x) == 0`) -> `'symplectic_midpoint'`
        - Fast gating subsystem (`rush_larsen_step`) -> `'exponential_euler'`
        - Stiff dissipative system (`dt * ||J|| > 0.85`) -> `'adaptive_rk'`
    """
    conservative = NonlinearPendulumAndViscoelasticBrick(damping_default=0.0)
    x0_c = torch.tensor([[0.9, 0.2]], dtype=torch.float64)
    p_c = conservative.default_params(batch_size=1, dtype=torch.float64)
    assert select_optimal_integrator(conservative, x0_c, p_c, dt=0.02) == "symplectic_midpoint"

    auto_roll = rollout_trajectories(conservative, x0_c, p_c, dt=0.02, num_steps=25, integrator="auto")
    assert auto_roll.max_energy_drift < 5e-4

    quad_con = CanonicalHamiltonianBrick(damping_default=0.0)
    auto_quad = rollout_trajectories(
        quad_con, x0_c, quad_con.default_params(batch_size=1, dtype=torch.float64), dt=0.02, num_steps=25, integrator="auto"
    )
    assert auto_quad.max_energy_drift < 1e-7

    circuit = NonlinearConductanceCircuitBrick()
    x0_e = circuit.default_state(batch_size=1, dtype=torch.float64)
    p_e = circuit.default_params(batch_size=1, dtype=torch.float64)
    assert select_optimal_integrator(circuit, x0_e, p_e, dt=0.1) == "exponential_euler"

    stiff_osc = CanonicalHamiltonianBrick(k_default=50.0, m_default=1.0, damping_default=25.0)
    x0_s = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    p_s = stiff_osc.default_params(batch_size=1, dtype=torch.float64)
    assert select_optimal_integrator(stiff_osc, x0_s, p_s, dt=0.1) == "adaptive_rk"


def test_invariant_subspace_equilibrium_root_finding_on_conserved_mass_and_diffusion() -> None:
    """
    Verify `find_steady_state_and_stability` on rank-deficient Jacobian systems with linear
    conservation laws (`m^T f(x) = 0`), ensuring `Range(J)` Newton steps preserve total mass
    and classify the equilibrium as `'stable_node'` with `num_conserved_modes >= 1`.
    """
    # 1. Chemical reaction A + B <=> C with initial concentrations [3.2, 2.0, 0.8]
    #    Two independent conserved moieties: (c_A + c_C = 4.0) and (c_B + c_C = 2.8)
    rxn = StoichiometricReactionBrick()
    c_init = torch.tensor([[3.2, 2.0, 0.8]], dtype=torch.float64)
    p_rxn = rxn.default_params(batch_size=1, dtype=torch.float64)
    eq_rxn = find_steady_state_and_stability(rxn, c_init, p_rxn, max_newton_iters=60, tol=1e-10)

    assert bool(eq_rxn.converged_mask[0].item())
    moiety_1_before = float((c_init[0, 0] + c_init[0, 2]).item())
    moiety_1_after = float((eq_rxn.equilibrium_state[0, 0] + eq_rxn.equilibrium_state[0, 2]).item())
    moiety_2_before = float((c_init[0, 1] + c_init[0, 2]).item())
    moiety_2_after = float((eq_rxn.equilibrium_state[0, 1] + eq_rxn.equilibrium_state[0, 2]).item())
    assert abs(moiety_1_after - moiety_1_before) < 1e-12
    assert abs(moiety_2_after - moiety_2_before) < 1e-12
    assert eq_rxn.stability_classes[0] == "stable_node"
    assert int(eq_rxn.num_conserved_modes[0].item()) == 2
    assert float(eq_rxn.max_real_eigenvalue[0].item()) < -0.1

    # 2. 4-compartment 1D diffusion chain with non-uniform initial amounts
    comps = [
        make_cylinder_compartment(f"c_{i}", radius=Quantity(2.0, um), length=Quantity(10.0, um))
        for i in range(4)
    ]
    graph = build_compartment_graph(comps, edges=[(0, 1), (1, 2), (2, 3)])
    diff_brick = graph.to_heterogeneous_transport_brick(name="conserved_diffusion_4", diffusivity_default=1.0)
    c_diff_init = torch.tensor([[4.0, 1.0, 0.5, 2.5]], dtype=torch.float64)  # mean = 2.0
    p_diff = diff_brick.default_params(batch_size=1, dtype=torch.float64)
    eq_diff = find_steady_state_and_stability(diff_brick, c_diff_init, p_diff, max_newton_iters=60, tol=1e-10)

    assert bool(eq_diff.converged_mask[0].item())
    assert torch.allclose(
        eq_diff.equilibrium_state,
        torch.full_like(c_diff_init, 2.0),
        atol=1e-8,
    )
    assert eq_diff.stability_classes[0] == "stable_node"
    assert int(eq_diff.num_conserved_modes[0].item()) == 1


def test_hidden_initial_state_and_extreme_out_of_box_parameter_recovery() -> None:
    """
    Verify `run_elimination_funnel` jointly recovers an unknown non-zero initial hidden state
    (`p(0) = 1.6`, `initial_full_state=None`) from partial observations (`observed_state_indices=[0]`)
    and recovers parameters far outside default initialization (`k = 18.5`, `observed_dx_dt=None`).
    """
    brick = CanonicalHamiltonianBrick(k_default=1.0, m_default=1.0, damping_default=0.15)
    dt = 0.03

    # Part 1: Unknown non-zero initial hidden state p(0) = 1.6 with only q(t) measured
    x0_hidden_nz = torch.tensor([[1.2, 1.6]], dtype=torch.float64)
    p_true_1 = torch.tensor([[2.2, 1.0, 0.25]], dtype=torch.float64)
    roll_1 = rollout_trajectories(brick, x0_hidden_nz, p_true_1, dt=dt, num_steps=30, integrator="rk4")
    obs_q_only = roll_1.states[0, :, :1]  # [31, 1]

    rep_hidden = run_elimination_funnel(
        brick=brick,
        observed_states=obs_q_only,
        dt=dt,
        observed_state_indices=[0],
        initial_full_state=None,  # Engine must discover p(0) jointly with parameters!
        num_initial_samples=128,
        num_collocation_survivors=24,
        num_gradient_starts=6,
        gradient_steps=80,
        learning_rate=0.1,
    )
    assert rep_hidden.trajectory_r2 > 0.985
    recovered_p0 = float(rep_hidden.best_rollout.states[0, 0, 1].item())
    assert recovered_p0 > 0.8  # Jointly moved away from default p(0)=0.0 toward true +1.6

    # Part 2: Extreme out-of-box parameter k = 18.5 (default = 1.0) with observed_dx_dt=None
    p_extreme = torch.tensor([[18.5, 1.0, 0.4]], dtype=torch.float64)
    roll_ext = rollout_trajectories(
        brick, torch.tensor([[1.4, 0.0]], dtype=torch.float64), p_extreme, dt=0.015, num_steps=35
    )
    rep_ext = run_elimination_funnel(
        brick=brick,
        observed_states=roll_ext.states[0],
        dt=0.015,
        observed_dx_dt=None,
        num_initial_samples=128,
        num_collocation_survivors=24,
        num_gradient_starts=6,
        gradient_steps=110,
        learning_rate=0.25,
    )
    assert rep_ext.trajectory_r2 > 0.98
    assert abs(float(rep_ext.best_params[0, 0].item()) - 18.5) < 2.0


def test_optimal_control_waveform_synthesis() -> None:
    """
    Verify `solve_optimal_control_waveform` synthesizes a smooth external control waveform `u*(t)`
    on `actuated_state_indices=[1]` that drives a damped oscillator to `q(T) = 1.5` within bounds.
    """
    brick = CanonicalHamiltonianBrick(k_default=2.0, m_default=1.0, damping_default=0.8)
    x0 = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    params = brick.default_params(batch_size=1, dtype=torch.float64)
    q_target = 1.5

    def _terminal_q_loss(roll: object) -> torch.Tensor:
        from cortivra.layer4_engine.rollout_arena import RolloutResult
        assert isinstance(roll, RolloutResult)
        return torch.mean((roll.states[:, -5:, 0] - q_target) ** 2)

    ctrl_res = solve_optimal_control_waveform(
        brick=brick,
        initial_state=x0,
        params=params,
        dt=0.05,
        num_steps=30,
        target_loss_fn=_terminal_q_loss,
        actuated_state_indices=[1],  # Only actuate force on momentum p (index 1); q (index 0) unactuated
        max_control_amplitude=20.0,
        max_iters=90,
        learning_rate=0.2,
    )
    assert ctrl_res.controlled_loss < ctrl_res.baseline_loss * 0.01
    assert ctrl_res.controlled_loss < 0.02
    assert ctrl_res.all_time_viable
    assert ctrl_res.control_energy > 0.0
    # Unactuated state index 0 must have identically zero control input across all time steps
    assert float(torch.max(torch.abs(ctrl_res.control_waveform[:, :, 0])).item()) == 0.0


def test_bistable_double_well_equilibrium_landscape_and_bifurcation_scan() -> None:
    """
    Verify `discover_equilibrium_landscape` finds all 3 fixed points (`q = -sqrt(2), 0, +sqrt(2)`)
    of a bistable Duffing double-well system, and `scan_parameter_bifurcation` detects the
    pitchfork bifurcation at `a_c == 0.0`.
    """
    double_well_spec = SymbolicSystemSpec(
        name="duffing_double_well",
        domain="mechanics",
        state_specs=[
            StateVarSpec("q", DIMENSIONLESS, default_value=0.5, homeostatic_min=-5.0, homeostatic_max=5.0),
            StateVarSpec("p", DIMENSIONLESS, default_value=0.0, homeostatic_min=-5.0, homeostatic_max=5.0),
        ],
        param_specs=[
            ParamSpec("a", DIMENSIONLESS, default_value=-2.0, min_value=-10.0, max_value=10.0),
            ParamSpec("b", DIMENSIONLESS, default_value=1.0, min_value=0.1, max_value=10.0),
            ParamSpec("m", DIMENSIONLESS, default_value=1.0, min_value=0.1, max_value=10.0, tunable=False),
            ParamSpec("gamma", DIMENSIONLESS, default_value=0.8, min_value=0.0, max_value=10.0),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.25 * b * (q^4) + 0.5 * a * (q^2) + 0.5 * (p^2) / m",
        skew_routing_exprs={(0, 1): "1.0"},
        dissipation_diag_exprs=["0.0", "gamma"],
    )
    dw_brick = compile_symbolic_system(double_well_spec)
    p_bistable = dw_brick.default_params(batch_size=1, dtype=torch.float64)

    landscape = discover_equilibrium_landscape(
        brick=dw_brick,
        params=p_bistable,
        num_probes=32,
        probe_span=2.5,
        max_newton_iters=45,
        tol=1e-8,
        cluster_tol=1e-2,
    )
    assert landscape.distinct_equilibria.shape == (3, 2)
    assert len(landscape.stable_attractor_indices) == 2
    assert len(landscape.unstable_fixed_point_indices) == 1

    expected_q = torch.tensor([-math.sqrt(2.0), 0.0, math.sqrt(2.0)], dtype=torch.float64)
    assert torch.allclose(landscape.distinct_equilibria[:, 0], expected_q, atol=1e-5)
    assert torch.allclose(landscape.distinct_equilibria[:, 1], torch.zeros(3, dtype=torch.float64), atol=1e-7)
    assert landscape.unstable_fixed_point_indices == (1,)

    # Scan control parameter 'a' across [-1.5, +1.5] at the origin (0, 0) to detect pitchfork bifurcation at a_c = 0.0
    a_grid = torch.linspace(-1.5, 1.5, 16, dtype=torch.float64)
    bif = scan_parameter_bifurcation(
        brick=dw_brick,
        param_name="a",
        param_values=a_grid,
        initial_guess=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
    )
    assert len(bif.bifurcation_indices) >= 1
    assert any(abs(tc - 0.0) < 0.15 for tc in bif.critical_param_estimates)


def test_multi_experiment_dose_response_ensemble_and_lm_machine_precision() -> None:
    """
    Verify that `calibrate_and_audit_residual` and `run_elimination_funnel`:
        1. Natively accept a 3D multi-experiment ensemble `observed_states` of shape `[E, T, D]`
           (`E = 4` distinct initial conditions).
        2. Use the second-order Levenberg-Marquardt polish to recover parameters `(k, m, damping)`
           to machine precision (`vector_field_mse < 1e-12`).
    """
    brick = CanonicalHamiltonianBrick(k_default=1.0, m_default=1.0, damping_default=0.1)
    true_p = torch.tensor([[3.75, 1.25, 0.42]], dtype=torch.float64).expand(4, -1)
    x0_ensemble = torch.tensor(
        [
            [1.5, 0.0],
            [0.0, 1.8],
            [-1.2, 0.9],
            [2.0, -1.1],
        ],
        dtype=torch.float64,
    )
    dt = 0.03
    ens_roll = rollout_trajectories(brick, x0_ensemble, true_p, dt=dt, num_steps=20, integrator="rk4")

    cal = calibrate_and_audit_residual(
        brick=brick,
        observed_states=ens_roll.states,     # [4, 21, 2] multi-experiment ensemble!
        dt=dt,
        observed_dx_dt=ens_roll.flows,       # [4, 21, 2]
        num_initial_samples=96,
        num_collocation_survivors=20,
        num_gradient_starts=4,
        gradient_steps=60,
    )
    assert cal.funnel_report.best_rollout.states.shape == (4, 21, 2)
    assert cal.funnel_report.vector_field_mse < 1e-12
    assert cal.funnel_report.trajectory_r2 > 0.999999
    assert not cal.latent_mechanism_required


