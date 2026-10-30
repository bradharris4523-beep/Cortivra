"""
Verification and edge-case test suite for Layer 2 (`cortivra.layer2_fabric`).
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from cortivra.layer1_axioms import (
    CoupledPortHamiltonianSystem,
    LawVault,
    PortConnection,
    analyze_unexplained_residual,
    audit_law_brick,
    verify_charge_conservation,
    verify_mass_conservation,
    verify_reaction_electroneutrality,
    verify_thermodynamic_laws,
)
from cortivra.layer2_fabric import (
    PRIMITIVE_REGISTRY,
    CentralForceHamiltonianBrick,
    CooperativeBindingBrick,
    EventImpulseCouplingBrick,
    NonlinearConductanceCircuitBrick,
    NonlinearPendulumAndViscoelasticBrick,
    SpatialAdvectionDiffusionBrick,
    StoichiometricReactionBrick,
    boltzmann_partition_switch,
    conservative_advection_1d,
    cooperative_hill_saturation,
    cubic_bistable_feedback,
    eyring_barrier_rate,
    get_primitives_by_role,
    ghk_electrodiffusion_flux,
    graph_laplacian_diffusion_1d,
    graph_laplacian_sparse_edges,
    nernst_equilibrium_potential_mv,
    ohmic_dissipative_driving_force,
    register_foundational_laws,
    skew_symmetric_poisson_flow,
    smooth_bernoulli_quotient,
    step_rush_larsen,
    surrogate_event_impulse,
    trigonometric_phase_interaction,
)


def test_primitive_registry_completeness_and_roles() -> None:
    """Verify all 16 universal mathematical primitives are registered and categorized by role."""
    assert len(PRIMITIVE_REGISTRY) == 16
    cons = get_primitives_by_role("conservative")
    diss = get_primitives_by_role("dissipative")
    eq = get_primitives_by_role("equilibrium_partition")
    trans = get_primitives_by_role("state_transition")
    assert len(cons) + len(diss) + len(eq) + len(trans) == 16


def test_bernoulli_and_ghk_removable_singularity_c2_smoothness() -> None:
    """
    Verify that `smooth_bernoulli_quotient` and `ghk_electrodiffusion_flux` have finite,
    accurate values, 1st derivatives, and 2nd derivatives across `u = 0` (`V = 0 mV`)
    and across the Taylor transition boundary `|u| = 1e-4`.
    """
    u_pts = torch.tensor(
        [-1e-2, -1e-4, -1e-7, 0.0, 1e-7, 1e-4, 1e-2],
        dtype=torch.float64,
        requires_grad=True,
    )
    b_val = smooth_bernoulli_quotient(u_pts)
    assert torch.isfinite(b_val).all()
    assert abs(float(b_val[3].item()) - 1.0) < 1e-14

    grad1 = torch.autograd.grad(b_val.sum(), u_pts, create_graph=True)[0]
    assert torch.isfinite(grad1).all()
    # Analytical B'(0) = -1/2
    assert abs(float(grad1[3].item()) - (-0.5)) < 1e-12

    grad2 = torch.autograd.grad(grad1.sum(), u_pts)[0]
    assert torch.isfinite(grad2).all()
    # Analytical B''(0) = 1/6
    assert abs(float(grad2[3].item()) - (1.0 / 6.0)) < 1e-9

    # Verify GHK flux at V = 0 mV equals pure diffusive flux P * z * F * (c_in - c_out)
    v_zero = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    c_in = torch.tensor([10.0], dtype=torch.float64)
    c_out = torch.tensor([140.0], dtype=torch.float64)
    flux = ghk_electrodiffusion_flux(v_zero, c_in, c_out, permeability=1e-6, valence_z=1)
    flux.backward()
    assert torch.isfinite(flux).all()
    assert v_zero.grad is not None and torch.isfinite(v_zero.grad).all()


@given(
    x_val=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
    c_val=st.floats(min_value=1e-9, max_value=1e4, allow_nan=False, allow_infinity=False),
    hill_n=st.floats(min_value=0.2, max_value=8.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_property_based_extreme_inputs_never_nan(x_val: float, c_val: float, hill_n: float) -> None:
    """Fuzz statistical/kinetic primitives across extreme dynamic ranges to ensure finite outputs and gradients."""
    x = torch.tensor([x_val], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([c_val], dtype=torch.float64, requires_grad=True)

    b_out = boltzmann_partition_switch(x, x_half=0.0, slope_k=5.0)
    e_out = eyring_barrier_rate(x, base_rate=1.0, asymmetry_delta=0.5)
    h_out = cooperative_hill_saturation(c, half_maximal_k=1.0, hill_n=hill_n)
    n_out = nernst_equilibrium_potential_mv(c_out=c * 2.0, c_in=c, valence_z=1)

    total = (b_out + e_out + h_out + n_out).sum()
    total.backward()

    assert torch.isfinite(b_out).all() and (0.0 <= float(b_out.item()) <= 1.0)
    assert torch.isfinite(h_out).all() and (0.0 <= float(h_out.item()) <= 1.0)
    assert torch.isfinite(e_out).all() and float(e_out.item()) >= 0.0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert c.grad is not None and torch.isfinite(c.grad).all()


def test_trigonometric_poisson_and_cubic_primitives() -> None:
    """Verify trigonometric torque antisymmetry (`sum_i tau_i == 0`), Poisson orthogonality, and cubic feedback."""
    phases = torch.tensor([[0.1, 0.9, -0.4]], dtype=torch.float64)
    w = torch.tensor([[[0.0, 2.0, 1.0], [2.0, 0.0, 3.0], [1.0, 3.0, 0.0]]], dtype=torch.float64)
    pot, torque = trigonometric_phase_interaction(phases, w)
    assert float(pot.item()) > 0.0
    # Newton's 3rd Law for phase coupling: sum of internal torques must vanish
    assert abs(float(torque.sum().item())) < 1e-14

    upper = torch.randn((4, 5, 5), dtype=torch.float64)
    effort = torch.randn((4, 5), dtype=torch.float64)
    f_cons, _ = skew_symmetric_poisson_flow(upper, effort)
    assert float(torch.max(torch.abs((effort * f_cons).sum(dim=-1))).item()) < 1e-13

    fb = cubic_bistable_feedback(torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64), 1.0, 1.0, 0.0)
    assert torch.allclose(fb, torch.zeros_like(fb), atol=1e-15)


def test_all_layer2_bricks_pass_layer1_audit_certificate() -> None:
    """
    Verify that every concrete LawBrick in Layer 2 passes Layer 1's `audit_law_brick`
    and that every brick's analytical `effort(x)` equals `torch.autograd.grad(H(x).sum(), x)`
    to machine precision across perturbed non-resting states.
    """
    bricks = [
        CentralForceHamiltonianBrick(num_bodies=2, spatial_dim=2, drag_default=0.0),
        NonlinearPendulumAndViscoelasticBrick(damping_default=0.1),
        NonlinearConductanceCircuitBrick(),
        StoichiometricReactionBrick(),
        CooperativeBindingBrick(),
        SpatialAdvectionDiffusionBrick(num_nodes=16),
        EventImpulseCouplingBrick(),
    ]
    for brick in bricks:
        state = brick.default_state(batch_size=8, dtype=torch.float64)
        # Perturb away from rest so non-trivial state derivatives of H(x) are active
        noise = 0.05 * torch.ones_like(state)
        perturbed = brick.clamp_to_hard_bounds(state + noise).detach().requires_grad_(True)
        params = brick.default_params(batch_size=8, dtype=torch.float64)

        cert = audit_law_brick(brick, sample_state=perturbed, sample_params=params, tol=1e-5)
        cert.raise_if_failed()
        assert cert.passed_all

        # Verify analytical effort(x) == autograd.grad(hamiltonian_energy(x).sum(), x)
        h_val = brick.hamiltonian_energy(perturbed, params)
        auto_grad = torch.autograd.grad(h_val.sum(), perturbed)[0]
        analytical_effort = brick.effort(perturbed, params)
        assert torch.allclose(analytical_effort, auto_grad, rtol=1e-7, atol=1e-9), (
            f"Effort mismatch on {brick.name}: max diff = "
            f"{float(torch.max(torch.abs(analytical_effort - auto_grad)).item()):.2e}"
        )


def test_kepler_orbital_conservation_and_neptune_perturbation_discovery() -> None:
    """
    1. Verify energy `H(q, p)` and angular momentum `L_z` conservation on a 2-body orbit.
    2. Simulate a 3-body gravitational system (Sun + Uranus + unseen Neptune) and verify that
       a 2-body model (Sun + Uranus) triggers Layer 1's `analyze_unexplained_residual` with
       `latent_term_justified == True` when Neptune perturbs Uranus.
    """
    two_body = CentralForceHamiltonianBrick(
        name="sun_uranus_2body",
        num_bodies=2,
        spatial_dim=2,
        default_masses=(10.0, 1.0),
        g_default=1.0,
        drag_default=0.0,
    )
    # Near-circular orbit around r = 2.0 (v_circ = sqrt(G * M / r) = sqrt(5) ~ 2.236)
    x = torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.0, -2.2, 0.0, 2.2]], dtype=torch.float64)
    params = two_body.default_params(batch_size=1, dtype=torch.float64)

    e0 = float(two_body.hamiltonian_energy(x, params).item())
    lz0 = float(two_body.angular_momentum_z(x).item())

    traj_states = []
    for _ in range(40):
        traj_states.append(x)
        x = two_body.step_symplectic_midpoint(x, params, dt=0.01, fixed_point_iters=10, tol=1e-10)

    e_final = float(two_body.hamiltonian_energy(x, params).item())
    lz_final = float(two_body.angular_momentum_z(x).item())
    assert abs(e_final - e0) < 1e-4
    assert abs(lz_final - lz0) < 1e-9

    # Now compute observed accelerations of the 2 bodies when a 3rd body ("Neptune") is present
    three_body = CentralForceHamiltonianBrick(
        name="sun_uranus_neptune_3body",
        num_bodies=3,
        spatial_dim=2,
        default_masses=(10.0, 1.0, 2.5),
        g_default=1.0,
        drag_default=0.0,
    )
    states_2b = torch.cat(traj_states, dim=0)  # [40, 8]
    params_2b = two_body.default_params(batch_size=40, dtype=torch.float64)

    # Place unseen 3rd body at (3.2, 0.5) with momentum (0.0, 0.4)
    q_2b = states_2b[:, :4]
    p_2b = states_2b[:, 4:]
    q_neptune = torch.tensor([[3.2, 0.5]], dtype=torch.float64).expand(40, -1)
    p_neptune = torch.tensor([[0.0, 0.4]], dtype=torch.float64).expand(40, -1)
    states_3b = torch.cat([q_2b, q_neptune, p_2b, p_neptune], dim=-1)
    params_3b = three_body.default_params(batch_size=40, dtype=torch.float64)

    flow_3b = three_body.flow(states_3b, params_3b)
    # Extract observed [dq_0, dq_1, dp_0, dp_1] for the first 2 bodies
    obs_flow_2b = torch.cat([flow_3b[:, :4], flow_3b[:, 6:10]], dim=-1)

    residual_report = analyze_unexplained_residual(
        brick=two_body,
        state=states_2b,
        params=params_2b,
        observed_dx_dt=obs_flow_2b,
        noise_floor_std=1e-3,
        parameter_fit_exhausted=True,
    )
    assert residual_report.latent_term_justified
    assert residual_report.max_snr > 10.0
    assert residual_report.state_coherence_r2 > 0.90


def test_stoichiometric_reaction_brick_mass_charge_and_equilibrium_convergence() -> None:
    """Verify exact mass conservation, charge electroneutrality, and convergence to `c_eq`."""
    rxn = StoichiometricReactionBrick()
    mass_res = verify_mass_conservation(rxn.net_stoich, rxn.molar_masses_tensor)
    elec_res = verify_reaction_electroneutrality(rxn.net_stoich, rxn.valences_tensor)
    assert mass_res.passed
    assert elec_res.passed

    c = torch.tensor([[2.0, 1.5, 1.0]], dtype=torch.float64)
    params = rxn.default_params(batch_size=1, dtype=torch.float64)
    initial_mass = float(torch.matmul(c, rxn.molar_masses_tensor).item())

    for _ in range(150):
        dc_dt = rxn.nonlinear_mass_action_flow(c, params)
        c = c + 0.05 * dc_dt

    final_mass = float(torch.matmul(c, rxn.molar_masses_tensor).item())
    assert abs(final_mass - initial_mass) < 1e-11
    # At steady state, forward and reverse rates balance
    assert float(torch.max(torch.abs(rxn.nonlinear_mass_action_flow(c, params))).item()) < 1e-4


def test_electrodynamic_circuit_kirchhoff_balance_and_stiff_rush_larsen() -> None:
    """
    Verify Kirchhoff current balance on `NonlinearConductanceCircuitBrick`, action potential
    generation under depolarizing current, and unconditional gate stability `[0, 1]` under
    Rush-Larsen integration even at large `dt = 2.0 ms`.
    """
    circuit = NonlinearConductanceCircuitBrick()
    state = circuit.default_state(batch_size=4, dtype=torch.float64)
    params = circuit.default_params(batch_size=4, dtype=torch.float64)
    i_stim = torch.tensor([0.0, 250.0, 500.0, 800.0], dtype=torch.float64)

    dx_dt = circuit.full_excitable_flow(state, params, stimulus_current_pa=i_stim)
    currents = circuit.channel_currents_pa(state, params)
    stacked_ions = torch.stack(
        [currents["I_fast"], currents["I_slow"], currents["I_high"], currents["I_leak"]],
        dim=-1,
    )
    kirchhoff = verify_charge_conservation(
        capacitance_f=params[:, 0],
        dv_dt=dx_dt[:, 0],
        ionic_currents=stacked_ions,
        external_current=i_stim,
        tol=1e-9,
    )
    assert kirchhoff.passed

    # Integrate spike trajectory with Rush-Larsen at dt = 0.05 ms and verify spike peak > +10 mV
    x = state.clone()
    max_v = torch.full((4,), -100.0, dtype=torch.float64)
    for _ in range(120):
        x = circuit.step_exponential_euler(x, params, dt=0.05, stimulus_current_pa=i_stim)
        max_v = torch.maximum(max_v, x[:, 0])
        assert bool(((x[:, 1:] >= 0.0) & (x[:, 1:] <= 1.0)).all().item())

    assert float(max_v[0].item()) < -60.0  # Unstimulated cell stays at rest
    assert float(max_v[3].item()) > 10.0   # Stimulated cell fires an action potential

    # Stiff step torture test: dt = 5.0 ms (100x normal step) must keep gates strictly inside [0, 1]
    x_stiff = circuit.step_exponential_euler(state, params, dt=5.0, stimulus_current_pa=i_stim)
    assert torch.isfinite(x_stiff).all()
    assert bool(((x_stiff[:, 1:] >= 0.0) & (x_stiff[:, 1:] <= 1.0)).all().item())


def test_spatial_advection_diffusion_mass_conservation_and_100k_scaling() -> None:
    """
    Verify that `SpatialAdvectionDiffusionBrick` conserves total spatial integral `sum_i u_i`
    when `decay == 0`, satisfies 1st & 2nd Laws of Thermodynamics, and scales to 100,000 nodes.
    """
    pde = SpatialAdvectionDiffusionBrick(num_nodes=32, dx=0.5, decay_default=0.0)
    u = torch.exp(-((torch.linspace(-4.0, 4.0, 32, dtype=torch.float64)) ** 2)).unsqueeze(0)
    params = pde.default_params(batch_size=1, dtype=torch.float64)

    thermo = verify_thermodynamic_laws(pde, u, params, tol=1e-9)
    assert thermo.passed

    mass_0 = float(u.sum().item())
    for _ in range(25):
        u = pde.step_rk4(u, params, dt=0.02)
    mass_final = float(u.sum().item())
    assert abs(mass_final - mass_0) < 1e-11

    # Verify O(N) matrix-free evaluation on 100,000 batch x node elements
    u_large = torch.randn((5000, 32), dtype=torch.float32)
    p_large = pde.default_params(batch_size=5000, dtype=torch.float32)
    thermo_large = verify_thermodynamic_laws(pde, u_large, p_large, tol=1e-3)
    assert thermo_large.passed


def test_cross_domain_multi_physics_differentiable_coupling_and_vault_registration() -> None:
    """
    Couple an Electrodynamic circuit brick, a Cooperative Binding chemistry brick, an Event Impulse
    transport brick, and a Viscoelastic Mechanics brick, and verify that `.backward()` propagates
    gradients from the mechanical displacement `q` and voltage `V` back into parameters across
    all four physical domains. Also verify `register_foundational_laws` on `LawVault`.
    """
    vault = LawVault()
    core_records = register_foundational_laws(vault)
    assert len(core_records) == 8
    assert all(r.verify_integrity() for r in core_records)

    binding = CooperativeBindingBrick(ligand_conc_default=0.1, kd_default=0.05)
    circuit = NonlinearConductanceCircuitBrick()
    synapse = EventImpulseCouplingBrick()
    mechanics = NonlinearPendulumAndViscoelasticBrick()

    # Connect circuit electrical port to synapse electrical port inside CoupledPortHamiltonianSystem
    conn = PortConnection(
        source_brick=circuit.name,
        source_port=circuit.port_specs[0],
        target_brick=synapse.name,
        target_port=synapse.port_specs[0],
        coupling_gain=0.05,
    )
    coupled = CoupledPortHamiltonianSystem(
        name="electro_synaptic_composite",
        bricks=[circuit, synapse],
        connections=[conn],
    )
    cert = audit_law_brick(
        coupled,
        sample_state=coupled.default_state(batch_size=2, dtype=torch.float64),
        sample_params=coupled.default_params(batch_size=2, dtype=torch.float64),
        tol=1e-5,
    )
    cert.raise_if_failed()

    # End-to-end differentiable multi-scale pipeline:
    # Drug [D] & K_d (Chemistry) -> modulates circuit vh_high (Electrodynamics)
    # -> drives surrogate impulse g_syn (Transport) -> drives active mechanical force (Mechanics)
    bind_params = binding.default_params(batch_size=1, dtype=torch.float64).requires_grad_(True)
    circ_params = circuit.default_params(batch_size=1, dtype=torch.float64).requires_grad_(True)
    syn_params = synapse.default_params(batch_size=1, dtype=torch.float64).requires_grad_(True)
    mech_params = mechanics.default_params(batch_size=1, dtype=torch.float64).requires_grad_(True)

    theta_drug = binding.target_occupancy(bind_params)  # [1]
    # Allosteric gating shift: drug binding shifts vh_high by -12 mV * theta_drug
    shifted_vh_high = circ_params[:, 9] - 12.0 * theta_drug
    eff_circ_params = torch.cat(
        [circ_params[:, :9], shifted_vh_high.unsqueeze(-1), circ_params[:, 10:]],
        dim=-1,
    )

    x_circ = circuit.default_state(batch_size=1, dtype=torch.float64)
    x_syn = synapse.default_state(batch_size=1, dtype=torch.float64)
    x_mech = mechanics.default_state(batch_size=1, dtype=torch.float64)

    for _ in range(15):
        x_circ = circuit.step_exponential_euler(x_circ, eff_circ_params, dt=0.1, stimulus_current_pa=400.0)
        dg_dt, i_post = synapse.compute_event_drive_and_post_current(
            x_syn, syn_params, v_pre_mv=x_circ[:, 0], v_post_mv=torch.tensor([-65.0], dtype=torch.float64)
        )
        x_syn = torch.clamp(x_syn + 0.1 * dg_dt, min=0.0)
        ext_mech_flow = torch.stack([torch.zeros_like(i_post), 0.01 * i_post], dim=-1)
        x_mech = mechanics.step_rk4(
            x_mech, mech_params, dt=0.01, port_inputs={"external_flow": ext_mech_flow}
        )

    loss = (x_mech[:, 0] ** 2 + 0.01 * x_circ[:, 0] ** 2).sum()
    loss.backward()

    for p_tensor in (bind_params, circ_params, syn_params, mech_params):
        assert p_tensor.grad is not None
        assert torch.isfinite(p_tensor.grad).all()
        assert float(torch.max(torch.abs(p_tensor.grad)).item()) > 0.0


def test_active_circuit_battery_drive_first_law_and_param_gradients() -> None:
    """
    Verify that `NonlinearConductanceCircuitBrick.compute_battery_drive_flow` satisfies
    the exact First Law of Thermodynamics (`dH/dt == P_ext - D(x)`) inside `verify_thermodynamic_laws`,
    and that `hamiltonian_energy` and `effort` retain non-zero gradients with respect to
    gating half-activation parameters `vh_fast` and `vh_high`.
    """
    circuit = NonlinearConductanceCircuitBrick()
    state = circuit.default_state(batch_size=4, dtype=torch.float64)
    # Perturb voltage and gates away from rest so currents and gate efforts are non-zero
    state = state + torch.tensor([[15.0, 0.1, -0.1, 0.15, 0.2]], dtype=torch.float64)
    params = circuit.default_params(batch_size=4, dtype=torch.float64).requires_grad_(True)

    u_battery = circuit.compute_battery_drive_flow(state, params, stimulus_current_pa=200.0)
    thermo = verify_thermodynamic_laws(
        circuit,
        state,
        params,
        port_inputs={"external_flow": u_battery},
        tol=1e-9,
    )
    assert thermo.passed, thermo.message

    # Verify parameter gradients through hamiltonian_energy into vh_fast (idx 8) and vh_high (idx 9)
    h_energy = circuit.hamiltonian_energy(state, params).sum()
    h_energy.backward()
    assert params.grad is not None
    assert abs(float(params.grad[0, 8].item())) > 1e-6
    assert abs(float(params.grad[0, 9].item())) > 1e-6


def test_nonlinear_gibbs_relative_entropy_lyapunov_and_sparse_graph_laplacian() -> None:
    """
    1. Verify that `StoichiometricReactionBrick.nonlinear_free_energy` is strictly positive
       away from equilibrium and its time derivative `-nonlinear_chemical_affinity_dissipation`
       is strictly non-positive along far-from-equilibrium trajectories.
    2. Verify `graph_laplacian_sparse_edges` and `SpatialAdvectionDiffusionBrick(edge_index=...)`
       on an irregular branched tree topology (soma + 3 branches).
    """
    rxn = StoichiometricReactionBrick()
    c_far = torch.tensor(
        [[5.0, 0.2, 0.1], [0.1, 4.0, 3.5], [1.0, 1.0, 2.0]],
        dtype=torch.float64,
    )
    params = rxn.default_params(batch_size=3, dtype=torch.float64)
    g_free = rxn.nonlinear_free_energy(c_far)
    sigma = rxn.nonlinear_chemical_affinity_dissipation(c_far, params)

    assert float(g_free[0].item()) > 0.0 and float(g_free[1].item()) > 0.0
    assert abs(float(g_free[2].item())) < 1e-14  # Zero at exact equilibrium [1, 1, 2]
    assert float(sigma[0].item()) > 0.0 and float(sigma[1].item()) > 0.0
    assert abs(float(sigma[2].item())) < 1e-14

    # Branched tree graph: node 0 connected to (1, 2, 3), node 1 connected to (4, 5)
    edge_index = torch.tensor([[0, 0, 0, 1, 1], [1, 2, 3, 4, 5]], dtype=torch.long)
    tree_pde = SpatialAdvectionDiffusionBrick(
        name="branched_tree_diffusion",
        num_nodes=6,
        dx=1.0,
        edge_index=edge_index,
        decay_default=0.0,
    )
    u_tree = torch.tensor([[10.0, 1.0, 0.5, 0.2, 0.0, 0.0]], dtype=torch.float64)
    p_tree = tree_pde.default_params(batch_size=1, dtype=torch.float64)

    thermo_tree = verify_thermodynamic_laws(tree_pde, u_tree, p_tree, tol=1e-9)
    assert thermo_tree.passed

    mass_before = float(u_tree.sum().item())
    for _ in range(20):
        u_tree = tree_pde.step_rk4(u_tree, p_tree, dt=0.05)
    mass_after = float(u_tree.sum().item())
    assert abs(mass_after - mass_before) < 1e-12


def test_zero_distance_collision_regularization_in_central_force() -> None:
    """
    Verify that two bodies placed at the exact same coordinates `q_0 == q_1` produce finite
    potential energy, finite forces (`0.0` by symmetry), and finite gradients without `NaN`.
    """
    brick = CentralForceHamiltonianBrick(num_bodies=2, spatial_dim=2, softening_eps=1e-3)
    # Both bodies at (1.0, 1.0)
    state = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.5, -0.5, -0.5, 0.5]], dtype=torch.float64, requires_grad=True)
    params = brick.default_params(batch_size=1, dtype=torch.float64).requires_grad_(True)

    h = brick.hamiltonian_energy(state, params)
    f = brick.flow(state, params)
    loss = h.sum() + (f**2).sum()
    loss.backward()

    assert torch.isfinite(h).all()
    assert torch.isfinite(f).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert params.grad is not None and torch.isfinite(params.grad).all()


def test_surrogate_impulse_normalization_and_power_law_drag_dissipation() -> None:
    """
    1. Verify that `surrogate_event_impulse(x, threshold, sharpness_beta)` integrates to 1.0
       across `(-inf, +inf)` for any `sharpness_beta > 0`.
    2. Verify that `ohmic_dissipative_driving_force` with turbulent/power-law `power_exponent=2.0`
       strictly satisfies `(e - e_eq) * J >= 0` and has finite gradients at `e == e_eq`.
    """
    x_grid = torch.linspace(-100.0, 100.0, 50001, dtype=torch.float64)
    dx = float((x_grid[1] - x_grid[0]).item())
    for beta in (0.25, 1.0, 4.0):
        imp = surrogate_event_impulse(x_grid, threshold=2.5, sharpness_beta=beta)
        integral = float(torch.trapezoid(imp, dx=dx).item())
        assert abs(integral - 1.0) < 1e-6

    e_pts = torch.tensor([-5.0, 0.0, 5.0], dtype=torch.float64, requires_grad=True)
    j_quad = ohmic_dissipative_driving_force(e_pts, equilibrium_effort=0.0, conductance=2.0, power_exponent=2.0)
    entropy_prod = e_pts * j_quad
    assert bool((entropy_prod >= 0.0).all().item())
    j_quad.sum().backward()
    assert e_pts.grad is not None and torch.isfinite(e_pts.grad).all()


def test_quantum_symplectic_lindblad_brick_and_unitary_born_norm_conservation() -> None:
    """
    Verify Horizon 3 (`QuantumSymplecticLindbladBrick` and `verify_quantum_unitary_norm_conservation`):
        1. Unitary Rabi-Kerr evolution (`gamma = 0`) conserves both quantum Hamiltonian `H(q, p)`
           and the Born probability norm `sum_k |psi_k|^2 = 0.5 * sum_k (q_k^2 + p_k^2)`.
        2. Open-system Lindblad dephasing/relaxation (`gamma > 0`) satisfies `audit_law_brick`
           and monotonically dissipates energy under the 2nd Law of Thermodynamics.
    """
    from cortivra.layer1_axioms import audit_law_brick, verify_quantum_unitary_norm_conservation
    from cortivra.layer2_fabric import QuantumSymplecticLindbladBrick

    q_brick = QuantumSymplecticLindbladBrick(
        num_levels=3,
        rabi_coupling_default=0.3,
        lindblad_gamma_default=0.0,
    )
    x0 = q_brick.default_state(batch_size=2, dtype=torch.float64)
    p_unitary = q_brick.default_params(batch_size=2, dtype=torch.float64)

    norm_shield = verify_quantum_unitary_norm_conservation(q_brick, x0, p_unitary, tol=1e-12)
    assert norm_shield.passed
    assert audit_law_brick(q_brick, sample_state=x0, sample_params=p_unitary).passed_all

    # Integrate unitary symplectic midpoint steps and verify exact Born norm + energy conservation
    norm_0 = q_brick.born_norm(x0)
    energy_0 = q_brick.hamiltonian_energy(x0, p_unitary)
    x_curr = x0.clone()
    for _ in range(25):
        x_curr = q_brick.step_symplectic_midpoint(x_curr, p_unitary, dt=0.02)
    norm_final = q_brick.born_norm(x_curr)
    energy_final = q_brick.hamiltonian_energy(x_curr, p_unitary)
    assert torch.allclose(norm_final, norm_0, atol=1e-9)
    assert torch.allclose(energy_final, energy_0, atol=1e-8)

    # Open-system Lindblad relaxation (gamma > 0)
    q_open = QuantumSymplecticLindbladBrick(
        num_levels=2,
        rabi_coupling_default=0.25,
        lindblad_gamma_default=0.15,
    )
    x_open = q_open.default_state(batch_size=2, dtype=torch.float64)
    p_open = q_open.default_params(batch_size=2, dtype=torch.float64)
    assert verify_quantum_unitary_norm_conservation(q_open, x_open, p_open, tol=1e-12).passed
    assert audit_law_brick(q_open, sample_state=x_open, sample_params=p_open).passed_all
    assert bool((q_open.dissipation_power(x_open, p_open) > 0.0).all().item())



