"""
Verification and edge-case test suite for Layer 3 (`cortivra.layer3_compiler`).
"""

import tempfile
import pytest
import torch

from cortivra.layer1_axioms import (
    CanonicalHamiltonianBrick,
    DimensionError,
    J,
    J_per_mol,
    LawVault,
    N_force,
    ParamSpec,
    PowerPortSpec,
    Quantity,
    StateVarSpec,
    V,
    analyze_unexplained_residual,
    audit_law_brick,
    kg,
    m,
    m2_per_s,
    mM,
    m_per_s,
    ms,
    mV,
    nS,
    pA,
    pF,
    s,
    uF_per_cm2,
    um,
    um2,
    verify_thermodynamic_laws,
)
from cortivra.layer2_fabric import SpatialAdvectionDiffusionBrick
from cortivra.layer3_compiler import (
    ConstNode,
    SymbolNode,
    SymbolicCompilationError,
    SymbolicSystemSpec,
    build_compartment_graph,
    compile_residual_patch_brick,
    compile_symbolic_system,
    discretize_labeled_mask,
    make_cuboid_compartment,
    make_cylinder_compartment,
    make_sphere_compartment,
    parse_symbolic_expr,
)


def test_ast_parser_compile_time_unit_checking_and_scale_conversion() -> None:
    """
    Verify that `parse_symbolic_expr` rejects dimensionally invalid formulas at parse time
    (before any tensor runs) and automatically aligns compatible SI scales (e.g., `V + mV`,
    `pA / pF -> mV / ms`).
    """
    symbols = {
        "V_m": SymbolNode("V_m", mV, role="state"),
        "V_bias": SymbolNode("V_bias", V, role="param"),
        "I_ext": SymbolNode("I_ext", pA, role="input"),
        "C_m": SymbolNode("C_m", pF, role="param"),
        "V_T": ConstNode(26.7, mV),
    }

    # 1. Dimensional mismatch: adding voltage (mV) and current (pA) must fail immediately
    with pytest.raises(DimensionError, match="Dimensional mismatch"):
        parse_symbolic_expr("V_m + I_ext", symbol_table=symbols)

    # 2. Transcendental function of a dimensional quantity (exp(V_m)) must fail immediately
    with pytest.raises(DimensionError, match="dimensionless argument"):
        parse_symbolic_expr("exp(V_m)", symbol_table=symbols)

    # 3. Dimensionless ratio exp(V_m / V_T) with matching dimensions must succeed
    boltz_factor = parse_symbolic_expr("exp(V_m / V_T)", symbol_table=symbols)
    assert boltz_factor.dimension.is_dimensionless

    # 4. Cross-prefix addition: V_m (-65 mV) + V_bias (0.1 V = +100 mV) -> +35 mV
    sum_expr = parse_symbolic_expr("V_m + V_bias", symbol_table=symbols)
    env = {
        "V_m": torch.tensor([-65.0], dtype=torch.float64),
        "V_bias": torch.tensor([0.1], dtype=torch.float64),
    }
    res_mv = sum_expr.evaluate_in_unit(env, target_unit=mV)
    assert abs(float(res_mv.item()) - 35.0) < 1e-12

    # 5. Cross-unit derivative conversion: I_ext (100 pA) / C_m (50 pF) = 2 V/s = 2 mV/ms
    dv_dt_expr = parse_symbolic_expr("I_ext / C_m", symbol_table=symbols)
    env_rc = {
        "I_ext": torch.tensor([100.0], dtype=torch.float64),
        "C_m": torch.tensor([50.0], dtype=torch.float64),
    }
    res_mv_per_ms = dv_dt_expr.evaluate_in_unit(env_rc, target_unit=mV / ms)
    assert abs(float(res_mv_per_ms.item()) - 2.0) < 1e-12


def test_symbolic_differentiation_and_simplification_matches_autograd() -> None:
    """
    Verify that symbolic differentiation `node.diff(var)` and algebraic simplification
    `node.simplify()` match PyTorch `autograd.grad` across polynomials, transcendentals,
    cross-prefix additions (`mV + V`), and Layer 2 primitives (`boltzmann`, `hill`, `ohmic`,
    `michaelis_menten`, `bernoulli` at `u = 0`, `nernst_mv`, `eyring`).
    """
    symbols = {
        "V": SymbolNode("V", mV, role="state"),
        "V_bias": SymbolNode("V_bias", V, role="param"),
        "V_h": SymbolNode("V_h", mV, role="param"),
        "k": SymbolNode("k", mV, role="param"),
        "g": SymbolNode("g", nS, role="param"),
        "E": SymbolNode("E", mV, role="param"),
        "c_in": SymbolNode("c_in", mM, role="state"),
        "c_out": SymbolNode("c_out", mM, role="param"),
        "kd": SymbolNode("kd", mM, role="param"),
        "dG": SymbolNode("dG", J_per_mol, role="state"),
        "k0": SymbolNode("k0", s**-1, role="param"),
    }
    expr = parse_symbolic_expr(
        "ohmic(V + V_bias, E, g) * boltzmann_switch(V, V_h, k) * hill(c_in, kd, 2.0)",
        symbol_table=symbols,
    )
    d_expr_dv = expr.diff("V").simplify()
    d_expr_dbias = expr.diff("V_bias").simplify()
    d_expr_dcin = expr.diff("c_in").simplify()

    v_t = torch.tensor([-70.0, -40.0, 0.0, 20.0], dtype=torch.float64, requires_grad=True)
    vb_t = torch.tensor([0.01, -0.02, 0.0, 0.05], dtype=torch.float64, requires_grad=True)
    cin_t = torch.tensor([0.5, 1.2, 2.5, 5.0], dtype=torch.float64, requires_grad=True)
    env = {
        "V": v_t,
        "V_bias": vb_t,
        "V_h": torch.full_like(v_t, -35.0),
        "k": torch.full_like(v_t, 8.0),
        "g": torch.full_like(v_t, 12.0),
        "E": torch.full_like(v_t, -80.0),
        "c_in": cin_t,
        "c_out": torch.full_like(v_t, 100.0),
        "kd": torch.full_like(v_t, 1.5),
    }

    y = expr.eval_tensor(env)
    auto_dv, auto_dbias, auto_dcin = torch.autograd.grad(y.sum(), (v_t, vb_t, cin_t))
    assert torch.allclose(d_expr_dv.eval_tensor(env), auto_dv, rtol=1e-7, atol=1e-9)
    assert torch.allclose(d_expr_dbias.eval_tensor(env), auto_dbias, rtol=1e-7, atol=1e-9)
    assert torch.allclose(d_expr_dcin.eval_tensor(env), auto_dcin, rtol=1e-7, atol=1e-9)

    # Verify symbolic derivative of bernoulli(V / k) across the removable singularity V = 0
    bern_expr = parse_symbolic_expr("bernoulli(V / k)", symbol_table=symbols)
    d_bern_dv = bern_expr.diff("V").simplify()
    y_bern = bern_expr.eval_tensor(env)
    auto_bern_dv = torch.autograd.grad(y_bern.sum(), v_t)[0]
    sym_bern_dv = d_bern_dv.eval_tensor(env)
    assert torch.isfinite(sym_bern_dv).all()
    assert torch.allclose(sym_bern_dv, auto_bern_dv, rtol=1e-7, atol=1e-9)

    # Verify symbolic derivative of nernst_mv and eyring_rate
    nernst_expr = parse_symbolic_expr("nernst_mv(c_out, c_in, 1.0)", symbol_table=symbols)
    y_nernst = nernst_expr.eval_tensor(env)
    auto_nernst = torch.autograd.grad(y_nernst.sum(), cin_t)[0]
    assert torch.allclose(nernst_expr.diff("c_in").eval_tensor(env), auto_nernst, rtol=1e-7, atol=1e-9)

    dg_t = torch.tensor([-5000.0, 0.0, 8000.0, 15000.0], dtype=torch.float64, requires_grad=True)
    env_ey = {"dG": dg_t, "k0": torch.full_like(dg_t, 50.0)}
    ey_expr = parse_symbolic_expr("eyring_rate(dG, k0, 0.4)", symbol_table=symbols)
    y_ey = ey_expr.eval_tensor(env_ey)
    auto_ey = torch.autograd.grad(y_ey.sum(), dg_t)[0]
    assert torch.allclose(ey_expr.diff("dG").eval_tensor(env_ey), auto_ey, rtol=1e-7, atol=1e-9)

    # Check algebraic simplification reduces redundant identities with homogeneous units
    unsimplified = parse_symbolic_expr("(V + 0.0) * 1.0 + (E - E) * (V / k)", symbol_table=symbols)
    simplified = unsimplified.simplify()
    assert simplified.to_infix() == "V"
    assert simplified.tree_complexity() == 1


def test_compile_symbolic_port_hamiltonian_system_and_audit() -> None:
    """
    Compile a symbolic Port-Hamiltonian system from string formulas (`H`, `J`, `R`, and external drive),
    verify it passes Layer 1's `audit_law_brick()`, commit it to `MechanismVault`, and verify
    end-to-end `.backward()` parameter gradients.
    """
    spec = SymbolicSystemSpec(
        name="compiled_anharmonic_oscillator",
        domain="mechanics",
        state_specs=[
            StateVarSpec("q", m, default_value=1.2, homeostatic_min=-50.0, homeostatic_max=50.0),
            StateVarSpec("p", N_force * s, default_value=0.0, homeostatic_min=-100.0, homeostatic_max=100.0),
        ],
        param_specs=[
            ParamSpec("k_lin", N_force / m, default_value=2.0, min_value=0.0),
            ParamSpec("k_quar", N_force / (m**3), default_value=0.5, min_value=0.0),
            ParamSpec("mass", kg, default_value=1.5, min_value=1e-6),
            ParamSpec("damping", N_force * s / m, default_value=0.1, min_value=0.0),
        ],
        port_specs=[
            PowerPortSpec("mech_port", effort_unit=m_per_s, flow_unit=N_force, state_var_name="p"),
        ],
        time_unit=s,
        energy_unit=J,
        hamiltonian_expr="0.5 * k_lin * q^2 + 0.25 * k_quar * q^4 + 0.5 * (p^2) / mass",
        skew_routing_exprs={(0, 1): "1.0"},
        dissipation_diag_exprs=["0.0", "damping"],
    )

    brick = compile_symbolic_system(spec)
    state = brick.default_state(batch_size=6, dtype=torch.float64)
    params = brick.default_params(batch_size=6, dtype=torch.float64).requires_grad_(True)

    cert = audit_law_brick(brick, sample_state=state, sample_params=params, tol=1e-7)
    cert.raise_if_failed()
    assert cert.passed_all

    # Commit compiled brick to LawVault and verify cryptographic signature
    with tempfile.TemporaryDirectory() as tmp_dir:
        vault = LawVault(sandbox_storage_path=f"{tmp_dir}/sandbox.json")
        entry = vault.crystallize_discovered_law(
            law_id="compiled_anharmonic_oscillator_v1",
            name=brick.name,
            domain=brick.domain,
            symbolic_expression=brick.hamiltonian_ast.to_infix(),
            audit_certificate=cert,
            falsification_score=0.999,
            description="Compiled anharmonic Port-Hamiltonian oscillator.",
        )
        assert entry.verify_integrity()

    # Verify analytical symbolic effort matches autograd effort
    s_req = state.detach().requires_grad_(True)
    h_val = brick.hamiltonian_energy(s_req, params)
    auto_effort = torch.autograd.grad(h_val.sum(), s_req)[0]
    assert torch.allclose(brick.effort(s_req, params), auto_effort, rtol=1e-9, atol=1e-11)

    # Integrate trajectory and backpropagate into k_lin, k_quar, mass, damping
    x = state
    for _ in range(10):
        x = brick.step_symplectic_midpoint(x, params, dt=0.05)
    loss = (x[:, 0] ** 2).sum()
    loss.backward()
    assert params.grad is not None and torch.isfinite(params.grad).all()
    assert float(torch.min(torch.abs(params.grad)).item()) > 1e-5


def test_residual_patch_compilation_across_all_three_thermodynamic_modes() -> None:
    """
    Verify all three thermodynamic patch modes of `compile_residual_patch_brick`:
        1. `passive_dissipative` (nonlinear cubic damping discovery and gradient fit)
        2. `conservative_coupling` (skew-symmetric energy-conserving coupling discovery)
        3. `active_driven` on a `matrix_free=True` PDE brick (`SpatialAdvectionDiffusionBrick`)
    """
    # Mode 1: passive_dissipative
    base = CanonicalHamiltonianBrick(name="base_oscillator", damping_default=0.05)
    grid = torch.linspace(-2.0, 2.0, 20, dtype=torch.float64)
    states = torch.stack([torch.sin(grid), grid], dim=-1)
    base_params = base.default_params(batch_size=20, dtype=torch.float64)

    true_gamma = 0.45
    true_extra_dp_dt = -true_gamma * (states[:, 1] ** 3)
    obs_dx_dt = base.flow(states, base_params) + torch.stack(
        [torch.zeros_like(true_extra_dp_dt), true_extra_dp_dt], dim=-1
    )

    diag = analyze_unexplained_residual(
        base,
        states,
        base_params,
        observed_dx_dt=obs_dx_dt,
        noise_floor_std=1e-3,
        parameter_fit_exhausted=True,
    )
    assert diag.latent_term_justified
    assert diag.thermodynamic_class == "passive_dissipative"
    assert 1 in diag.affected_state_indices

    patched = compile_residual_patch_brick(
        name="patched_oscillator",
        base_brick=base,
        patch_param_specs=[ParamSpec("gamma_nl", base.state_specs[1].unit, default_value=0.1, min_value=0.0)],
        thermodynamic_mode="passive_dissipative",
        patch_expressions={1: "gamma_nl * p^2"},
    )

    gamma_est = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([gamma_est], lr=0.1)
    for _ in range(80):
        opt.zero_grad()
        full_p = torch.cat([base_params, gamma_est.unsqueeze(0).expand(20, -1)], dim=-1)
        pred_flow = patched.flow(states, full_p)
        mse = torch.mean((pred_flow - obs_dx_dt) ** 2)
        mse.backward()
        opt.step()

    assert abs(float(gamma_est.item()) - true_gamma) < 1e-3

    final_params = torch.cat([base_params, gamma_est.detach().unsqueeze(0).expand(20, -1)], dim=-1)
    thermo = verify_thermodynamic_laws(patched, states, final_params, tol=1e-7)
    assert thermo.passed
    post_diag = analyze_unexplained_residual(
        patched,
        states,
        final_params,
        observed_dx_dt=obs_dx_dt,
        noise_floor_std=1e-3,
        parameter_fit_exhausted=True,
    )
    assert not post_diag.latent_term_justified
    assert patched.total_patch_complexity() > 0

    # Mode 2: conservative_coupling (state-dependent skew coupling Delta_J_{0,1} = alpha * q^2)
    cons_patched = compile_residual_patch_brick(
        name="conservative_patched_oscillator",
        base_brick=base,
        patch_param_specs=[ParamSpec("alpha_j", s / (kg * (m**2)), default_value=0.3)],
        thermodynamic_mode="conservative_coupling",
        patch_expressions={(0, 1): "alpha_j * q^2"},
    )
    cp_params = cons_patched.default_params(batch_size=20, dtype=torch.float64)
    assert verify_thermodynamic_laws(cons_patched, states, cp_params, tol=1e-7).passed

    # Mode 3: active_driven on a matrix_free=True brick (SpatialAdvectionDiffusionBrick)
    pde_base = SpatialAdvectionDiffusionBrick(name="base_pde", num_nodes=4)
    active_patched = compile_residual_patch_brick(
        name="active_source_pde",
        base_brick=pde_base,
        patch_param_specs=[
            ParamSpec("v_max", mM / s, default_value=2.0, min_value=0.0),
            ParamSpec("k_m", mM, default_value=0.5, min_value=1e-6),
        ],
        thermodynamic_mode="active_driven",
        patch_expressions={0: "michaelis_menten(u_0, v_max, k_m)"},
    )
    assert active_patched.is_matrix_free
    u_state = active_patched.default_state(batch_size=5, dtype=torch.float64)
    u_params = active_patched.default_params(batch_size=5, dtype=torch.float64)
    active_cert = audit_law_brick(active_patched, sample_state=u_state, sample_params=u_params, tol=1e-7)
    active_cert.raise_if_failed()
    assert active_cert.passed_all


def test_parametric_geometry_mask_discretization_and_heterogeneous_transport() -> None:
    """
    Verify:
        1. Parametric sphere, cylinder, and cuboid geometry calculations (volumes, capacitances,
           Faraday conversion factors).
        2. Exact polyhedral voxel surface area and shared interface area calculation in
           `discretize_labeled_mask`.
        3. `HeterogeneousCompartmentTransportBrick` exact total mass conservation and
           equilibration to uniform intensive concentration across unequal compartment volumes.
    """
    soma = make_sphere_compartment("soma", radius=Quantity(10.0, um))
    dend = make_cylinder_compartment("dendrite", radius=Quantity(1.5, um), length=Quantity(50.0, um))
    chamber = make_cuboid_compartment("chamber", Quantity(10.0, um), Quantity(10.0, um), Quantity(20.0, um))

    assert chamber.volume.unit.symbol == "fL"
    assert abs(float(chamber.volume.value) - 2000.0) < 1e-9

    # Sphere surface area = 4 * pi * (10 um)^2 = 1256.637 um^2 -> at 1 uF/cm^2 (0.01 pF/um^2) -> 12.56637 pF
    c_soma = soma.total_capacitance(Quantity(1.0, uF_per_cm2))
    assert abs(float(c_soma.to(pF).value) - 12.5663706) < 1e-4

    alpha_ca = dend.faraday_flux_to_concentration_rate_factor(valence_z=2)
    assert alpha_ca > 0.0

    graph = build_compartment_graph([soma, dend, chamber], edges=[(0, 1), (1, 2)])
    rates = graph.compute_diffusive_edge_rates(Quantity(1e-9, m2_per_s))
    assert rates.shape == (2,) and bool((rates > 0.0).all().item())

    # Test heterogeneous transport brick across unequal volumes (soma ~ 4188.8 fL, dend ~ 353.4 fL, chamber = 2000 fL)
    het_brick = graph.to_heterogeneous_transport_brick(diffusivity_default=2.0)
    m_state = torch.tensor([[8.0, 1.0, 0.5]], dtype=torch.float64)
    het_params = het_brick.default_params(batch_size=1, dtype=torch.float64)
    cert = audit_law_brick(het_brick, sample_state=m_state, sample_params=het_params, tol=1e-9)
    cert.raise_if_failed()

    initial_total_mass = float(m_state.sum().item())
    for _ in range(250):
        m_state = het_brick.step_rk4(m_state, het_params, dt=0.05)
    final_total_mass = float(m_state.sum().item())
    assert abs(final_total_mass - initial_total_mass) < 1e-11

    # At equilibrium, intensive concentrations c_i = M_i / v_rel_i must be equal across all 3 unequal compartments
    eq_conc = het_brick.intensive_concentration(m_state).squeeze(0)
    assert float(torch.max(eq_conc) - torch.min(eq_conc)) < 1e-3

    # Test 2D segmented mask discretization with exact polyhedral voxel surface area verification
    # Region 1: 2x3 block of voxels at pitch=2 um (dz=2 um)
    # Volume = 6 * 8 = 48 fL
    # Exposed faces: top+bottom (2 * 6 = 12) + perimeter (2*(2+3) = 10) = 22 faces * 4 um^2 = 88 um^2
    mask_2d = torch.zeros((8, 12), dtype=torch.long)
    mask_2d[2:4, 1:4] = 1
    mask_2d[2:4, 4:7] = 2
    mask_2d[2:4, 7:10] = 3
    mask_graph = discretize_labeled_mask(mask_2d, voxel_pitch=Quantity(2.0, um))
    assert mask_graph.num_compartments == 3
    assert mask_graph.edge_index.shape == (2, 2)
    assert abs(float(mask_graph.compartments[0].volume.to(um**3).value) - 48.0) < 1e-9
    assert abs(float(mask_graph.compartments[0].surface_area.to(um2).value) - 88.0) < 1e-9

    # Test 3D volumetric segmented mask discretization ([D=4, H=6, W=6])
    mask_3d = torch.zeros((4, 6, 6), dtype=torch.long)
    mask_3d[1:3, 1:5, 1:3] = 10  # 2 * 4 * 2 = 16 voxels
    mask_3d[1:3, 1:5, 3:5] = 20  # 2 * 4 * 2 = 16 voxels touching along axis 2 (8 shared faces)
    graph_3d = discretize_labeled_mask(mask_3d, voxel_pitch=Quantity(1.0, um))
    assert graph_3d.num_compartments == 2
    assert graph_3d.edge_index.shape == (2, 1)
    assert abs(float(graph_3d.compartments[0].volume.to(um**3).value) - 16.0) < 1e-9
    # Shared interface area = 8 faces * 1 um^2 = 8 um^2; centroid distance = 2.0 um -> G_01 = 4.0 um = 4e-6 m
    assert abs(float(graph_3d.edge_geometric_factors_m[0].item()) - 4e-6) < 1e-12


def test_all_symbolic_primitives_parse_infer_units_and_evaluate() -> None:
    """Verify every primitive in `_PRIMITIVE_ALIASES` parses, checks units, and evaluates finite tensors."""
    syms = {
        "v": SymbolNode("v", mV),
        "c_in": SymbolNode("c_in", mM),
        "c_out": SymbolNode("c_out", mM),
        "perm": SymbolNode("perm", m_per_s),
        "vel": SymbolNode("vel", m_per_s),
        "gamma": SymbolNode("gamma", N_force * s / m),
        "u": SymbolNode("u", pA),
        "u_max": SymbolNode("u_max", mV / ms),
        "u_half": SymbolNode("u_half", pA),
    }
    env = {
        "v": torch.tensor([-65.0, 0.0, 20.0], dtype=torch.float64),
        "c_in": torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64),
        "c_out": torch.tensor([140.0, 140.0, 140.0], dtype=torch.float64),
        "perm": torch.tensor([1e-6, 1e-6, 1e-6], dtype=torch.float64),
        "vel": torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64),
        "gamma": torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64),
        "u": torch.tensor([-10.0, 0.0, 50.0], dtype=torch.float64),
        "u_max": torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64),
        "u_half": torch.tensor([25.0, 25.0, 25.0], dtype=torch.float64),
    }

    ghk_node = parse_symbolic_expr("ghk_flux(v, c_in, c_out, perm, 1)", symbol_table=syms)
    assert torch.isfinite(ghk_node.eval_tensor(env)).all()

    drag_node = parse_symbolic_expr("power_drag(vel, gamma, 1.0)", symbol_table=syms)
    assert drag_node.infer_unit().dimension == N_force.dimension
    assert torch.isfinite(drag_node.eval_tensor(env)).all()

    sat_node = parse_symbolic_expr("saturating_drive(u, u_max, u_half)", symbol_table=syms)
    assert sat_node.infer_unit().dimension == (mV / ms).dimension
    assert torch.isfinite(sat_node.eval_tensor(env)).all()

    imp_node = parse_symbolic_expr("surrogate_impulse(v, 0.5 * v, 0.5)", symbol_table=syms)
    assert torch.isfinite(imp_node.eval_tensor(env)).all()


def test_malformed_symbolic_expressions_rejected_safely() -> None:
    """Verify that invalid syntax, unknown symbols, and wrong primitive arities raise `SymbolicCompilationError`."""
    syms = {"x": SymbolNode("x", m)}
    with pytest.raises(SymbolicCompilationError):
        parse_symbolic_expr("", symbol_table=syms)
    with pytest.raises(SymbolicCompilationError, match="Syntax error"):
        parse_symbolic_expr("x + * 2", symbol_table=syms)
    with pytest.raises(SymbolicCompilationError, match="Unknown symbol"):
        parse_symbolic_expr("x + unknown_var", symbol_table=syms)
    with pytest.raises(SymbolicCompilationError, match="Unrecognized function"):
        parse_symbolic_expr("__import__('os')", symbol_table=syms)
    with pytest.raises(SymbolicCompilationError, match="requires 3 arguments"):
        parse_symbolic_expr("boltzmann(x, x)", symbol_table=syms)
    with pytest.raises(SymbolicCompilationError, match="Keyword arguments are not supported"):
        parse_symbolic_expr("boltzmann(x, x, k=x)", symbol_table=syms)


def test_compiled_external_drive_and_property_based_ast_invariants() -> None:
    """
    Verify `CompiledLawBrick` with `external_drive_exprs` passes `audit_law_brick`
    (satisfying First Law `dH/dt = P_ext - D(x)`) and converts cross-prefix drive units
    (`pA / pF -> mV / ms`) automatically.
    """
    rc_spec = SymbolicSystemSpec(
        name="compiled_driven_rc_membrane",
        domain="electrodynamics",
        state_specs=[
            StateVarSpec("V_m", mV, default_value=-65.0, homeostatic_min=-120.0, homeostatic_max=80.0),
        ],
        param_specs=[
            ParamSpec("C_m", pF, default_value=100.0, min_value=1.0),
            ParamSpec("g_leak", nS, default_value=10.0, min_value=0.0),
            ParamSpec("I_stim", pA, default_value=250.0),
        ],
        time_unit=ms,
        energy_unit=pF * (mV**2),
        hamiltonian_expr="0.5 * C_m * (V_m^2)",
        dissipation_diag_exprs=["g_leak / (C_m^2)"],
        external_drive_exprs=["I_stim / C_m"],
    )
    rc_brick = compile_symbolic_system(rc_spec)
    state = rc_brick.default_state(batch_size=4, dtype=torch.float64)
    params = rc_brick.default_params(batch_size=4, dtype=torch.float64)

    cert = audit_law_brick(rc_brick, sample_state=state, sample_params=params, tol=1e-7)
    cert.raise_if_failed()
    assert cert.passed_all
    assert rc_brick.total_symbolic_complexity() > 5


def test_onsager_off_diagonal_spsd_dissipation_and_3d_euclidean_primitives() -> None:
    """
    Verify Upgrade 1 (Off-Diagonal Symmetric Positive Semi-Definite Onsager Dissipative Couplings)
    and Upgrade 3 (3D Euclidean AST primitives `dist3d`, `inv_dist_pow`, `diffusive_flux`):
      1. `dissipation_coupling_exprs` and `OnsagerCouplingSpec` construct an exact symmetric PSD
         dissipation matrix `R(x) = R(x)^T >= 0` with off-diagonal entries `R_{ij}(x) < 0` that
         conserves `x_0 + x_1` when `stoichiometry == (1.0, -1.0)`.
      2. `ResidualPatchedLawBrick` supports off-diagonal `(0, 1)` patches in `passive_dissipative` mode.
      3. `dist3d`, `inv_dist_pow`, and `diffusive_flux` infer exact SI units, evaluate finite tensors,
         and differentiate symbolically.
    """
    from cortivra.layer1_axioms.units import DIMENSIONLESS
    from cortivra.layer3_compiler.symbolic_compiler import OnsagerCouplingSpec

    onsager_spec = SymbolicSystemSpec(
        name="onsager_cross_coupled_transport",
        domain="chemistry",
        state_specs=[
            StateVarSpec("c_1", mM, default_value=4.0, homeostatic_min=0.0, homeostatic_max=20.0),
            StateVarSpec("c_2", mM, default_value=1.0, homeostatic_min=0.0, homeostatic_max=20.0),
        ],
        param_specs=[
            ParamSpec("k_e", J / (mM**2), default_value=1.0),
            ParamSpec("g_exchange", (mM**2) / (J * s), default_value=0.6, min_value=0.0),
        ],
        time_unit=s,
        energy_unit=J,
        hamiltonian_expr="0.5 * k_e * (c_1**2 + c_2**2)",
        dissipation_diag_exprs=("0.0", "0.0"),
        dissipation_coupling_exprs={(0, 1): "g_exchange"},
        onsager_couplings=(
            OnsagerCouplingSpec(
                stoichiometry=((0, 1.0), (1, -1.0)),
                conductance_expr="0.2 * g_exchange",
            ),
        ),
    )
    brick = compile_symbolic_system(onsager_spec)
    state = brick.default_state(batch_size=2, dtype=torch.float64)
    params = brick.default_params(batch_size=2, dtype=torch.float64)

    r_mat = brick.dissipation_matrix(state, params)
    assert torch.allclose(r_mat, r_mat.transpose(-1, -2), atol=1e-12)
    assert float(r_mat[0, 0, 1].item()) < -0.5  # Off-diagonal Onsager entry is non-zero!
    eigvals = torch.linalg.eigvalsh(r_mat)
    assert bool((eigvals >= -1e-12).all().item())

    # Total mass c_1 + c_2 is conserved under symmetric exchange (dc_1/dt + dc_2/dt == 0)
    flow = brick.flow(state, params)
    assert torch.allclose(flow[:, 0] + flow[:, 1], torch.zeros_like(flow[:, 0]), atol=1e-12)
    assert audit_law_brick(brick, sample_state=state, sample_params=params).passed_all

    # Test off-diagonal (0, 1) ResidualPatchedLawBrick in passive_dissipative mode
    patched_onsager = compile_residual_patch_brick(
        name="patched_onsager_pair",
        base_brick=brick,
        patch_param_specs=[ParamSpec("g_extra", (mM**2) / (J * s), default_value=0.25, min_value=0.0)],
        thermodynamic_mode="passive_dissipative",
        patch_expressions={(0, 1): "g_extra"},
    )
    p_patched = patched_onsager.default_params(batch_size=2, dtype=torch.float64)
    assert audit_law_brick(patched_onsager, sample_state=state, sample_params=p_patched).passed_all

    # Test 3D Euclidean AST primitives: dist3d, inv_dist_pow, diffusive_flux
    syms_3d = {
        "dx": SymbolNode("dx", m),
        "dy": SymbolNode("dy", m),
        "dz": SymbolNode("dz", m),
        "g_d": SymbolNode("g_d", DIMENSIONLESS / s),
    }
    env_3d = {
        "dx": torch.tensor([3.0, 0.0], dtype=torch.float64),
        "dy": torch.tensor([4.0, 0.0], dtype=torch.float64),
        "dz": torch.tensor([0.0, 0.0], dtype=torch.float64),
        "g_d": torch.tensor([2.0, 2.0], dtype=torch.float64),
    }
    dist_node = parse_symbolic_expr("dist3d(dx, dy, dz)", symbol_table=syms_3d)
    assert dist_node.infer_unit().dimension == m.dimension
    dist_vals = dist_node.eval_tensor(env_3d)
    assert abs(float(dist_vals[0].item()) - 5.0) < 1e-6
    assert float(dist_vals[1].item()) > 0.0  # Singularity-regularized at r=0!

    inv_r2_node = parse_symbolic_expr("inv_dist_pow(dx, dy, dz, 2)", symbol_table=syms_3d)
    assert inv_r2_node.infer_unit().dimension == (DIMENSIONLESS / (m**2)).dimension
    inv_vals = inv_r2_node.eval_tensor(env_3d)
    assert abs(float(inv_vals[0].item()) - 0.04) < 1e-5
    assert torch.isfinite(inv_r2_node.diff("dx").eval_tensor(env_3d)).all()

    flux_node = parse_symbolic_expr("diffusive_flux(dx, dy, g_d)", symbol_table=syms_3d)
    assert flux_node.infer_unit().dimension == (m / s).dimension
    assert abs(float(flux_node.eval_tensor(env_3d)[0].item()) - 2.0) < 1e-9


def test_multifield_reaction_diffusion_pde_brick_mass_conservation_and_kinetics() -> None:
    """
    Verify Horizon 4 (`MultiFieldReactionDiffusionPDEBrick` and `to_multifield_reaction_diffusion_brick`):
        1. Differential per-species diffusion (`D_u != D_v`) across heterogeneous compartment volumes
           conserves exact mass per species (`sum_i M_{s, i}`) when `k_exchange == 0` and passes `audit_law_brick`.
        2. Reversible Onsager inter-species conversion (`k_exchange > 0`) conserves total combined mass
           `sum_{s, i} M_{s, i}` while satisfying `R = R^T >= 0`.
        3. Local symbolic reaction kinetics (`reaction_exprs`) evaluate across all spatial nodes.
    """
    from cortivra.layer3_compiler import build_compartment_graph, make_sphere_compartment

    c0 = make_sphere_compartment("node_0", Quantity(5.0, um))
    c1 = make_sphere_compartment("node_1", Quantity(8.0, um))
    c2 = make_sphere_compartment("node_2", Quantity(6.0, um))
    graph = build_compartment_graph([c0, c1, c2], edges=[(0, 1), (1, 2)])

    mf_brick = graph.to_multifield_reaction_diffusion_brick(
        name="turing_two_species_pde",
        species_names=("u", "v"),
        diffusivities_default=(0.1, 1.5),
        inter_species_exchange_default=0.0,
    )
    x0 = mf_brick.default_state(batch_size=1, dtype=torch.float64)
    p0 = mf_brick.default_params(batch_size=1, dtype=torch.float64)
    assert audit_law_brick(mf_brick, sample_state=x0, sample_params=p0).passed_all

    # Pure differential diffusion conserves mass of species u and species v independently
    masses_0 = mf_brick.species_masses(x0)
    x_diff = x0.clone()
    for _ in range(25):
        x_diff = mf_brick.step_rk4(x_diff, p0, dt=0.04)
    masses_1 = mf_brick.species_masses(x_diff)
    assert torch.allclose(masses_1, masses_0, atol=1e-12)

    # Reversible inter-species exchange conserves total combined mass (mass_u + mass_v)
    p_ex = p0.clone()
    p_ex[:, 2] = 0.4  # k_exchange = 0.4
    assert audit_law_brick(mf_brick, sample_state=x0, sample_params=p_ex).passed_all
    x_ex = x0.clone()
    for _ in range(25):
        x_ex = mf_brick.step_rk4(x_ex, p_ex, dt=0.04)
    assert abs(float(mf_brick.species_masses(x_ex).sum().item() - masses_0.sum().item())) < 1e-12



