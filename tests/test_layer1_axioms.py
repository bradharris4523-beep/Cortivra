"""
Verification and edge-case test suite for Layer 1 (`cortivra.layer1_axioms`).
"""

import json
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from cortivra.layer1_axioms import (
    BODY_TEMPERATURE_K,
    DIM_DIFFUSIVITY,
    DIM_DIMENSIONLESS,
    DIM_LENGTH,
    DIM_POWER,
    DIM_VOLTAGE,
    DIMENSIONLESS,
    FARADAY,
    R_GAS,
    THERMAL_VOLTAGE_37C,
    VACUUM_PERMEABILITY,
    AuditCertificate,
    CanonicalHamiltonianBrick,
    ConservationViolationError,
    CoupledPortHamiltonianSystem,
    Dimension,
    DimensionError,
    J_per_K,
    K,
    LawBrick,
    LawVault,
    MOhm,
    N_force,
    Pa,
    ParamSpec,
    PortConnection,
    PowerPortSpec,
    Quantity,
    StateVarSpec,
    Unit,
    V,
    analyze_unexplained_residual,
    audit_law_brick,
    compute_viable_universe_mask,
    discover_dimensionless_groups,
    kJ_per_mol,
    m,
    m2_per_s,
    m3,
    mM,
    mol_per_s,
    m_per_s,
    ms,
    mV,
    nA,
    nS,
    pA,
    parse_unit,
    pF,
    s,
    verify_charge_conservation,
    verify_detailed_balance_cycle,
    verify_dimensional_consistency,
    verify_equation_dimensions,
    verify_homeostatic_viability,
    verify_mass_conservation,
    verify_reaction_electroneutrality,
    verify_thermodynamic_laws,
)


# --- 1. Dimensional Algebra, Constants, and Property-Based Fuzzing ---

def test_nernst_equilibrium_dimensional_derivation() -> None:
    k_out = mM(5.0)
    k_in = mM(140.0)
    z_k = 1.0

    assert k_in > k_out
    assert k_out < k_in
    assert V(0.05) >= mV(50.0)
    assert V(0.05) <= mV(50.0)
    assert V(0.05) == mV(50.0)
    assert mV(50.0) != mM(50.0)

    ratio = k_out / k_in
    assert ratio.dimension == DIM_DIMENSIONLESS

    e_k = ((R_GAS * BODY_TEMPERATURE_K) / (z_k * FARADAY)) * ratio.log()
    assert e_k.dimension == DIM_VOLTAGE

    e_k_mv = e_k.to(mV)
    assert pytest.approx(float(e_k_mv.value), abs=0.05) == -89.06
    assert pytest.approx(float(THERMAL_VOLTAGE_37C.value), abs=0.01) == 26.73


def test_sqrt_and_spatial_derivative_dimensions() -> None:
    diffusivity = m2_per_s(1e-9)
    time_window = s(4.0)
    rms_displacement = (2.0 * diffusivity * time_window).sqrt()
    assert rms_displacement.dimension == m.dimension
    assert pytest.approx(float(rms_displacement.value), rel=1e-6) == (8e-9) ** 0.5

    laplacian_c_dim = mM.dimension.d_dx(order=2)
    diffusion_term_dim = DIM_DIFFUSIVITY * laplacian_c_dim
    assert diffusion_term_dim == mM.dimension.d_dt(order=1)


@settings(max_examples=60)
@given(
    exps_a=st.tuples(*[st.integers(min_value=-4, max_value=4) for _ in range(7)]),
    exps_b=st.tuples(*[st.integers(min_value=-4, max_value=4) for _ in range(7)]),
    log_scale_a=st.floats(min_value=-9.0, max_value=9.0),
    log_scale_b=st.floats(min_value=-9.0, max_value=9.0),
    val_a=st.floats(min_value=1e-4, max_value=1e4),
    val_b=st.floats(min_value=1e-4, max_value=1e4),
)
def test_property_based_dimensional_group_invariants(
    exps_a: tuple[int, ...],
    exps_b: tuple[int, ...],
    log_scale_a: float,
    log_scale_b: float,
    val_a: float,
    val_b: float,
) -> None:
    dim_a = Dimension(*exps_a)
    dim_b = Dimension(*exps_b)
    u_a = Unit("uA", dim_a, scale=10.0**log_scale_a)
    u_b = Unit("uB", dim_b, scale=10.0**log_scale_b)

    q_a = u_a(val_a)
    q_b = u_b(val_b)

    # (q_a * q_b) / q_b must recover the exact dimension and SI value of q_a
    recovered = (q_a * q_b) / q_b
    assert recovered.dimension == q_a.dimension
    assert pytest.approx(float(recovered.si_value), rel=1e-5) == float(q_a.si_value)

    # Self-ratio q_a / q_a must always be dimensionless with value 1.0
    self_ratio = q_a / q_a
    assert self_ratio.dimension.is_dimensionless
    assert pytest.approx(float(self_ratio.si_value), rel=1e-6) == 1.0


def test_dimensional_and_spec_validation_errors() -> None:
    with pytest.raises(DimensionError, match="finite"):
        Dimension(M=float("nan"))

    with pytest.raises(DimensionError, match="finite"):
        DIM_LENGTH ** float("inf")

    with pytest.raises(ValueError, match="strictly positive"):
        Unit("bad", DIM_LENGTH, scale=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        Unit("bad_neg", DIM_LENGTH, scale=-1.0)

    with pytest.raises(TypeError, match="Cannot wrap a Quantity"):
        Quantity(mV(10.0), mV)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="hard_min"):
        StateVarSpec("bad_state", mV, default_value=0.0, hard_min=10.0, hard_max=5.0)

    with pytest.raises(ValueError, match="outside hard bounds"):
        StateVarSpec("out_of_bounds", mV, default_value=20.0, hard_min=0.0, hard_max=10.0)

    with pytest.raises(ValueError, match="outside bounds"):
        ParamSpec("bad_param", mV, default_value=-5.0, min_value=0.0, max_value=10.0)


def test_dynamic_unit_parser_and_malformed_inputs() -> None:
    # Nested SI base expression for Volt: (kg * m^2) / (A * s^3)
    u_volt = parse_unit("(kg * m^2) / (A * s^3)")
    assert u_volt.dimension == V.dimension
    assert pytest.approx(u_volt.scale) == 1.0

    # Left-associative division: J / mol / K == J / (mol * K)
    u_ent = parse_unit("kJ / mol / K")
    assert u_ent.dimension == R_GAS.dimension
    assert pytest.approx(u_ent.scale) == 1000.0

    # Negative and fractional powers
    u_acc = parse_unit("m * s^-2")
    assert u_acc.dimension == (m / (s**2)).dimension
    u_frac = parse_unit("m^(1.5) / s^(0.5)")
    assert u_frac.dimension == Dimension(L=1.5, T=-0.5)

    # Malformed expressions must raise DimensionError cleanly
    malformed_cases = [
        "",
        "   ",
        "mV / / ms",
        "((kJ / mol)",
        "kJ / mol)",
        "()",
        "mV ^",
        "mV ^ ()",
        "mV ^ abc",
        "mV ms",
        "* mV",
        "not_a_unit / s",
    ]
    for bad_expr in malformed_cases:
        with pytest.raises(DimensionError):
            parse_unit(bad_expr)


def test_buckingham_pi_edge_cases_and_multi_group_nullspace() -> None:
    # 1. Full-rank independent variables (mass, length, time) -> no dimensionless group exists
    independent = [("mass", parse_unit("kg")), ("length", m), ("time", s)]
    assert discover_dimensionless_groups(independent) == []

    # 2. Duplicate variable names must raise ValueError
    with pytest.raises(ValueError, match="unique"):
        discover_dimensionless_groups([("x", m), ("x", s)])

    # 3. Multi-group system: [V, F, R, T, t, R_mem, C_mem] has rank 5 and 7 variables -> 2 independent Pi groups
    multi_vars = [
        ("V", mV),
        ("F", FARADAY),
        ("R", R_GAS),
        ("T", BODY_TEMPERATURE_K),
        ("t", ms),
        ("R_mem", MOhm),
        ("C_mem", pF),
    ]
    groups = discover_dimensionless_groups(multi_vars)
    assert len(groups) == 2


# --- 2. Port-Hamiltonian, Symplectic Integration & Thermodynamic Fuzzing ---

def test_symplectic_midpoint_and_coupled_dirac_energy_conservation() -> None:
    port_a = PowerPortSpec("port_out", effort_unit=N_force, flow_unit=m_per_s, state_var_name="p")
    port_b = PowerPortSpec("port_in", effort_unit=mV, flow_unit=nA, state_var_name="p")

    brick_a = CanonicalHamiltonianBrick(
        name="osc_A", k_default=2.0, m_default=1.0, damping_default=0.0, port_specs=[port_a]
    )
    brick_b = CanonicalHamiltonianBrick(
        name="osc_B", k_default=2.0, m_default=1.0, damping_default=0.0, port_specs=[port_b]
    )
    conn = PortConnection(
        source_brick="osc_A",
        source_port=port_a,
        target_brick="osc_B",
        target_port=port_b,
        coupling_gain=0.5,
    )
    coupled = CoupledPortHamiltonianSystem(
        name="coupled_AB",
        bricks=[brick_a, brick_b],
        connections=[conn],
    )

    state = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    params = coupled.default_params(batch_size=1, dtype=torch.float64)

    initial_total_energy = float(coupled.hamiltonian_energy(state, params).item())
    assert audit_law_brick(coupled, state, params).passed_all is True

    curr = state
    for _ in range(200):
        curr = coupled.step_symplectic_midpoint(curr, params, dt=0.02)

    final_total_energy = float(coupled.hamiltonian_energy(curr, params).item())
    final_b_energy = float(brick_b.hamiltonian_energy(curr[:, 2:4], params[:, 3:6]).item())

    assert final_b_energy > 0.15
    assert pytest.approx(final_total_energy, rel=1e-6) == initial_total_energy


@settings(max_examples=30)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n_dim=st.integers(min_value=2, max_value=6),
)
def test_property_based_random_port_hamiltonian_thermodynamics(seed: int, n_dim: int) -> None:
    """
    For any random skew-symmetric matrix J = A - A^T and any random positive semi-definite
    dissipation matrix R = B^T B, the 1st and 2nd Laws of Thermodynamics must hold identically.
    """
    gen = torch.Generator().manual_seed(seed)
    a_mat = torch.randn((n_dim, n_dim), generator=gen, dtype=torch.float64)
    b_mat = torch.randn((n_dim, n_dim), generator=gen, dtype=torch.float64)
    j_fixed = a_mat - a_mat.T
    r_fixed = b_mat.T @ b_mat

    class RandomPHBrick(LawBrick):
        def __init__(self) -> None:
            specs = [StateVarSpec(f"x_{i}", DIMENSIONLESS, default_value=0.0) for i in range(n_dim)]
            super().__init__(name="rand_ph", domain="math", state_specs=specs)

        def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            return 0.5 * torch.sum(state**2, dim=-1)

        def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            return state

        def structure_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            return j_fixed.unsqueeze(0).expand(state.shape[0], -1, -1)

        def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            return r_fixed.unsqueeze(0).expand(state.shape[0], -1, -1)

    brick = RandomPHBrick()
    states = torch.randn((8, n_dim), generator=gen, dtype=torch.float64)
    params = brick.default_params(batch_size=8, dtype=torch.float64)
    res = verify_thermodynamic_laws(brick, states, params, tol=1e-7)
    assert res.passed is True


# --- 3. Extreme Scale Conditioning & Conservation Shields ---

def test_extreme_mass_scale_and_charge_conservation() -> None:
    """
    Test mass conservation when a 250,000 Da macromolecule binds a 1.008 Da proton
    across reaction rates spanning 12 orders of magnitude (1e-6 to 1e6).
    """
    s_mat = torch.tensor([[-1.0], [-1.0], [1.0]], dtype=torch.float64)
    masses = torch.tensor([250_000.0, 1.008, 250_001.008], dtype=torch.float64)
    valences = torch.tensor([-1.0, 1.0, 0.0], dtype=torch.float64)
    extreme_rates = torch.tensor([[1e-6], [1.0], [1e6]], dtype=torch.float64)

    assert verify_mass_conservation(s_mat, masses, reaction_rates=extreme_rates, tol=1e-9).passed is True
    assert verify_reaction_electroneutrality(s_mat, valences, tol=1e-9).passed is True

    # Non-finite stoichiometric input must fail cleanly without crashing
    nan_masses = torch.tensor([float("nan"), 1.0, 1.0])
    assert verify_mass_conservation(s_mat, nan_masses).passed is False


def test_wegscheider_detailed_balance_cycle_shield() -> None:
    k_fwd_valid = torch.tensor([[10.0, 2.0, 5.0, 1.0]])
    k_bwd_valid = torch.tensor([[4.0, 5.0, 5.0, 1.0]])
    assert verify_detailed_balance_cycle(k_fwd_valid, k_bwd_valid, 0.0).passed is True

    # Negative or zero rate must fail cleanly
    k_zero = torch.tensor([[0.0, 2.0, 5.0, 1.0]])
    assert verify_detailed_balance_cycle(k_zero, k_bwd_valid, 0.0).passed is False

    # Unbalanced perpetual cycle must fail
    k_fwd_bad = torch.tensor([[50.0, 2.0, 5.0, 1.0]])
    assert verify_detailed_balance_cycle(k_fwd_bad, k_bwd_valid, 0.0).passed is False


# --- 4. Unmodeled Residual Gate Tests ---

def test_unmodeled_residual_gate_blocks_uncalibrated_and_noisy_trajectories() -> None:
    """
    Verify that `analyze_unexplained_residual` rejects unmodeled-term discovery when:
      1. `parameter_fit_exhausted=False` (known parameters not yet tuned).
      2. Discrepancy is within the measurement/thermal noise floor (SNR < 3.0).
      3. Discrepancy is large (10x noise floor) but is uncorrelated random noise (low state R^2).
    And passes when a deterministic state-dependent force acts above the noise floor.
    """
    brick = CanonicalHamiltonianBrick(q_unit=m, p_unit=N_force * s, k_default=2.0, m_default=1.0, damping_default=0.1)
    gen = torch.Generator().manual_seed(42)
    states = torch.linspace(-2.0, 2.0, 64, dtype=torch.float64).unsqueeze(1).repeat(1, 2)
    states[:, 0] = torch.sin(states[:, 0])
    params = brick.default_params(batch_size=64, dtype=torch.float64)
    known_dx_dt = brick.flow(states, params)

    # True unmodeled mechanism: nonlinear cubic damping force Phi_p = -0.8 * p^3
    unmodeled_force = torch.stack([torch.zeros_like(states[:, 0]), -0.8 * (states[:, 1] ** 3)], dim=-1)
    obs_with_true_law = known_dx_dt + unmodeled_force

    # Gate 1 Test: parameter_fit_exhausted=False -> MUST REJECT even though unmodeled_force is present
    gate1 = analyze_unexplained_residual(
        brick, states, params, obs_with_true_law, noise_floor_std=0.01, parameter_fit_exhausted=False
    )
    assert gate1.latent_term_justified is False
    assert gate1.rejection_reason == "known_parameters_not_yet_exhausted"

    # Gate 2 Test: small thermal wobble below 3-sigma noise floor -> MUST REJECT
    small_wobble = 0.005 * torch.randn(states.shape, generator=gen, dtype=torch.float64)
    gate2 = analyze_unexplained_residual(
        brick, states, params, known_dx_dt + small_wobble, noise_floor_std=0.01, parameter_fit_exhausted=True
    )
    assert gate2.latent_term_justified is False
    assert gate2.rejection_reason == "residual_within_noise_floor"

    # Gate 3 Test: huge 10-sigma random static glitch (uncorrelated with state x) -> MUST REJECT
    huge_random_glitch = 0.15 * torch.randn(states.shape, generator=gen, dtype=torch.float64)
    gate3 = analyze_unexplained_residual(
        brick, states, params, known_dx_dt + huge_random_glitch, noise_floor_std=0.01, parameter_fit_exhausted=True
    )
    assert gate3.latent_term_justified is False
    assert gate3.rejection_reason == "residual_is_uncorrelated_noise"

    # All 3 Gates Pass: genuine state-dependent unmodeled cubic damping above noise floor
    discovery = analyze_unexplained_residual(
        brick, states, params, obs_with_true_law, noise_floor_std=0.01, parameter_fit_exhausted=True
    )
    assert discovery.latent_term_justified is True
    assert discovery.rejection_reason is None
    assert discovery.affected_state_indices == (1,)
    assert discovery.required_derivative_units[1].dimension == N_force.dimension
    assert discovery.thermodynamic_class == "passive_dissipative"
    assert discovery.state_coherence_r2 > 0.85


# --- 5. Matrix-Free 100k Batch, Mixed Timescales & Vault Tamper Detection ---

def test_matrix_free_100k_and_mixed_timescales() -> None:
    class MatrixFreeOscillator(LawBrick):
        def __init__(self) -> None:
            super().__init__(
                name="matrix_free_osc",
                domain="mechanics",
                state_specs=[
                    StateVarSpec("q", m, default_value=1.0),
                    StateVarSpec("p", N_force * s, default_value=0.0),
                ],
                matrix_free=True,
            )

        def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            return 0.5 * (state[:, 0] ** 2 + state[:, 1] ** 2)

        def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            return state

        def conservative_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            return torch.stack([state[:, 1], -state[:, 0]], dim=-1)

        def dissipative_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.zeros_like(state[:, 0]), 0.25 * state[:, 1]], dim=-1)

    mf_brick = MatrixFreeOscillator()
    batch_100k = torch.randn((100_000, 2), dtype=torch.float32)
    empty_params = mf_brick.default_params(batch_size=100_000)
    assert verify_thermodynamic_laws(mf_brick, batch_100k, empty_params).passed is True

    slow_brick = CanonicalHamiltonianBrick(name="slow_chem", damping_default=1.0)
    slow_brick.time_unit = s
    fast_brick = CanonicalHamiltonianBrick(name="fast_membrane", damping_default=1.0)
    fast_brick.time_unit = ms

    coupled_ms = CoupledPortHamiltonianSystem(
        name="chem_membrane_bridge",
        bricks=[slow_brick, fast_brick],
        time_unit=ms,
    )
    state = torch.tensor([[0.0, 2.0, 0.0, 2.0]], dtype=torch.float64)
    params = coupled_ms.default_params(batch_size=1, dtype=torch.float64)
    dx_dt = coupled_ms.flow(state, params)
    assert pytest.approx(float(dx_dt[0, 1].item()), rel=1e-6) == -0.002
    assert pytest.approx(float(dx_dt[0, 3].item()), rel=1e-6) == -2.0
    assert verify_thermodynamic_laws(coupled_ms, state, params).passed is True


def test_vault_isolation_lineage_and_disk_tamper_detection(tmp_path: Path) -> None:
    sandbox_file = tmp_path / "sandbox_vault.json"
    vault = LawVault(sandbox_storage_path=sandbox_file)

    vault.register_core_law(
        law_id="nernst_planck_v1",
        name="Nernst-Planck Electrodiffusion",
        domain="thermodynamics",
        symbolic_expression="E_ion = (R * T / (z * F)) * ln(c_out / c_in)",
    )

    brick = CanonicalHamiltonianBrick()
    passing_cert = audit_law_brick(brick)

    with pytest.raises(KeyError, match="unknown parent_law_id"):
        vault.crystallize_discovered_law(
            law_id="bad_lineage",
            name="Unverified Lineage",
            domain="neuroscience",
            symbolic_expression="x = 1",
            audit_certificate=passing_cert,
            falsification_score=0.95,
            parent_law_ids=("missing_parent",),
        )

    vault.crystallize_discovered_law(
        law_id="kv31_rescue_law_v1",
        name="Kv3.1 + Kv7 Cooperative Rescue Law",
        domain="neuroscience",
        symbolic_expression="dV_half_Kv31 = -4.37 mV, g_Kv7_scale = 1.184",
        audit_certificate=passing_cert,
        falsification_score=0.96,
        parent_law_ids=("nernst_planck_v1",),
    )

    # Tamper with the saved JSON file on disk and verify LawVault detects SHA-256 corruption on load
    raw_data = json.loads(sandbox_file.read_text(encoding="utf-8"))
    raw_data["kv31_rescue_law_v1"]["symbolic_expression"] = "dV_half_Kv31 = +999.0 mV"
    sandbox_file.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")

    with pytest.raises(ConservationViolationError, match="SHA-256 mismatch"):
        LawVault(sandbox_storage_path=sandbox_file)


def test_end_to_end_differentiable_inverse_gradients_through_coupled_bricks() -> None:
    """
    Prove that PyTorch `.backward()` flows cleanly through `LawBrick.effort`,
    `CoupledPortHamiltonianSystem.conservative_flow`, and `step_symplectic_midpoint`
    into tunable `params` even when `state.requires_grad` is False.
    """
    class AutogradOscillator(LawBrick):
        """Uses base `LawBrick.effort` (torch.autograd.grad) without closed-form override."""

        def __init__(self, name: str, port: PowerPortSpec) -> None:
            super().__init__(
                name=name,
                domain="mechanics",
                state_specs=[
                    StateVarSpec("q", m, default_value=1.0),
                    StateVarSpec("p", N_force * s, default_value=0.0),
                ],
                param_specs=[
                    ParamSpec("k", DIMENSIONLESS, default_value=2.0, min_value=0.1),
                    ParamSpec("m", DIMENSIONLESS, default_value=1.0, min_value=0.1),
                ],
                port_specs=[port],
            )

        def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            return 0.5 * params[:, 0] * (state[:, 0] ** 2) + 0.5 * (state[:, 1] ** 2) / params[:, 1]

        def structure_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            j = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=state.dtype, device=state.device)
            return j.unsqueeze(0).expand(state.shape[0], -1, -1).clone()

    port_a = PowerPortSpec("p_a", effort_unit=N_force, flow_unit=m_per_s, state_var_name="p")
    port_b = PowerPortSpec("p_b", effort_unit=N_force, flow_unit=m_per_s, state_var_name="p")
    b_a = AutogradOscillator("auto_A", port_a)
    b_b = AutogradOscillator("auto_B", port_b)
    conn = PortConnection("auto_A", port_a, "auto_B", port_b, coupling_gain=0.4)
    coupled = CoupledPortHamiltonianSystem("coupled_auto", [b_a, b_b], [conn])

    # Fixed initial state (requires_grad=False), tunable parameters (requires_grad=True)
    x0 = torch.tensor([[1.5, 0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=False)
    params = coupled.default_params(batch_size=1, dtype=torch.float64).requires_grad_(True)

    x = x0
    for _ in range(10):
        x = coupled.step_symplectic_midpoint(x, params, dt=0.05)

    target_q_b = torch.tensor([0.5], dtype=torch.float64)
    loss = ((x[:, 2] - target_q_b) ** 2).sum()
    loss.backward()

    assert params.grad is not None
    assert torch.isfinite(params.grad).all()
    assert float(torch.max(torch.abs(params.grad)).item()) > 1e-4


def test_vault_multi_generation_ancestry_chain_dag() -> None:
    """Verify topological ancestry traversal from a Generation-2 discovery back to Tier 1 axioms."""
    vault = LawVault()
    vault.register_core_law("axiom_nernst", "Nernst Equilibrium", "thermodynamics", "E = RT/zF ln(c_o/c_i)")
    vault.register_core_law("axiom_boltzmann", "Boltzmann Gate", "thermodynamics", "m_inf = 1/(1+exp((V_h-V)/k))")

    cert = audit_law_brick(CanonicalHamiltonianBrick())
    vault.crystallize_discovered_law(
        law_id="gen1_hh_channel",
        name="Voltage-Gated Conductance Law",
        domain="biophysics",
        symbolic_expression="I = g_max * m_inf^3 * (V - E)",
        audit_certificate=cert,
        falsification_score=0.95,
        parent_law_ids=("axiom_nernst", "axiom_boltzmann"),
    )
    vault.crystallize_discovered_law(
        law_id="gen2_kv31_rescue",
        name="Interneuron Rescue Law",
        domain="neuroscience",
        symbolic_expression="dV_h = -4.37 mV",
        audit_certificate=cert,
        falsification_score=0.97,
        parent_law_ids=("gen1_hh_channel",),
    )

    chain = vault.get_ancestry_chain("gen2_kv31_rescue")
    assert [r.law_id for r in chain] == [
        "axiom_nernst",
        "axiom_boltzmann",
        "gen1_hh_channel",
        "gen2_kv31_rescue",
    ]
    assert len(chain[-1].audit_summary) == 3

    # Verify an empty AuditCertificate with passed_all=True is rejected
    empty_cert = AuditCertificate(target_name="forged", passed_all=True, results=())
    with pytest.raises(ConservationViolationError, match="no shield results"):
        vault.crystallize_discovered_law(
            law_id="forged_law",
            name="Forged Law",
            domain="biophysics",
            symbolic_expression="x = 0",
            audit_certificate=empty_cert,
            falsification_score=0.99,
        )


def test_cross_energy_and_time_scale_coupled_hamiltonian_and_driven_first_law() -> None:
    """
    Verify that coupling a picojoule/millisecond brick with a joule/second brick
    accurately scales energy, effort, J, and R, and that `verify_thermodynamic_laws`
    accounts for external `port_inputs` in the First Law power balance.
    """
    from cortivra.layer1_axioms.units import J, ms, pJ

    class ScaledOscillator(CanonicalHamiltonianBrick):
        def __init__(self, name: str, port: PowerPortSpec, t_unit: Unit, e_unit: Unit) -> None:
            super().__init__(name=name, port_specs=[port])
            self.time_unit = t_unit
            self.energy_unit = e_unit

    p_micro = PowerPortSpec("p_micro", effort_unit=N_force, flow_unit=m_per_s, state_var_name="p")
    p_macro = PowerPortSpec("p_macro", effort_unit=N_force, flow_unit=m_per_s, state_var_name="p")

    b_micro = ScaledOscillator("micro", p_micro, t_unit=ms, e_unit=pJ)
    b_macro = ScaledOscillator("macro", p_macro, t_unit=s, e_unit=J)

    conn = PortConnection("micro", p_micro, "macro", p_macro, coupling_gain=0.25)
    coupled = CoupledPortHamiltonianSystem(
        name="multi_scale_energy_time",
        bricks=[b_micro, b_macro],
        connections=[conn],
        time_unit=s,
        energy_unit=J,
    )

    state = torch.tensor([[2.0, -1.0, 0.5, 1.0]], dtype=torch.float64)
    params = coupled.default_params(batch_size=1, dtype=torch.float64)

    # Check matrix vs vector conservative flow consistency across 1e-12 energy and 1e-3 time scales
    e = coupled.effort(state, params)
    f_cons_vec = coupled.conservative_flow(state, params)
    f_cons_mat = torch.bmm(coupled.structure_matrix(state, params), e.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(f_cons_vec, f_cons_mat, atol=1e-12)

    # Check First Law under external port input drive
    ext_flow = torch.tensor([[0.0, 0.5, 0.0, -0.3]], dtype=torch.float64)
    driven_res = verify_thermodynamic_laws(
        coupled,
        state,
        params,
        port_inputs={"external_flow": ext_flow},
        tol=1e-9,
    )
    assert driven_res.passed, driven_res.message


def test_bilinear_residual_detection_and_param_validation() -> None:
    """
    Verify that `analyze_unexplained_residual` detects pure bilinear coupling (`q * p`)
    whose marginal projection onto `[1, q, p, q^2, p^2]` alone is zero, and verify
    `validate_params` / `clamp_params` and magnetic unit parsing (`H/m`, `mT`).
    """
    assert parse_unit("H/m") == VACUUM_PERMEABILITY.unit
    assert parse_unit("mT").dimension == parse_unit("V*s/m^2").dimension

    brick = CanonicalHamiltonianBrick(damping_default=0.1)
    bad_params = torch.tensor([[-1.0, 1.0, 0.1]], dtype=torch.float64)
    with pytest.raises(ValueError, match="out of bounds"):
        brick.validate_params(bad_params)
    clamped = brick.clamp_params(bad_params)
    brick.validate_params(clamped)
    assert float(clamped[0, 0].item()) == 0.0

    # Symmetric grid where E[q * p * q] = E[q * p * q^2] = 0, so only bilinear cross-term q*p captures it
    grid = torch.linspace(-2.0, 2.0, 10, dtype=torch.float64)
    q_grid, p_grid = torch.meshgrid(grid, grid, indexing="ij")
    states = torch.stack([q_grid.reshape(-1), p_grid.reshape(-1)], dim=-1)  # [100, 2]
    params = brick.default_params(batch_size=100, dtype=torch.float64)

    known_dx_dt = brick.flow(states, params)
    bilinear_perturbation = torch.stack(
        [torch.zeros_like(states[:, 0]), -0.8 * states[:, 0] * states[:, 1]],
        dim=-1,
    )
    obs_dx_dt = known_dx_dt + bilinear_perturbation

    res = analyze_unexplained_residual(
        brick,
        states,
        params,
        observed_dx_dt=obs_dx_dt,
        noise_floor_std=0.01,
        parameter_fit_exhausted=True,
    )
    assert res.latent_term_justified
    assert res.state_coherence_r2 > 0.99


def test_generic_first_law_thermal_sink_conservation_and_3d_spatial_nodes() -> None:
    """
    Verify Upgrade 2 (GENERIC 1st-Law Thermal & Entropy Conservation) and
    Upgrade 3 (Universal 3D Euclidean SE(3) SpatialNode3DSpec):
      1. Dissipated mechanical free energy D(x) = grad_H . R grad_H >= 0 is routed 100% into
         the internal thermal sink state E_therm so H_total = H_work + E_therm is conserved.
      2. SpatialNode3DSpec extracts [Batch, NumNodes, 3] 3D Cartesian coordinates and validates
         common spatial length dimensions.
    """
    from cortivra.layer1_axioms.hamiltonian import SpatialNode3DSpec
    from cortivra.layer1_axioms.units import J, kg
    from cortivra.layer3_compiler.symbolic_compiler import SymbolicSystemSpec, compile_symbolic_system

    thermo_spec = SymbolicSystemSpec(
        name="generic_thermo_oscillator",
        domain="thermodynamics",
        state_specs=[
            StateVarSpec("rx", m, default_value=2.0),
            StateVarSpec("ry", m, default_value=1.0),
            StateVarSpec("rz", m, default_value=-0.5),
            StateVarSpec("px", N_force * s, default_value=0.0),
            StateVarSpec("py", N_force * s, default_value=1.5),
            StateVarSpec("pz", N_force * s, default_value=-0.8),
            StateVarSpec("E_therm", J, default_value=5.0, homeostatic_min=0.0, homeostatic_max=500.0, hard_min=0.0),
        ],
        param_specs=[
            ParamSpec("k_s", N_force / m, default_value=2.5),
            ParamSpec("inv_m", DIMENSIONLESS / kg, default_value=1.0, tunable=False),
            ParamSpec("gamma_d", kg / s, default_value=0.8, min_value=0.0),
        ],
        time_unit=s,
        energy_unit=J,
        hamiltonian_expr="0.5 * k_s * (rx**2 + ry**2 + rz**2) + 0.5 * inv_m * (px**2 + py**2 + pz**2)",
        dissipation_diag_exprs=("0.0", "0.0", "0.0", "gamma_d", "gamma_d", "gamma_d", "0.0"),
        skew_routing_exprs={(0, 3): "1.0", (1, 4): "1.0", (2, 5): "1.0"},
        thermal_sink_state_name="E_therm",
        thermal_capacity_scale=1.0,
        spatial_nodes_3d=(
            SpatialNode3DSpec(
                node_id="particle_3d",
                position_state_names=("rx", "ry", "rz"),
                momentum_state_names=("px", "py", "pz"),
            ),
        ),
    )
    brick = compile_symbolic_system(thermo_spec)
    x0 = brick.default_state(batch_size=1, dtype=torch.float64)
    params = brick.default_params(batch_size=1, dtype=torch.float64)

    coords_3d = brick.extract_3d_coordinates(x0)
    assert coords_3d.shape == (1, 1, 3)
    assert torch.allclose(coords_3d[0, 0], torch.tensor([2.0, 1.0, -0.5], dtype=torch.float64))

    cert = audit_law_brick(brick, sample_state=x0, sample_params=params, tol=1e-9)
    cert.raise_if_failed()
    assert cert.passed_all

    # Roll out 40 RK4 steps and verify H_work decreases, E_therm increases, and H_total is conserved!
    x_curr = x0.clone()
    h_work_0 = float(brick.hamiltonian_energy(x_curr, params).item())
    e_therm_0 = float(x_curr[0, 6].item())
    h_tot_0 = float(brick.total_conserved_energy(x_curr, params).item())
    assert abs(h_tot_0 - (h_work_0 + e_therm_0)) < 1e-12

    for _ in range(40):
        x_curr = brick.step_rk4(x_curr, params, dt=0.02)

    h_work_end = float(brick.hamiltonian_energy(x_curr, params).item())
    e_therm_end = float(x_curr[0, 6].item())
    h_tot_end = float(brick.total_conserved_energy(x_curr, params).item())

    assert h_work_end < h_work_0 - 1.0
    assert e_therm_end > e_therm_0 + 1.0
    assert abs(h_tot_end - h_tot_0) < 1e-5




