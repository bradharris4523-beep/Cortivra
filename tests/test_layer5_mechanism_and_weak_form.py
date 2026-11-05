"""
Verification and edge-case test suite for Layer 5 symbolic mechanism invention
from Pontryagin optimal control and variational bump weak-form multi-experiment
discovery (`cortivra.layer5_discovery`).
"""

import torch

from cortivra.layer1_axioms import (
    CanonicalHamiltonianBrick,
    DIMENSIONLESS,
    ParamSpec,
    StateVarSpec,
    s,
)
from cortivra.layer3_compiler import (
    SymbolicSystemSpec,
    compile_residual_patch_brick,
    compile_symbolic_system,
)
from cortivra.layer4_engine import rollout_trajectories
from cortivra.layer5_discovery import (
    build_variational_bump_integral_system,
    discover_and_compile_law,
    generate_symbolic_basis_library,
    synthesize_symbolic_feedback_mechanism,
)


def test_inverse_symbolic_mechanism_invention_from_pontryagin_control() -> None:
    """
    Verify Horizon 1 (`synthesize_symbolic_feedback_mechanism`):
    When `base_brick` has frozen/exhausted parameters that decay state `x` to `0.0` instead of
    the desired homeostatic target `x_target = 2.0`, `synthesize_symbolic_feedback_mechanism`:
        1. Computes the optimal Pontryagin control waveform `u^*(t)`.
        2. Projects `u^*(t)` onto Layer 5's Symbolic Basis Library to invent a closed-form
           feedback mechanism `u_catalyst(x, theta_new)`.
        3. Compiles and polishes an autonomous closed-loop `LawBrick` (`u_ext = 0`) that slashes
           target loss by > 90% while passing `audit_law_brick`.
    """
    spec = SymbolicSystemSpec(
        name="unactuated_decay_pathway",
        domain="chemistry",
        state_specs=[
            StateVarSpec("x", DIMENSIONLESS, default_value=0.4, homeostatic_min=0.0, homeostatic_max=10.0),
        ],
        param_specs=[
            ParamSpec("k_leak", DIMENSIONLESS / s, default_value=1.0, min_value=0.1, max_value=5.0, tunable=False),
        ],
        time_unit=s,
        energy_unit=DIMENSIONLESS,
        hamiltonian_expr="0.5 * (x^2)",
        dissipation_diag_exprs=["k_leak"],
    )
    base_brick = compile_symbolic_system(spec)
    x0 = torch.tensor([[0.4]], dtype=torch.float64)
    p_base = base_brick.default_params(batch_size=1, dtype=torch.float64)

    def _target_loss(traj: torch.Tensor) -> torch.Tensor:
        if traj.ndim == 3:
            return torch.mean((traj[:, -5:, 0] - 2.0) ** 2)
        return torch.mean((traj[-5:, 0] - 2.0) ** 2)

    inv_rep = synthesize_symbolic_feedback_mechanism(
        base_brick=base_brick,
        initial_state=x0,
        base_params=p_base,
        dt=0.05,
        num_steps=30,
        target_loss_fn=_target_loss,
        max_control_amplitude=5.0,
        control_optimization_steps=30,
        polish_optimization_steps=30,
    )
    assert inv_rep.audit_certificate.passed_all
    assert inv_rep.viable_across_trajectory
    assert inv_rep.autonomous_closed_loop_loss < inv_rep.baseline_target_loss * 0.10
    assert inv_rep.loss_reduction_ratio > 10.0
    assert "u_x(x) =" in inv_rep.invented_expression


def test_variational_bump_weak_form_and_multi_experiment_joint_discovery() -> None:
    """
    Verify the Adversarial Hardening upgrades in Layer 5:
        1. `build_variational_bump_integral_system` & `discover_and_compile_law` under additive
           state measurement noise (`observed_dx_dt=None`): integration-by-parts against compactly
           supported $C^4$ polynomial bump test functions $\\phi(t) = (1 - \\tau^2)^4$ plus
           Savitzky-Golay denoising recovers the exact missing nonlinear law (`diss_1_p_sq`)
           without ever needing clean ground-truth derivatives.
        2. Multi-Experiment / Multi-Initial-Condition Joint Discovery (`[NumExperiments, TimeSteps, StateDim]`):
           given a 3D tensor `[3, 26, 2]` of 3 distinct experimental trajectories, `discover_and_compile_law`
           and `solve_sparse_pareto_front` respect experiment boundaries (`num_experiments=3`) so no
           sliding integral window ever crosses from the end of experiment `e` into the start of `e+1`,
           recovering the exact missing law with `out_of_sample_r2 > 0.98`.
    """
    base = CanonicalHamiltonianBrick(k_default=1.5, m_default=1.0, damping_default=0.05)
    true_patched = compile_residual_patch_brick(
        name="true_cubic_damped_oscillator",
        base_brick=base,
        patch_param_specs=[ParamSpec("gamma_nl", base.state_specs[1].unit, default_value=0.6, min_value=0.0)],
        thermodynamic_mode="passive_dissipative",
        patch_expressions={1: "gamma_nl * p^2"},
    )
    p_true = torch.tensor([[1.5, 1.0, 0.05, 0.6]], dtype=torch.float64)
    dt = 0.025

    # Part 1: Noisy state-only trajectory (observed_dx_dt=None)
    x0_single = torch.tensor([[1.6, 0.9]], dtype=torch.float64)
    gt_single = rollout_trajectories(true_patched, x0_single, p_true, dt=dt, num_steps=64, integrator="rk4")
    clean_states = gt_single.states[0]  # [65, 2]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(777)
    noisy_states = clean_states + 0.025 * torch.randn(clean_states.shape, generator=gen, dtype=torch.float64)

    p_base = torch.tensor([[1.5, 1.0, 0.05]], dtype=torch.float64)
    base_flow_on_noisy = base.flow(noisy_states, p_base.expand(noisy_states.shape[0], -1))
    lib_noisy = generate_symbolic_basis_library(
        brick=base,
        states=noisy_states,
        base_params=p_base,
        thermodynamic_slot="passive_dissipative",
        affected_state_indices=[1],
        include_rational_primitives=False,
    )
    phi_cols = torch.stack([t.flow_column for t in lib_noisy.terms], dim=-1)
    phi_bump, y_bump = build_variational_bump_integral_system(
        flow_columns=phi_cols,
        target_residual=torch.zeros_like(noisy_states),
        dt=dt,
        window_steps=10,
        bump_degree=4,
        states=noisy_states,
        base_flow=base_flow_on_noisy,
    )
    assert phi_bump.shape[0] > 10 and phi_bump.shape[1] == len(lib_noisy.terms)
    assert y_bump.shape == (phi_bump.shape[0],)

    disc_noisy = discover_and_compile_law(
        base_brick=base,
        observed_states=noisy_states,
        dt=dt,
        observed_dx_dt=None,  # Force automatic noise detection + Savitzky-Golay + Variational Bump Weak Form!
        max_active_terms=1,
        custom_base_bounds={"k": (1.5, 1.5), "m": (1.0, 1.0), "damping": (0.05, 0.05)},
    )
    assert disc_noisy.discovery_triggered
    assert disc_noisy.winning_candidate is not None
    assert "diss_1_p_sq" in disc_noisy.winning_candidate.active_term_ids
    assert disc_noisy.audit_certificate.passed_all

    # Part 2: 3D Multi-Experiment / Multi-Initial-Condition Ensemble [3, 26, 2]
    x0_ensemble = torch.tensor(
        [
            [1.6, 0.0],
            [0.0, 1.5],
            [-1.3, -0.8],
        ],
        dtype=torch.float64,
    )
    gt_ens = rollout_trajectories(
        true_patched,
        x0_ensemble,
        p_true.expand(3, -1),
        dt=dt,
        num_steps=25,
        integrator="rk4",
    )
    ens_states = gt_ens.states  # [3, 26, 2]
    assert ens_states.shape == (3, 26, 2)

    disc_ens = discover_and_compile_law(
        base_brick=base,
        observed_states=ens_states,
        dt=dt,
        observed_dx_dt=None,
        max_active_terms=1,
        custom_base_bounds={"k": (1.5, 1.5), "m": (1.0, 1.0), "damping": (0.05, 0.05)},
    )
    assert disc_ens.discovery_triggered
    assert disc_ens.winning_candidate is not None
    assert "diss_1_p_sq" in disc_ens.winning_candidate.active_term_ids
    assert disc_ens.out_of_sample_r2 > 0.98
    assert disc_ens.audit_certificate.passed_all
