"""
End-to-end tests for the 5 Universal Blind Scientific Reconstruction Benchmarks
(`tests/test_blind_scientific_reconstructions.py`).
"""

from __future__ import annotations

from cortivra.layer6_agents import (
    run_benchmark_celestial_3d_unseen_perturber_inversion,
    run_benchmark_excitable_bioelectrical_and_collateral_cure,
    run_benchmark_locked_pathway_autonomous_catalyst_invention,
    run_benchmark_non_markovian_hidden_state_and_3d_multifield_pde,
    run_benchmark_quantum_symplectic_lindblad_and_kramers_escape,
    run_blind_scientific_reconstruction_suite,
)


def test_benchmark_1_excitable_bioelectrical_and_collateral_cure() -> None:
    res = run_benchmark_excitable_bioelectrical_and_collateral_cure()
    assert res.passed
    assert res.reconstruction_r2 >= 0.95
    assert res.target_loss_reduction_ratio >= 2.0
    assert res.lyapunov_basin_radius > 0.0
    assert res.passed_falsification


def test_benchmark_2_celestial_3d_unseen_perturber_inversion() -> None:
    res = run_benchmark_celestial_3d_unseen_perturber_inversion()
    assert res.passed
    assert res.metrics["direction_3d_error"] < 1e-5
    assert res.reconstruction_r2 >= 0.95
    assert res.passed_falsification


def test_benchmark_3_locked_pathway_autonomous_catalyst_invention() -> None:
    res = run_benchmark_locked_pathway_autonomous_catalyst_invention()
    assert res.passed
    assert res.target_loss_reduction_ratio > 8.0
    assert res.passed_falsification
    assert "u_x_biomarker(x) =" in res.discovered_symbolic_law


def test_benchmark_4_quantum_symplectic_lindblad_and_kramers_escape() -> None:
    res = run_benchmark_quantum_symplectic_lindblad_and_kramers_escape()
    assert res.passed
    assert res.metrics["born_norm_max_violation"] <= 1e-12
    assert abs(res.metrics["kramers_barrier_delta_h"] - 0.25) < 1e-4


def test_benchmark_5_non_markovian_hidden_state_and_3d_multifield_pde() -> None:
    res = run_benchmark_non_markovian_hidden_state_and_3d_multifield_pde()
    assert res.passed
    assert res.metrics["reconstructed_state_dim"] >= 2
    assert res.reconstruction_r2 >= 0.94
    assert res.metrics["multifield_pde_mass_drift"] < 1e-12


def test_full_blind_scientific_reconstruction_suite() -> None:
    suite = run_blind_scientific_reconstruction_suite()
    assert suite.passed_all
    assert suite.num_passed == 5
    assert suite.num_total == 5
    assert suite.mean_reconstruction_r2 >= 0.95
