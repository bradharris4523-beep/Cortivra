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


def test_full_blind_scientific_reconstruction_suite_and_all_five_benchmarks() -> None:
    suite = run_blind_scientific_reconstruction_suite()
    assert suite.passed_all
    assert suite.num_passed == 5
    assert suite.num_total == 5
    assert suite.mean_reconstruction_r2 >= 0.95

    r1, r2, r3, r4, r5 = suite.results
    assert r1.passed and r1.reconstruction_r2 >= 0.95 and r1.target_loss_reduction_ratio >= 2.0
    assert r1.lyapunov_basin_radius > 0.0 and r1.passed_falsification

    assert r2.passed and r2.metrics["direction_3d_error"] < 1e-5 and r2.reconstruction_r2 >= 0.95
    assert r3.passed and r3.target_loss_reduction_ratio > 8.0 and "u_x_biomarker(x) =" in r3.discovered_symbolic_law
    assert r4.passed and r4.metrics["born_norm_max_violation"] <= 1e-12
    assert abs(r4.metrics["kramers_barrier_delta_h"] - 0.25) < 1e-4
    assert r5.passed and r5.metrics["reconstructed_state_dim"] >= 2 and r5.reconstruction_r2 >= 0.94
    assert r5.metrics["multifield_pde_mass_drift"] < 1e-12

