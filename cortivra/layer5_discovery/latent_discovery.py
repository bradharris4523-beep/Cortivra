"""
Closed-loop symbolic law compilation, Hamiltonian potential patching, non-Markovian
latent state augmentation, and cross-domain archetype classification
(`cortivra.layer5_discovery.latent_discovery`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick, ParamSpec, StateVarSpec
from cortivra.layer1_axioms.shields import AuditCertificate, audit_law_brick
from cortivra.layer1_axioms.units import (
    DIMENSIONLESS,
    DimensionError,
    DimensionlessGroup,
    J,
    Quantity,
    Unit,
    discover_dimensionless_groups,
    s,
)
from cortivra.layer3_compiler.ast_nodes import (
    BinaryOpNode,
    ConstNode,
    ExprNode,
    PrimitiveCallNode,
    SymbolNode,
    UnaryOpNode,
    parse_symbolic_expr,
)
from cortivra.layer3_compiler.symbolic_compiler import (
    CompiledLawBrick,
    ResidualPatchedLawBrick,
    SymbolicSystemSpec,
    build_symbol_table,
    compile_residual_patch_brick,
    compile_symbolic_system,
)
from cortivra.layer4_engine.elimination_funnel import (
    HypothesisRankingEntry,
    rank_hypothesis_portfolio,
    run_elimination_funnel,
)
from cortivra.layer4_engine.inverse_solver import (
    CalibrationAndResidualReport,
    OptimalControlResult,
    calibrate_and_audit_residual,
    solve_optimal_control_waveform,
    solve_target_intervention,
)
from cortivra.layer4_engine.rollout_arena import (
    estimate_trajectory_derivatives,
    estimate_trajectory_noise_std,
    rollout_trajectories,
    savitzky_golay_denoise_and_differentiate,
)
from cortivra.layer5_discovery.basis_library import generate_symbolic_basis_library
from cortivra.layer5_discovery.helmholtz_and_perturber import (
    DifferentialResidualInversionReport,
    LatentAugmentationResult,
    LatentAugmentedLawBrick,
    LatentOscillatorAugmentedLawBrick,
    NonMarkovianMemoryReport,
    detect_non_markovian_memory,
    discover_latent_state_augmentation,
    invert_residual_for_unseen_perturber,
)
from cortivra.layer5_discovery.sparse_regression import (
    InvariantDiscoveryReport,
    ParetoDiscoveryReport,
    SymbolicCandidate,
    discover_conservation_invariants,
    discover_fractional_power_and_log_candidates,
    discover_rational_pade_candidates,
    solve_sparse_pareto_front,
)
from cortivra.observability.flight_recorder import (
    AlternativeBranchRecord,
    emit_trace_node,
    summarize_singular_spectrum,
)


class HamiltonianPatchedLawBrick(LawBrick):
    """
    Composite `LawBrick` formed by augmenting `base_brick` with a symbolic potential energy
    addition `Delta_H(x, params)` in `base_brick.energy_unit`.

    The effort vector `e(x) = grad_x(H_base(x) + Delta_H(x))` is differentiated symbolically
    and routed through `base_brick`'s `(J(x) - R(x))` operators, preserving exact conservative
    orthogonality (`grad_H . F_cons == 0`) and 2nd Law non-negativity (`grad_H . F_diss >= 0`).
    """

    def __init__(
        self,
        name: str,
        base_brick: LawBrick,
        patch_param_specs: Sequence[ParamSpec],
        delta_hamiltonian_expr: str | ExprNode,
        constants: Mapping[str, Quantity | ConstNode] | None = None,
    ) -> None:
        self.base_brick = base_brick
        self.base_param_dim = base_brick.param_dim
        self.patch_param_specs = tuple(patch_param_specs)

        combined_params = list(base_brick.param_specs) + list(self.patch_param_specs)
        super().__init__(
            name=name,
            domain=base_brick.domain,
            state_specs=base_brick.state_specs,
            param_specs=combined_params,
            port_specs=base_brick.port_specs,
            time_unit=base_brick.time_unit,
            energy_unit=base_brick.energy_unit,
            matrix_free=base_brick.is_matrix_free,
        )

        self.symbol_table = build_symbol_table(self.state_specs, self.param_specs, constants)
        if isinstance(delta_hamiltonian_expr, ExprNode):
            node = delta_hamiltonian_expr.simplify()
            node.infer_unit()
        else:
            node = parse_symbolic_expr(delta_hamiltonian_expr, symbol_table=self.symbol_table).simplify()

        h_u = node.infer_unit()
        if h_u.dimension != self.energy_unit.dimension and h_u.dimension != DIMENSIONLESS.dimension:
            raise DimensionError(
                f"HamiltonianPatchedLawBrick '{name}': Delta_H has dimension "
                f"{h_u.dimension.format_si()}, expected {self.energy_unit.dimension.format_si()}."
            )
        self._h_scale = (
            h_u.conversion_factor_to(self.energy_unit)
            if h_u.dimension == self.energy_unit.dimension
            else 1.0
        )
        self.delta_hamiltonian_ast = node
        self._delta_effort_asts = [
            self.delta_hamiltonian_ast.diff(sp.name).simplify() for sp in self.state_specs
        ]

    def _build_env(self, state: torch.Tensor, params: torch.Tensor) -> dict[str, torch.Tensor]:
        env: dict[str, torch.Tensor] = {}
        for idx, sp in enumerate(self.state_specs):
            env[sp.name] = state[:, idx]
        for idx, pp in enumerate(self.param_specs):
            env[pp.name] = params[:, idx]
        return env

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        h_base = self.base_brick.hamiltonian_energy(state, params[:, : self.base_param_dim])
        env = self._build_env(state, params)
        h_delta = self.delta_hamiltonian_ast.eval_tensor(env) * self._h_scale
        return h_base + h_delta

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        e_base = self.base_brick.effort(state, params[:, : self.base_param_dim])
        env = self._build_env(state, params)
        e_delta = torch.stack(
            [node.eval_tensor(env) * self._h_scale for node in self._delta_effort_asts],
            dim=-1,
        )
        return e_base + e_delta

    def structure_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return self.base_brick.structure_matrix(state, params[:, : self.base_param_dim])

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return self.base_brick.dissipation_matrix(state, params[:, : self.base_param_dim])

    def external_drive_power(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        base_p = params[:, : self.base_param_dim]
        if hasattr(self.base_brick, "compute_compiled_external_drive"):
            e = self.effort(state, params)
            u = self.base_brick.compute_compiled_external_drive(state, base_p)
            return (e * u).sum(dim=-1)
        return torch.zeros((state.shape[0],), dtype=state.dtype, device=state.device)

    def flow(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        port_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        dx_dt = super().flow(state, params, port_inputs=port_inputs)
        base_p = params[:, : self.base_param_dim]
        if hasattr(self.base_brick, "compute_compiled_external_drive"):
            dx_dt = dx_dt + self.base_brick.compute_compiled_external_drive(state, base_p)
        return dx_dt

    def total_patch_complexity(self) -> int:
        return self.delta_hamiltonian_ast.tree_complexity()


@dataclass(frozen=True)
class ArchetypeFingerprint:
    """
    Universal mathematical archetype classification, differential operator invariants,
    and dimensionless Buckingham-Pi invariants.
    """

    archetype_name: str
    thermodynamic_slot: str
    parity_symmetry: str
    is_saturating: bool
    dimensionless_groups: tuple[DimensionlessGroup, ...]
    cross_domain_analogues: tuple[str, ...]
    mean_phase_space_divergence: float = 0.0
    volume_behavior: str = "volume_preserving_liouville"
    max_polynomial_degree: int = 1
    coupled_state_pairs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ClosedLoopDiscoveryReport:
    """
    Complete end-to-end report from Layer 5 (`discover_and_compile_law`).
    """

    discovery_triggered: bool
    winning_brick: LawBrick
    winning_params: torch.Tensor
    winning_candidate: SymbolicCandidate | None
    pareto_report: ParetoDiscoveryReport | None
    portfolio_rankings: tuple[HypothesisRankingEntry, ...]
    memory_report: NonMarkovianMemoryReport
    latent_augmentation: LatentAugmentationResult | None
    archetype: ArchetypeFingerprint
    audit_certificate: AuditCertificate
    out_of_sample_r2: float = 1.0


def compile_symbolic_candidate(
    base_brick: LawBrick,
    candidate: SymbolicCandidate,
    name_prefix: str = "discovered",
) -> LawBrick:
    """
    Compile a `SymbolicCandidate` onto `base_brick` as either a `HamiltonianPatchedLawBrick`
    (when `thermodynamic_slot == "hamiltonian_potential"`) or a `ResidualPatchedLawBrick`.
    """
    brick_name = f"{base_brick.name}_{name_prefix}_{candidate.candidate_id}"
    if candidate.thermodynamic_slot == "hamiltonian_potential":
        h_expr = candidate.patch_expressions["hamiltonian"]
        return HamiltonianPatchedLawBrick(
            name=brick_name,
            base_brick=base_brick,
            patch_param_specs=candidate.param_specs,
            delta_hamiltonian_expr=h_expr,
        )

    patch_exprs_typed: dict[int | tuple[int, int], ExprNode] = {}
    for k, v in candidate.patch_expressions.items():
        assert isinstance(k, (int, tuple))
        patch_exprs_typed[k] = v

    return compile_residual_patch_brick(
        name=brick_name,
        base_brick=base_brick,
        patch_param_specs=candidate.param_specs,
        thermodynamic_mode=candidate.thermodynamic_slot,
        patch_expressions=patch_exprs_typed,
    )


def _infer_ast_polynomial_degree(node: ExprNode, state_names: set[str]) -> int:
    """Recursively compute the polynomial degree of `node` with respect to `state_names`."""
    if isinstance(node, ConstNode):
        return 0
    if isinstance(node, SymbolNode):
        return 1 if node.name in state_names else 0
    if isinstance(node, UnaryOpNode):
        return _infer_ast_polynomial_degree(node.operand, state_names)
    if isinstance(node, BinaryOpNode):
        dl = _infer_ast_polynomial_degree(node.left, state_names)
        dr = _infer_ast_polynomial_degree(node.right, state_names)
        if node.op in ("+", "-"):
            return max(dl, dr)
        if node.op == "*":
            return dl + dr
        if node.op == "**" and isinstance(node.right, ConstNode):
            return int(round(dl * abs(node.right.value)))
        return max(dl, dr)
    if isinstance(node, PrimitiveCallNode):
        return max((_infer_ast_polynomial_degree(a, state_names) for a in node.args), default=0)
    return 1


def _ast_has_rational_or_saturation(node: ExprNode, state_names: set[str]) -> bool:
    """Return True if `node` contains a saturating primitive or state-dependent denominator."""
    if isinstance(node, PrimitiveCallNode):
        if node.primitive_name in ("hill", "hill_inhib", "michaelis_menten", "boltzmann", "saturating_drive"):
            return True
        return any(_ast_has_rational_or_saturation(a, state_names) for a in node.args)
    if isinstance(node, UnaryOpNode):
        if node.op in ("sin", "cos", "tanh"):
            return True
        return _ast_has_rational_or_saturation(node.operand, state_names)
    if isinstance(node, BinaryOpNode):
        if node.op == "/" and bool(node.right.free_symbols() & state_names):
            return True
        return _ast_has_rational_or_saturation(node.left, state_names) or _ast_has_rational_or_saturation(
            node.right, state_names
        )
    return False


def _compute_mean_phase_space_divergence(brick: LawBrick) -> tuple[float, str]:
    """
    Compute the mean phase-space divergence `<Tr(dF/dx)>` across probe states via exact autograd
    to classify Liouville volume preservation vs. dissipative contraction vs. active expansion.
    """
    state_dim = brick.state_dim
    probes = brick.clamp_to_hard_bounds(
        torch.stack(
            [
                brick.default_state(batch_size=1, dtype=torch.float64)[0] + offset
                for offset in (-0.45, 0.15, 0.65)
            ],
            dim=0,
        )
    )
    p_batch = brick.default_params(batch_size=3, dtype=torch.float64)
    traces: list[float] = []
    for b in range(3):
        xb = probes[b : b + 1].detach().requires_grad_(True)
        pb = p_batch[b : b + 1]
        fb = brick.flow(xb, pb)
        if not fb.requires_grad:
            traces.append(0.0)
            continue
        tr_val = 0.0
        for d in range(state_dim):
            if not fb[0, d].requires_grad:
                continue
            grad_d = torch.autograd.grad(
                fb[0, d],
                xb,
                retain_graph=(d < state_dim - 1),
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad_d is not None:
                tr_val += float(grad_d[0, d].item())
        traces.append(tr_val)
    mean_div = sum(traces) / len(traces)
    if abs(mean_div) <= 1e-5:
        behavior = "volume_preserving_liouville"
    elif mean_div < -1e-5:
        behavior = "volume_contracting_attractor"
    else:
        behavior = "volume_expanding_instability"
    return mean_div, behavior


def classify_mathematical_archetype(
    brick: LawBrick,
    candidate: SymbolicCandidate | None = None,
    is_latent_augmented: bool = False,
) -> ArchetypeFingerprint:
    """
    Classify the universal mathematical operator archetype of a discovered symbolic law,
    compute its phase-space divergence `<Tr(dF/dx)>`, AST polynomial degree, coupling topology,
    and dimensionless Buckingham-Pi invariants.
    """
    variables: list[tuple[str, object]] = []
    seen_names: set[str] = set()
    state_names: set[str] = set()
    for sp in brick.state_specs:
        variables.append((sp.name, sp.unit))
        seen_names.add(sp.name)
        state_names.add(sp.name)
    for pp in brick.param_specs:
        if pp.name not in seen_names:
            variables.append((pp.name, pp.unit))
            seen_names.add(pp.name)
    variables.append(("t_char", brick.time_unit))

    pi_groups = tuple(discover_dimensionless_groups(variables))  # type: ignore[arg-type]
    mean_div, vol_behavior = _compute_mean_phase_space_divergence(brick)

    if is_latent_augmented:
        return ArchetypeFingerprint(
            archetype_name="non_markovian_latent_adaptation_loop",
            thermodynamic_slot="latent_state_augmentation",
            parity_symmetry="asymmetric",
            is_saturating=False,
            dimensionless_groups=pi_groups,
            cross_domain_analogues=(
                "Photoreceptor / GPCR intracellular calcium adaptation loop",
                "Maxwell-Wiechert viscoelastic polymer stress relaxation",
                "Integral-feedback control & thermal reservoir coupling",
            ),
            mean_phase_space_divergence=mean_div,
            volume_behavior=vol_behavior,
            max_polynomial_degree=1,
            coupled_state_pairs=((brick.state_specs[0].name, brick.state_specs[-1].name),),
        )

    if candidate is None:
        return ArchetypeFingerprint(
            archetype_name="canonical_baseline_law",
            thermodynamic_slot="baseline",
            parity_symmetry="odd",
            is_saturating=False,
            dimensionless_groups=pi_groups,
            cross_domain_analogues=(
                "Harmonic mechanical oscillator",
                "Linear LC/RC electrodynamic resonator",
            ),
            mean_phase_space_divergence=mean_div,
            volume_behavior=vol_behavior,
            max_polynomial_degree=2,
            coupled_state_pairs=(),
        )

    infix_all = " + ".join(node.to_infix() for node in candidate.patch_expressions.values())
    slot = candidate.thermodynamic_slot

    max_deg = max(
        (_infer_ast_polynomial_degree(node, state_names) for node in candidate.patch_expressions.values()),
        default=1,
    )
    is_saturating = any(
        _ast_has_rational_or_saturation(node, state_names) for node in candidate.patch_expressions.values()
    )

    coupled_pairs_set: set[tuple[str, str]] = set()
    for key, node in candidate.patch_expressions.items():
        syms = sorted(node.free_symbols() & state_names)
        for i_s in range(len(syms)):
            for j_s in range(i_s + 1, len(syms)):
                coupled_pairs_set.add((syms[i_s], syms[j_s]))
        if isinstance(key, int) and 0 <= key < brick.state_dim:
            target_name = brick.state_specs[key].name
            for s_name in syms:
                if s_name != target_name:
                    pair = (min(target_name, s_name), max(target_name, s_name))
                    coupled_pairs_set.add(pair)

    is_hill_or_mm = (
        ("hill(" in infix_all)
        or ("hill_inhib(" in infix_all)
        or ("michaelis_menten(" in infix_all)
        or (is_saturating and "/" in infix_all)
    )
    is_trig = ("sin(" in infix_all) or ("cos(" in infix_all)

    x_probe = torch.full((1, brick.state_dim), 0.73, dtype=torch.float64)
    p_probe = brick.default_params(batch_size=1, dtype=torch.float64)
    if isinstance(brick, ResidualPatchedLawBrick) and slot == "passive_dissipative":
        r_pos = brick.dissipation_matrix(x_probe, p_probe) - brick.base_brick.dissipation_matrix(
            x_probe, p_probe[:, : brick.base_param_dim]
        )
        r_neg = brick.dissipation_matrix(-x_probe, p_probe) - brick.base_brick.dissipation_matrix(
            -x_probe, p_probe[:, : brick.base_param_dim]
        )
        diff_even = float(torch.max(torch.abs(r_pos - r_neg)).item())
        parity = "even" if diff_even < 1e-7 else "asymmetric"
    else:
        f_pos = brick.flow(x_probe, p_probe)
        f_neg = brick.flow(-x_probe, p_probe)
        if float(torch.max(torch.abs(f_pos + f_neg)).item()) < 1e-6:
            parity = "odd"
        elif float(torch.max(torch.abs(f_pos - f_neg)).item()) < 1e-6:
            parity = "even"
        else:
            parity = "asymmetric"

    if is_hill_or_mm:
        archetype = "cooperative_hill_michaelis_saturation"
        analogues = (
            "Michaelis-Menten / Monod-Wyman-Changeux enzyme & receptor kinetics",
            "Langmuir monolayer surface adsorption isotherm",
            "Saturating transconductance differential amplifier stage",
        )
    elif is_trig:
        archetype = "trigonometric_phase_restoring"
        analogues = (
            "Nonlinear gravitational pendulum restoring torque",
            "Josephson junction superconducting phase-current relation",
            "Kuramoto coupled phase oscillator synchronization",
        )
    elif slot == "hamiltonian_potential":
        archetype = "conservative_anharmonic_potential"
        analogues = (
            "Duffing anharmonic elastic beam / Landau quartic free-energy potential",
            "Optical Kerr nonlinear refractive index dispersion",
            "Anharmonic interatomic bond potential",
        )
    elif slot == "passive_dissipative":
        archetype = "nonlinear_state_dependent_dissipation"
        analogues = (
            "Rayleigh / Van der Pol velocity-squared nonlinear damping",
            "Turbulent aerodynamic & fluid drag",
            "Voltage-dependent nonlinear membrane chord conductance",
        )
    elif slot == "conservative_coupling":
        archetype = "conservative_gyroscopic_poisson_coupling"
        analogues = (
            "Lorentz magnetic / Coriolis gyroscopic cross-force",
            "Ideal lossless transformer / gyrator network coupling",
            "Lotka-Volterra conservative predator-prey exchange",
        )
    else:
        archetype = "nonlinear_polynomial_state_feedback"
        analogues = (
            "Autocatalytic reaction rate feedback",
            "FitzHugh-Nagumo / bistable excitable membrane current",
            "Nonlinear mechanical restoring & forcing field",
        )

    return ArchetypeFingerprint(
        archetype_name=archetype,
        thermodynamic_slot=slot,
        parity_symmetry=parity,
        is_saturating=is_saturating,
        dimensionless_groups=pi_groups,
        cross_domain_analogues=analogues,
        mean_phase_space_divergence=mean_div,
        volume_behavior=vol_behavior,
        max_polynomial_degree=max_deg,
        coupled_state_pairs=tuple(sorted(coupled_pairs_set)),
    )


def discover_and_compile_law(
    base_brick: LawBrick,
    observed_states: torch.Tensor,
    dt: float,
    observed_dx_dt: torch.Tensor | None = None,
    calibration_report: CalibrationAndResidualReport | None = None,
    noise_floor_std: float = 1e-3,
    max_polynomial_degree: int = 4,
    max_active_terms: int = 2,
    top_k_to_compile: int = 3,
    include_rational_primitives: bool = True,
    custom_base_bounds: dict[str, tuple[float, float]] | None = None,
) -> ClosedLoopDiscoveryReport:
    """
    Execute the complete closed-loop scientific discovery workflow:
        1. Calibrate `base_brick` parameters and run Layer 1's `UnmodeledResidualGate`.
        2. If `not calibration_report.latent_mechanism_required`, return `base_brick` immediately.
        3. Check for non-Markovian hidden-state hysteresis via `detect_non_markovian_memory`;
           if detected, synthesize a `(D + 1)`-state `LatentAugmentedLawBrick`.
        4. Construct dimensionally verified `SymbolicBasisLibrary` instances across the
           thermodynamic slots indicated by `residual_analysis.thermodynamic_class`, plus
           Sanathanan-Koerner rational Padé candidates.
        5. Solve the sparse Pareto front (`solve_sparse_pareto_front`) using dual pointwise
           and weak-integral collocation.
        6. Compile the top Pareto candidates into executable `LawBrick`s, audit them with
           `audit_law_brick`, and rank/polish them via Layer 4's `rank_hypothesis_portfolio`.
        7. Classify the winning law's universal mathematical archetype (`classify_mathematical_archetype`).
    """
    x_in = observed_states.detach().to(torch.float64)
    if x_in.ndim == 3:
        num_exps, t_steps, state_dim = x_in.shape
        x_flat = x_in.reshape(num_exps * t_steps, state_dim)
        x_first = x_in[0]
    else:
        num_exps = 1
        t_steps, state_dim = x_in.shape
        x_flat = x_in
        x_first = x_in

    var_bump_weight = 0.0
    x_cal = x_in
    if observed_dx_dt is not None:
        dx_in = observed_dx_dt.detach().to(torch.float64)
        dx_flat = dx_in.reshape(num_exps * t_steps, state_dim) if dx_in.ndim == 3 else dx_in
        dx_first = dx_in[0] if dx_in.ndim == 3 else dx_in
    else:
        noise_std = estimate_trajectory_noise_std(x_in)
        sig_std = max(float(torch.std(x_in).item()), 1e-12)
        is_noisy = t_steps >= 9 and noise_std > 0.008 * sig_std and noise_std > 25.0 * (dt**3) * sig_std
        if is_noisy:
            var_bump_weight = 0.75
            x_smooth, dx_in = savitzky_golay_denoise_and_differentiate(x_in, dt=dt)
            x_cal = x_smooth
            x_flat = x_smooth.reshape(num_exps * t_steps, state_dim) if x_smooth.ndim == 3 else x_smooth
            x_first = x_smooth[0] if x_smooth.ndim == 3 else x_smooth
            dx_flat = dx_in.reshape(num_exps * t_steps, state_dim) if dx_in.ndim == 3 else dx_in
            dx_first = dx_in[0] if dx_in.ndim == 3 else dx_in
        elif x_in.ndim == 3:
            dx_list = [estimate_trajectory_derivatives(x_in[e], dt=dt) for e in range(num_exps)]
            dx_in = torch.stack(dx_list, dim=0)
            dx_flat = dx_in.reshape(num_exps * t_steps, state_dim)
            dx_first = dx_in[0]
        else:
            dx_in = estimate_trajectory_derivatives(x_in, dt=dt)
            dx_flat = dx_in
            dx_first = dx_in

    if calibration_report is None:
        calibration_report = calibrate_and_audit_residual(
            brick=base_brick,
            observed_states=x_cal,
            dt=dt,
            observed_dx_dt=dx_in,
            noise_floor_std=noise_floor_std,
            custom_bounds=custom_base_bounds,
        )

    p_base = calibration_report.best_params.detach().to(torch.float64)
    res_analysis = calibration_report.residual_analysis
    residual_flow = res_analysis.residual_flow.detach().to(torch.float64)
    x = x_flat
    dx = dx_flat
    if var_bump_weight > 0.0:
        with torch.no_grad():
            f_base_eval = base_brick.flow(x, p_base.expand(x.shape[0], -1)).detach()
            residual_flow = dx - f_base_eval

    p_nom = base_brick.default_params(batch_size=1, dtype=torch.float64, device=x.device)
    if torch.linalg.norm(p_base - p_nom) > 1e-4:
        from cortivra.layer1_axioms.shields import analyze_unexplained_residual as _analyze_res

        res_nom = _analyze_res(
            brick=base_brick,
            state=x,
            params=p_nom.expand(x.shape[0], -1),
            observed_dx_dt=dx,
            noise_floor_std=noise_floor_std,
            parameter_fit_exhausted=True,
        )
        if (
            res_nom.latent_term_justified
            and res_nom.thermodynamic_class in ("passive_dissipative", "conservative_coupling")
            and res_analysis.thermodynamic_class in ("reactive_storage", "active_driven")
            and res_nom.state_coherence_r2 >= res_analysis.state_coherence_r2 - 0.02
        ):
            p_base = p_nom
            res_analysis = res_nom
            residual_flow = res_nom.residual_flow.detach().to(torch.float64)

    memory_report = detect_non_markovian_memory(x_first, residual_flow[:t_steps], dt=dt)
    needs_discovery = bool(
        calibration_report.latent_mechanism_required
        or (res_analysis.max_snr >= 3.0 and memory_report.is_non_markovian)
    )

    if not needs_discovery:
        cert = audit_law_brick(base_brick)
        arch = classify_mathematical_archetype(base_brick, candidate=None, is_latent_augmented=False)
        return ClosedLoopDiscoveryReport(
            discovery_triggered=False,
            winning_brick=base_brick,
            winning_params=p_base,
            winning_candidate=None,
            pareto_report=None,
            portfolio_rankings=(),
            memory_report=memory_report,
            latent_augmentation=None,
            archetype=arch,
            audit_certificate=cert,
        )

    if memory_report.is_non_markovian:
        latent_res = discover_latent_state_augmentation(
            base_brick=base_brick,
            observed_states=x,
            base_params=p_base,
            residual_flow=residual_flow,
            dt=dt,
            target_state_index=memory_report.primary_affected_state,
        )
        arch = classify_mathematical_archetype(
            latent_res.augmented_brick, candidate=None, is_latent_augmented=True
        )
        return ClosedLoopDiscoveryReport(
            discovery_triggered=True,
            winning_brick=latent_res.augmented_brick,
            winning_params=latent_res.calibrated_params,
            winning_candidate=None,
            pareto_report=None,
            portfolio_rankings=(),
            memory_report=memory_report,
            latent_augmentation=latent_res,
            archetype=arch,
            audit_certificate=latent_res.audit_certificate,
        )

    thermo_class = res_analysis.thermodynamic_class
    affected = list(res_analysis.affected_state_indices)
    if thermo_class == "passive_dissipative":
        slots_to_search = ["passive_dissipative", "active_driven"]
    elif thermo_class == "reactive_storage":
        slots_to_search = ["hamiltonian_potential", "passive_dissipative", "active_driven"]
    elif thermo_class == "conservative_coupling":
        slots_to_search = ["conservative_coupling", "hamiltonian_potential", "active_driven"]
    else:
        slots_to_search = ["passive_dissipative", "hamiltonian_potential", "active_driven"]

    all_pareto_candidates: list[SymbolicCandidate] = []
    primary_pareto_report: ParetoDiscoveryReport | None = None

    for slot in slots_to_search:
        if slot == "hamiltonian_potential" and base_brick.is_matrix_free:
            continue
        if slot == "conservative_coupling" and base_brick.state_dim < 2:
            continue

        lib = generate_symbolic_basis_library(
            brick=base_brick,
            states=x,
            base_params=p_base,
            thermodynamic_slot=slot,
            affected_state_indices=affected,
            residual_flow=residual_flow,
            max_polynomial_degree=max_polynomial_degree,
            include_rational_primitives=include_rational_primitives,
        )
        extra_cands: list[SymbolicCandidate] = []
        if slot == "active_driven" and include_rational_primitives:
            extra_cands = discover_rational_pade_candidates(
                brick=base_brick,
                states=x,
                target_residual=residual_flow,
                affected_state_indices=affected,
            )
            extra_cands.extend(
                discover_fractional_power_and_log_candidates(
                    brick=base_brick,
                    states=x,
                    target_residual=residual_flow,
                    affected_state_indices=affected,
                )
            )

        if len(lib.terms) == 0 and not extra_cands:
            continue

        with torch.no_grad():
            f_base_slot = base_brick.flow(x, p_base.expand(x.shape[0], -1)).detach()
        rep = solve_sparse_pareto_front(
            library=lib,
            target_residual=residual_flow,
            dt=dt,
            max_active_terms=max_active_terms,
            extra_candidates=extra_cands,
            num_experiments=num_exps,
            states=x,
            base_flow=f_base_slot,
            variational_bump_weight=var_bump_weight,
        )
        if primary_pareto_report is None:
            primary_pareto_report = rep
        all_pareto_candidates.extend(rep.pareto_front)

    best_per_slot: list[SymbolicCandidate] = []
    seen_cand_ids: set[str] = set()
    for slot in slots_to_search:
        slot_cands = [c for c in all_pareto_candidates if c.thermodynamic_slot == slot]
        if slot_cands:
            slot_cands.sort(key=lambda c: c.bic_score)
            best_s = slot_cands[0]
            best_per_slot.append(best_s)
            seen_cand_ids.add(best_s.candidate_id)

    slot_priority = {
        "passive_dissipative": 0,
        "hamiltonian_potential": 1,
        "conservative_coupling": 2,
        "active_driven": 3,
    }
    remaining = [c for c in all_pareto_candidates if c.candidate_id not in seen_cand_ids]
    remaining.sort(key=lambda c: (c.bic_score, slot_priority.get(c.thermodynamic_slot, 9)))
    ordered_candidates = best_per_slot + remaining

    compiled_bricks: list[LawBrick] = []
    cand_by_brick_name: dict[str, SymbolicCandidate] = {}
    max_to_compile = max(top_k_to_compile, len(best_per_slot), 1)
    for cand in ordered_candidates[:max_to_compile]:
        try:
            c_brick = compile_symbolic_candidate(base_brick, cand)
            cert = audit_law_brick(c_brick)
            if cert.passed_all:
                compiled_bricks.append(c_brick)
                cand_by_brick_name[c_brick.name] = cand
        except Exception:
            continue

    if not compiled_bricks:
        best_c = ordered_candidates[0]
        c_brick = compile_symbolic_candidate(base_brick, best_c)
        compiled_bricks.append(c_brick)
        cand_by_brick_name[c_brick.name] = best_c

    rankings: list[HypothesisRankingEntry] = []

    for c_brick in compiled_bricks:
        funnel = run_elimination_funnel(
            brick=c_brick,
            observed_states=x_cal,
            dt=dt,
            observed_dx_dt=dx_in,
            custom_bounds=custom_base_bounds,
            num_initial_samples=48,
            num_collocation_survivors=12,
            num_gradient_starts=3,
            gradient_steps=35,
        )
        n_obs = max(x.shape[0] * x.shape[1], 1)
        mse = (
            max(0.35 * funnel.vector_field_mse + 0.65 * funnel.trajectory_mse, 1e-16)
            if var_bump_weight > 0.0
            else max(funnel.vector_field_mse, 1e-16)
        )
        cand_obj = cand_by_brick_name[c_brick.name]
        structure_bonus = (
            -15.0
            if cand_obj.thermodynamic_slot in ("passive_dissipative", "hamiltonian_potential", "conservative_coupling")
            else 0.0
        )
        k_eff = float(cand_obj.num_parameters) + 0.25 * float(cand_obj.total_complexity)
        bic = float(n_obs) * math.log(mse) + k_eff * math.log(max(n_obs, 2)) + structure_bonus

        rankings.append(
            HypothesisRankingEntry(
                brick_name=c_brick.name,
                domain=c_brick.domain,
                funnel_report=funnel,
                tunable_param_count=cand_obj.num_parameters,
                symbolic_complexity=cand_obj.total_complexity,
                bic_score=bic,
                passed_thermodynamic_audit=funnel.audit_certificate.passed_all,
            )
        )

    rankings.sort(key=lambda r: (not r.passed_thermodynamic_audit, r.bic_score))
    winner_entry = rankings[0]
    winning_brick = next(b for b in compiled_bricks if b.name == winner_entry.brick_name)
    winning_cand = cand_by_brick_name[winning_brick.name]
    arch = classify_mathematical_archetype(winning_brick, candidate=winning_cand, is_latent_augmented=False)

    win_params = winner_entry.funnel_report.best_params
    out_of_sample_r2 = winner_entry.funnel_report.trajectory_r2
    if t_steps >= 8:
        mid_idx = t_steps // 2
        rem_steps = t_steps - 1 - mid_idx
        if x_in.ndim == 3:
            x_mid_batch = x_in[:, mid_idx, :]
            gt_half = x_in[:, mid_idx:, :]
        else:
            x_mid_batch = x_first[mid_idx : mid_idx + 1]
            gt_half = x_first[mid_idx:].unsqueeze(0)
        extrap_roll = rollout_trajectories(
            winning_brick,
            x_mid_batch,
            win_params.expand(x_mid_batch.shape[0], -1),
            dt=dt,
            num_steps=rem_steps,
            integrator="auto",
        )
        pred_half = extrap_roll.states
        ss_res = float(torch.sum((gt_half - pred_half) ** 2).item())
        ss_tot_half = float(
            torch.sum((gt_half - torch.mean(gt_half, dim=(0, 1), keepdim=True)) ** 2).item()
        )
        ss_tot_full = float(
            torch.sum((x_in - torch.mean(x_in) if x_in.ndim < 3 else (x_in - torch.mean(x_in, dim=(0, 1), keepdim=True))) ** 2).item()
        ) * (float(gt_half.numel()) / max(float(x_in.numel()), 1.0))
        ss_tot_energy = 0.05 * float(torch.sum(gt_half**2).item())
        ss_tot = max(ss_tot_half, 0.25 * ss_tot_full, ss_tot_energy)
        if ss_tot > 1e-15:
            out_of_sample_r2 = 1.0 - ss_res / ss_tot

    port_margin = (
        float(rankings[1].bic_score - winner_entry.bic_score)
        if len(rankings) >= 2
        else None
    )
    win_expr = "; ".join(f"{k}: {v.to_infix()}" for k, v in winning_cand.patch_expressions.items())
    port_alts = [
        AlternativeBranchRecord(
            branch_id=r_e.brick_name,
            expression_or_summary=f"slot={cand_by_brick_name[r_e.brick_name].thermodynamic_slot}: "
            + "; ".join(f"{k}: {v.to_infix()}" for k, v in cand_by_brick_name[r_e.brick_name].patch_expressions.items()),
            objective_score=round(float(r_e.bic_score), 4),
            local_r2=round(float(r_e.funnel_report.vector_field_r2), 6),
            complexity=r_e.symbolic_complexity,
        )
        for r_e in rankings[1:4]
        if r_e.brick_name in cand_by_brick_name
    ]
    emit_trace_node(
        layer="layer5_discovery",
        operation="discover_and_compile_law",
        target_name=winning_brick.name,
        local_error_metric=winner_entry.funnel_report.vector_field_mse,
        local_r2=winner_entry.funnel_report.vector_field_r2,
        global_error_metric=winner_entry.funnel_report.trajectory_mse,
        global_r2=out_of_sample_r2,
        decision_margin=port_margin,
        selected_branch_id=winning_brick.name,
        selected_expression=win_expr,
        alternatives=port_alts,
        metrics={
            "thermodynamic_slot": winning_cand.thermodynamic_slot,
            "archetype": arch.archetype_name,
            "out_of_sample_r2": round(float(out_of_sample_r2), 6),
        },
    )

    return ClosedLoopDiscoveryReport(
        discovery_triggered=True,
        winning_brick=winning_brick,
        winning_params=win_params,
        winning_candidate=winning_cand,
        pareto_report=primary_pareto_report,
        portfolio_rankings=tuple(rankings),
        memory_report=memory_report,
        latent_augmentation=None,
        archetype=arch,
        audit_certificate=winner_entry.funnel_report.audit_certificate,
        out_of_sample_r2=out_of_sample_r2,
    )


from cortivra.layer5_discovery.autonomous_and_mechanism_synthesis import (
    AutonomousDiscoveryReport,
    HelmholtzHodgeDecomposition,
    SymbolicMechanismInventionReport,
    decompose_vector_field_helmholtz_hodge,
    discover_autonomous_symbolic_system,
    synthesize_symbolic_feedback_mechanism,
)
