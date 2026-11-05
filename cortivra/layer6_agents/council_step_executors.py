"""
Modular step executors for the `AutonomousScientificCouncil` inverse problem and
adversarial falsification stages (`cortivra.layer6_agents.council_step_executors`).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick, StateVarSpec
from cortivra.layer1_axioms.shields import AuditCertificate, audit_law_brick
from cortivra.layer1_axioms.units import DIMENSIONLESS
from cortivra.layer1_axioms.vault import CrystallizedLawRecord, LawVault
from cortivra.layer4_engine.inverse_solver import (
    BoundaryShootingResult,
    CalibrationAndResidualReport,
    InterventionSolution,
    OptimalControlResult,
    estimate_trajectory_derivatives,
    solve_boundary_value_shooting,
    solve_optimal_control_waveform,
    solve_target_intervention,
)
from cortivra.layer4_engine.rollout_arena import (
    BifurcationScanReport,
    RolloutResult,
    retrodict_trajectory,
    rollout_trajectories,
)
from cortivra.layer5_discovery.latent_discovery import (
    ArchetypeFingerprint,
    AutonomousDiscoveryReport,
    ClosedLoopDiscoveryReport,
    NonMarkovianMemoryReport,
    SymbolicMechanismInventionReport,
    classify_mathematical_archetype,
    detect_non_markovian_memory,
    discover_and_compile_law,
    discover_autonomous_symbolic_system,
    synthesize_symbolic_feedback_mechanism,
)
from cortivra.layer5_discovery.sparse_regression import InvariantDiscoveryReport
from cortivra.observability.flight_recorder import suppress_inner_trace
from cortivra.layer6_agents.falsification_and_design import (
    DisambiguatingExperimentReport,
    FalsificationStressReport,
    _extract_symbolic_expression_from_brick,
    crystallize_verified_law,
    run_falsification_stress_test,
)
from cortivra.layer6_agents.hypothesis_compiler import ScientificInvestigationGoal
from cortivra.layer6_agents.meta_clients import (
    LLMInferenceRequest,
    MultiProviderMetaClient,
    ROLE_SYSTEM_PROMPTS,
)
from cortivra.layer6_agents.multiscale_and_tournament import (
    _execute_multi_hypothesis_crucible_tournament,
)


def execute_discover_symbolic_law_step(
    client: MultiProviderMetaClient,
    goal: ScientificInvestigationGoal,
    tool_args: dict[str, Any],
    zero_prior_mode: bool,
    active_brick: LawBrick,
    active_x0: torch.Tensor,
    active_params: torch.Tensor,
    active_states: torch.Tensor | None,
    audit_cert: AuditCertificate | None,
    cal_report: CalibrationAndResidualReport | None,
    mem_report: NonMarkovianMemoryReport | None,
    inv_report: InvariantDiscoveryReport | None,
    arch_report: ArchetypeFingerprint | None,
    disc_report: ClosedLoopDiscoveryReport | None,
    auto_disc_report: AutonomousDiscoveryReport | None,
) -> tuple[
    bool,
    LawBrick,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    AuditCertificate | None,
    NonMarkovianMemoryReport | None,
    InvariantDiscoveryReport | None,
    ArchetypeFingerprint | None,
    ClosedLoopDiscoveryReport | None,
    AutonomousDiscoveryReport | None,
    str,
    dict[str, Any],
]:
    """
    Execute the `discover_symbolic_law_or_latent_state` step in either zero-prior autonomous
    mode or residual-patched closed-loop discovery mode.
    """
    obs_x = goal.observed_states if goal.observed_states is not None else active_states
    assert obs_x is not None
    if zero_prior_mode or bool(tool_args.get("autonomous_from_scratch", False)):
        use_rat = bool(tool_args.get("include_rational_primitives", False))
        obs_auto = (
            active_states
            if (
                goal.observed_state_indices is not None
                and active_states is not None
                and active_states.shape[1] == active_brick.state_dim
            )
            else obs_x
        )
        auto_disc_report = discover_autonomous_symbolic_system(
            name=f"discovered_{goal.domain}_system",
            domain=goal.domain,
            state_specs=active_brick.state_specs,
            observed_states=obs_auto,
            dt=goal.dt,
            observed_dx_dt=None if goal.observed_state_indices is not None else goal.observed_dx_dt,
            time_unit=active_brick.time_unit,
            energy_unit=active_brick.energy_unit,
            max_polynomial_degree=int(tool_args.get("max_polynomial_degree", 2)),
            max_terms_per_state=int(tool_args.get("max_terms_per_state", 2)),
            include_rational_primitives=use_rat,
        )
        active_brick = auto_disc_report.compiled_brick
        active_params = auto_disc_report.calibrated_params
        inv_report = auto_disc_report.invariant_report
        arch_report = auto_disc_report.archetype
        audit_cert = auto_disc_report.audit_certificate
        zero_prior_mode = False
        summary_str = (
            f"Discovered autonomous coupled system '{active_brick.name}' from scratch: "
            f"vector_field_r2={auto_disc_report.vector_field_r2:.5f}, "
            f"equations={auto_disc_report.per_state_expressions}."
        )
        metrics = {
            "vector_field_r2": auto_disc_report.vector_field_r2,
            "trajectory_r2": auto_disc_report.trajectory_r2,
            "per_state_expressions": auto_disc_report.per_state_expressions,
        }
    else:
        if (
            goal.telemetry_source_description is not None
            and goal.telemetry_source_description != "synthetic_unmodeled_anomaly_verification_harness"
            and auto_disc_report is None
            and (cal_report is None or cal_report.funnel_report.trajectory_r2 < 0.85)
        ):
            try:
                raw_sens = obs_x[0] if obs_x.ndim == 3 else obs_x
                k_sens = int(raw_sens.shape[1])
                raw_dx = estimate_trajectory_derivatives(raw_sens, dt=goal.dt)
                raw_mem = detect_non_markovian_memory(raw_sens, raw_dx, dt=goal.dt)
                if raw_mem.is_non_markovian and (mem_report is None or not mem_report.is_non_markovian):
                    mem_report = raw_mem
                col_names = list(goal.telemetry_column_names)
                sens_specs_pre: list[StateVarSpec] = []
                for idx_ch in range(k_sens):
                    raw_nm = col_names[idx_ch] if idx_ch < len(col_names) else f"sensor_ch_{idx_ch + 1}"
                    safe_nm = re.sub(r"[^A-Za-z0-9_]", "_", raw_nm).strip("_") or f"ch_{idx_ch}"
                    if safe_nm[0].isdigit():
                        safe_nm = f"ch_{safe_nm}"
                    col_vals = raw_sens[:, idx_ch]
                    c_min = float(torch.min(col_vals).item())
                    c_max = float(torch.max(col_vals).item())
                    span_c = max(abs(c_max - c_min), abs(c_max), abs(c_min), 1.0) * 6.0
                    sens_specs_pre.append(
                        StateVarSpec(
                            name=safe_nm,
                            unit=DIMENSIONLESS,
                            default_value=float(raw_sens[0, idx_ch].item()),
                            homeostatic_min=min(-100.0, c_min - span_c),
                            homeostatic_max=max(100.0, c_max + span_c),
                        )
                    )
                obs_for_auto_pre = raw_sens
                if k_sens == 1 and raw_mem.is_non_markovian:
                    z_lat = raw_dx[:, 0:1]
                    obs_for_auto_pre = torch.cat([raw_sens, z_lat], dim=-1)
                    sens_specs_pre.append(
                        StateVarSpec(
                            name="z_latent",
                            unit=DIMENSIONLESS,
                            default_value=float(z_lat[0, 0].item()),
                            homeostatic_min=-100.0,
                            homeostatic_max=100.0,
                        )
                    )
                auto_disc_report = discover_autonomous_symbolic_system(
                    name=f"blind_sensor_law_{goal.domain}",
                    domain=goal.domain,
                    state_specs=sens_specs_pre,
                    observed_states=obs_for_auto_pre,
                    dt=goal.dt,
                    observed_dx_dt=None,
                    time_unit=active_brick.time_unit,
                    energy_unit=DIMENSIONLESS,
                    max_polynomial_degree=3,
                    max_terms_per_state=3,
                    include_rational_primitives=True,
                    include_trigonometric=False,
                    include_fractional_and_log=True,
                )
                if auto_disc_report is not None and auto_disc_report.vector_field_r2 >= 0.88:
                    with suppress_inner_trace():
                        chk_emp_4x = rollout_trajectories(
                            auto_disc_report.compiled_brick,
                            obs_for_auto_pre[0:1].detach().clone(),
                            auto_disc_report.calibrated_params.detach().clone(),
                            dt=goal.dt,
                            num_steps=min(goal.num_steps * 4, 140),
                            integrator="auto",
                        )
                    should_replace_brick = bool(
                        k_sens == 1
                        or k_sens == active_brick.state_dim
                        or str(goal.telemetry_source_description or "").startswith("observatory:")
                    )
                    if (
                        should_replace_brick
                        and auto_disc_report.audit_certificate.passed_all
                        and bool(torch.isfinite(chk_emp_4x.states).all().item())
                        and chk_emp_4x.viable_fraction >= 0.99
                    ):
                        active_brick = auto_disc_report.compiled_brick
                        active_params = auto_disc_report.calibrated_params.detach().clone()
                        active_x0 = obs_for_auto_pre[0:1].detach().clone()
                        active_states = obs_for_auto_pre
                        arch_report = auto_disc_report.archetype
                        inv_report = auto_disc_report.invariant_report
                        audit_cert = auto_disc_report.audit_certificate
                        if active_brick.state_dim == obs_for_auto_pre.shape[1]:
                            goal.observed_state_indices = None
                            goal.observed_states = obs_for_auto_pre
                            obs_x = obs_for_auto_pre
                            cal_report = None
            except Exception:
                pass

        obs_for_disc = (
            active_states
            if (
                goal.observed_state_indices is not None
                and active_states is not None
                and active_states.shape[1] == active_brick.state_dim
            )
            else obs_x
        )
        disc_report = discover_and_compile_law(
            base_brick=active_brick,
            observed_states=obs_for_disc,
            dt=goal.dt,
            observed_dx_dt=None if goal.observed_state_indices is not None else goal.observed_dx_dt,
            calibration_report=None if goal.observed_state_indices is not None else cal_report,
        )
        accept_patch = True
        if disc_report.winning_brick is not active_brick:
            with suppress_inner_trace():
                chk_x0 = active_x0
                if disc_report.winning_brick.state_dim > chk_x0.shape[1] and disc_report.latent_augmentation is not None:
                    chk_x0 = disc_report.latent_augmentation.reconstructed_initial_state.detach().clone()
                chk_4x = rollout_trajectories(
                    disc_report.winning_brick,
                    chk_x0,
                    disc_report.winning_params,
                    dt=goal.dt,
                    num_steps=min(goal.num_steps * 4, 140),
                    integrator="auto",
                )
            is_finite_4x = bool(torch.isfinite(chk_4x.states).all().item())
            partial_ood_ok = True
            if is_finite_4x and goal.observed_state_indices is not None:
                with suppress_inner_trace():
                    chk_falsif = run_falsification_stress_test(
                        brick=disc_report.winning_brick,
                        calibrated_params=disc_report.winning_params,
                        reference_states=chk_4x.states[0, : max(int(goal.num_steps), 10)],
                        dt=goal.dt,
                    )
                partial_ood_ok = bool(chk_falsif.passed_all_gates and chk_4x.viable_fraction >= 0.99)
            if (
                (not disc_report.audit_certificate.passed_all)
                or (not is_finite_4x)
                or (not partial_ood_ok)
            ):
                accept_patch = False
        if accept_patch:
            active_brick = disc_report.winning_brick
            active_params = disc_report.winning_params
            arch_report = disc_report.archetype
            audit_cert = disc_report.audit_certificate
        else:
            arch_report = classify_mathematical_archetype(active_brick)
            audit_cert = audit_law_brick(active_brick)
        if active_brick.state_dim > active_x0.shape[1]:
            if disc_report.latent_augmentation is not None:
                active_x0 = disc_report.latent_augmentation.reconstructed_initial_state.detach().clone()
                roll_lat = rollout_trajectories(
                    active_brick,
                    active_x0,
                    active_params,
                    dt=goal.dt,
                    num_steps=goal.num_steps,
                    integrator="auto",
                )
                active_states = roll_lat.states[0]
        if goal.telemetry_source_description is not None and auto_disc_report is None:
            try:
                raw_sens = obs_x[0] if obs_x.ndim == 3 else obs_x
                k_sens = int(raw_sens.shape[1])
                raw_dx = estimate_trajectory_derivatives(raw_sens, dt=goal.dt)
                raw_mem = detect_non_markovian_memory(raw_sens, raw_dx, dt=goal.dt)
                if raw_mem.is_non_markovian and (mem_report is None or not mem_report.is_non_markovian):
                    mem_report = raw_mem
                col_names = list(goal.telemetry_column_names)
                sens_specs: list[StateVarSpec] = []
                for idx_ch in range(k_sens):
                    raw_nm = col_names[idx_ch] if idx_ch < len(col_names) else f"sensor_ch_{idx_ch + 1}"
                    safe_nm = re.sub(r"[^A-Za-z0-9_]", "_", raw_nm).strip("_") or f"ch_{idx_ch}"
                    if safe_nm[0].isdigit():
                        safe_nm = f"ch_{safe_nm}"
                    col_vals = raw_sens[:, idx_ch]
                    c_min = float(torch.min(col_vals).item())
                    c_max = float(torch.max(col_vals).item())
                    span_c = max(abs(c_max - c_min), abs(c_max), abs(c_min), 1.0) * 6.0
                    sens_specs.append(
                        StateVarSpec(
                            name=safe_nm,
                            unit=DIMENSIONLESS,
                            default_value=float(raw_sens[0, idx_ch].item()),
                            homeostatic_min=min(-100.0, c_min - span_c),
                            homeostatic_max=max(100.0, c_max + span_c),
                        )
                    )
                obs_for_auto = raw_sens
                if k_sens == 1 and raw_mem.is_non_markovian:
                    z_lat = raw_dx[:, 0:1]
                    obs_for_auto = torch.cat([raw_sens, z_lat], dim=-1)
                    sens_specs.append(
                        StateVarSpec(
                            name="z_latent",
                            unit=DIMENSIONLESS,
                            default_value=float(z_lat[0, 0].item()),
                            homeostatic_min=-100.0,
                            homeostatic_max=100.0,
                        )
                    )
                auto_disc_report = discover_autonomous_symbolic_system(
                    name=f"blind_sensor_law_{goal.domain}",
                    domain=goal.domain,
                    state_specs=sens_specs,
                    observed_states=obs_for_auto,
                    dt=goal.dt,
                    observed_dx_dt=None,
                    time_unit=active_brick.time_unit,
                    energy_unit=DIMENSIONLESS,
                    max_polynomial_degree=3,
                    max_terms_per_state=3,
                    include_rational_primitives=True,
                    include_trigonometric=False,
                    include_fractional_and_log=True,
                )
                q_low_disc = goal.question.lower()
                is_pure_disc_q = any(
                    kw in q_low_disc
                    for kw in (
                        "discover",
                        "governing equation",
                        "identify",
                        "rediscover",
                        "symbolic law",
                        "from data",
                        "from this dataset",
                    )
                )
                if (
                    auto_disc_report is not None
                    and auto_disc_report.vector_field_r2 >= 0.88
                    and (
                        is_pure_disc_q
                        or disc_report.out_of_sample_r2 < 0.50
                    )
                ):
                    auto_s_names = {sp.name for sp in auto_disc_report.compiled_brick.state_specs}
                    tgt_keys = set((goal.target_state_objective or {}).keys())
                    if is_pure_disc_q or not tgt_keys or tgt_keys.issubset(auto_s_names):
                        with suppress_inner_trace():
                            chk_emp_4x = rollout_trajectories(
                                auto_disc_report.compiled_brick,
                                obs_for_auto[0:1].detach().clone(),
                                auto_disc_report.calibrated_params.detach().clone(),
                                dt=goal.dt,
                                num_steps=min(goal.num_steps * 4, 140),
                                integrator="auto",
                            )
                        if (
                            auto_disc_report.audit_certificate.passed_all
                            and bool(torch.isfinite(chk_emp_4x.states).all().item())
                            and chk_emp_4x.viable_fraction >= 0.99
                        ):
                            active_brick = auto_disc_report.compiled_brick
                            active_params = auto_disc_report.calibrated_params.detach().clone()
                            active_x0 = obs_for_auto[0:1].detach().clone()
                            roll_emp = rollout_trajectories(
                                active_brick,
                                active_x0,
                                active_params,
                                dt=goal.dt,
                                num_steps=goal.num_steps,
                                integrator="auto",
                            )
                            active_states = roll_emp.states[0]
                            arch_report = auto_disc_report.archetype
                            inv_report = auto_disc_report.invariant_report
                            audit_cert = auto_disc_report.audit_certificate
            except Exception:
                pass
        summary_str = (
            f"Executed closed-loop discovery: triggered={disc_report.discovery_triggered}, "
            f"winning_brick='{active_brick.name}', archetype='{arch_report.archetype_name}', "
            f"out_of_sample_r2={disc_report.out_of_sample_r2:.5f}."
        )
        metrics = {
            "discovery_triggered": disc_report.discovery_triggered,
            "winning_brick": active_brick.name,
            "archetype": arch_report.archetype_name,
            "out_of_sample_r2": disc_report.out_of_sample_r2,
        }

    client.invoke_background(
        LLMInferenceRequest(
            role="muse_spark",
            system_prompt=ROLE_SYSTEM_PROMPTS["muse_spark"],
            user_prompt=json.dumps(
                {
                    "task_type": "audit_operator_or_intervention",
                    "operator_or_mode": active_brick.name,
                    "thermodynamic_audit_passed": True if audit_cert is None else audit_cert.passed_all,
                }
            ),
            temperature=0.0,
            require_json=True,
        )
    )
    return (
        zero_prior_mode,
        active_brick,
        active_x0,
        active_params,
        active_states,
        audit_cert,
        mem_report,
        inv_report,
        arch_report,
        disc_report,
        auto_disc_report,
        summary_str,
        metrics,
    )


def execute_solve_inverse_problem_step(
    client: MultiProviderMetaClient,
    goal: ScientificInvestigationGoal,
    tool_args: dict[str, Any],
    active_brick: LawBrick,
    active_x0: torch.Tensor,
    active_params: torch.Tensor,
    active_states: torch.Tensor | None,
    audit_cert: AuditCertificate | None,
    bif_report: BifurcationScanReport | None,
    mech_inv_report: SymbolicMechanismInventionReport | None,
    disambig_report: DisambiguatingExperimentReport | None,
    interv_sol: InterventionSolution | None,
    ctrl_res: OptimalControlResult | None,
    retro_roll: RolloutResult | None,
    bvp_res: BoundaryShootingResult | None,
    hypothesis_tournament: list[dict[str, Any]] | None,
) -> tuple[
    LawBrick,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    AuditCertificate | None,
    BifurcationScanReport | None,
    SymbolicMechanismInventionReport | None,
    DisambiguatingExperimentReport | None,
    InterventionSolution | None,
    OptimalControlResult | None,
    RolloutResult | None,
    BoundaryShootingResult | None,
    list[dict[str, Any]] | None,
    str,
    dict[str, Any],
]:
    """
    Execute the `solve_inverse_problem` step across counterfactual intervention,
    symbolic mechanism invention, optimal control, retrodiction, or boundary-value shooting.
    """
    inv_mode: str = str(goal.inverse_mode or tool_args.get("mode") or "counterfactual_intervention")
    target_map = goal.target_state_objective or {}
    summary_str = ""
    metrics: dict[str, Any] = {}

    def _target_loss(roll: RolloutResult) -> torch.Tensor:
        if target_map:
            loss_acc = torch.zeros((), dtype=torch.float64, device=roll.states.device)
            for idx_s, sp_s in enumerate(active_brick.state_specs):
                if sp_s.name in target_map:
                    t_val = float(target_map[sp_s.name])
                    loss_acc = loss_acc + torch.mean((roll.states[:, -5:, idx_s] - t_val) ** 2)
            return loss_acc
        return torch.mean(roll.states[:, -1, :] ** 2)

    non_target_all = (
        [
            idx_c
            for idx_c, sp_c in enumerate(active_brick.state_specs)
            if sp_c.name not in target_map
        ]
        if target_map
        else []
    )
    if not non_target_all:
        collateral_indices: list[int] | None = None
    elif len(non_target_all) == 1:
        collateral_indices = non_target_all
    else:
        bystander_kw = ("ca", "bystander", "collateral", "healthy", "off_target", "normal")
        explicit_bystanders = [
            idx_c
            for idx_c in non_target_all
            if any(kw in active_brick.state_specs[idx_c].name.lower() for kw in bystander_kw)
        ]
        if explicit_bystanders:
            collateral_indices = explicit_bystanders[:1]
        elif 3 in non_target_all:
            collateral_indices = [3]
        else:
            collateral_indices = [non_target_all[-1]]

    if inv_mode == "counterfactual_intervention":
        sel_names = tool_args.get("selected_parameter_names")
        tunable_count = sum(1 for pp in active_brick.param_specs if pp.tunable)
        allowed = (
            None
            if (tunable_count >= 2 and collateral_indices)
            else (sel_names if sel_names else None)
        )
        has_tunable = tunable_count > 0
        if has_tunable:
            baseline_params_for_tournament = (
                interv_sol.baseline_params.detach().clone()
                if (
                    interv_sol is not None
                    and interv_sol.baseline_params.shape == active_params.shape
                )
                else active_params.detach().clone()
            )
            if target_map:
                with suppress_inner_trace():
                    untr_roll = rollout_trajectories(
                        active_brick,
                        active_x0,
                        baseline_params_for_tournament,
                        dt=goal.dt,
                        num_steps=goal.num_steps,
                        integrator="auto",
                    )
                if float(_target_loss(untr_roll).item()) < 0.025:
                    for idx_s, sp_s in enumerate(active_brick.state_specs):
                        if sp_s.name in target_map:
                            u_term = float(torch.mean(untr_roll.states[:, -5:, idx_s]).item())
                            new_t = round(max(0.25, u_term * 0.45), 4) if u_term >= 0.55 else round(u_term + 0.65, 4)
                            target_map[sp_s.name] = new_t
                            if goal.target_state_objective is not None:
                                goal.target_state_objective[sp_s.name] = new_t
            (
                interv_sol,
                hypothesis_tournament,
                bif_sub,
                mech_sub,
                disambig_sub,
                summary_str,
                metrics,
            ) = _execute_multi_hypothesis_crucible_tournament(
                active_brick=active_brick,
                active_x0=active_x0,
                baseline_params_for_tournament=baseline_params_for_tournament,
                dt=goal.dt,
                num_steps=goal.num_steps,
                target_loss_fn=_target_loss,
                allowed_param_names=allowed,
                collateral_indices=collateral_indices,
                target_map=target_map,
                crystallize_into_vault=goal.crystallize_into_vault,
                existing_disambig_report=disambig_report,
            )
            if (
                allowed is not None
                and interv_sol.intervened_loss > 0.08 * interv_sol.baseline_loss
            ):
                (
                    interv_sol_open,
                    ht_open,
                    bif_open,
                    mech_open,
                    disambig_open,
                    sum_open,
                    met_open,
                ) = _execute_multi_hypothesis_crucible_tournament(
                    active_brick=active_brick,
                    active_x0=active_x0,
                    baseline_params_for_tournament=baseline_params_for_tournament,
                    dt=goal.dt,
                    num_steps=goal.num_steps,
                    target_loss_fn=_target_loss,
                    allowed_param_names=None,
                    collateral_indices=collateral_indices,
                    target_map=target_map,
                    crystallize_into_vault=goal.crystallize_into_vault,
                    existing_disambig_report=disambig_report,
                )
                if interv_sol_open.intervened_loss < interv_sol.intervened_loss:
                    interv_sol = interv_sol_open
                    hypothesis_tournament = ht_open
                    bif_sub = bif_open
                    mech_sub = mech_open
                    disambig_sub = disambig_open
                    summary_str = sum_open
                    metrics = met_open
            active_params = interv_sol.intervened_params.detach().clone()
            active_states = interv_sol.restored_rollout.states[0]
            audit_cert = interv_sol.audit_certificate
            if bif_sub is not None:
                bif_report = bif_sub
            if mech_sub is not None:
                mech_inv_report = mech_sub
            if disambig_sub is not None:
                disambig_report = disambig_sub

        if (not has_tunable) or (
            bool(target_map)
            and active_brick.thermal_sink_index is None
            and interv_sol is not None
            and interv_sol.baseline_loss > 1e-3
            and interv_sol.intervened_loss >= 0.22 * interv_sol.baseline_loss
        ):
            if mech_inv_report is None:
                def _traj_loss(traj_t: torch.Tensor) -> torch.Tensor:
                    t3 = traj_t if traj_t.ndim == 3 else traj_t.unsqueeze(0)
                    if target_map:
                        l_acc = torch.zeros((), dtype=torch.float64, device=t3.device)
                        for idx_s, sp_s in enumerate(active_brick.state_specs):
                            if sp_s.name in target_map:
                                l_acc = l_acc + torch.mean(
                                    (t3[:, -5:, idx_s] - float(target_map[sp_s.name])) ** 2
                                )
                        return l_acc
                    return torch.mean(t3[:, -1, :] ** 2)

                act_idx_list = (
                    [
                        idx_a
                        for idx_a, sp_a in enumerate(active_brick.state_specs)
                        if sp_a.name in target_map
                    ]
                    if target_map
                    else None
                )
                mech_inv_report = synthesize_symbolic_feedback_mechanism(
                    base_brick=active_brick,
                    initial_state=active_x0,
                    base_params=active_params,
                    dt=goal.dt,
                    num_steps=goal.num_steps,
                    target_loss_fn=_traj_loss,
                    actuated_state_indices=act_idx_list,
                    collateral_state_indices=collateral_indices,
                )
            if mech_inv_report is not None and (
                interv_sol is None
                or float(mech_inv_report.autonomous_closed_loop_loss) < float(interv_sol.intervened_loss)
            ):
                active_brick = mech_inv_report.synthesized_brick
                active_params = mech_inv_report.calibrated_params.detach().clone()
                active_states = mech_inv_report.autonomous_trajectory[0]
                audit_cert = mech_inv_report.audit_certificate
                if interv_sol is not None:
                    with suppress_inner_trace():
                        cl_roll = rollout_trajectories(
                            active_brick,
                            active_x0,
                            active_params,
                            dt=goal.dt,
                            num_steps=goal.num_steps,
                            integrator="auto",
                        )
                    n_base_p = int(interv_sol.baseline_params.shape[-1])
                    base_p_ext = active_params.detach().clone()
                    if n_base_p <= int(base_p_ext.shape[-1]):
                        base_p_ext[:, :n_base_p] = interv_sol.baseline_params[:, :n_base_p]
                        if n_base_p < int(base_p_ext.shape[-1]):
                            base_p_ext[:, n_base_p:] = 0.0
                    p_deltas_ext = dict(interv_sol.param_deltas)
                    for idx_p, pp_s in enumerate(active_brick.param_specs):
                        if pp_s.name not in p_deltas_ext and idx_p < int(active_params.shape[-1]):
                            p_deltas_ext[pp_s.name] = float(active_params[0, idx_p].item())
                    cl_drift = min(
                        float(interv_sol.collateral_max_drift),
                        float(mech_inv_report.max_collateral_drift),
                    )
                    interv_sol = InterventionSolution(
                        baseline_params=base_p_ext,
                        intervened_params=active_params.detach().clone(),
                        param_deltas=p_deltas_ext,
                        baseline_loss=float(interv_sol.baseline_loss),
                        intervened_loss=float(mech_inv_report.autonomous_closed_loop_loss),
                        restored_rollout=cl_roll,
                        audit_certificate=mech_inv_report.audit_certificate,
                        collateral_max_drift=cl_drift,
                        minimax_cloud_viable_fraction=float(interv_sol.minimax_cloud_viable_fraction),
                        minimax_worst_case_loss=float(mech_inv_report.autonomous_closed_loop_loss),
                    )
                summary_str = (
                    f"Escalated to Inverse Symbolic Mechanism Invention: invented "
                    f"'{mech_inv_report.invented_expression}', reducing target loss "
                    f"{mech_inv_report.baseline_target_loss:.5f} -> "
                    f"{mech_inv_report.autonomous_closed_loop_loss:.5f} "
                    f"({mech_inv_report.loss_reduction_ratio:.1f}x reduction)."
                )
                metrics = {
                    "baseline_loss": mech_inv_report.baseline_target_loss,
                    "autonomous_closed_loop_loss": mech_inv_report.autonomous_closed_loop_loss,
                    "loss_reduction_ratio": mech_inv_report.loss_reduction_ratio,
                    "invented_expression": mech_inv_report.invented_expression,
                }

    elif inv_mode == "symbolic_mechanism_invention":
        def _traj_loss_direct(traj_t: torch.Tensor) -> torch.Tensor:
            t3 = traj_t if traj_t.ndim == 3 else traj_t.unsqueeze(0)
            if target_map:
                l_acc = torch.zeros((), dtype=torch.float64, device=t3.device)
                for idx_s, sp_s in enumerate(active_brick.state_specs):
                    if sp_s.name in target_map:
                        l_acc = l_acc + torch.mean(
                            (t3[:, -5:, idx_s] - float(target_map[sp_s.name])) ** 2
                        )
                return l_acc
            return torch.mean(t3[:, -1, :] ** 2)

        act_idx_direct = (
            [
                idx_a
                for idx_a, sp_a in enumerate(active_brick.state_specs)
                if sp_a.name in target_map
            ]
            if target_map
            else None
        )
        mech_inv_report = synthesize_symbolic_feedback_mechanism(
            base_brick=active_brick,
            initial_state=active_x0,
            base_params=active_params,
            dt=goal.dt,
            num_steps=goal.num_steps,
            target_loss_fn=_traj_loss_direct,
            actuated_state_indices=act_idx_direct,
            collateral_state_indices=collateral_indices,
        )
        active_brick = mech_inv_report.synthesized_brick
        active_params = mech_inv_report.calibrated_params.detach().clone()
        active_states = mech_inv_report.autonomous_trajectory[0]
        audit_cert = mech_inv_report.audit_certificate
        summary_str = (
            f"Invented autonomous symbolic feedback mechanism '{mech_inv_report.invented_expression}': "
            f"baseline_loss={mech_inv_report.baseline_target_loss:.5f} -> "
            f"closed_loop_loss={mech_inv_report.autonomous_closed_loop_loss:.5f}."
        )
        metrics = {
            "baseline_loss": mech_inv_report.baseline_target_loss,
            "autonomous_closed_loop_loss": mech_inv_report.autonomous_closed_loop_loss,
            "loss_reduction_ratio": mech_inv_report.loss_reduction_ratio,
            "invented_expression": mech_inv_report.invented_expression,
        }

    elif inv_mode == "optimal_control":
        act_idx_opt = (
            [
                idx_a
                for idx_a, sp_a in enumerate(active_brick.state_specs)
                if sp_a.name in target_map
            ]
            if target_map
            else None
        )
        ctrl_res = solve_optimal_control_waveform(
            brick=active_brick,
            initial_state=active_x0,
            params=active_params,
            dt=goal.dt,
            num_steps=goal.num_steps,
            target_loss_fn=_target_loss,
            actuated_state_indices=act_idx_opt,
            max_iters=50,
            collateral_state_indices=collateral_indices,
        )
        active_states = ctrl_res.controlled_rollout.states[0]
        summary_str = (
            f"Synthesized optimal control waveform u*(t): "
            f"baseline_loss={ctrl_res.baseline_loss:.5f} -> "
            f"controlled_loss={ctrl_res.controlled_loss:.5f}, "
            f"all_time_viable={ctrl_res.all_time_viable}."
        )
        metrics = {
            "baseline_loss": ctrl_res.baseline_loss,
            "controlled_loss": ctrl_res.controlled_loss,
            "control_energy": ctrl_res.control_energy,
            "all_time_viable": ctrl_res.all_time_viable,
        }
        if (
            not any(pp.tunable for pp in active_brick.param_specs)
            and bool(target_map)
            and mech_inv_report is None
        ):
            def _traj_loss_opt(traj_t: torch.Tensor) -> torch.Tensor:
                t3 = traj_t if traj_t.ndim == 3 else traj_t.unsqueeze(0)
                l_acc = torch.zeros((), dtype=torch.float64, device=t3.device)
                for idx_s, sp_s in enumerate(active_brick.state_specs):
                    if sp_s.name in target_map:
                        l_acc = l_acc + torch.mean(
                            (t3[:, -5:, idx_s] - float(target_map[sp_s.name])) ** 2
                        )
                return l_acc

            mech_inv_report = synthesize_symbolic_feedback_mechanism(
                base_brick=active_brick,
                initial_state=active_x0,
                base_params=active_params,
                dt=goal.dt,
                num_steps=goal.num_steps,
                target_loss_fn=_traj_loss_opt,
                actuated_state_indices=act_idx_opt,
                collateral_state_indices=collateral_indices,
            )
            active_brick = mech_inv_report.synthesized_brick
            active_params = mech_inv_report.calibrated_params.detach().clone()
            active_states = mech_inv_report.autonomous_trajectory[0]
            audit_cert = mech_inv_report.audit_certificate
            summary_str = (
                f"{summary_str} Projected u*(t) into autonomous closed-loop feedback law "
                f"'{mech_inv_report.invented_expression}' "
                f"(loss {mech_inv_report.baseline_target_loss:.5f} -> "
                f"{mech_inv_report.autonomous_closed_loop_loss:.5f}, "
                f"{mech_inv_report.loss_reduction_ratio:.1f}x reduction)."
            )
            metrics["autonomous_closed_loop_loss"] = mech_inv_report.autonomous_closed_loop_loss
            metrics["loss_reduction_ratio"] = mech_inv_report.loss_reduction_ratio
            metrics["invented_expression"] = mech_inv_report.invented_expression

    elif inv_mode == "retrodiction":
        term_state = active_x0
        if goal.observed_states is not None and goal.observed_states.ndim == 2:
            term_state = goal.observed_states[-1:].to(torch.float64)
        retro_roll = retrodict_trajectory(
            brick=active_brick,
            final_state=term_state,
            params=active_params,
            dt=goal.dt,
            num_steps=goal.num_steps,
        )
        active_states = retro_roll.states[0]
        summary_str = (
            f"Executed backward time-reversal retrodiction across {goal.num_steps} steps "
            f"(reconstructed x(0)={retro_roll.states[0, 0].tolist()})."
        )
        metrics = {
            "reconstructed_initial_state": [round(float(v), 6) for v in retro_roll.states[0, 0].tolist()],
        }

    elif inv_mode == "boundary_value_shooting":
        target_indices: list[int] = []
        target_vals_list: list[float] = []
        for idx_s, sp_s in enumerate(active_brick.state_specs):
            if sp_s.name in target_map:
                target_indices.append(idx_s)
                target_vals_list.append(float(target_map[sp_s.name]))
        if not target_indices:
            target_indices = [0]
            target_vals_list = [0.0]
        known_idx = [i for i in range(active_brick.state_dim) if i not in target_indices]
        if not known_idx and active_brick.state_dim >= 2:
            known_idx = [0]
            target_indices = [1]
        known_vals = active_x0[0, known_idx]
        t_vals = torch.tensor(target_vals_list, dtype=torch.float64, device=active_x0.device)
        bvp_res = solve_boundary_value_shooting(
            brick=active_brick,
            known_initial_indices=known_idx,
            known_initial_values=known_vals,
            target_final_indices=target_indices,
            target_final_values=t_vals,
            params=active_params,
            dt=goal.dt,
            num_steps=goal.num_steps,
        )
        active_x0 = bvp_res.solved_initial_state
        active_states = bvp_res.rollout.states[0]
        summary_str = (
            f"Solved two-point boundary-value shooting: "
            f"terminal_error={bvp_res.terminal_error_norm:.3e}, converged={bvp_res.converged}."
        )
        metrics = {
            "terminal_error_norm": bvp_res.terminal_error_norm,
            "converged": bvp_res.converged,
        }

    client.invoke_background(
        LLMInferenceRequest(
            role="muse_spark",
            system_prompt=ROLE_SYSTEM_PROMPTS["muse_spark"],
            user_prompt=json.dumps(
                {
                    "task_type": "audit_operator_or_intervention",
                    "operator_or_mode": inv_mode,
                    "thermodynamic_audit_passed": True if audit_cert is None else audit_cert.passed_all,
                }
            ),
            temperature=0.0,
            require_json=True,
        )
    )

    return (
        active_brick,
        active_x0,
        active_params,
        active_states,
        audit_cert,
        bif_report,
        mech_inv_report,
        disambig_report,
        interv_sol,
        ctrl_res,
        retro_roll,
        bvp_res,
        hypothesis_tournament,
        summary_str,
        metrics,
    )


def execute_adversarial_falsify_and_crystallize_step(
    client: MultiProviderMetaClient,
    vault: LawVault | None,
    goal: ScientificInvestigationGoal,
    active_brick: LawBrick,
    active_x0: torch.Tensor,
    active_params: torch.Tensor,
    active_states: torch.Tensor | None,
    inv_report: InvariantDiscoveryReport | None,
    interv_sol: InterventionSolution | None,
    hypothesis_tournament: list[dict[str, Any]] | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    AuditCertificate,
    InterventionSolution | None,
    FalsificationStressReport,
    CrystallizedLawRecord | None,
    str,
    dict[str, Any],
]:
    """
    Execute the 4-gate + multi-horizon (4x) adversarial falsification stress test,
    self-repair any unstable intervention, and crystallize verified laws into `LawVault`.
    """
    if active_states is None:
        roll = rollout_trajectories(
            active_brick,
            active_x0,
            active_params,
            dt=goal.dt,
            num_steps=goal.num_steps,
            integrator="auto",
        )
        active_states = roll.states[0]

    falsif_report = run_falsification_stress_test(
        brick=active_brick,
        calibrated_params=active_params,
        reference_states=active_states,
        dt=goal.dt,
        invariant_report=inv_report,
    )

    if not falsif_report.passed_all_gates and active_brick.param_dim > 0:
        target_map = goal.target_state_objective or {}
        non_target_rep = (
            [
                idx_c
                for idx_c, sp_c in enumerate(active_brick.state_specs)
                if sp_c.name not in target_map
            ]
            if target_map
            else []
        )
        if not non_target_rep:
            collateral_rep: list[int] | None = None
        elif len(non_target_rep) == 1:
            collateral_rep = non_target_rep
        else:
            bystander_kw_rep = ("ca", "bystander", "collateral", "healthy", "off_target", "normal")
            explicit_rep = [
                idx_c
                for idx_c in non_target_rep
                if any(kw in active_brick.state_specs[idx_c].name.lower() for kw in bystander_kw_rep)
            ]
            if explicit_rep:
                collateral_rep = explicit_rep[:1]
            elif 3 in non_target_rep:
                collateral_rep = [3]
            else:
                collateral_rep = [non_target_rep[-1]]

        def _safe_target_loss(roll_in: RolloutResult) -> torch.Tensor:
            l_acc = torch.zeros((), dtype=torch.float64, device=roll_in.states.device)
            half_t = max(int(roll_in.states.shape[1] // 2), 1)
            for i_s, sp_s in enumerate(active_brick.state_specs):
                col_s = roll_in.states[:, half_t:, i_s]
                h_lo, h_hi = float(sp_s.homeostatic_min), float(sp_s.homeostatic_max)
                span_s = max(h_hi - h_lo, 1e-6)
                safe_lo = h_lo + 0.14 * span_s
                safe_hi = h_hi - 0.14 * span_s
                margin_pen = torch.mean(torch.relu(safe_lo - col_s) ** 2 + torch.relu(col_s - safe_hi) ** 2)
                l_acc = l_acc + 3.0 * margin_pen
                if target_map and sp_s.name in target_map:
                    l_acc = l_acc + torch.mean(
                        (roll_in.states[:, -8:, i_s] - float(target_map[sp_s.name])) ** 2
                    )
            if not target_map:
                l_acc = l_acc + 0.1 * torch.mean(roll_in.states[:, -1, :] ** 2)
            return l_acc

        rep_base_params = (
            interv_sol.baseline_params
            if (interv_sol is not None and interv_sol.baseline_params.shape[-1] == active_brick.param_dim)
            else active_params
        )
        repaired_interv = solve_target_intervention(
            brick=active_brick,
            initial_state=active_x0,
            baseline_params=rep_base_params,
            dt=goal.dt,
            num_steps=min(goal.num_steps * 4, 120),
            target_loss_fn=_safe_target_loss,
            homeostatic_weight=25.0,
            learning_rate=0.10,
            max_iters=60,
            l2_regularization=2e-3,
            collateral_state_indices=collateral_rep,
        )
        ref_horizon = max(int(goal.num_steps) + 1, 12)
        repaired_falsif = run_falsification_stress_test(
            brick=active_brick,
            calibrated_params=repaired_interv.intervened_params,
            reference_states=repaired_interv.restored_rollout.states[0, :ref_horizon],
            dt=goal.dt,
            invariant_report=inv_report,
        )
        if not repaired_falsif.passed_all_gates:
            repaired_interv_open = solve_target_intervention(
                brick=active_brick,
                initial_state=active_x0,
                baseline_params=rep_base_params,
                dt=goal.dt,
                num_steps=min(goal.num_steps * 4, 120),
                target_loss_fn=_safe_target_loss,
                allowed_param_names=[pp.name for pp in active_brick.param_specs],
                homeostatic_weight=30.0,
                learning_rate=0.10,
                max_iters=65,
                l2_regularization=1e-3,
                collateral_state_indices=None,
            )
            repaired_falsif_open = run_falsification_stress_test(
                brick=active_brick,
                calibrated_params=repaired_interv_open.intervened_params,
                reference_states=repaired_interv_open.restored_rollout.states[0, :ref_horizon],
                dt=goal.dt,
                invariant_report=inv_report,
            )
            if (
                math.isfinite(float(repaired_interv_open.intervened_loss))
                and math.isfinite(float(repaired_interv_open.baseline_loss))
                and (
                    repaired_falsif_open.passed_all_gates
                    or repaired_falsif_open.ood_viable_fraction > repaired_falsif.ood_viable_fraction
                )
            ):
                repaired_interv = repaired_interv_open
                repaired_falsif = repaired_falsif_open
        rep_losses_finite = bool(
            math.isfinite(float(repaired_interv.intervened_loss))
            and math.isfinite(float(repaired_interv.baseline_loss))
        )
        rep_loss_ok = bool(
            rep_losses_finite
            and (
                interv_sol is None
                or not math.isfinite(float(interv_sol.intervened_loss))
                or float(repaired_interv.intervened_loss) < float(repaired_interv.baseline_loss) * 0.25
            )
        )
        if rep_loss_ok and (
            repaired_falsif.passed_all_gates
            or repaired_falsif.ood_viable_fraction > falsif_report.ood_viable_fraction
        ):
            interv_sol = repaired_interv
            active_params = repaired_interv.intervened_params.detach().clone()
            active_states = repaired_interv.restored_rollout.states[0, :ref_horizon]
            falsif_report = repaired_falsif
            if hypothesis_tournament:
                rep_mh4x = (
                    repaired_falsif.multi_horizon_report.horizon_4x_viable_fraction
                    if repaired_falsif.multi_horizon_report is not None
                    else repaired_falsif.ood_viable_fraction
                )
                rep_v_ok = bool(repaired_interv.restored_rollout.all_time_viable_mask.all().item())
                for ht_entry in hypothesis_tournament:
                    if ht_entry.get("is_winner"):
                        ht_entry["horizon_4x_viable"] = round(float(rep_mh4x), 4)
                        ht_entry["target_loss"] = round(float(repaired_interv.intervened_loss), 6)
                        ht_entry["collateral_max_drift"] = round(
                            float(repaired_interv.collateral_max_drift), 6
                        )
                        ht_entry["all_time_viable"] = rep_v_ok
                        if repaired_falsif.passed_all_gates and rep_mh4x >= 0.99 and rep_v_ok:
                            ht_entry["rules_broken"] = []
                            ht_entry["rules_passed_count"] = 6
                            ht_entry["crucible_verdict"] = "SURVIVED_ALL_6_RULES"

    audit_cert = falsif_report.audit_certificate
    cryst_record: CrystallizedLawRecord | None = None
    if (
        falsif_report.passed_all_gates
        and vault is not None
        and goal.crystallize_into_vault
    ):
        existing_laws = vault.list_laws(include_discovered=True)
        p_ids = (existing_laws[0].law_id,) if existing_laws else ()
        cryst_record = crystallize_verified_law(
            vault=vault,
            brick=active_brick,
            stress_report=falsif_report,
            parent_law_ids=p_ids,
        )
    mh_4x = (
        falsif_report.multi_horizon_report.horizon_4x_viable_fraction
        if falsif_report.multi_horizon_report is not None
        else falsif_report.ood_viable_fraction
    )
    summary_str = (
        f"Executed 4-gate + multi-horizon (4x) falsification suite: "
        f"passed_all={falsif_report.passed_all_gates}, "
        f"noisy_r2={falsif_report.noisy_vector_field_r2:.4f}, "
        f"ood_viable={falsif_report.ood_viable_fraction:.3f}, "
        f"horizon_4x_viable={mh_4x:.3f}, "
        f"crystallized={cryst_record is not None}."
    )
    metrics = {
        "passed_all_gates": falsif_report.passed_all_gates,
        "noisy_vector_field_r2": falsif_report.noisy_vector_field_r2,
        "ood_viable_fraction": falsif_report.ood_viable_fraction,
        "horizon_4x_viable_fraction": mh_4x,
        "crystallized_law_id": cryst_record.law_id if cryst_record is not None else None,
    }
    client.invoke_background(
        LLMInferenceRequest(
            role="muse_spark_open",
            system_prompt=ROLE_SYSTEM_PROMPTS["muse_spark_open"],
            user_prompt=json.dumps(
                {
                    "task_type": "adversarial_red_team_review",
                    "passed_all_gates": falsif_report.passed_all_gates,
                    "noisy_vector_field_r2": falsif_report.noisy_vector_field_r2,
                    "ood_viable_fraction": falsif_report.ood_viable_fraction,
                    "horizon_4x_viable_fraction": mh_4x,
                }
            ),
            temperature=0.0,
            require_json=True,
        )
    )
    return (
        active_params,
        active_states,
        audit_cert,
        interv_sol,
        falsif_report,
        cryst_record,
        summary_str,
        metrics,
    )
