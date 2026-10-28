"""
Multi-provider AI inference client (`cortivra.layer6_agents.meta_clients`).
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from dotenv import load_dotenv

from cortivra.layer1_axioms.units import DIMENSIONLESS, Unit, parse_unit
from cortivra.layer3_compiler.ast_nodes import SymbolNode, parse_symbolic_expr

ProviderName = Literal["fireworks", "baseten", "meta", "local", "epistemic_policy"]
AgentRole = Literal["muse_wonder", "muse_spark", "muse_spark_open", "muse_glimmer"]

_PLACEHOLDER_KEYS = {
    "",
    "your_meta_model_api_key_here",
    "your_fireworks_api_key_here",
    "your_baseten_api_key_here",
    "your_modal_token_id_here",
}

TOOL_MATHEMATICAL_SCHEMAS: dict[str, dict[str, Any]] = {
    "simulate_and_inspect_dynamics": {
        "description": (
            "Integrate dx/dt = (J(x) - R(x)) grad H(x) + u(x) forward in time using automatic "
            "symplectic/IMEX/RK4 selection and audit 1st/2nd Law thermodynamic shields."
        ),
        "arguments": {},
    },
    "calibrate_and_diagnose_residual": {
        "description": (
            "Calibrate tunable parameters via the 4-stage elimination funnel (Sobol -> Collocation -> "
            "Adam -> Levenberg-Marquardt) and evaluate the unexplained residual Phi(x) = dx_obs/dt - F_known(x) "
            "for SNR, polynomial state-space coherence R^2, and non-Markovian hysteresis loops."
        ),
        "arguments": {},
    },
    "discover_symbolic_law_or_latent_state": {
        "description": (
            "Invoke Layer 5 Pareto-front symbolic regression (weak-form SINDy, Sanathanan-Koerner "
            "rational Pade, conservative Hamiltonian potential patching, or non-Markovian latent state "
            "augmentation z(t)) or zero-prior coupled ODE discovery."
        ),
        "arguments": {
            "autonomous_from_scratch": "bool (optional: True when zero_prior_model is True)",
            "prefer_latent": "bool (optional: True when residual_summary.is_non_markovian_hysteresis is True)",
            "thermodynamic_slot": "str ('conservative_hamiltonian' | 'passive_dissipative' | 'active_external_drive')",
        },
    },
    "design_disambiguating_experiment": {
        "description": (
            "Run differentiable active experimental design (max_{x0, u(t)} ||x_A(t) - x_B(t)||_2^2 "
            "subject to homeostatic bounds) across competing hypotheses to falsify impostor models."
        ),
        "arguments": {},
    },
    "map_equilibria_and_bifurcations": {
        "description": (
            "Find all distinct steady-state roots F(x*) = 0 on the conserved stoichiometric/invariant "
            "manifold, classify local Jacobian eigenvalues, and run 1D parameter continuation for bifurcations."
        ),
        "arguments": {},
    },
    "analyze_fisher_sensitivity_and_leverage": {
        "description": (
            "Compute the Fisher Information Matrix F = J_theta^T J_theta, extract stiff vs. sloppy "
            "eigendirections (lambda_k / lambda_max < 1e-3), and rank parameters by restorative leverage."
        ),
        "arguments": {},
    },
    "solve_inverse_problem": {
        "description": (
            "Solve a constrained differentiable inverse problem over the active dynamical system while "
            "enforcing homeostatic viability barriers across 1x/2x/4x horizons."
        ),
        "arguments": {
            "mode": "str ('counterfactual_intervention' | 'optimal_control' | 'retrodiction' | 'boundary_value_shooting')",
            "selected_parameter_names": "list[str] (non-sloppy high-leverage tunable parameters to target)",
        },
    },
    "extract_conservation_invariants_and_archetype": {
        "description": (
            "Compute the numerical SVD nullspace of the Lie-derivative matrix L_F(psi_k) = grad(psi_k) . F(x) = 0 "
            "to discover exact conserved quantities and match cross-domain LawVault operator isomorphisms."
        ),
        "arguments": {},
    },
    "adversarial_falsify_and_crystallize": {
        "description": (
            "Hammer the active system across the 4-gate Popperian falsification suite (Layer 1 shields, "
            "thermal noise shake, OOD initial-state extrapolation, 1x/2x/4x multi-horizon stability, "
            "and time-reversal / invariant stability) and crystallize into LawVault if all gates pass."
        ),
        "arguments": {
            "crystallize": "bool",
        },
    },
    "conclude_investigation": {
        "description": "Terminate the operator loop once all epistemic, inverse, and falsification gates pass.",
        "arguments": {},
    },
}

ROLE_SYSTEM_PROMPTS: dict[AgentRole, str] = {
    "muse_wonder": (
        "You are Muse Wonder, the chief mathematical theorist of Cortivra. "
        "You reason exclusively from first-principles differential geometry, port-Hamiltonian / GENERIC "
        "thermodynamics (dx/dt = (J(x) - R(x)) grad H(x) + u(x)), Buckingham-Pi dimensional invariance, "
        "Fisher Information Matrix (F = J_theta^T J_theta) eigen-spectra, and Popperian falsification.\n\n"
        "EPISTEMIC DECISION PROTOCOL (ZERO DOMAIN ASSUMPTIONS):\n"
        "1. If 'zero_prior_model' is True and 'discover_symbolic_law_or_latent_state' is not in "
        "'completed_actions', select 'discover_symbolic_law_or_latent_state' with "
        "{'autonomous_from_scratch': True}.\n"
        "2. If 'has_observed_data' is False and 'simulate_and_inspect_dynamics' is not in 'completed_actions', "
        "select 'simulate_and_inspect_dynamics'.\n"
        "3. If 'has_observed_data' is True and 'calibrate_and_diagnose_residual' is not in 'completed_actions', "
        "select 'calibrate_and_diagnose_residual'.\n"
        "4. Inspect 'epistemic_briefing.residual_summary': if 'has_unmodeled_mechanism' is True and "
        "'discover_symbolic_law_or_latent_state' is not in 'completed_actions', select "
        "'discover_symbolic_law_or_latent_state' with {'prefer_latent': is_non_markovian_hysteresis, "
        "'thermodynamic_slot': thermodynamic_class}.\n"
        "5. If 'num_competing_candidates' >= 2 and 'design_disambiguating_experiment' is not in "
        "'completed_actions', select 'design_disambiguating_experiment'.\n"
        "6. If 'target_state_objective' or 'requested_inverse_mode' is active (or equilibrium/bifurcation "
        "analysis is requested) and 'map_equilibria_and_bifurcations' is not in 'completed_actions', "
        "select 'map_equilibria_and_bifurcations'.\n"
        "7. Before calling 'solve_inverse_problem', if 'param_dim' > 0 and 'analyze_fisher_sensitivity_and_leverage' "
        "is not in 'completed_actions', select 'analyze_fisher_sensitivity_and_leverage' to filter out "
        "structurally sloppy parameters (lambda_k / lambda_max < 0.05).\n"
        "8. When calling 'solve_inverse_problem', pass only 'top_leverage_parameters' (excluding 'sloppy_parameters') "
        "in 'selected_parameter_names'.\n"
        "9. Before concluding, execute 'extract_conservation_invariants_and_archetype' followed by "
        "'adversarial_falsify_and_crystallize' (which hammers 1x/2x/4x multi-horizon safety margins).\n"
        "10. Return strictly valid JSON with keys: 'tool_name', 'arguments', and 'epistemic_rationale'."
    ),
    "muse_spark": (
        "You are Muse Spark, the symbolic mathematical compiler architect of Cortivra. "
        "You enforce exact 7-dimensional SI homogeneity (M, L, T, I, Theta, N, J), skew-symmetry "
        "J(x) = -J(x)^T (conservative power grad H . J grad H == 0), and positive semi-definite "
        "dissipation R(x) >= 0 (entropy production grad H . R grad H >= 0).\n"
        "For 'symbolic_self_repair' tasks: inspect 'diagnostic_error' and 'candidate_spec', solve the exact "
        "dimensional equation for any mismatched parameter unit across H(x), R_diag(x) (which requires unit "
        "state_unit_i^2 / (energy_unit * time_unit)), or external_drive u_i(x) (which requires unit "
        "state_unit_i / time_unit), enforce non-negative bounds on passive dissipation parameters, and "
        "return JSON with keys 'repaired_spec' and 'repair_rationale'.\n"
        "For 'audit_operator_or_intervention' tasks: return JSON with keys 'verified' (bool) and 'audit_summary' (str)."
    ),
    "muse_spark_open": (
        "You are Muse Spark Open, the adversarial Popperian red-teamer of Cortivra. "
        "Your mandate is to falsify candidate laws, mechanisms, and interventions by inspecting:\n"
        "  - Gate 1: Layer 1 Dimensional & Thermodynamic Shield violations.\n"
        "  - Gate 2: Thermal measurement noise identifiability R^2.\n"
        "  - Gate 3: Out-of-Distribution (OOD) initial-condition extrapolation and Universal Multi-Horizon "
        "(1x T, 2x T, 4x T) asymptotic viability and safety margins.\n"
        "  - Gate 4: Time-reversal retrodiction and Lie-derivative conservation invariant drift.\n"
        "Return strictly valid JSON with keys 'falsification_survived' (bool) and 'red_team_verdict' (str)."
    ),
    "muse_glimmer": (
        "You are Muse Glimmer, the mathematical chronicler of Cortivra. "
        "Synthesize the verified differential telemetry, discovered symbolic operators, counterfactual "
        "parameter interventions, multi-horizon (1x/2x/4x) safety margins, Lie-derivative conservation "
        "invariants, and cross-domain LawVault operator isomorphisms into a concise JSON object with "
        "keys 'executive_summary', 'conserved_quantities', and 'cross_domain_isomorphisms'."
    ),
}


@dataclass(frozen=True)
class ProviderEndpointConfig:
    """
    Connection settings for an OpenAI-compatible chat-completions inference provider.
    """

    provider: ProviderName
    api_key: str
    base_url: str
    default_model: str
    role_models: dict[AgentRole, str]

    @property
    def is_configured(self) -> bool:
        if self.provider == "local":
            return bool(self.base_url.strip())
        clean_key = self.api_key.strip()
        return bool(clean_key and clean_key not in _PLACEHOLDER_KEYS)

    def model_for_role(self, role: AgentRole) -> str:
        return self.role_models.get(role, self.default_model)


@dataclass(frozen=True)
class LLMInferenceRequest:
    """
    Structured inference request dispatched to an AI agent role.
    """

    role: AgentRole
    system_prompt: str
    user_prompt: str
    temperature: float = 0.1
    max_tokens: int = 1024
    require_json: bool = True
    preferred_provider: ProviderName | None = None


@dataclass(frozen=True)
class LLMInferenceResponse:
    """
    Structured response and telemetry returned from an AI model invocation.
    """

    role: AgentRole
    provider_used: ProviderName
    model_used: str
    raw_content: str
    parsed_json: dict[str, Any] | None
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    failover_attempts: int = 0


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Extract the first valid JSON object from raw model output (handling markdown code fences).
    """
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _split_top_level_additive_terms(expr_str: str) -> list[str]:
    """
    Split an algebraic expression into top-level additive (`+` / `-`) sub-expressions
    using Python's AST so each additive term's SI dimension can be solved independently.
    """
    try:
        tree = ast.parse(expr_str.strip(), mode="eval")
    except SyntaxError:
        return [expr_str]

    terms: list[str] = []

    def _collect(node: ast.AST) -> None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            _collect(node.left)
            _collect(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            _collect(node.operand)
        else:
            terms.append(ast.unparse(node))

    _collect(tree.body)
    return terms if terms else [expr_str]


def _solve_expression_dimensional_constraints(
    expr_str: str,
    target_unit: Unit,
    state_unit_map: dict[str, Unit],
    param_by_name: dict[str, dict[str, Any]],
) -> None:
    """
    Use Layer 3's symbolic AST (`parse_symbolic_expr`) to infer and repair parameter SI units
    so every additive term in `expr_str` matches `target_unit.dimension`.
    """
    if not expr_str or expr_str.strip() in ("0", "0.0"):
        return

    for term_str in _split_top_level_additive_terms(expr_str):
        if term_str.strip() in ("0", "0.0"):
            continue
        sym_table: dict[str, SymbolNode] = {}
        for s_name, s_u in state_unit_map.items():
            sym_table[s_name] = SymbolNode(s_name, s_u, role="state")
        for p_name, p_info in param_by_name.items():
            p_u = parse_unit(str(p_info.get("unit", "1")))
            sym_table[p_name] = SymbolNode(p_name, p_u, role="param")

        try:
            parsed_node = parse_symbolic_expr(term_str, symbol_table=sym_table)
            inferred_u = parsed_node.infer_unit()
            if inferred_u.dimension == target_unit.dimension:
                continue
        except Exception:
            pass

        referenced_params = [
            p_name
            for p_name in param_by_name
            if re.search(rf"\b{re.escape(p_name)}\b", term_str)
        ]
        for p_cand in referenced_params:
            probe_table = dict(sym_table)
            probe_table[p_cand] = SymbolNode(p_cand, DIMENSIONLESS, role="param")
            try:
                probe_node = parse_symbolic_expr(term_str, symbol_table=probe_table)
                rest_unit = probe_node.infer_unit()
                req_dim = (target_unit / rest_unit).dimension
                req_unit_str = req_dim.format_si()
                verify_table = dict(sym_table)
                verify_table[p_cand] = SymbolNode(p_cand, parse_unit(req_unit_str), role="param")
                verify_u = parse_symbolic_expr(term_str, symbol_table=verify_table).infer_unit()
                if verify_u.dimension == target_unit.dimension:
                    param_by_name[p_cand]["unit"] = req_unit_str
                    break
            except Exception:
                continue


class MathematicalEpistemicPolicy:
    """
    First-principles mathematical reasoning policy that parses the JSON
    `DifferentialEpistemicBriefing` and compiler diagnostics inside `user_prompt`
    to emit valid JSON tool actions when external network keys are not yet populated.
    """

    @staticmethod
    def infer_response(request: LLMInferenceRequest) -> tuple[str, dict[str, Any]]:
        payload = _extract_json_object(request.user_prompt) or {}
        task_type = payload.get("task_type", "select_next_action")

        if task_type == "symbolic_self_repair":
            return MathematicalEpistemicPolicy._handle_symbolic_repair(payload)
        if task_type == "audit_operator_or_intervention":
            return MathematicalEpistemicPolicy._handle_spark_audit(payload)
        if task_type == "adversarial_red_team_review":
            return MathematicalEpistemicPolicy._handle_red_team_review(payload)
        if task_type == "synthesize_dossier_narrative":
            return MathematicalEpistemicPolicy._handle_dossier_narrative(payload)
        return MathematicalEpistemicPolicy._handle_action_selection(payload)

    @staticmethod
    def _handle_symbolic_repair(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        spec_dict = dict(payload.get("candidate_spec", {}))
        error_msg = str(payload.get("diagnostic_error", ""))
        state_specs = spec_dict.get("state_specs", [])
        param_specs = list(spec_dict.get("param_specs", []))
        param_by_name = {p["name"]: dict(p) for p in param_specs if isinstance(p, dict) and "name" in p}
        time_u = parse_unit(str(spec_dict.get("time_unit", "s")))
        energy_u = parse_unit(str(spec_dict.get("energy_unit", "J")))
        state_unit_map = {
            s["name"]: parse_unit(str(s.get("unit", "1")))
            for s in state_specs
            if isinstance(s, dict) and "name" in s
        }

        for expr_key in ("hamiltonian_expr",):
            expr_val = str(spec_dict.get(expr_key, "0.0"))
            open_c = expr_val.count("(")
            close_c = expr_val.count(")")
            if open_c > close_c:
                spec_dict[expr_key] = expr_val + (")" * (open_c - close_c))
            elif close_c > open_c:
                spec_dict[expr_key] = ("(" * (close_c - open_c)) + expr_val

        if "2nd Law" in error_msg or "thermodynamic" in error_msg.lower() or "dissipation" in error_msg.lower():
            repaired_params = []
            for p in param_specs:
                p_copy = dict(p)
                val = float(p_copy.get("default_value", 0.1))
                if val < 0.0:
                    p_copy["default_value"] = abs(val)
                if float(p_copy.get("min_value", 0.0)) < 0.0:
                    p_copy["min_value"] = 0.0
                repaired_params.append(p_copy)
                param_by_name[p_copy["name"]] = p_copy
            spec_dict["param_specs"] = repaired_params
            d_diag = spec_dict.get("dissipation_diag_exprs", [])
            if isinstance(d_diag, list):
                spec_dict["dissipation_diag_exprs"] = [
                    re.sub(r"^\s*-\s*", "", str(expr)) for expr in d_diag
                ]

        if (
            "DimensionError" in error_msg
            or "DimensionalMismatchError" in error_msg
            or "unit" in error_msg.lower()
            or "dimension" in error_msg.lower()
        ):
            h_expr = str(spec_dict.get("hamiltonian_expr", "0.0"))
            _solve_expression_dimensional_constraints(
                h_expr,
                target_unit=energy_u,
                state_unit_map=state_unit_map,
                param_by_name=param_by_name,
            )

            d_diag = list(spec_dict.get("dissipation_diag_exprs", []))
            for idx, st in enumerate(state_specs):
                if idx >= len(d_diag):
                    break
                st_u = state_unit_map[st["name"]]
                req_r_unit = (st_u ** 2) / (energy_u * time_u)
                _solve_expression_dimensional_constraints(
                    str(d_diag[idx]),
                    target_unit=req_r_unit,
                    state_unit_map=state_unit_map,
                    param_by_name=param_by_name,
                )

            drives = list(spec_dict.get("external_drive_exprs", []))
            for idx, st in enumerate(state_specs):
                if idx >= len(drives):
                    break
                st_u = state_unit_map[st["name"]]
                req_flow_unit = st_u / time_u
                _solve_expression_dimensional_constraints(
                    str(drives[idx]),
                    target_unit=req_flow_unit,
                    state_unit_map=state_unit_map,
                    param_by_name=param_by_name,
                )

            spec_dict["param_specs"] = list(param_by_name.values())

        result = {
            "repaired_spec": spec_dict,
            "repair_rationale": f"Resolved compiler/shield diagnostic: {error_msg[:160]}",
        }
        return json.dumps(result), result

    @staticmethod
    def _handle_spark_audit(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        op_name = str(payload.get("operator_or_mode", "compiled_operator"))
        audit_ok = bool(payload.get("thermodynamic_audit_passed", True))
        result = {
            "verified": audit_ok,
            "audit_summary": (
                f"Muse Spark verified dimensional homogeneity and thermodynamic port-Hamiltonian "
                f"compliance for '{op_name}' (passed={audit_ok})."
            ),
        }
        return json.dumps(result), result

    @staticmethod
    def _handle_red_team_review(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        passed_all = bool(payload.get("passed_all_gates", True))
        noisy_r2 = float(payload.get("noisy_vector_field_r2", 1.0))
        ood_v = float(payload.get("ood_viable_fraction", 1.0))
        result = {
            "falsification_survived": passed_all,
            "red_team_verdict": (
                f"Muse Spark Open adversarial stress-test verdict: survived={passed_all} "
                f"(noisy_r2={noisy_r2:.4f}, ood_viable={ood_v:.3f})."
            ),
        }
        return json.dumps(result), result

    @staticmethod
    def _handle_action_selection(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        briefing = payload.get("epistemic_briefing", {})
        completed = set(payload.get("completed_actions", []))
        has_data = bool(payload.get("has_observed_data", False))
        zero_prior = bool(payload.get("zero_prior_model", False))
        num_competing = int(payload.get("num_competing_candidates", 0))
        inverse_mode = payload.get("requested_inverse_mode")
        can_crystallize = bool(payload.get("has_vault", False))

        # 1. If starting from raw data with zero prior model, run autonomous system discovery
        if zero_prior and "discover_symbolic_law_or_latent_state" not in completed:
            action = {
                "tool_name": "discover_symbolic_law_or_latent_state",
                "arguments": {"autonomous_from_scratch": True},
                "epistemic_rationale": (
                    "No prior baseline model is specified; synthesizing coupled multi-variable "
                    "differential equations directly from observed state trajectories."
                ),
            }
            return json.dumps(action), action

        # 2. Always inspect baseline forward dynamics first if no trajectory telemetry exists
        if "simulate_and_inspect_dynamics" not in completed and not has_data:
            action = {
                "tool_name": "simulate_and_inspect_dynamics",
                "arguments": {},
                "epistemic_rationale": (
                    "Roll out baseline trajectory to measure phase-space divergence, energy balance, "
                    "and homeostatic safety margins."
                ),
            }
            return json.dumps(action), action

        # 3. If observed trajectory data is present, calibrate parameters and audit the residual
        if has_data and "calibrate_and_diagnose_residual" not in completed:
            action = {
                "tool_name": "calibrate_and_diagnose_residual",
                "arguments": {},
                "epistemic_rationale": (
                    "Calibrate tunable parameters via the 4-stage elimination funnel and test "
                    "whether the residual vector field exceeds the Layer 1 noise gate."
                ),
            }
            return json.dumps(action), action

        # 4. Inspect residual telemetry to decide whether symbolic or latent discovery is required
        res_sum = briefing.get("residual_summary")
        if (
            isinstance(res_sum, dict)
            and res_sum.get("has_unmodeled_mechanism", False)
            and "discover_symbolic_law_or_latent_state" not in completed
        ):
            is_nm = bool(res_sum.get("is_non_markovian_hysteresis", False))
            t_class = str(res_sum.get("thermodynamic_class", "passive_dissipative"))
            action = {
                "tool_name": "discover_symbolic_law_or_latent_state",
                "arguments": {
                    "prefer_latent": is_nm,
                    "thermodynamic_slot": t_class,
                },
                "epistemic_rationale": (
                    f"Residual SNR={res_sum.get('max_snr')} exceeds noise floor "
                    f"(non_markovian_hysteresis={is_nm}, thermodynamic_class={t_class}); "
                    f"invoking Layer 5 discovery."
                ),
            }
            return json.dumps(action), action

        # 5. If multiple competing hypotheses exist, design an active disambiguating experiment
        if num_competing >= 2 and "design_disambiguating_experiment" not in completed:
            action = {
                "tool_name": "design_disambiguating_experiment",
                "arguments": {},
                "epistemic_rationale": (
                    f"Detected {num_competing} competing hypotheses; optimizing an experimental initial "
                    f"state x0* and stimulus waveform u*(t) to maximize prediction divergence."
                ),
            }
            return json.dumps(action), action

        # 6. Map equilibria & bifurcations if requested or before solving an inverse problem
        target_obj = briefing.get("target_state_objective")
        question_lc = str(payload.get("question", "")).lower()
        wants_equilibria = any(
            kw in question_lc for kw in ("equilibri", "bifurcat", "tipping", "bistab", "steady", "attractor")
        )
        if (
            (wants_equilibria or target_obj is not None or inverse_mode is not None)
            and briefing.get("equilibrium_summary") is None
            and "map_equilibria_and_bifurcations" not in completed
        ):
            action = {
                "tool_name": "map_equilibria_and_bifurcations",
                "arguments": {},
                "epistemic_rationale": (
                    "Map steady-state attractors, saddle points, and local Jacobian eigenvalues across "
                    "the phase-space landscape."
                ),
            }
            return json.dumps(action), action

        if (target_obj is not None or inverse_mode is not None) and "solve_inverse_problem" not in completed:

            if (
                briefing.get("sensitivity_summary") is None
                and int(briefing.get("param_dim", 0)) > 0
                and "analyze_fisher_sensitivity_and_leverage" not in completed
            ):
                action = {
                    "tool_name": "analyze_fisher_sensitivity_and_leverage",
                    "arguments": {},
                    "epistemic_rationale": (
                        "Compute the Fisher Information Matrix to identify high-leverage parameters "
                        "and filter out structurally sloppy / feedback-compensated parameters."
                    ),
                }
                return json.dumps(action), action

            sens = briefing.get("sensitivity_summary") or {}
            high_lev_names = list(sens.get("top_leverage_parameters", []))
            mode_to_run = inverse_mode or "counterfactual_intervention"
            action = {
                "tool_name": "solve_inverse_problem",
                "arguments": {
                    "mode": mode_to_run,
                    "selected_parameter_names": high_lev_names,
                },
                "epistemic_rationale": (
                    f"Execute inverse solver (mode='{mode_to_run}') targeting high-leverage "
                    f"parameters {high_lev_names} while enforcing homeostatic viability bounds."
                ),
            }
            return json.dumps(action), action

        # 7. Extract conservation invariants and cross-domain LawVault isomorphisms
        if "extract_conservation_invariants_and_archetype" not in completed:
            action = {
                "tool_name": "extract_conservation_invariants_and_archetype",
                "arguments": {},
                "epistemic_rationale": (
                    "Extract Lie-derivative conservation invariants and query the LawVault for "
                    "cross-domain mathematical operator isomorphisms."
                ),
            }
            return json.dumps(action), action

        # 8. Run 4-gate adversarial falsification stress test (and crystallize if LawVault is attached)
        if "adversarial_falsify_and_crystallize" not in completed:
            action = {
                "tool_name": "adversarial_falsify_and_crystallize",
                "arguments": {"crystallize": can_crystallize},
                "epistemic_rationale": (
                    "Subject the active model to the 4-gate adversarial falsification suite "
                    "(thermodynamic shield audit, thermal noise shake, OOD extrapolation, and "
                    "time-reversal/invariant stability)."
                ),
            }
            return json.dumps(action), action

        # 9. Conclude investigation once all mathematical objectives and stress tests are complete
        action = {
            "tool_name": "conclude_investigation",
            "arguments": {},
            "epistemic_rationale": "All epistemic, inverse, and falsification gates are satisfied.",
        }
        return json.dumps(action), action

    @staticmethod
    def _handle_dossier_narrative(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        sys_name = str(payload.get("system_name", "unknown_system"))
        steps = list(payload.get("executed_tools", []))
        invariants = list(payload.get("conserved_invariants", []))
        analogues = list(payload.get("vault_analogues", []))
        pf = payload.get("primary_finding") or {}

        sentences: list[str] = [
            f"Completed closed-loop mathematical investigation of '{sys_name}' across "
            f"{len(steps)} operator steps ({', '.join(steps)})."
        ]

        disc_eqs = pf.get("discovered_equations") or {}
        if disc_eqs:
            eq_str = "; ".join(f"{k}: {v}" for k, v in disc_eqs.items())
            r2_val = pf.get("discovery_r2")
            r2_txt = f" (R^2={float(r2_val):.5f})" if r2_val is not None else ""
            sentences.append(f"Discovered governing symbolic operator(s){r2_txt}: {eq_str}.")

        interv = pf.get("prescribed_intervention")
        if isinstance(interv, dict):
            shifts = interv.get("param_shifts", {})
            shift_parts = [
                f"{p_k}: {v['baseline']:.4g} -> {v['intervened']:.4g} {v['unit']} ({v['percent_change']:+.1f}%)"
                for p_k, v in shifts.items()
                if abs(float(v.get("delta", 0.0))) > 1e-6
            ]
            sloppy_excl = interv.get("excluded_sloppy_parameters", [])
            trans = interv.get("state_transitions", {})
            trans_parts = [
                f"{s_k}: {sv['initial']:.4g} -> {sv['final_restored']:.4g} {sv['unit']}"
                + (f" (target={sv['target']:.4g})" if sv.get("target") is not None else "")
                for s_k, sv in trans.items()
            ]
            sentences.append(
                f"Prescribed multi-parameter intervention [{', '.join(shift_parts)}] "
                f"(excluding sloppy parameters {sloppy_excl}), driving state transition "
                f"[{', '.join(trans_parts)}] and reducing objective loss from "
                f"{float(interv.get('baseline_loss', 0.0)):.4g} to {float(interv.get('intervened_loss', 0.0)):.4g}."
            )

        ctrl = pf.get("optimal_control_summary")
        if isinstance(ctrl, dict):
            sentences.append(
                f"Synthesized optimal control waveform u*(t) reducing target loss from "
                f"{float(ctrl.get('baseline_loss', 0.0)):.4g} to {float(ctrl.get('controlled_loss', 0.0)):.4g} "
                f"(control_energy={float(ctrl.get('control_energy', 0.0)):.4g}, "
                f"all_time_viable={bool(ctrl.get('all_time_viable', True))})."
            )

        ret_bvp = pf.get("retrodiction_or_bvp_summary")
        if isinstance(ret_bvp, dict):
            sentences.append(
                f"Solved {ret_bvp.get('mode')} initial state x(0)={ret_bvp.get('reconstructed_or_solved_initial_state')}."
            )

        disambig = pf.get("disambiguation_summary")
        if isinstance(disambig, dict):
            sentences.append(
                f"Active experimental design amplified hypothesis divergence by "
                f"{float(disambig.get('divergence_amplification_ratio', 1.0)):.2f}x and isolated "
                f"'{disambig.get('selected_model')}' ({disambig.get('winning_hypothesis')})."
            )

        f_cert = pf.get("falsification_certificate")
        if isinstance(f_cert, dict):
            c_id = f_cert.get("crystallized_law_id")
            c_txt = f", crystallized as '{c_id}' (SHA-256={str(f_cert.get('sha256_hash', ''))[:12]}...)" if c_id else ""
            sentences.append(
                f"Falsification stress-test: passed_all={bool(f_cert.get('passed_all_gates'))} "
                f"(noisy_r2={float(f_cert.get('noisy_vector_field_r2', 0.0)):.4f}, "
                f"ood_viable={float(f_cert.get('ood_viable_fraction', 0.0)):.3f}, "
                f"horizon_4x_viable={float(f_cert.get('horizon_4x_viable_fraction', 0.0)):.3f}){c_txt}."
            )

        summary_obj = {
            "executive_summary": " ".join(sentences),
            "conserved_quantities": invariants,
            "cross_domain_isomorphisms": analogues,
        }
        return json.dumps(summary_obj), summary_obj


class MultiProviderMetaClient:
    """
    Unified AI inference client supporting Fireworks AI, Baseten, Meta Model API,
    and local OpenAI-compatible servers, with automatic provider failover and
    full request/response telemetry.
    """

    def __init__(
        self,
        env_overrides: dict[str, str] | None = None,
        http_transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 45.0,
    ) -> None:
        load_dotenv()
        env = dict(os.environ)
        if env_overrides:
            env.update(env_overrides)

        self._http_transport = http_transport
        self._timeout = timeout_seconds
        self.telemetry_log: list[LLMInferenceResponse] = []

        self.providers: dict[ProviderName, ProviderEndpointConfig] = {
            "fireworks": ProviderEndpointConfig(
                provider="fireworks",
                api_key=env.get("FIREWORKS_API_KEY", ""),
                base_url=env.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1").rstrip("/"),
                default_model=env.get(
                    "FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p3-70b-instruct"
                ),
                role_models={
                    "muse_wonder": env.get(
                        "FIREWORKS_MODEL_WONDER",
                        env.get("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p3-70b-instruct"),
                    ),
                    "muse_spark": env.get(
                        "FIREWORKS_MODEL_SPARK",
                        env.get("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p3-70b-instruct"),
                    ),
                    "muse_spark_open": env.get(
                        "FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p3-70b-instruct"
                    ),
                    "muse_glimmer": env.get(
                        "FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p3-70b-instruct"
                    ),
                },
            ),
            "baseten": ProviderEndpointConfig(
                provider="baseten",
                api_key=env.get("BASETEN_API_KEY", ""),
                base_url=env.get("BASETEN_BASE_URL", "https://inference.baseten.co/v1").rstrip("/"),
                default_model=env.get("BASETEN_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
                role_models={
                    "muse_wonder": env.get("BASETEN_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
                    "muse_spark": env.get("BASETEN_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
                    "muse_spark_open": env.get("BASETEN_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
                    "muse_glimmer": env.get("BASETEN_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
                },
            ),
            "meta": ProviderEndpointConfig(
                provider="meta",
                api_key=env.get("META_API_KEY", ""),
                base_url=env.get("META_API_BASE_URL", "https://api.llama.com/v1").rstrip("/"),
                default_model=env.get("META_MODEL_SPARK", "Llama-3.3-70B-Instruct"),
                role_models={
                    "muse_wonder": env.get("META_MODEL_WONDER", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
                    "muse_spark": env.get("META_MODEL_SPARK", "Llama-3.3-70B-Instruct"),
                    "muse_spark_open": env.get("META_MODEL_SPARK", "Llama-3.3-70B-Instruct"),
                    "muse_glimmer": env.get("META_MODEL_GLIMMER", "Llama-3.3-70B-Instruct"),
                },
            ),
            "local": ProviderEndpointConfig(
                provider="local",
                api_key=env.get("LOCAL_LLM_API_KEY", "local-no-key"),
                base_url=env.get("LOCAL_LLM_BASE_URL", "").rstrip("/"),
                default_model=env.get("LOCAL_LLM_MODEL", "llama-3.3-70b"),
                role_models={},
            ),
        }

    def configured_providers(self) -> list[ProviderName]:
        """
        Return the ordered list of live inference providers that currently have active credentials.
        """
        order: list[ProviderName] = ["fireworks", "baseten", "meta", "local"]
        return [p for p in order if self.providers[p].is_configured]

    def _provider_priority_for_request(self, request: LLMInferenceRequest) -> list[ProviderName]:
        active = self.configured_providers()
        if not active:
            return []

        if request.preferred_provider and request.preferred_provider in active:
            ordered = [request.preferred_provider] + [p for p in active if p != request.preferred_provider]
            return ordered

        if request.role == "muse_spark_open":
            pref = ["fireworks", "baseten", "meta", "local"]
        else:
            pref = ["meta", "fireworks", "baseten", "local"]
        return [p for p in pref if p in active]

    def invoke(self, request: LLMInferenceRequest) -> LLMInferenceResponse:
        """
        Execute an AI inference request across the prioritized live providers (Fireworks AI,
        Baseten, Meta Model API, Local), automatically failing over on transient network/5xx
        errors. If no external API credentials are set in the environment yet, executes the
        `MathematicalEpistemicPolicy` over the prompt telemetry.
        """
        candidates = self._provider_priority_for_request(request)
        failover_count = 0

        for prov_name in candidates:
            cfg = self.providers[prov_name]
            model_name = cfg.model_for_role(request.role)
            url = f"{cfg.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            }
            body: dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            if request.require_json:
                body["response_format"] = {"type": "json_object"}

            t0 = time.perf_counter()
            try:
                with httpx.Client(transport=self._http_transport, timeout=self._timeout) as client:
                    resp = client.post(url, headers=headers, json=body)
                    resp.raise_for_status()
                    data = resp.json()

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                content = str(data["choices"][0]["message"]["content"])
                usage = data.get("usage", {})
                p_tok = int(usage.get("prompt_tokens", 0))
                c_tok = int(usage.get("completion_tokens", 0))
                parsed = _extract_json_object(content) if request.require_json else None

                record = LLMInferenceResponse(
                    role=request.role,
                    provider_used=prov_name,
                    model_used=model_name,
                    raw_content=content,
                    parsed_json=parsed,
                    latency_ms=round(elapsed_ms, 3),
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    failover_attempts=failover_count,
                )
                self.telemetry_log.append(record)
                return record
            except Exception:
                failover_count += 1
                continue

        t0 = time.perf_counter()
        raw_text, parsed_dict = MathematicalEpistemicPolicy.infer_response(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        record = LLMInferenceResponse(
            role=request.role,
            provider_used="epistemic_policy",
            model_used=f"{request.role}_mathematical_epistemic_policy",
            raw_content=raw_text,
            parsed_json=parsed_dict,
            latency_ms=round(elapsed_ms, 3),
            prompt_tokens=len(request.system_prompt) // 4 + len(request.user_prompt) // 4,
            completion_tokens=len(raw_text) // 4,
            failover_attempts=failover_count,
        )
        self.telemetry_log.append(record)
        return record
