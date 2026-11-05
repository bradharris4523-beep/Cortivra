"""
Multi-provider AI inference client (`cortivra.layer6_agents.meta_clients`).
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from dotenv import load_dotenv


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """
    Extract and parse the first valid JSON object from an LLM response string,
    handling markdown code fences or leading/trailing commentary.
    """
    text = raw_text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if 0 <= first_brace < last_brace:
        try:
            parsed = json.loads(text[first_brace : last_brace + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None

ProviderName = Literal["fireworks", "baseten", "meta", "local"]
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
        "You are Muse Glimmer, the mathematical chronicler and physical realization architect of Cortivra. "
        "Synthesize the verified differential telemetry, discovered symbolic operators, counterfactual "
        "parameter interventions (including Ashby-Eyring multiplicative factorization alpha = alpha_mat * alpha_geom * alpha_act), "
        "multi-horizon (1x/2x/4x) safety margins, Lie-derivative conservation invariants, and cross-domain "
        "LawVault operator isomorphisms into a concise JSON object with keys: "
        "'executive_summary' (str: 2-3 dense quantitative sentences, <= 75 words), 'conserved_quantities' (list[str]), "
        "'cross_domain_isomorphisms' (list[str]), and 'physical_realization_protocol' (dict with keys: 'bottleneck_lever', "
        "'realizability_tier', 'material_or_molecular_mechanism', 'micro_geometric_architecture', "
        "'active_feedback_or_firmware_role', and 'step_by_step_verification_protocol' (list of 4 concise 1-sentence steps)). "
        "Keep the entire JSON output compact (< 400 tokens) with zero thinking prose."
    ),
}

WORLD_MODEL_SYNTHESIS_PROTOCOL: str = (
    "\nUNIVERSAL N-COMPARTMENT CAUSAL BLAST-RADIUS & BYSTANDER SHIELD PROTOCOL FOR 'synthesize_world_model' (DOMAIN-AUTHENTIC NONLINEAR PHYSICS):\n"
    "CRITICAL EFFICIENCY & ANTI-TEMPLATE RULE: Do NOT restate these rules or write long prose in your thinking. "
    "Use concise snake_case state names (<= 18 chars) and output the compact JSON object immediately (< 1200 tokens). "
    "If 'observed_telemetry_columns' is provided in the prompt payload, you MUST name the leading state variables in 'state_specs' using those exact column names. "
    "When the user asks an open-ended engineering, medical, or control question without restricting the state dimension, choose N = 5 coupled state variables (5 compartments) covering the full causal blast radius, and output the JSON object directly. "
    "When the user explicitly asks to discover/identify the governing equations of an uploaded K-column dataset or specifies an exact K-state canonical mathematical system, you may use the exact K state variables of that dataset/system.\n"
    "AUTONOMOUS FAILURE-MODE & DOMAIN-AUTHENTIC NONLINEARITY MANDATE: "
    "You MUST deduce the real physical failure modes, local stresses, and vulnerable neighboring bystander compartments in that specific scientific domain. "
    "NEVER reuse generic placeholder numbers like [2.85, 1.45, 0.72, 1.92, 3.4] or [0.42, 1.65, 0.88, 0.31, 1.15]; every problem MUST have unique, domain-authentic stiffnesses, relaxation rates, and nonlinear interaction laws reflecting its real physical mechanism:\n"
    "  - Quantum / Superconductivity / Condensed Matter: Include quartic Ginzburg-Landau condensate potential ('0.06 * k_1 * x_1**4') and biquadratic competing order-parameter coupling ('0.12 * k_2 * x_1**2 * x_2**2') in 'hamiltonian_expr', plus Josephson/phase skew routing.\n"
    "  - Fusion / Plasma / MHD: Include anharmonic pedestal pressure energy ('0.05 * k_1 * x_1**4') in 'hamiltonian_expr', strong E x B zonal-shear skew-symmetric exchange ('j_osc'), and shear-suppressed transport.\n"
    "  - Prebiotic Chemistry / Origin of Life / Catalysis: Include bimolecular template-substrate hybridization & autocatalytic replication kinetics ('u_1 * x_1 * x_3 / (0.45 + x_3 + 0.35 * x_1**2)') and chelator-shielded hydrolysis.\n"
    "  - Oncology / Immunology: Include predator-prey effector-tumor cytolytic kinetics modulated by hypoxic adenosine/TOX exhaustion and peripheral Treg tolerance.\n"
    "  - Neurobiology / Proteostasis: Include cooperative oligomer nucleation-fragmentation, chaperone/autophagic clearance, and microglial NLRP3 threshold gating.\n"
    "Construct a nonlinear N-state Port-Hamiltonian dynamical system dx/dt = (J(x) - R(x)) grad H(x) + u(x) with N >= 5:\n"
    "  1. AXIOM I, II & III (N >= 5 Coupled Physical States in 'state_specs', all with 'unit': '1'):\n"
    "     - State 0 (Primary Target Order Parameter x_1): primary physical mechanism targeted by the hypothesis "
    "(choose domain-authentic degraded 'default_value' in [1.8, 4.2], 'homeostatic_min': 0.05, 'homeostatic_max': 8.0, 'causal_role': 'primary_target').\n"
    "     - State 1 (Local Stress / Competing Mode x_2): direct local thermodynamic, oxidative, stripe, or mechanical stress inside the primary compartment "
    "(choose 'default_value' in [0.7, 1.5], 'homeostatic_min': 0.05, 'homeostatic_max': 8.0, 'causal_role': 'local_stress').\n"
    "     - State 2 (Energetic / Resource Supply Reservoir x_3): energetic carrier, fuel manifold, or potential reservoir feeding the system "
    "(choose 'default_value' in [0.6, 1.4], 'homeostatic_min': 0.05, 'homeostatic_max': 8.0, 'causal_role': 'energy_reservoir').\n"
    "     - State 3 (Co-Located Bystander Compartment x_4): neighboring cell type, adjacent mechanical component, "
    "or parallel subsystem sharing the physical medium that must be shielded from off-target friendly fire "
    "(choose 'default_value' in [0.8, 1.3], 'homeostatic_min': 0.05, 'homeostatic_max': 8.0, 'causal_role': 'co_located_bystander').\n"
    "     - State 4 (Downstream Functional Receiver x_5): downstream organ, neural pathway, structural frame, or exhaust stage receiving output flux from State 0 "
    "(choose 'default_value' in [0.8, 1.3], 'homeostatic_min': 0.05, 'homeostatic_max': 8.0, 'causal_role': 'downstream_receiver').\n"
    "  2. AXIOM III (Heterogeneous Domain-Specific Parameters in 'param_specs'):\n"
    "     - Stiffnesses in H: k_1 (tunable: true), k_2 (tunable: true), k_3..k_N (tunable: false) with 'unit': '1', min_value 0.1, max_value 10.0, and DISTINCT domain-specific 'default_value' in [0.5, 3.8].\n"
    "     - Dissipation/clearance rates: g_1 (tunable: true), g_2 (tunable: true), g_3..g_N (tunable: false) with 'unit': '1/s', min_value 0.05, max_value 10.0, and DISTINCT domain-specific 'default_value' in [0.25, 2.2].\n"
    "     - Onsager cross-coupling rate: g_cpl (tunable: true, 'unit': '1/s', choose 'default_value' in [0.08, 0.45], min_value: 0.0, max_value: 5.0).\n"
    "     - Conservative skew-symmetric routing rate: j_osc (tunable: true, 'unit': '1/s', 'default_value' in [0.05, 0.6], min_value: 0.0, max_value: 5.0).\n"
    "     - Basal/active drives: u_1 (tunable: true), u_2..u_N (tunable: false) with 'unit': '1/s', min_value 0.0, max_value 10.0, and DISTINCT 'default_value' in [0.3, 2.4].\n"
    "  3. AXIOM IV (Domain-Specific Nonlinear Mechanistic Kinetics & N-Node 3D Compartment Geometry):\n"
    "     - 'name': descriptive snake_case model name, 'domain': specific scientific domain string (e.g. 'neurobiology', 'immunology_oncology', 'fusion_plasma_physics', 'prebiotic_chemistry', 'quantum_condensed_matter'), 'time_unit': 's', 'energy_unit': '1'\n"
    "     - 'hamiltonian_expr': domain-specific energy functional combining quadratic terms '0.5 * k_i * x_i**2' with domain-appropriate quartic ('0.05 * k_1 * x_1**4') or biquadratic cross-potential ('0.08 * k_2 * x_1**2 * x_2**2') terms\n"
    "     - 'dissipation_diag_exprs': length-N list with domain-specific nonlinear state-modulated clearance/damping on State 0 and State 1\n"
    "     - 'dissipation_coupling_exprs': symmetric Onsager cross-dissipation dict coupling State 0 across the blast radius {'0,1': '...', '0,2': '...', '0,3': '...', '0,4': '...'}\n"
    "     - 'skew_routing_exprs': skew-symmetric conservative energy exchange dict e.g. {'0,1': 'j_osc', '0,2': '0.5 * j_osc'}\n"
    "     - 'external_drive_exprs': length-N list with domain-specific nonlinear reaction/drive kinetics on State 0\n"
    "     - 'spatial_nodes_3d': list of N 3D compartments (one per state) [{'node_id': str, 'fixed_position_m': [x, y, z], "
    "'volume_m3': float, 'radius_m': float, 'associated_state_names': [str]}].\n"
    "  - Allowed algebraic operators: '+', '-', '*', '/', '**'. All symbols in expressions MUST be in 'state_specs' or 'param_specs'.\n"
    "Return strictly valid JSON with keys: 'candidate_spec', 'target_state_objective' (dict mapping primary target "
    "state name to a domain-appropriate healthy numeric target value in [0.5, 1.2]), and 'inverse_mode' (default 'counterfactual_intervention'; "
    "use 'symbolic_mechanism_invention' only if asked to invent a feedback catalyst)."
)


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
    max_tokens: int = 16384
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
        timeout_seconds: float = 120.0,
    ) -> None:
        load_dotenv()
        env = dict(os.environ)
        if env_overrides:
            env.update(env_overrides)

        self._http_transport = http_transport
        self._timeout = timeout_seconds
        self.telemetry_log: list[LLMInferenceResponse] = []
        self._telemetry_lock = threading.Lock()
        self._bg_pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=10) if self._http_transport is None else None
        )
        self._pending_futures: list[Future[LLMInferenceResponse]] = []
        self._shared_http_client = httpx.Client(
            transport=self._http_transport,
            timeout=self._timeout,
            limits=httpx.Limits(max_keepalive_connections=16, max_connections=32),
        )

        baseten_default_model = env.get(
            "BASETEN_MODEL",
            env.get("BASETEN_MODEL_SPARK_OPEN", "meta-models/muse-spark-open-1026"),
        )

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
                default_model=baseten_default_model,
                role_models={
                    "muse_wonder": env.get("BASETEN_MODEL_WONDER", baseten_default_model),
                    "muse_spark": env.get("BASETEN_MODEL_SPARK", baseten_default_model),
                    "muse_spark_open": env.get("BASETEN_MODEL_SPARK_OPEN", baseten_default_model),
                    "muse_glimmer": env.get("BASETEN_MODEL_GLIMMER", baseten_default_model),
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
        order: list[ProviderName] = ["baseten", "fireworks", "meta", "local"]
        return [p for p in order if self.providers[p].is_configured]

    def _provider_priority_for_request(self, request: LLMInferenceRequest) -> list[ProviderName]:
        active = self.configured_providers()
        if not active:
            return []

        if request.preferred_provider and request.preferred_provider in active:
            ordered = [request.preferred_provider] + [p for p in active if p != request.preferred_provider]
            return ordered

        pref: list[ProviderName] = ["baseten", "fireworks", "meta", "local"]
        return [p for p in pref if p in active]

    def has_cached_response(self, request: LLMInferenceRequest) -> bool:
        """Always returns False (100% live AI inference, zero warm/disk caching)."""
        return False

    def invoke_background(self, request: LLMInferenceRequest) -> Future[LLMInferenceResponse]:
        """
        Dispatch a live AI inference request asynchronously on `_bg_pool` when in live
        network mode (`_http_transport is None`) so CPU math and HTTP inference overlap,
        or synchronously when `_http_transport is not None` (unit-test transport mode).
        """
        if self._bg_pool is None:
            fut: Future[LLMInferenceResponse] = Future()
            try:
                fut.set_result(self.invoke(request))
            except Exception as exc:
                fut.set_exception(exc)
            return fut

        from cortivra.observability.flight_recorder import submit_with_recorder

        fut = submit_with_recorder(self._bg_pool, self.invoke, request)
        with self._telemetry_lock:
            self._pending_futures.append(fut)
        return fut

    def flush_background_invocations(self) -> None:
        """Wait for all in-flight background live AI requests to complete."""
        with self._telemetry_lock:
            futs = list(self._pending_futures)
            self._pending_futures.clear()
        for f in futs:
            try:
                f.result(timeout=self._timeout)
            except Exception:
                pass

    def invoke(self, request: LLMInferenceRequest) -> LLMInferenceResponse:
        """
        Execute a live AI inference request across the prioritized live providers (Baseten,
        Fireworks AI, Meta Model API, Local), automatically failing over on transient network/5xx
        errors. Raises a RuntimeError if no live AI provider is configured or if all live
        providers fail (zero offline LLM fallback, zero warm caching).
        """
        candidates = self._provider_priority_for_request(request)
        if not candidates:
            raise RuntimeError(
                "No live AI provider is configured in the environment (.env). "
                "Please set BASETEN_API_KEY, FIREWORKS_API_KEY, META_API_KEY, or LOCAL_LLM_BASE_URL."
            )
        failover_count = 0
        last_error: Exception | None = None

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
            if prov_name == "baseten" and self._http_transport is None:
                body["reasoning_effort"] = "low"
            if request.require_json:
                body["response_format"] = {"type": "json_object"}

            max_attempts = 2 if self._http_transport is None else 1
            for attempt_idx in range(max_attempts):
                t0 = time.perf_counter()
                try:
                    resp = self._shared_http_client.post(url, headers=headers, json=body)
                    resp.raise_for_status()
                    data = resp.json()

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    msg_obj = data["choices"][0]["message"]
                    raw_c = msg_obj.get("content")
                    if not raw_c:
                        raw_c = msg_obj.get("reasoning_content") or msg_obj.get("reasoning") or ""
                    content = str(raw_c)
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
                    with self._telemetry_lock:
                        self.telemetry_log.append(record)
                    return record
                except Exception as exc:
                    last_error = exc
                    if attempt_idx + 1 < max_attempts:
                        body.pop("reasoning_effort", None)
                        continue
            failover_count += 1

        raise RuntimeError(
            f"All configured live AI providers ({candidates}) failed after {failover_count} failover(s). "
            f"Last error: {last_error}"
        )
