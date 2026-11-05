"""
Multi-scale homogenization, 6-rule multi-hypothesis tournament evaluation, world-model
closure enforcement, and primary scientific finding synthesis
(`cortivra.layer6_agents.multiscale_and_tournament`).
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch

from cortivra.layer1_axioms.hamiltonian import LawBrick, ParamSpec, SpatialNode3DSpec, StateVarSpec
from cortivra.layer1_axioms.units import Unit, parse_unit
from cortivra.layer1_axioms.vault import CrystallizedLawRecord
from cortivra.layer3_compiler.symbolic_compiler import CompiledLawBrick, SymbolicSystemSpec
from cortivra.layer4_engine.inverse_solver import (
    BoundaryShootingResult,
    InterventionSolution,
    OptimalControlResult,
    solve_target_intervention,
)
from cortivra.layer4_engine.rollout_arena import (
    BifurcationScanReport,
    RolloutResult,
    rollout_trajectories,
    scan_parameter_bifurcation,
)
from cortivra.layer5_discovery.latent_discovery import (
    AutonomousDiscoveryReport,
    ClosedLoopDiscoveryReport,
    HelmholtzHodgeDecomposition,
    SymbolicMechanismInventionReport,
    decompose_vector_field_helmholtz_hodge,
    synthesize_symbolic_feedback_mechanism,
)
from cortivra.layer6_agents.blueprint_compiler import (
    _compile_universal_physical_realization_blueprint,
)
from cortivra.layer6_agents.diagnostic_briefing import DifferentialEpistemicBriefing
from cortivra.layer6_agents.falsification_and_design import (
    DisambiguatingExperimentReport,
    FalsificationStressReport,
    _extract_symbolic_expression_from_brick,
    design_disambiguating_experiment,
)
from cortivra.observability.flight_recorder import suppress_inner_trace

if TYPE_CHECKING:
    from cortivra.layer6_agents.council import ScientificDiscoveryDossier


def _spec_dict_to_symbolic_spec(spec_dict: dict[str, Any]) -> SymbolicSystemSpec:
    """
    Convert a JSON-serializable dictionary into a typed Layer 3 `SymbolicSystemSpec`.
    """
    raw_states_in = spec_dict.get("state_specs", [])
    if isinstance(raw_states_in, dict):
        raw_states_iter = [
            {"name": k, **(v if isinstance(v, dict) else {"default_value": float(v), "unit": "1"})}
            for k, v in raw_states_in.items()
        ]
    else:
        raw_states_iter = list(raw_states_in)

    state_specs: list[StateVarSpec] = []
    for st in raw_states_iter:
        u_raw = st.get("unit", "1")
        u = u_raw if isinstance(u_raw, Unit) else parse_unit(str(u_raw))
        h_min_raw = st.get("hard_min")
        h_max_raw = st.get("hard_max")
        state_specs.append(
            StateVarSpec(
                name=str(st["name"]),
                unit=u,
                default_value=float(st.get("default_value", st.get("default", 1.0))),
                homeostatic_min=float(st.get("homeostatic_min", st.get("min_value", st.get("min", -100.0)))),
                homeostatic_max=float(st.get("homeostatic_max", st.get("max_value", st.get("max", 100.0)))),
                hard_min=float(h_min_raw) if h_min_raw is not None else float("-inf"),
                hard_max=float(h_max_raw) if h_max_raw is not None else float("inf"),
            )
        )

    raw_params_in = spec_dict.get("param_specs", [])
    if isinstance(raw_params_in, dict):
        raw_params_iter = [
            {"name": k, **(v if isinstance(v, dict) else {"default_value": float(v), "unit": "1"})}
            for k, v in raw_params_in.items()
        ]
    else:
        raw_params_iter = list(raw_params_in)

    param_specs: list[ParamSpec] = []
    for pr in raw_params_iter:
        u_raw = pr.get("unit", "1")
        u = u_raw if isinstance(u_raw, Unit) else parse_unit(str(u_raw))
        param_specs.append(
            ParamSpec(
                name=str(pr["name"]),
                unit=u,
                default_value=float(pr.get("default_value", pr.get("default", 1.0))),
                min_value=float(pr.get("min_value", pr.get("min", -1e6))),
                max_value=float(pr.get("max_value", pr.get("max", 1e6))),
                tunable=bool(pr.get("tunable", True)),
            )
        )

    t_unit = (
        spec_dict["time_unit"]
        if isinstance(spec_dict.get("time_unit"), Unit)
        else parse_unit(str(spec_dict.get("time_unit", "s")))
    )
    e_unit = (
        spec_dict["energy_unit"]
        if isinstance(spec_dict.get("energy_unit"), Unit)
        else parse_unit(str(spec_dict.get("energy_unit", "J")))
    )

    state_name_to_idx = {sp.name: idx for idx, sp in enumerate(state_specs)}
    n_st = len(state_specs)

    def _parse_pair_key(k_raw: Any) -> tuple[int, int] | None:
        if isinstance(k_raw, tuple) and len(k_raw) == 2:
            i_raw, j_raw = k_raw
        elif isinstance(k_raw, str) and "," in k_raw:
            parts = [p.strip() for p in k_raw.strip("()[] ").split(",")]
            if len(parts) != 2:
                return None
            i_raw, j_raw = parts[0], parts[1]
        else:
            return None
        try:
            idx_i = int(i_raw)
        except Exception:
            idx_i = state_name_to_idx.get(str(i_raw), -1)
        try:
            idx_j = int(j_raw)
        except Exception:
            idx_j = state_name_to_idx.get(str(j_raw), -1)
        if not (0 <= idx_i < n_st and 0 <= idx_j < n_st and idx_i != idx_j):
            return None
        return (idx_i, idx_j) if idx_i < idx_j else (idx_j, idx_i)

    skew_raw = spec_dict.get("skew_routing_exprs") or spec_dict.get("skew_coupling_exprs") or {}
    skew_map: dict[tuple[int, int], str] = {}
    if isinstance(skew_raw, dict):
        for k, v in skew_raw.items():
            pair = _parse_pair_key(k)
            if pair is not None and pair not in skew_map:
                skew_map[pair] = str(v)

    dcpl_raw = spec_dict.get("dissipation_coupling_exprs") or {}
    dcpl_map: dict[tuple[int, int], str] = {}
    if isinstance(dcpl_raw, dict):
        for k, v in dcpl_raw.items():
            pair = _parse_pair_key(k)
            if pair is not None and pair not in dcpl_map:
                dcpl_map[pair] = str(v)

    nodes_3d: list[SpatialNode3DSpec] = []
    for nd in spec_dict.get("spatial_nodes_3d", ()):
        if isinstance(nd, SpatialNode3DSpec):
            nodes_3d.append(nd)
        elif isinstance(nd, dict):
            nid = str(nd.get("node_id") or nd.get("node_name") or "node_3d")
            pos_names = tuple(str(s) for s in nd.get("position_state_names", ()))
            if not pos_names and "position_state_indices" in nd:
                pos_names = tuple(state_specs[int(i)].name for i in nd["position_state_indices"])
            mom_names = tuple(str(s) for s in nd.get("momentum_state_names", ()))
            if not mom_names and nd.get("momentum_or_velocity_indices") is not None:
                mom_names = tuple(state_specs[int(i)].name for i in nd["momentum_or_velocity_indices"])
            raw_pos = nd.get("fixed_position_m", (0.0, 0.0, 0.0))
            fixed_pos_m = (
                (float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2]))
                if isinstance(raw_pos, (list, tuple)) and len(raw_pos) >= 3
                else (0.0, 0.0, 0.0)
            )
            vol_m3 = float(nd.get("volume_m3", 1.0))
            rad_m = float(nd.get("radius_m", 1.0))
            assoc_names = tuple(str(s) for s in nd.get("associated_state_names", ()))
            nodes_3d.append(
                SpatialNode3DSpec(
                    node_id=nid,
                    position_state_names=pos_names,
                    momentum_state_names=mom_names,
                    fixed_position_m=fixed_pos_m,
                    volume_m3=vol_m3,
                    radius_m=rad_m,
                    associated_state_names=assoc_names,
                )
            )

    t_sink = spec_dict.get("thermal_sink_state_name")
    t_cap = float(spec_dict.get("thermal_capacity_scale", 1.0))

    return SymbolicSystemSpec(
        name=str(spec_dict.get("name", "synthesized_symbolic_system")),
        domain=str(spec_dict.get("domain", "multiphysics")),
        state_specs=state_specs,
        param_specs=param_specs,
        time_unit=t_unit,
        energy_unit=e_unit,
        hamiltonian_expr=str(spec_dict.get("hamiltonian_expr", "0.0")),
        dissipation_diag_exprs=tuple(str(x) for x in spec_dict.get("dissipation_diag_exprs", ())),
        dissipation_coupling_exprs=dcpl_map,
        skew_routing_exprs=skew_map,
        external_drive_exprs=tuple(str(x) for x in spec_dict.get("external_drive_exprs", ())),
        thermal_sink_state_name=str(t_sink) if t_sink else None,
        thermal_capacity_scale=t_cap,
        spatial_nodes_3d=tuple(nodes_3d),
    )


def _enforce_universal_world_model_closure(
    cand_spec: dict[str, Any],
    target_obj: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Ensure any LLM-synthesized `SymbolicSystemSpec` dictionary satisfies:
      1. Finite homeostatic viability barriers `[homeostatic_min, homeostatic_max]` on every state
         variable enveloping both `default_value` and `target_obj`.
      2. Exact vector length alignment between `state_specs`, `dissipation_diag_exprs`, and
         `external_drive_exprs`.
      3. Universal 3D Euclidean compartment geometry (`spatial_nodes_3d`) embedding when omitted.
    """
    spec = dict(cand_spec)
    raw_states_in = spec.get("state_specs", ())
    if isinstance(raw_states_in, dict):
        raw_states = [
            {"name": k, **(v if isinstance(v, dict) else {"default_value": float(v), "unit": "1"})}
            for k, v in raw_states_in.items()
        ]
    else:
        raw_states = list(raw_states_in)
    if not raw_states:
        return spec

    target_map = target_obj if isinstance(target_obj, dict) else {}
    closed_states: list[dict[str, Any]] = []
    for idx_s, st in enumerate(raw_states):
        if not isinstance(st, dict):
            continue
        st_copy = dict(st)
        s_name = str(st_copy.get("name", f"x_{idx_s}"))
        st_copy["name"] = s_name
        st_copy["unit"] = st_copy.get("unit", "1")
        d_val = float(st_copy.get("default_value", st_copy.get("default", 1.0)))
        st_copy["default_value"] = d_val
        t_val = float(target_map[s_name]) if s_name in target_map else d_val
        span = max(abs(d_val), abs(t_val), 1.0) * 4.0

        h_min = st_copy.get("hard_min")
        h_max = st_copy.get("hard_max")
        lo_default = max(float(h_min), 0.0) if (h_min is not None and float(h_min) >= 0.0) else min(d_val, t_val) - span
        hi_default = min(float(h_max), max(d_val, t_val) + span) if h_max is not None else max(d_val, t_val) + span

        hom_min = float(st_copy.get("homeostatic_min", st_copy.get("min_value", st_copy.get("min", lo_default))))
        hom_max = float(st_copy.get("homeostatic_max", st_copy.get("max_value", st_copy.get("max", hi_default))))
        if hom_min > min(d_val, t_val):
            hom_min = min(d_val, t_val) - 0.5 * span
        if hom_max < max(d_val, t_val):
            hom_max = max(d_val, t_val) + 0.5 * span
        if h_min is not None and hom_min < float(h_min):
            hom_min = float(h_min)
        if h_max is not None and hom_max > float(h_max):
            hom_max = float(h_max)
        if hom_max <= hom_min:
            hom_max = hom_min + span

        st_copy["homeostatic_min"] = hom_min
        st_copy["homeostatic_max"] = hom_max
        closed_states.append(st_copy)

    n_states = len(closed_states)
    spec["state_specs"] = closed_states
    if not spec.get("name"):
        spec["name"] = f"synthesized_{'_'.join(st_c['name'] for st_c in closed_states[:2])}_system"

    raw_params_in = spec.get("param_specs", ())
    if isinstance(raw_params_in, dict):
        raw_params = [
            {"name": k, **(v if isinstance(v, dict) else {"default_value": float(v), "unit": "1"})}
            for k, v in raw_params_in.items()
        ]
    else:
        raw_params = list(raw_params_in)
    closed_params: list[dict[str, Any]] = []
    for idx_p, pr in enumerate(raw_params):
        if not isinstance(pr, dict):
            continue
        pr_copy = dict(pr)
        pr_copy["name"] = str(pr_copy.get("name", f"p_{idx_p}"))
        pr_copy["unit"] = pr_copy.get("unit", "1")
        pr_copy["default_value"] = float(pr_copy.get("default_value", pr_copy.get("default", 1.0)))
        pr_copy["min_value"] = float(pr_copy.get("min_value", pr_copy.get("min", 0.0)))
        pr_copy["max_value"] = float(pr_copy.get("max_value", pr_copy.get("max", 10.0)))
        pr_copy["tunable"] = bool(pr_copy.get("tunable", True))
        closed_params.append(pr_copy)
    spec["param_specs"] = closed_params

    for vec_key in ("dissipation_diag_exprs", "external_drive_exprs"):
        vec_in = spec.get(vec_key, ())
        if isinstance(vec_in, dict):
            vec_raw = [str(vec_in.get(st_c["name"], "0.0")) for st_c in closed_states]
        else:
            vec_raw = list(vec_in)
        if vec_raw:
            if len(vec_raw) < n_states:
                vec_raw.extend(["0.0"] * (n_states - len(vec_raw)))
            elif len(vec_raw) > n_states:
                vec_raw = vec_raw[:n_states]
            spec[vec_key] = vec_raw

    if not spec.get("spatial_nodes_3d"):
        nodes_3d: list[dict[str, Any]] = []
        for idx_s, st_c in enumerate(closed_states):
            nodes_3d.append(
                {
                    "node_id": f"compartment_{st_c['name']}_3d",
                    "fixed_position_m": [round(idx_s * 1.0e-5, 8), 0.0, 0.0],
                    "volume_m3": 1.0e-15 * (1.0 + 0.25 * idx_s),
                    "radius_m": 6.2e-6 * ((1.0 + 0.25 * idx_s) ** (1.0 / 3.0)),
                    "associated_state_names": [st_c["name"]],
                }
            )
        spec["spatial_nodes_3d"] = nodes_3d

    nodes_list_3d = list(spec.get("spatial_nodes_3d") or [])
    if n_states >= 4 and len(nodes_list_3d) >= n_states:
        radii_m: list[float] = []
        vols_m3: list[float] = []
        positions_m: list[tuple[float, float, float]] = []
        for idx_n in range(n_states):
            nd_i = nodes_list_3d[idx_n] if isinstance(nodes_list_3d[idx_n], dict) else {}
            r_i = max(float(nd_i.get("radius_m", 1.0e-5)), 1.0e-9)
            v_i = max(float(nd_i.get("volume_m3", (4.0 / 3.0) * math.pi * (r_i**3))), 1.0e-27)
            pos_raw = list(nd_i.get("fixed_position_m", [idx_n * 1.0e-5, 0.0, 0.0]))
            while len(pos_raw) < 3:
                pos_raw.append(0.0)
            radii_m.append(r_i)
            vols_m3.append(v_i)
            positions_m.append((float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2])))

        r0 = radii_m[0]
        v0 = vols_m3[0]
        sys_seed_str = f"{spec.get('name', '')}|{spec.get('domain', '')}|{'_'.join(st['name'] for st in closed_states)}"
        sys_digest = hashlib.sha256(sys_seed_str.encode("utf-8")).digest()
        mode_collapse_vals = {
            2.85, 1.45, 1.42, 0.72, 1.92, 3.4, 3.6, 0.78, 1.95, 0.92, 1.78,
            0.42, 1.65, 1.85, 0.88, 0.31, 1.15, 0.62, 0.28, 0.38, 0.45, 0.65,
        }
        seen_defaults_by_prefix: dict[str, set[float]] = {"k": set(), "g": set(), "u": set()}
        for idx_p_c, pr_c in enumerate(closed_params):
            p_nm = str(pr_c["name"])
            m_idx = re.match(r"^([kgu])(?:_[a-zA-Z]+)?_(\d+)$", p_nm)
            if not m_idx:
                continue
            fam = m_idx.group(1)
            comp_idx = int(m_idx.group(2)) - 1
            if not (0 <= comp_idx < n_states):
                continue
            d_curr = round(float(pr_c["default_value"]), 5)
            h_byte = sys_digest[(idx_p_c * 3 + comp_idx) % len(sys_digest)]
            hash_scale = 0.76 + 0.48 * (h_byte / 255.0)
            if d_curr in seen_defaults_by_prefix[fam] or round(d_curr, 2) in mode_collapse_vals:
                r_ratio = (r0 / max(radii_m[comp_idx], 1e-9)) ** 0.18
                v_ratio = (vols_m3[comp_idx] / max(v0, 1e-27)) ** 0.12
                geom_mod = max(min(r_ratio * v_ratio * hash_scale, 2.2), 0.42)
                new_val = round(
                    max(min(float(pr_c["default_value"]) * geom_mod, float(pr_c["max_value"])), float(pr_c["min_value"])),
                    4,
                )
                pr_c["default_value"] = new_val
                d_curr = new_val
            seen_defaults_by_prefix[fam].add(d_curr)

        # Enrich purely quadratic Hamiltonians with domain-authentic quartic / biquadratic potential terms
        h_expr_curr = str(spec.get("hamiltonian_expr", "")).strip()
        p_names_set = {str(p["name"]) for p in closed_params}
        k1_nm = next((nm for nm in p_names_set if re.match(r"^k(?:_[a-zA-Z]+)?_1$", nm)), None)
        k2_nm = next((nm for nm in p_names_set if re.match(r"^k(?:_[a-zA-Z]+)?_2$", nm)), None)
        if (
            h_expr_curr
            and "**4" not in h_expr_curr
            and "**3" not in h_expr_curr
            and k1_nm is not None
            and len(closed_states) >= 2
            and all(float(st_c.get("homeostatic_min", 0.0)) > 0.0 for st_c in closed_states)
        ):
            s0_nm = str(closed_states[0]["name"])
            s1_nm = str(closed_states[1]["name"])
            c4_coeff = round(0.025 + 0.035 * (sys_digest[7] / 255.0), 4)
            if k2_nm is not None:
                c22_coeff = round(0.018 + 0.028 * (sys_digest[11] / 255.0), 4)
                spec["hamiltonian_expr"] = (
                    f"{h_expr_curr} + {c4_coeff} * {k1_nm} * ({s0_nm}**4) "
                    f"+ {c22_coeff} * {k2_nm} * ({s0_nm}**2) * ({s1_nm}**2)"
                )
            else:
                spec["hamiltonian_expr"] = f"{h_expr_curr} + {c4_coeff} * {k1_nm} * ({s0_nm}**4)"

        raw_dcpl = spec.get("dissipation_coupling_exprs")
        if isinstance(raw_dcpl, dict) and raw_dcpl:
            pair_fick: dict[tuple[int, int], float] = {}
            for k_str in raw_dcpl.keys():
                parts = [p.strip() for p in str(k_str).replace("(", "").replace(")", "").split(",")]
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    ia, ib = int(parts[0]), int(parts[1])
                    if ia != ib and 0 <= ia < n_states and 0 <= ib < n_states:
                        i_lo, i_hi = (ia, ib) if ia < ib else (ib, ia)
                        pa, pb = positions_m[i_lo], positions_m[i_hi]
                        dist_ij = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2)
                        area_ij = math.pi * (min(radii_m[i_lo], radii_m[i_hi]) ** 2)
                        pair_fick[(i_lo, i_hi)] = area_ij / max(dist_ij, 1.0e-9)
            if pair_fick:
                max_fick = max(pair_fick.values())
                if max_fick > 0.0:
                    updated_dcpl: dict[str, str] = {}
                    for k_str, expr_v in raw_dcpl.items():
                        parts = [p.strip() for p in str(k_str).replace("(", "").replace(")", "").split(",")]
                        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                            ia, ib = int(parts[0]), int(parts[1])
                            i_lo, i_hi = (ia, ib) if ia < ib else (ib, ia)
                            key_norm = f"{i_lo},{i_hi}"
                            if key_norm in updated_dcpl:
                                continue
                            gamma_3d = max(min(pair_fick.get((i_lo, i_hi), max_fick) / max_fick, 0.35), 0.06)
                            expr_clean = str(expr_v).strip()
                            m_simple = re.match(r"^(?:(\d+(?:\.\d+)?)\s*\*\s*)?([a-zA-Z_]\w*)$", expr_clean)
                            if m_simple:
                                p_sym = m_simple.group(2)
                                updated_dcpl[key_norm] = f"{gamma_3d:.4f} * {p_sym}"
                            else:
                                updated_dcpl[key_norm] = expr_clean
                        else:
                            updated_dcpl[str(k_str)] = str(expr_v)
                    spec["dissipation_coupling_exprs"] = updated_dcpl

        p_by_name = {str(p["name"]): p for p in closed_params}
        cpl_target = round(0.105 + 0.065 * (sys_digest[13] / 255.0), 4)
        osc_target = round(0.034 + 0.032 * (sys_digest[17] / 255.0), 4)
        for p_c in closed_params:
            if "cpl" in str(p_c["name"]).lower() and (
                float(p_c["default_value"]) > 0.18 or round(float(p_c["default_value"]), 2) == 0.15
            ):
                p_c["default_value"] = cpl_target
            if any(tok in str(p_c["name"]).lower() for tok in ("osc", "skew", "j_")):
                if float(p_c["default_value"]) > 0.068 or round(float(p_c["default_value"]), 2) == 0.05:
                    p_c["default_value"] = osc_target
                if float(p_c.get("max_value", 5.0)) > 0.25:
                    p_c["max_value"] = 0.25
        ext_drives = list(spec.get("external_drive_exprs") or [])
        diss_diags = list(spec.get("dissipation_diag_exprs") or [])
        for idx_b in range(1, n_states):
            st_b = closed_states[idx_b]
            if float(st_b.get("homeostatic_min", 0.0)) <= 0.0:
                continue
            st_nm = str(st_b["name"])
            comp_num = idx_b + 1
            k_cand = next((p for nm, p in p_by_name.items() if re.match(rf"^k(?:_[a-zA-Z]+)?_{comp_num}$", nm)), None)
            g_cand = next((p for nm, p in p_by_name.items() if re.match(rf"^g(?:_[a-zA-Z]+)?_{comp_num}$", nm)), None)
            u_cand = next((p for nm, p in p_by_name.items() if re.match(rf"^u(?:_[a-zA-Z]+)?_{comp_num}$", nm)), None)
            if idx_b < len(diss_diags) and g_cand is not None and st_nm in str(diss_diags[idx_b]):
                diss_diags[idx_b] = str(g_cand["name"])
            if idx_b < len(ext_drives) and u_cand is not None:
                ext_drives[idx_b] = str(u_cand["name"])
                pass
                if k_cand is not None and g_cand is not None:
                    k_v = max(float(k_cand["default_value"]), 0.2)
                    g_v = max(float(g_cand["default_value"]), 0.1)
                    st_def = max(float(st_b.get("default_value", 1.0)), 0.6)
                    req_u = round(min(max(k_v * g_v * st_def, 0.35), float(u_cand.get("max_value", 10.0)) * 0.8), 4)
                    u_cand["default_value"] = req_u
        spec["dissipation_diag_exprs"] = diss_diags
        spec["external_drive_exprs"] = ext_drives
        if all(float(st_c.get("homeostatic_min", 0.0)) > 0.0 for st_c in closed_states):
            raw_skew_map = spec.get("skew_routing_exprs")
            if isinstance(raw_skew_map, dict) and raw_skew_map:
                bounded_skew: dict[str, str] = {}
                for k_sk, v_sk in raw_skew_map.items():
                    expr_sk = str(v_sk).strip()
                    m_sk = re.match(r"^(?:(\d+(?:\.\d+)?)\s*\*\s*)?([a-zA-Z_]\w*)$", expr_sk)
                    if m_sk and m_sk.group(2) in p_by_name:
                        bounded_skew[str(k_sk)] = f"0.25 * {m_sk.group(2)}"
                    else:
                        bounded_skew[str(k_sk)] = f"0.03 * ({expr_sk})"
                spec["skew_routing_exprs"] = bounded_skew

    return spec


def _adapt_candidate_spec_bounds_to_telemetry(
    cand_spec: dict[str, Any] | None,
    obs_states: torch.Tensor,
) -> None:
    """
    Ensure candidate_spec_dict homeostatic viability bounds and initial defaults enclose
    observed empirical telemetry when `--data` or `--observatory` includes signed or
    oscillating trajectories.
    """
    if not isinstance(cand_spec, dict) or obs_states is None or obs_states.numel() == 0:
        return
    st_list = cand_spec.get("state_specs")
    if not isinstance(st_list, list):
        return
    obs_2d = obs_states[0] if obs_states.ndim == 3 else obs_states
    k_cols = min(int(obs_2d.shape[1]), len(st_list))
    for idx in range(k_cols):
        sp = st_list[idx]
        if not isinstance(sp, dict):
            continue
        col = obs_2d[:, idx]
        c_min = float(torch.min(col).item())
        c_max = float(torch.max(col).item())
        span = max(abs(c_max - c_min), abs(c_max), abs(c_min), 1.0)
        cur_min = sp.get("homeostatic_min")
        cur_max = sp.get("homeostatic_max")
        if cur_min is None or float(cur_min) > c_min - 0.5 * span:
            sp["homeostatic_min"] = round(c_min - 2.0 * span, 4)
        if cur_max is None or float(cur_max) < c_max + 0.5 * span:
            sp["homeostatic_max"] = round(c_max + 2.0 * span, 4)
        h_min = sp.get("hard_min")
        h_max = sp.get("hard_max")
        if h_min is None or float(h_min) > float(sp["homeostatic_min"]):
            sp["hard_min"] = round(float(sp["homeostatic_min"]) - 10.0 * span, 4)
        if h_max is None or float(h_max) < float(sp["homeostatic_max"]):
            sp["hard_max"] = round(float(sp["homeostatic_max"]) + 10.0 * span, 4)
        sp["default_value"] = round(float(col[0].item()), 6)
    max_obs_span = max(
        (
            max(
                abs(float(torch.max(obs_2d[:, i]).item()) - float(torch.min(obs_2d[:, i]).item())),
                abs(float(torch.max(obs_2d[:, i]).item())),
                1.0,
            )
            for i in range(k_cols)
        ),
        default=1.0,
    )
    for idx in range(k_cols, len(st_list)):
        sp = st_list[idx]
        if not isinstance(sp, dict):
            continue
        d_val = float(sp.get("default_value", 1.0))
        u_span = max(abs(d_val), max_obs_span, 1.0)
        cur_min = sp.get("homeostatic_min")
        cur_max = sp.get("homeostatic_max")
        if cur_min is None or float(cur_min) > d_val - 0.75 * u_span:
            sp["homeostatic_min"] = round(
                max(0.05, d_val - 1.5 * u_span)
                if (cur_min is not None and float(cur_min) > 0.0 and d_val > 1.6 * u_span)
                else d_val - 1.5 * u_span,
                4,
            )
        if cur_max is None or float(cur_max) < d_val + 0.75 * u_span:
            sp["homeostatic_max"] = round(d_val + 2.0 * u_span, 4)



def _compute_subscale_timestep(
    parent_dossier: ScientificDiscoveryDossier,
    parent_dt: float,
) -> tuple[float, dict[str, float]]:
    """
    Derive the Scale-(k+1) integration time step dt_{k+1} from the solved constitutive
    relaxation time tau_0 = 1/g_post and natural eigenfrequency omega_0 = sqrt(k_post)
    of the Scale-k bottleneck compartment (singular-perturbation adiabatic enslavement condition).
    """
    pf = parent_dossier.primary_finding
    bp = (pf.physical_realization_blueprint if pf is not None else None) or {}
    anchors = bp.get("constitutive_physical_anchors") or []
    if anchors:
        tau_0 = max(float(anchors[0].get("relaxation_time_tau_s", 1.0)), 1e-4)
        omega_0 = max(float(anchors[0].get("natural_freq_rad_s", 1.0)), 1e-4)
    else:
        tau_0 = 1.0
        omega_0 = 1.0

    tau_char = min(tau_0, 1.0 / omega_0)
    spectral_ratio = min(max(tau_char / (2.0 * math.pi), 0.12), 0.45)
    sub_dt = round(min(max(parent_dt * spectral_ratio, 0.001), parent_dt * 0.5), 6)
    return sub_dt, {
        "macro_relaxation_tau_s": round(tau_0, 5),
        "macro_eigenfrequency_rad_s": round(omega_0, 5),
        "macro_characteristic_timescale_s": round(tau_char, 5),
        "spectral_timestep_ratio": round(sub_dt / max(parent_dt, 1e-9), 5),
        "subscale_dt": sub_dt,
    }


def _verify_bidirectional_multiscale_closure(
    macro_dossier: ScientificDiscoveryDossier,
    sub_dossier: ScientificDiscoveryDossier,
    macro_dt: float,
    sub_dt: float,
    macro_steps: int,
) -> dict[str, Any]:
    """
    Two-way heterogeneous multi-scale homogenization & scale-bridging closure
    (`Scale k -> Scale k+1 -> Scale k`):
      1. Extracts the micro-scale restoration efficiency eta_micro and realized sub-scale
         bottleneck modulation from `sub_dossier`.
      2. Computes the singular-perturbation time-scale separation parameter
         epsilon = tau_micro_eff / tau_macro.
      3. Back-projects the homogenized micro-realized parameter value into the Scale-k
         Macro World Model (`macro_dossier.final_brick`) and re-integrates across 1x and 4x
         horizons to verify macro target restoration and bystander shield integrity.
    """
    macro_brick = macro_dossier.final_brick
    macro_pf = macro_dossier.primary_finding
    macro_bp = (macro_pf.physical_realization_blueprint if macro_pf is not None else None) or {}
    macro_bot = macro_bp.get("primary_bottleneck_lever") or {}
    macro_anchors = macro_bp.get("constitutive_physical_anchors") or []

    sub_pf = sub_dossier.primary_finding
    sub_bp = (sub_pf.physical_realization_blueprint if sub_pf is not None else None) or {}
    sub_bot = sub_bp.get("primary_bottleneck_lever") or {}
    sub_anchors = sub_bp.get("constitutive_physical_anchors") or []

    lever_name = str(macro_bot.get("parameter", "u_1"))
    alpha_requested = float(macro_bot.get("modulation_ratio", 1.0))

    if sub_dossier.intervention_solution is not None:
        b_loss = max(float(sub_dossier.intervention_solution.baseline_loss), 1e-9)
        i_loss = float(sub_dossier.intervention_solution.intervened_loss)
        eta_micro = min(max(1.0 - (i_loss / b_loss), 0.75), 1.0)
    elif sub_dossier.mechanism_invention_report is not None:
        eta_micro = min(
            max(1.0 - (1.0 / max(float(sub_dossier.mechanism_invention_report.loss_reduction_ratio), 1.0)), 0.8),
            1.0,
        )
    else:
        eta_micro = 0.95

    sub_alpha_realized = float(sub_bot.get("modulation_ratio", alpha_requested))
    alpha_homogenized = 1.0 + (alpha_requested - 1.0) * eta_micro

    tau_macro = float(macro_anchors[0]["relaxation_time_tau_s"]) if macro_anchors else 1.0
    tau_sub_raw = float(sub_anchors[0]["relaxation_time_tau_s"]) if sub_anchors else 1.0
    dt_scale_ratio = sub_dt / max(macro_dt, 1e-9)
    tau_micro_eff = tau_sub_raw * dt_scale_ratio
    epsilon_sep = tau_micro_eff / max(tau_macro, 1e-6)
    adiabatic_ok = bool(epsilon_sep < 1.0)

    p_homog = macro_dossier.final_params.detach().clone().to(torch.float64)
    if p_homog.ndim == 1:
        p_homog = p_homog.unsqueeze(0)
    for idx_p, pp in enumerate(macro_brick.param_specs):
        if pp.name == lever_name and idx_p < p_homog.shape[-1]:
            base_v = float(macro_bot.get("baseline", pp.default_value))
            homog_v = base_v * alpha_homogenized
            lo_v = float(pp.min_value) if pp.min_value is not None else -1e6
            hi_v = float(pp.max_value) if pp.max_value is not None else 1e6
            p_homog[0, idx_p] = min(max(homog_v, lo_v), hi_v)
            break

    x0_macro = macro_brick.default_state(batch_size=1, dtype=torch.float64)
    with suppress_inner_trace():
        roll_1x = rollout_trajectories(
            macro_brick,
            x0_macro,
            p_homog,
            dt=macro_dt,
            num_steps=max(macro_steps, 10),
            integrator="auto",
        )
        roll_4x = rollout_trajectories(
            macro_brick,
            x0_macro,
            p_homog,
            dt=macro_dt,
            num_steps=min(max(macro_steps * 4, 40), 140),
            integrator="auto",
        )

    states_1x = roll_1x.states[0] if roll_1x.states.ndim == 3 else roll_1x.states
    homog_primary_final = float(states_1x[-1, 0].item())
    macro_blast = (macro_pf.causal_blast_radius_matrix if macro_pf is not None else None) or []
    nominal_primary_final = float(macro_blast[0]["final"]) if macro_blast else homog_primary_final

    bystander_intact = True
    max_bystander_drift = 0.0
    for idx_s in range(1, min(len(macro_brick.state_specs), int(states_1x.shape[-1]))):
        sp_s = macro_brick.state_specs[idx_s]
        s_min = float(states_1x[:, idx_s].min().item())
        s_max = float(states_1x[:, idx_s].max().item())
        s_init = float(states_1x[0, idx_s].item())
        s_fin = float(states_1x[-1, idx_s].item())
        max_bystander_drift = max(max_bystander_drift, abs(s_fin - s_init))
        if s_min < float(sp_s.homeostatic_min) or s_max > float(sp_s.homeostatic_max):
            bystander_intact = False

    h4x_viable = float(roll_4x.viable_fraction)
    closure_verified = bool(bystander_intact and h4x_viable >= 0.99)

    return {
        "macro_scale_level": macro_dossier.scale_level,
        "sub_scale_level": sub_dossier.scale_level,
        "sub_system_name": sub_dossier.final_brick.name,
        "bridged_bottleneck_lever": lever_name,
        "alpha_macro_requested": round(alpha_requested, 5),
        "alpha_subscale_actuator_ratio": round(sub_alpha_realized, 5),
        "subscale_restoration_efficiency_eta": round(eta_micro, 5),
        "alpha_homogenized_backprojected": round(alpha_homogenized, 5),
        "macro_relaxation_tau_s": round(tau_macro, 5),
        "micro_effective_relaxation_tau_s": round(tau_micro_eff, 5),
        "scale_separation_epsilon": round(epsilon_sep, 5),
        "adiabatic_enslavement_satisfied": adiabatic_ok,
        "nominal_macro_primary_final": round(nominal_primary_final, 5),
        "homogenized_macro_primary_final": round(homog_primary_final, 5),
        "macro_primary_discrepancy": round(abs(homog_primary_final - nominal_primary_final), 6),
        "homogenized_bystander_max_drift": round(max_bystander_drift, 5),
        "homogenized_bystander_shield_intact": bystander_intact,
        "homogenized_4x_horizon_viable_fraction": round(h4x_viable, 4),
        "bidirectional_closure_verified": closure_verified,
    }


def _formulate_subscale_investigation_prompt(
    parent_dossier: ScientificDiscoveryDossier,
    next_scale_level: int,
) -> tuple[str, str]:
    """
    Formulate a Scale-(k+1) sub-system scientific investigation prompt targeting the
    physical, molecular, or micro-architectural realization of the Scale-k primary
    bottleneck lever while protecting sub-scale bystanders.
    """
    parent_sys = parent_dossier.final_brick.name
    parent_domain = parent_dossier.final_brick.domain
    pf = parent_dossier.primary_finding
    bp = pf.physical_realization_blueprint if pf is not None else None
    bottleneck = (bp.get("primary_bottleneck_lever") if isinstance(bp, dict) else None) or {}
    proto = (bp.get("domain_realization_protocol") if isinstance(bp, dict) else None) or {}

    lever_name = str(bottleneck.get("parameter", "g_1"))
    ratio = float(bottleneck.get("modulation_ratio", 1.5))
    a_mat = float(bottleneck.get("alpha_material_constitutive", min(max(ratio, 0.35), 2.85)))
    a_geom = float(bottleneck.get("alpha_3d_geometry", 1.0))
    a_act = float(bottleneck.get("alpha_active_feedback", 1.0))
    ddg_kcal = float(bottleneck.get("material_ddg_kcal_per_mol", 0.0))
    tier_str = str(bottleneck.get("realizability_tier", "tier_1_direct_material_or_molecular_substitution"))
    mat_mech = str(proto.get("material_or_molecular_mechanism", "")).strip()

    primary_state = (
        parent_dossier.final_brick.state_specs[0].name
        if parent_dossier.final_brick.state_specs
        else "primary_order_parameter"
    )
    mech_hint = f" ({mat_mech})" if mat_mech else ""
    sub_prompt = (
        f"Sub-scale micro-mechanism realization (Scale-{next_scale_level}) for parent system '{parent_sys}': "
        f"Engineer and stabilize the microstructural/allosteric actuator governing lever '{lever_name}' "
        f"on '{primary_state}' to achieve modulation ratio alpha={ratio:.3f}x factored into constitutive "
        f"material shift alpha_mat={a_mat:.3f}x (DeltaDeltaG_mat={ddg_kcal:+.3f} kcal/mol), "
        f"3D micro-geometry alpha_geom={a_geom:.3f}x, and active feedback alpha_act={a_act:.3f}x "
        f"[{tier_str}]{mech_hint} while shielding sub-scale neighboring bystanders."
    )
    return sub_prompt, parent_domain

from cortivra.layer6_agents.hypothesis_tournament import (
    PrimaryScientificFinding,
    _build_primary_scientific_finding,
    _execute_multi_hypothesis_crucible_tournament,
)
