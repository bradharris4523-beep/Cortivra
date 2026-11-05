"""
Command-line entry point (`python -m cortivra "<one-sentence scientific hypothesis>"`).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

from cortivra.layer6_agents.council import investigate
from cortivra.layer6_agents.meta_clients import MultiProviderMetaClient
from cortivra.observability.forensic_report import serialize_trace_to_jsonl


def _build_cli_client(
    provider: str,
    model_override: str | None = None,
    base_url_override: str | None = None,
) -> MultiProviderMetaClient:
    """
    Construct a `MultiProviderMetaClient` configured according to CLI flags.
    """
    overrides: dict[str, str] = {}
    if provider == "local":
        overrides["FIREWORKS_API_KEY"] = ""
        overrides["BASETEN_API_KEY"] = ""
        overrides["META_API_KEY"] = ""
        if base_url_override:
            overrides["LOCAL_LLM_BASE_URL"] = base_url_override
        if model_override:
            overrides["LOCAL_LLM_MODEL"] = model_override
    elif provider == "baseten":
        overrides["FIREWORKS_API_KEY"] = ""
        overrides["META_API_KEY"] = ""
        overrides["LOCAL_LLM_BASE_URL"] = ""
        if base_url_override:
            overrides["BASETEN_BASE_URL"] = base_url_override
        if model_override:
            overrides["BASETEN_MODEL"] = model_override
    elif provider == "fireworks":
        overrides["BASETEN_API_KEY"] = ""
        overrides["META_API_KEY"] = ""
        overrides["LOCAL_LLM_BASE_URL"] = ""
        if base_url_override:
            overrides["FIREWORKS_BASE_URL"] = base_url_override
        if model_override:
            overrides["FIREWORKS_MODEL"] = model_override
    elif provider == "meta":
        overrides["FIREWORKS_API_KEY"] = ""
        overrides["BASETEN_API_KEY"] = ""
        overrides["LOCAL_LLM_BASE_URL"] = ""
        if base_url_override:
            overrides["META_API_BASE_URL"] = base_url_override
        if model_override:
            overrides["META_MODEL_SPARK"] = model_override
            overrides["META_MODEL_WONDER"] = model_override
            overrides["META_MODEL_GLIMMER"] = model_override

    return MultiProviderMetaClient(env_overrides=overrides if overrides else None)


def _save_run_artifacts(
    dossier,
    log_dir: Path,
) -> Path:
    """
    Persist `dossier.md`, `dossier.json`, and `flight_recorder.jsonl` into a timestamped
    run directory under `log_dir`.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_sys = re.sub(r"[^A-Za-z0-9_.-]+", "_", dossier.final_brick.name).strip("_")[:48]
    run_folder = log_dir / f"{ts}_{safe_sys}"
    run_folder.mkdir(parents=True, exist_ok=True)

    md_content = dossier.to_markdown_report() + "\n\n" + dossier.to_forensic_xray_report() + "\n"
    (run_folder / "dossier.md").write_text(md_content, encoding="utf-8")
    (run_folder / "dossier.html").write_text(dossier.to_html_report(), encoding="utf-8")
    (run_folder / "dossier.json").write_text(
        json.dumps(dossier.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    if dossier.causal_trace_dag is not None:
        jsonl_content = serialize_trace_to_jsonl(
            dossier.causal_trace_dag,
            dossier.contradiction_report,
        )
        (run_folder / "flight_recorder.jsonl").write_text(jsonl_content + "\n", encoding="utf-8")

    dossier.export_deliverables(run_folder)
    return run_folder


def _safe_stdout_write(text: str) -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.write(text.encode("ascii", errors="replace").decode("ascii"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cortivra",
        description=(
            "Cortivra — Universal One-Sentence Hypothesis-to-Physics Discovery & Inverse Control Engine."
        ),
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="One-sentence or short-paragraph scientific hypothesis, anomaly, or intervention question.",
    )
    parser.add_argument(
        "--verify-oracle-suite",
        action="store_true",
        help="Execute the 5-domain blind ground-truth oracle rescue and +/-25%% mismatched-reality verification suite.",
    )
    parser.add_argument(
        "--run-benchmark-suite",
        action="store_true",
        help="Execute the 7-domain blind scientific law reconstruction benchmark suite.",
    )
    parser.add_argument(
        "--domain",
        default="multiphysics",
        help="Optional scientific domain label (default: multiphysics).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=35,
        help="Number of dynamical rollout steps (default: 35).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Integration time step dt (default: 0.02).",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "local", "baseten", "fireworks", "meta"),
        default="auto",
        help="Inference provider routing mode (default: auto).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model identifier override for the selected provider.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL override (e.g. http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional path to empirical CSV/JSON telemetry file or inline JSON array for partial/full sensor calibration.",
    )
    parser.add_argument(
        "--observatory",
        default=None,
        help="Optional open-science observatory stream ID (e.g. 'noaa_solar_wind', 'ruffini_sa2', 'auto').",
    )
    parser.add_argument(
        "--xray",
        action="store_true",
        help="Print the causal trace diagnostic report (also printed automatically if any gate fails).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Automatically execute a chained Scale-1 sub-system investigation targeting the primary physical bottleneck lever.",
    )
    parser.add_argument(
        "--recursive-depth",
        type=int,
        default=0,
        help="Number of recursive multi-scale sub-system levels K to chain and homogenize (e.g. 1 or 2).",
    )
    parser.add_argument(
        "--html",
        default=None,
        help="Optional output file path to write the self-contained visual HTML5 interactive dossier.",
    )
    parser.add_argument(
        "--save-logs",
        action="store_true",
        help="Save dossier.md, dossier.html, dossier.json, and flight_recorder.jsonl under --log-dir.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/runs",
        help="Directory for saved run output files (default: logs/runs).",
    )
    parser.add_argument(
        "--serve",
        "--ui",
        dest="serve",
        action="store_true",
        help="Launch the interactive Cortivra Scientific Workbench Web UI server.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for --serve (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Bind port for --serve (default: 8080).",
    )
    args = parser.parse_args(argv)
    client = _build_cli_client(
        provider=args.provider,
        model_override=args.model,
        base_url_override=args.base_url,
    )

    if args.serve:
        from cortivra.ui_server import serve_workbench

        serve_workbench(
            host=args.host,
            port=args.port,
            provider=args.provider,
            runs_dir=str(Path(args.log_dir) / "ui_workbench"),
        )
        return 0

    if args.verify_oracle_suite:
        from cortivra.layer6_agents.blind_oracle_rescue_suite import run_blind_oracle_rescue_suite

        suite_rep = run_blind_oracle_rescue_suite(client=client)
        _safe_stdout_write(json.dumps(suite_rep.to_dict(), indent=2) + "\n")
        return 0 if suite_rep.passed_all else 1

    if args.run_benchmark_suite:
        from cortivra.layer6_agents.blind_reconstruction_benchmarks import (
            run_blind_scientific_reconstruction_suite,
        )

        bench_rep = run_blind_scientific_reconstruction_suite()
        _safe_stdout_write(json.dumps(bench_rep.to_dict(), indent=2) + "\n")
        return 0 if bench_rep.passed_all else 1

    question = " ".join(args.prompt).strip()
    if not question:
        parser.error("A scientific hypothesis prompt is required unless --verify-oracle-suite or --run-benchmark-suite is set.")

    rec_depth = max(int(args.recursive_depth), 1 if args.recursive else 0)
    dossier = investigate(
        prompt=question,
        domain=args.domain,
        client=client,
        dt=args.dt,
        num_steps=args.steps,
        observed_states=args.data,
        observatory_stream=args.observatory,
        recursive_depth=rec_depth,
    )

    if args.html:
        html_path = Path(args.html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(dossier.to_html_report(), encoding="utf-8")

    has_failure = bool(
        (dossier.falsification_report is not None and not dossier.falsification_report.passed_all_gates)
        or (dossier.contradiction_report is not None and dossier.contradiction_report.contradiction_detected)
    )
    saved_dir: Path | None = None
    if args.save_logs or has_failure:
        saved_dir = _save_run_artifacts(dossier, Path(args.log_dir))

    _safe_stdout_write(dossier.to_markdown_report() + "\n")

    if args.xray or has_failure:
        _safe_stdout_write("\n" + dossier.to_forensic_xray_report() + "\n")

    if saved_dir is not None:
        _safe_stdout_write("\n" + dossier.to_flight_recorder_log() + "\n")
        _safe_stdout_write(
            f"\n[Flight Recorder & 20-File Deliverable Bundle Saved] -> {saved_dir.resolve()}\n"
            f"  - Markdown Scientific Dossier:{(saved_dir / 'dossier.md').resolve()}\n"
            f"  - Interactive HTML5 Dossier:  {(saved_dir / 'dossier.html').resolve()}\n"
            f"  - Full JSON Telemetry:        {(saved_dir / 'dossier.json').resolve()}\n"
            f"  - 8-Stage Flight Recorder:    {(saved_dir / 'execution_flight_recorder.log').resolve()}\n"
            f"  - 8-Stage Structured Ledger:  {(saved_dir / 'execution_ledger.jsonl').resolve()}\n"
            f"  - Merkle Trace DAG JSONL:     {(saved_dir / 'flight_recorder.jsonl').resolve()}\n"
            f"  - 3D Printable STL Solid:     {(saved_dir / 'deliverables' / 'cad_assembly.stl').resolve()}\n"
            f"  - 3D Wavefront OBJ Mesh:      {(saved_dir / 'deliverables' / 'cad_assembly.obj').resolve()}\n"
            f"  - Vector Engineering SVG:     {(saved_dir / 'deliverables' / 'engineering_blueprint.svg').resolve()}\n"
            f"  - Compilable C99 Firmware:    {(saved_dir / 'deliverables' / 'controller_firmware.c').resolve()}\n"
            f"  - Runnable Python Simulator:  {(saved_dir / 'deliverables' / 'digital_twin_simulator.py').resolve()}\n"
            f"  - Wet-Lab & Manufacturing SOP:{(saved_dir / 'deliverables' / 'wetlab_and_manufacturing_sop.md').resolve()}\n"
            f"  - FMECA, SOA & BOM JSON:      {(saved_dir / 'deliverables' / 'fmeca_and_bom.json').resolve()}\n"
            f"  - Pass-3 Molecular 3D XYZ:    {(saved_dir / 'deliverables' / 'pass3_molecule.xyz').resolve()}\n"
            f"  - Pass-3 Lattice & Graph SVG: {(saved_dir / 'deliverables' / 'pass3_structure.svg').resolve()}\n"
            f"  - Pass-3 Matter Spec JSON:    {(saved_dir / 'deliverables' / 'pass3_matter_and_lattice_spec.json').resolve()}\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
