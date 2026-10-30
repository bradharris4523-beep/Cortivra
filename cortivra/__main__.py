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
    if provider == "policy":
        overrides = {
            "FIREWORKS_API_KEY": "",
            "BASETEN_API_KEY": "",
            "META_API_KEY": "",
            "LOCAL_LLM_BASE_URL": "",
        }
    elif provider == "local":
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
        nargs="+",
        help="One-sentence or short-paragraph scientific hypothesis, anomaly, or intervention question.",
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
        choices=("auto", "policy", "local", "baseten", "fireworks", "meta"),
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
        help="Print the Forensic Causal X-Ray report (also printed automatically if any gate fails).",
    )
    parser.add_argument(
        "--save-logs",
        action="store_true",
        help="Save dossier.md, dossier.json, and flight_recorder.jsonl under --log-dir.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/runs",
        help="Directory for saved forensic run artifacts (default: logs/runs).",
    )
    args = parser.parse_args(argv)
    question = " ".join(args.prompt).strip()

    client = _build_cli_client(
        provider=args.provider,
        model_override=args.model,
        base_url_override=args.base_url,
    )

    dossier = investigate(
        prompt=question,
        domain=args.domain,
        client=client,
        dt=args.dt,
        num_steps=args.steps,
        observed_states=args.data,
        observatory_stream=args.observatory,
    )

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
        _safe_stdout_write(
            f"\n[Forensic Flight Recorder Artifacts Saved] -> {saved_dir.resolve()}\n"
            f"  - Markdown Dossier + X-Ray: {(saved_dir / 'dossier.md').resolve()}\n"
            f"  - Full JSON Telemetry:      {(saved_dir / 'dossier.json').resolve()}\n"
            f"  - Merkle Trace DAG JSONL:   {(saved_dir / 'flight_recorder.jsonl').resolve()}\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
