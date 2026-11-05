"""
End-to-end integration tests for the Cortivra Scientific Workbench Web UI Server (`cortivra.ui_server`).

Verifies:
1. Clean Workbench HTML & Security Headers (`GET /`, `GET /api/health`, path-traversal protection):
   - Single question textbox + optional dataset upload + Run Investigation button.
   - Zero preset mystery/benchmark buttons, zero parameter re-run sliders, zero buzzwords.
   - Strict HTTP security headers (`X-Content-Type-Options`, `X-Frame-Options`, `Content-Security-Policy`).
2. Live SSE Streaming + Empirical Dataset Upload (`POST /api/investigate/stream`):
   - Real-time `trace_node` events translated into plain-English explanations (`title`, `plain_explanation`, `thought_process`, `key_result`, `metrics`), including `ingest_uploaded_dataset`.
   - Final `complete` event containing the full `ScientificDiscoveryDossier`, trajectory series, 8-stage flight recorder logs, forensic causal X-ray report, and downloadable deliverable URLs.
3. Deliverable Artifact Serving (`GET /api/deliverables/...`):
   - Serves generated CAD, blueprint SVG, C99 firmware, simulator, and dossier files with strict path-traversal and extension allowlist protection.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

from cortivra.layer6_agents.meta_clients import MultiProviderMetaClient
from cortivra.ui_server import start_workbench_server


def _build_fast_test_client() -> MultiProviderMetaClient:
    return MultiProviderMetaClient()


def _parse_sse_stream(raw_bytes: bytes) -> list[tuple[str, dict]]:
    text = raw_bytes.decode("utf-8")
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        event_type = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            payload = json.loads("\n".join(data_lines))
            events.append((event_type, payload))
    return events


def test_workbench_html_security_headers_and_traversal_protection(tmp_path: Path) -> None:
    server = start_workbench_server(
        host="127.0.0.1",
        port=0,
        runs_dir=str(tmp_path / "ui_runs"),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"
    try:
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=10) as resp:
            assert resp.status == 200
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert resp.headers.get("X-Frame-Options") == "DENY"
            assert "default-src 'self'" in (resp.headers.get("Content-Security-Policy") or "")
            health = json.loads(resp.read().decode("utf-8"))
            assert health["status"] == "ok"
            assert health["engine"] == "cortivra"

        with urllib.request.urlopen(f"{base_url}/", timeout=10) as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "CORTIVRA" in html
            assert "Physics Discovery, Verification &amp; Control Workbench" in html
            assert 'id="hypothesisInput"' in html
            assert 'id="datasetFileInput"' in html
            assert 'id="runBtn"' in html
            assert 'id="cancelBtn"' in html
            assert 'id="liveProgressFooter"' in html
            assert "Stage 1: Primary Control Levers &amp; Required Thermodynamic Energy Shifts" in html
            assert "Saved &amp; Active Runs:" in html
            assert 'type="range"' not in html
            assert "1. Overview &amp; Findings" in html
            assert "2. Equations, Proofs &amp; Shields" in html
            assert "3. Trajectories &amp; Blast Radius" in html
            assert "4. Physical Realization &amp; Materials" in html
            assert "5. 3D CAD, Firmware &amp; Simulator" in html
            assert "6. Flight Recorder &amp; Deliverables" in html

        # Verify path traversal attempts are strictly rejected with 403/404
        for bad_path in (
            "/api/deliverables/../../pyproject.toml",
            "/api/deliverables/..%2F..%2Fpyproject.toml",
            "/api/deliverables/run1/secret.exe",
        ):
            try:
                urllib.request.urlopen(f"{base_url}{bad_path}", timeout=5)
                raise AssertionError(f"Expected HTTPError for {bad_path}")
            except urllib.error.HTTPError as err:
                assert err.code in (403, 404)
    finally:
        server.shutdown()
        server.server_close()


def test_workbench_live_sse_investigation_stream_with_uploaded_dataset(tmp_path: Path) -> None:
    server = start_workbench_server(
        host="127.0.0.1",
        port=0,
        runs_dir=str(tmp_path / "ui_runs"),
        client=_build_fast_test_client(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"
    try:
        # Build a clean 20-step empirical CSV dataset to upload with the question
        csv_lines = ["time,v_mem,ca_cyt"]
        for k in range(20):
            t_val = k * 0.05
            v_val = 2.2 * (0.92 ** k) + 0.8 * (1.0 - (0.92 ** k))
            c_val = 1.0 + 0.1 * (0.90 ** k)
            csv_lines.append(f"{t_val:.4f},{v_val:.6f},{c_val:.6f}")
        csv_text = "\n".join(csv_lines)

        req_body = json.dumps(
            {
                "prompt": (
                    "How can we restore healthy resting potential in a depolarized sensory neuron "
                    "without triggering intracellular calcium overload?"
                ),
                "observed_data_text": csv_text,
                "observed_data_filename": "neuron_telemetry.csv",
                "steps": 15,
                "max_council_steps": 4,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/investigate/stream",
            data=req_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            assert resp.status == 200
            raw_stream = resp.read()

        events = _parse_sse_stream(raw_stream)
        event_types = [etype for etype, _ in events]

        assert "error" not in event_types, f"SSE stream returned error: {events}"
        assert "started" in event_types
        assert "trace_node" in event_types
        assert "complete" in event_types

        trace_events = [payload for etype, payload in events if etype == "trace_node"]
        assert len(trace_events) >= 9
        ops = [card["operation"] for card in trace_events]
        assert "initialize_question_analysis" in ops
        assert "ingest_uploaded_dataset" in ops
        assert "compile_physical_blueprint_and_deliverables" in ops

        for node_card in trace_events:
            assert node_card["title"]
            assert node_card["plain_explanation"]
            assert node_card["thought_process"]
            assert "key_result" in node_card
            assert node_card["layer_label"]
            assert node_card["merkle_hash"]

        complete_payload = [payload for etype, payload in events if etype == "complete"][-1]
        assert complete_payload["merkle_integrity_verified"] is True
        dossier = complete_payload["dossier"]
        assert dossier["system_name"]
        assert dossier["primary_finding"]["falsification_certificate"]["passed_all_gates"] is True
        channel_names = [ch["name"] for ch in complete_payload["trajectory_series"]["channels"]]
        assert channel_names[0] == "v_mem"
        assert channel_names[1] == "ca_cyt"
        disc_eqs = dossier["primary_finding"]["discovered_equations"]
        assert "d(v_mem)/dt" in disc_eqs
        assert "d(ca_cyt)/dt" in disc_eqs
        assert len(complete_payload["trajectory_series"]["channels"]) > 0
        assert len(complete_payload["flight_recorder_log_text"]) > 100
        assert len(complete_payload["forensic_xray_markdown"]) > 100

        links = complete_payload["deliverable_links"]
        assert len(links) == 20
        assert "GOVERNING LAW:" in complete_payload["flight_recorder_log_text"]
        assert "FALSIFICATION GATES:" in complete_payload["flight_recorder_log_text"]
        for key in (
            "cad_stl",
            "cad_obj",
            "blueprint_svg",
            "firmware_c",
            "simulator_py",
            "dossier_md",
            "formal_lean4",
            "formal_smt2",
            "pass3_molecule_xyz",
            "pass3_structure_svg",
            "pass3_matter_spec_json",
            "deliverable_manifest_json",
        ):
            url_path = links[key]
            with urllib.request.urlopen(f"{base_url}{url_path}", timeout=10) as dl_resp:
                assert dl_resp.status == 200
                assert len(dl_resp.read()) > 50

        with urllib.request.urlopen(f"{base_url}/api/runs", timeout=10) as runs_resp:
            assert runs_resp.status == 200
            runs_data = json.loads(runs_resp.read().decode("utf-8"))
            assert len(runs_data["runs"]) >= 1
            first_run_id = runs_data["runs"][0]["run_id"]

        with urllib.request.urlopen(f"{base_url}/api/runs/{first_run_id}", timeout=10) as run_detail_resp:
            assert run_detail_resp.status == 200
            run_detail = json.loads(run_detail_resp.read().decode("utf-8"))
            assert run_detail["merkle_integrity_verified"] is True
            assert len(run_detail["deliverable_links"]) == 20
            assert len(run_detail["trace_cards"]) >= 5

        rig_proofs = dossier["primary_finding"]["falsification_certificate"]["rigorous_mathematical_proofs"]
        assert rig_proofs["all_rigorous_proofs_verified"] is True
        ded_thm = rig_proofs["formal_deductive_theorem"]
        assert ded_thm["qed_verified"] is True
        assert len(ded_thm["lemmas"]) == 5
        assert ded_thm["complex_analytic_certificate"]["complex_rhp_zero_free_proven"] is True
        assert ded_thm["complex_analytic_certificate"]["cauchy_rhp_winding_number"] == 0
        assert ded_thm["complex_analytic_certificate"]["imaginary_axis_resolvent_verified"] is True
        assert "theorem" in ded_thm["lean4_smt_certificate"]["lean4_source_code"]
        assert "(set-logic QF_NRA)" in ded_thm["lean4_smt_certificate"]["smt2_source_code"]
    finally:
        server.shutdown()
        server.server_close()
