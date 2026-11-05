"""
Layer 7 Editorial Scientific Workbench HTML/CSS/JS Frontend (`cortivra.ui_template`).

Designed with a clean, progressive-disclosure, plain-English scientific workbench layout:
  1. Single Prompt Input Box + Optional Dataset Upload (.csv / .json) + Cancel button at the top.
  2. Live Process, Reasoning & Step Log Stream that updates in real time via SSE,
     translating every mathematical operation across Layers 1–6 into plain English
     alongside a live elapsed-second timer and expandable raw mathematical telemetry.
  3. Structured 6-Tab Scientific Dossier Inspector organized into collapsible
     dropdown sections (`<details class="dossier-dropdown">`), where every tab,
     every dropdown, and every table row includes a dynamic plain-English explanation
     of the exact numbers from that run.
  4. Built-in "Ask Co-Scientist AI" Chatbot drawer (`POST /api/chat`) grounded 100%
     in the active run's `dossier.json` so any user or judge can ask questions about
     any section, molecule, equation, or proof.
"""

from __future__ import annotations

from cortivra.ui_frontend_js import WORKBENCH_FRONTEND_JS


def get_workbench_html() -> str:
    """
    Return the complete, self-contained HTML5 document for the Cortivra Workbench.
    """
    return _WORKBENCH_HTML.replace("<!-- __WORKBENCH_FRONTEND_JS__ -->", WORKBENCH_FRONTEND_JS)


_WORKBENCH_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cortivra — Physics Discovery, Verification &amp; Control Workbench</title>
  <style>
    :root {
      --bg-page: #f8fafc;
      --bg-surface: #ffffff;
      --bg-subtle: #f1f5f9;
      --bg-code: #0f172a;
      --text-primary: #0f172a;
      --text-secondary: #334155;
      --text-muted: #64748b;
      --border-light: #e2e8f0;
      --border-strong: #cbd5e1;
      --accent: #0f172a;
      --accent-blue: #1d4ed8;
      --accent-blue-bg: #eff6ff;
      --pass-green: #047857;
      --pass-green-bg: #ecfdf5;
      --pass-green-border: #a7f3d0;
      --warn-amber: #b45309;
      --warn-amber-bg: #fffbeb;
      --fail-red: #b91c1c;
      --fail-red-bg: #fef2f2;
      --font-sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      --font-mono: "JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 0;
      background-color: var(--bg-page); color: var(--text-primary);
      font-family: var(--font-sans); font-size: 14px; line-height: 1.55;
      -webkit-font-smoothing: antialiased;
    }
    .topbar {
      background: var(--bg-surface); border-bottom: 1px solid var(--border-light);
      padding: 14px 28px; display: flex; align-items: center; justify-content: space-between;
      position: sticky; top: 0; z-index: 50;
    }
    .brand-group { display: flex; align-items: baseline; gap: 12px; }
    .brand-title { font-size: 16px; font-weight: 800; letter-spacing: -0.02em; color: var(--text-primary); }
    .brand-subtitle { font-size: 13px; color: var(--text-muted); }
    .topbar-right { display: flex; align-items: center; gap: 12px; }
    .status-pill {
      display: inline-flex; align-items: center; gap: 7px; font-size: 12px; font-weight: 500;
      padding: 5px 11px; border-radius: 6px; background: var(--bg-subtle);
      color: var(--text-secondary); border: 1px solid var(--border-light); font-family: var(--font-mono);
    }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #10b981; }
    .status-dot.running { background: #2563eb; animation: pulse 1.2s infinite ease-in-out; }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.88); }
    }
    .co-scientist-top-btn {
      display: inline-flex; align-items: center; gap: 7px;
      padding: 6px 14px; border-radius: 7px; font-size: 12.5px; font-weight: 700;
      background: linear-gradient(135deg, #1d4ed8 0%, #0f172a 100%);
      color: #ffffff; border: none; cursor: pointer; box-shadow: 0 2px 6px rgba(29, 78, 216, 0.25);
      transition: transform 0.12s ease, opacity 0.12s ease;
    }
    .co-scientist-top-btn:hover { transform: translateY(-1px); opacity: 0.95; }
    .container { max-width: 1460px; margin: 0 auto; padding: 24px 28px 80px; }
    .prompt-card {
      background: var(--bg-surface); border: 1px solid var(--border-strong); border-radius: 10px;
      padding: 20px 22px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04); margin-bottom: 24px;
    }
    .prompt-label { display: block; font-size: 13.5px; font-weight: 700; color: var(--text-primary); margin-bottom: 8px; }
    .prompt-input-wrap { display: flex; gap: 10px; align-items: stretch; }
    .prompt-textarea {
      flex: 1; min-height: 64px; padding: 12px 14px; font-family: var(--font-sans); font-size: 15px;
      line-height: 1.45; color: var(--text-primary); background: var(--bg-page);
      border: 1px solid var(--border-strong); border-radius: 8px; resize: vertical; outline: none;
      transition: border-color 0.15s ease, background-color 0.15s ease;
    }
    .prompt-textarea:focus { background: var(--bg-surface); border-color: var(--accent-blue); }
    .run-button {
      padding: 0 24px; background: var(--accent); color: #ffffff; border: none; border-radius: 8px;
      font-family: var(--font-sans); font-size: 14px; font-weight: 700; cursor: pointer;
      white-space: nowrap; transition: background 0.15s ease, opacity 0.15s ease;
    }
    .run-button:hover:not(:disabled) { background: #1e293b; }
    .run-button:disabled { opacity: 0.55; cursor: not-allowed; }
    .cancel-button {
      padding: 0 18px; background: var(--fail-red-bg); color: var(--fail-red);
      border: 1px solid #fecaca; border-radius: 8px; font-family: var(--font-sans);
      font-size: 13.5px; font-weight: 600; cursor: pointer; white-space: nowrap;
    }
    .cancel-button:hover { background: #fee2e2; }
    .data-upload-row {
      display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;
      gap: 10px; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border-light);
      font-size: 12.5px; color: var(--text-secondary);
    }
    .upload-controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .upload-btn {
      display: inline-flex; align-items: center; gap: 6px; padding: 5px 11px;
      border: 1px solid var(--border-strong); border-radius: 6px; background: var(--bg-page);
      color: var(--text-primary); font-size: 12px; font-weight: 600; cursor: pointer;
    }
    .upload-btn:hover { background: var(--bg-subtle); border-color: var(--text-secondary); }
    .data-badge {
      display: inline-flex; align-items: center; gap: 8px; padding: 4px 10px; border-radius: 6px;
      background: var(--pass-green-bg); color: var(--pass-green);
      border: 1px solid var(--pass-green-border); font-family: var(--font-mono); font-size: 11.5px;
    }
    .clear-data-btn {
      background: none; border: none; color: var(--fail-red); font-family: var(--font-sans);
      font-size: 11.5px; font-weight: 600; cursor: pointer; padding: 0 2px;
    }
    .workspace-grid { display: grid; grid-template-columns: 420px 1fr; gap: 24px; align-items: start; }
    @media (max-width: 1120px) { .workspace-grid { grid-template-columns: 1fr; } }
    .panel {
      background: var(--bg-surface); border: 1px solid var(--border-light); border-radius: 10px;
      overflow: hidden; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .panel-header {
      padding: 14px 18px; border-bottom: 1px solid var(--border-light);
      display: flex; justify-content: space-between; align-items: center; background: var(--bg-surface);
    }
    .panel-title {
      font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
      color: var(--text-secondary); margin: 0;
    }
    .step-counter { font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); }
    .process-stream {
      padding: 16px 18px; max-height: 820px; overflow-y: auto;
      display: flex; flex-direction: column; gap: 12px;
    }
    .empty-state { padding: 40px 20px; text-align: center; color: var(--text-muted); font-size: 13.5px; }
    .step-card {
      border: 1px solid var(--border-light); border-left: 3px solid var(--accent-blue);
      border-radius: 8px; padding: 12px 14px; background: var(--bg-surface);
    }
    .step-card.layer1 { border-left-color: #059669; }
    .step-card.layer3 { border-left-color: #4f46e5; }
    .step-card.layer4 { border-left-color: #0284c7; }
    .step-card.layer5 { border-left-color: #d97706; }
    .step-card.layer6 { border-left-color: #0f172a; }
    .step-meta {
      display: flex; justify-content: space-between; align-items: center;
      font-size: 11px; font-family: var(--font-mono); color: var(--text-muted); margin-bottom: 5px;
    }
    .step-layer-tag { font-weight: 600; color: var(--text-secondary); }
    .step-title { font-size: 13.5px; font-weight: 700; color: var(--text-primary); margin-bottom: 4px; }
    .step-plain { font-size: 13px; color: var(--text-secondary); margin-bottom: 8px; }
    .step-thought {
      font-size: 12px; color: var(--text-muted); background: var(--bg-subtle);
      padding: 7px 10px; border-radius: 6px; margin-bottom: 8px; border: 1px solid var(--border-light);
    }
    .step-thought strong { color: var(--text-secondary); font-weight: 600; }
    .step-result-box {
      font-family: var(--font-mono); font-size: 11.5px; color: var(--accent-blue);
      background: var(--accent-blue-bg); padding: 6px 9px; border-radius: 5px; word-break: break-word;
    }
    .step-details { margin-top: 8px; font-size: 11.5px; }
    .step-details summary { cursor: pointer; color: var(--text-muted); font-family: var(--font-mono); user-select: none; }
    .step-details pre {
      margin: 6px 0 0; padding: 8px; background: var(--bg-code); color: #e2e8f0;
      border-radius: 5px; overflow-x: auto; font-family: var(--font-mono); font-size: 11px;
    }
    .live-progress-footer {
      display: none; align-items: center; justify-content: space-between;
      padding: 10px 18px; background: var(--accent-blue-bg); border-top: 1px solid #bfdbfe;
      font-size: 12px; color: var(--accent-blue); font-family: var(--font-mono);
    }
    .session-history-bar {
      display: none; align-items: center; gap: 8px; flex-wrap: wrap;
      padding: 9px 18px; background: var(--bg-page); border-bottom: 1px solid var(--border-light); font-size: 12px;
    }
    .session-history-label { font-weight: 700; color: var(--text-muted); margin-right: 4px; }
    .history-pill {
      background: var(--bg-surface); border: 1px solid var(--border-strong); border-radius: 6px;
      padding: 4px 10px; font-size: 12px; color: var(--text-secondary); cursor: pointer; font-family: var(--font-sans);
    }
    .history-pill.active {
      background: var(--accent-blue-bg); border-color: var(--accent-blue); color: var(--accent-blue); font-weight: 700;
    }
    .tabs-bar {
      display: flex; flex-wrap: wrap; gap: 4px; padding: 10px 16px 0;
      background: var(--bg-subtle); border-bottom: 1px solid var(--border-light);
    }
    .tab-btn {
      padding: 10px 14px; font-family: var(--font-sans); font-size: 13px; font-weight: 600;
      color: var(--text-muted); background: transparent; border: 1px solid transparent;
      border-bottom: none; border-radius: 6px 6px 0 0; cursor: pointer;
    }
    .tab-btn:hover { color: var(--text-primary); }
    .tab-btn.active {
      background: var(--bg-surface); color: var(--accent-blue); font-weight: 700;
      border-color: var(--border-light); margin-bottom: -1px; padding-bottom: 11px;
    }
    .tab-pane { display: none; padding: 20px 22px; }
    .tab-pane.active { display: block; }

    /* Big Bold Plain-English Takeaway Banner at the Top of Every Tab */
    .tab-takeaway-banner {
      background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
      border: 2px solid #93c5fd; border-left: 6px solid var(--accent-blue);
      border-radius: 10px; padding: 16px 18px; margin-bottom: 18px;
      box-shadow: 0 2px 6px rgba(29, 78, 216, 0.06);
    }
    .tab-takeaway-header {
      display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;
      margin-bottom: 8px;
    }
    .tab-takeaway-kicker {
      font-size: 11.5px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--accent-blue); background: #dbeafe; padding: 3px 9px; border-radius: 999px;
    }
    .tab-takeaway-headline {
      font-size: 16px; font-weight: 800; color: var(--text-primary); line-height: 1.45; margin-bottom: 6px;
    }
    .tab-takeaway-sub {
      font-size: 13.5px; color: var(--text-secondary); line-height: 1.55;
    }
    .tab-controls-bar {
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;
      gap: 8px; margin-bottom: 14px; padding: 8px 12px; background: var(--bg-subtle);
      border-radius: 8px; border: 1px solid var(--border-light); font-size: 12px; color: var(--text-secondary);
    }
    .tab-control-btn {
      background: var(--bg-surface); border: 1px solid var(--border-strong); border-radius: 6px;
      padding: 4px 10px; font-size: 11.5px; font-weight: 600; color: var(--text-primary); cursor: pointer;
    }
    .tab-control-btn:hover { border-color: var(--accent-blue); color: var(--accent-blue); }

    /* Progressive-Disclosure Collapsible Dropdown Cards */
    details.dossier-dropdown {
      background: var(--bg-surface); border: 1px solid var(--border-strong);
      border-radius: 10px; margin-bottom: 14px; overflow: hidden;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
      transition: border-color 0.15s ease;
    }
    details.dossier-dropdown[open] {
      border-color: #93c5fd;
      box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05);
    }
    details.dossier-dropdown > summary.dropdown-header {
      list-style: none; cursor: pointer; user-select: none;
      padding: 14px 18px; background: var(--bg-page);
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
      border-bottom: 1px solid transparent; transition: background 0.15s ease;
    }
    details.dossier-dropdown > summary.dropdown-header::-webkit-details-marker { display: none; }
    details.dossier-dropdown[open] > summary.dropdown-header {
      background: #f8fafc; border-bottom-color: var(--border-light);
    }
    details.dossier-dropdown > summary.dropdown-header:hover { background: var(--bg-subtle); }
    .dropdown-title-group { display: flex; align-items: center; gap: 10px; flex: 1; min-width: 0; }
    .dropdown-chevron {
      display: inline-flex; align-items: center; justify-content: center;
      width: 24px; height: 24px; border-radius: 6px; background: #dbeafe;
      color: var(--accent-blue); font-size: 12px; font-weight: 800; flex-shrink: 0;
      transition: transform 0.18s ease;
    }
    details.dossier-dropdown[open] .dropdown-chevron { transform: rotate(90deg); background: var(--accent-blue); color: #ffffff; }
    .dropdown-title-text { display: flex; flex-direction: column; min-width: 0; }
    .dropdown-main-title { font-size: 14.5px; font-weight: 800; color: var(--text-primary); line-height: 1.3; }
    .dropdown-sub-caption { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
    .dropdown-right-controls { display: flex; align-items: center; gap: 8px; flex-shrink: 0; flex-wrap: wrap; }
    .ask-ai-pill {
      display: inline-flex; align-items: center; gap: 5px;
      padding: 4px 10px; border-radius: 6px; font-size: 11.5px; font-weight: 700;
      background: var(--accent-blue-bg); color: var(--accent-blue);
      border: 1px solid #bfdbfe; cursor: pointer; transition: all 0.12s ease;
    }
    .ask-ai-pill:hover { background: var(--accent-blue); color: #ffffff; border-color: var(--accent-blue); }
    .dropdown-body { padding: 16px 18px; background: var(--bg-surface); }

    /* Big Bold Plain-English Section Callout Inside Every Dropdown */
    .plain-english-callout {
      background: #f8fafc; border: 1.5px solid #cbd5e1; border-left: 5px solid var(--accent-blue);
      border-radius: 8px; padding: 12px 15px; margin-bottom: 14px;
    }
    .plain-english-callout-title {
      font-size: 11.5px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;
      color: var(--accent-blue); margin-bottom: 4px; display: flex; align-items: center; justify-content: space-between;
    }
    .plain-english-callout-body {
      font-size: 14px; font-weight: 600; color: var(--text-primary); line-height: 1.5;
    }
    .plain-english-callout-sub {
      font-size: 12.5px; color: var(--text-secondary); margin-top: 5px; line-height: 1.5;
    }

    .kpi-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px; margin-bottom: 18px;
    }
    .kpi-card {
      background: var(--bg-page); border: 1px solid var(--border-strong); border-radius: 8px; padding: 13px 15px;
    }
    .kpi-label { font-size: 11.5px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }
    .kpi-value { font-family: var(--font-mono); font-size: 17px; font-weight: 800; color: var(--text-primary); }
    .kpi-sub { font-size: 12px; color: var(--text-secondary); margin-top: 4px; line-height: 1.4; }
    .summary-prose {
      background: var(--bg-page); border: 1px solid var(--border-light); border-radius: 8px;
      padding: 14px 16px; font-size: 13.5px; color: var(--text-secondary); line-height: 1.6;
    }
    .eq-box {
      background: var(--bg-code); color: #f8fafc; font-family: var(--font-mono); font-size: 12.5px;
      padding: 13px 16px; border-radius: 8px; overflow-x: auto; margin-bottom: 12px; line-height: 1.55;
    }
    .eq-label {
      font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
      color: #38bdf8; margin-bottom: 4px; display: block;
    }
    .eq-plain-note {
      font-family: var(--font-sans); font-size: 12.5px; color: #cbd5e1;
      background: rgba(255, 255, 255, 0.06); border-left: 3px solid #38bdf8;
      padding: 7px 10px; border-radius: 4px; margin-top: 8px; line-height: 1.45;
    }
    table.data-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
    table.data-table th, table.data-table td {
      padding: 10px 11px; border-bottom: 1px solid var(--border-light); text-align: left; vertical-align: top;
    }
    table.data-table th {
      background: var(--bg-subtle); color: var(--text-secondary); font-weight: 700;
      font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.03em;
    }
    table.data-table code {
      font-family: var(--font-mono); font-size: 12px; background: var(--bg-subtle); padding: 2px 5px; border-radius: 4px;
    }
    .row-plain-meaning {
      font-size: 12px; color: #1e293b; background: #f8fafc;
      border-left: 3px solid #3b82f6; padding: 5px 8px; border-radius: 4px;
      margin-top: 4px; line-height: 1.45;
    }
    .badge {
      display: inline-block; padding: 2px 8px; border-radius: 4px;
      font-family: var(--font-mono); font-size: 11px; font-weight: 700;
    }
    .badge-pass { background: var(--pass-green-bg); color: var(--pass-green); border: 1px solid var(--pass-green-border); }
    .badge-warn { background: var(--warn-amber-bg); color: var(--warn-amber); border: 1px solid #fde68a; }
    .badge-fail { background: var(--fail-red-bg); color: var(--fail-red); border: 1px solid #fecaca; }
    .code-viewer {
      background: var(--bg-code); color: #e2e8f0; padding: 14px 16px; border-radius: 8px;
      font-family: var(--font-mono); font-size: 12px; overflow-x: auto; max-height: 460px; line-height: 1.5; margin: 0;
    }
    .deliverable-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; }
    .deliverable-card {
      display: flex; flex-direction: column; justify-content: space-between;
      border: 1px solid var(--border-light); border-radius: 8px; padding: 12px 14px;
      background: var(--bg-page); text-decoration: none; color: var(--text-primary);
      transition: border-color 0.15s ease, background 0.15s ease;
    }
    .deliverable-card:hover { border-color: var(--accent-blue); background: var(--bg-surface); }
    .deliverable-name { font-family: var(--font-mono); font-size: 12.5px; font-weight: 700; color: var(--accent-blue); margin-bottom: 4px; }
    .deliverable-desc { font-size: 12px; color: var(--text-secondary); line-height: 1.45; }
    .canvas-box {
      background: radial-gradient(circle at 50% 35%, #0f2442 0%, #060d1a 100%);
      border-radius: 10px; border: 1px solid #1e3a5f;
      overflow: hidden; display: flex; flex-direction: column; align-items: center; justify-content: space-between; padding: 12px;
      box-shadow: inset 0 0 28px rgba(14, 165, 233, 0.08);
    }
    .cad-toolbar {
      display: flex; flex-wrap: wrap; gap: 6px; width: 100%; justify-content: space-between; align-items: center;
      margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #1e293b;
    }
    .cad-btn {
      background: #0f172a; color: #cbd5e1; border: 1px solid #334155; border-radius: 5px;
      padding: 4px 8px; font-family: var(--font-mono); font-size: 11px; cursor: pointer;
      transition: all 0.15s ease;
    }
    .cad-btn:hover { border-color: #38bdf8; color: #38bdf8; background: #1e293b; }
    .cad-btn.active { background: #0284c7; color: #ffffff; border-color: #38bdf8; }
    .cad-legend {
      display: flex; flex-wrap: wrap; gap: 10px; width: 100%; justify-content: center;
      margin-top: 8px; padding-top: 6px; border-top: 1px solid #1e293b;
      font-family: var(--font-mono); font-size: 10.5px; color: #cbd5e1;
    }
    .cad-legend-dot {
      display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; vertical-align: middle;
    }
    .summary-card-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px;
    }
    .summary-card {
      background: var(--bg-page); border: 1px solid var(--border-light); border-left: 4px solid var(--accent-blue);
      border-radius: 8px; padding: 12px 14px; font-size: 12.5px; color: var(--text-secondary); line-height: 1.55;
    }
    .summary-card-title {
      font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em;
      color: var(--text-primary); margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;
    }
    .svg-blueprint-container svg { max-width: 100%; height: auto; border-radius: 8px; display: block; }
    .example-chips-row {
      display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
      margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--border-light);
      font-size: 11.5px; color: var(--text-muted);
    }
    .example-chip {
      background: var(--bg-page); color: var(--text-secondary); border: 1px solid var(--border-light);
      border-radius: 999px; padding: 3px 10px; font-size: 11.5px; cursor: pointer;
      transition: all 0.15s ease;
    }
    .example-chip:hover { border-color: var(--accent-blue); color: var(--accent-blue); background: var(--accent-blue-bg); }
    .filter-pills-bar {
      display: flex; gap: 4px; flex-wrap: wrap; padding: 6px 12px;
      background: var(--bg-page); border-bottom: 1px solid var(--border-light);
    }
    .filter-pill {
      background: var(--bg-surface); color: var(--text-secondary); border: 1px solid var(--border-light);
      border-radius: 4px; padding: 2px 8px; font-size: 11px; font-family: var(--font-mono); cursor: pointer;
    }
    .filter-pill:hover { border-color: var(--accent-blue); color: var(--accent-blue); }
    .filter-pill.active { background: var(--accent-blue); color: #ffffff; border-color: var(--accent-blue); }
    .phase-divider-banner {
      background: linear-gradient(90deg, #eff6ff 0%, #f8fafc 100%);
      border: 1px solid #bfdbfe; border-left: 4px solid var(--accent-blue);
      border-radius: 7px; padding: 7px 12px; margin-top: 6px;
      font-size: 11.5px; font-weight: 700; color: var(--accent-blue);
      font-family: var(--font-mono); letter-spacing: 0.02em;
      display: flex; justify-content: space-between; align-items: center;
    }
    .step-metrics-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
    .step-metric-chip {
      font-family: var(--font-mono); font-size: 10.5px; padding: 2px 6px;
      border-radius: 4px; background: #f1f5f9; color: #334155; border: 1px solid #e2e8f0;
    }

    /* Live Co-Scientist AI Slide-Out Chatbot Drawer */
    .chat-fab-btn {
      position: fixed; bottom: 22px; right: 26px; z-index: 90;
      display: inline-flex; align-items: center; gap: 8px;
      padding: 12px 18px; border-radius: 999px;
      background: linear-gradient(135deg, #1d4ed8 0%, #0f172a 100%);
      color: #ffffff; border: 2px solid #60a5fa;
      font-family: var(--font-sans); font-size: 13.5px; font-weight: 800;
      cursor: pointer; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.28);
      transition: transform 0.15s ease;
    }
    .chat-fab-btn:hover { transform: translateY(-2px); }
    .chat-drawer {
      position: fixed; top: 0; right: -460px; width: 440px; max-width: 94vw; height: 100vh;
      background: var(--bg-surface); border-left: 2px solid var(--border-strong);
      box-shadow: -8px 0 32px rgba(15, 23, 42, 0.18); z-index: 100;
      display: flex; flex-direction: column; transition: right 0.22s ease;
    }
    .chat-drawer.open { right: 0; }
    .chat-drawer-header {
      padding: 15px 18px; background: #0f172a; color: #ffffff;
      display: flex; align-items: center; justify-content: space-between;
    }
    .chat-drawer-title { font-size: 14.5px; font-weight: 800; letter-spacing: -0.01em; }
    .chat-drawer-sub { font-size: 11.5px; color: #93c5fd; margin-top: 2px; }
    .chat-close-btn {
      background: rgba(255,255,255,0.12); color: #ffffff; border: none; border-radius: 6px;
      padding: 5px 10px; font-size: 12px; font-weight: 700; cursor: pointer;
    }
    .chat-close-btn:hover { background: rgba(255,255,255,0.22); }
    .chat-context-banner {
      padding: 8px 14px; background: #eff6ff; border-bottom: 1px solid #bfdbfe;
      font-size: 11.5px; color: #1e3a8a; font-weight: 600;
      display: flex; justify-content: space-between; align-items: center; gap: 6px;
    }
    .chat-quick-chips {
      padding: 10px 14px; background: var(--bg-page); border-bottom: 1px solid var(--border-light);
      display: flex; flex-wrap: wrap; gap: 6px;
    }
    .chat-quick-chip {
      background: var(--bg-surface); color: var(--accent-blue); border: 1px solid #bfdbfe;
      border-radius: 999px; padding: 4px 10px; font-size: 11.5px; font-weight: 600; cursor: pointer;
      transition: all 0.12s ease;
    }
    .chat-quick-chip:hover { background: var(--accent-blue); color: #ffffff; }
    .chat-messages {
      flex: 1; overflow-y: auto; padding: 14px 16px; display: flex; flex-direction: column; gap: 12px;
      background: #f8fafc;
    }
    .chat-bubble {
      padding: 11px 14px; border-radius: 10px; font-size: 13px; line-height: 1.55; max-width: 92%;
      word-break: break-word;
    }
    .chat-bubble.user {
      align-self: flex-end; background: var(--accent-blue); color: #ffffff;
      border-bottom-right-radius: 3px; font-weight: 500;
    }
    .chat-bubble.assistant {
      align-self: flex-start; background: var(--bg-surface); color: var(--text-primary);
      border: 1px solid var(--border-strong); border-left: 4px solid var(--accent-blue);
      border-bottom-left-radius: 3px;
    }
    .chat-input-area {
      padding: 12px 14px; background: var(--bg-surface); border-top: 1px solid var(--border-light);
      display: flex; gap: 8px; align-items: stretch;
    }
    .chat-input {
      flex: 1; padding: 9px 12px; border: 1px solid var(--border-strong); border-radius: 8px;
      font-family: var(--font-sans); font-size: 13px; outline: none; resize: none; height: 42px;
    }
    .chat-input:focus { border-color: var(--accent-blue); }
    .chat-send-btn {
      padding: 0 16px; background: var(--accent-blue); color: #ffffff; border: none;
      border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer;
    }
    .chat-send-btn:hover:not(:disabled) { background: #1e40af; }
    .chat-send-btn:disabled { opacity: 0.55; cursor: not-allowed; }
  </style>
</head>
<body>

  <!-- Header -->
  <header class="topbar">
    <div class="brand-group">
      <span class="brand-title">CORTIVRA</span>
      <span class="brand-subtitle">First-Principles Scientific Discovery, Proof &amp; Molecule/Hardware Design Workbench</span>
    </div>
    <div class="topbar-right">
      <button type="button" class="co-scientist-top-btn" onclick="window.openCoScientistChat && window.openCoScientistChat('Entire Discovery Run')">
        &#128172; Ask Co-Scientist AI
      </button>
      <div class="status-pill" id="engineStatusPill">
        <span class="status-dot" id="engineStatusDot"></span>
        <span id="engineStatusText">Ready &middot; Layers 1&ndash;6 Active</span>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="container">

    <!-- Single Clean Question Input Card + Optional Dataset Upload -->
    <section class="prompt-card">
      <label class="prompt-label" for="hypothesisInput">
        Ask Any Scientific, Medical, Energy, or Engineering Question
      </label>
      <div class="prompt-input-wrap">
        <textarea
          id="hypothesisInput"
          class="prompt-textarea"
          placeholder="Type any scientific problem, disease target, clean-energy challenge, or physical anomaly here..."
        ></textarea>
        <button id="runBtn" class="run-button" onclick="startInvestigation()">
          Run Investigation
        </button>
        <button id="cancelBtn" type="button" class="cancel-button" style="display:none;" onclick="cancelInvestigation()">
          Cancel
        </button>
      </div>
      <div class="data-upload-row">
        <div class="upload-controls">
          <input
            type="file"
            id="datasetFileInput"
            accept=".csv,.json,.txt"
            style="display:none;"
            onchange="handleDatasetFileSelect(event)"
          />
          <button type="button" class="upload-btn" onclick="document.getElementById('datasetFileInput').click()">
            + Attach Dataset (.csv or .json &mdash; optional)
          </button>
          <span id="datasetHintText" style="color:var(--text-muted);">
            Optional: attach experimental sensor data so the engine discovers equations directly from your measurements.
          </span>
          <span id="datasetLoadedBadge" class="data-badge" style="display:none;">
            <span id="datasetBadgeText"></span>
            <button type="button" class="clear-data-btn" onclick="clearUploadedDataset()">Remove</button>
          </span>
        </div>
      </div>
      <div class="example-chips-row">
        <span style="font-weight:700;color:var(--text-secondary);">Sample Prompts:</span>
        <button type="button" class="example-chip" onclick="fillExamplePrompt(0)">Spacecraft Hypersonic Plasma Blackout</button>
        <button type="button" class="example-chip" onclick="fillExamplePrompt(1)">Alzheimer's Plaque Clearance</button>
        <button type="button" class="example-chip" onclick="fillExamplePrompt(2)">CAR-T Solid Tumor vs. Cytokine Storm</button>
        <button type="button" class="example-chip" onclick="fillExamplePrompt(3)">ITER Fusion Plasma Heat Shield</button>
        <button type="button" class="example-chip" onclick="fillExamplePrompt(4)">Origin-of-Life RNA Replication</button>
        <button type="button" class="example-chip" onclick="fillExamplePrompt(5)">Room-Temp Superconductor Stability</button>
      </div>
    </section>

    <!-- Two-Column Workbench: Live Step Stream (Left) + Progressive-Disclosure Plain-English Dossier (Right) -->
    <div class="workspace-grid">

      <!-- Left Column: Step-by-Step Process, Thoughts & Plain-English Logs -->
      <aside class="panel">
        <div class="panel-header">
          <h2 class="panel-title">Live Step-by-Step Reasoning Stream</h2>
          <span class="step-counter" id="stepCountLabel">0 steps</span>
        </div>
        <div class="filter-pills-bar" id="stepFilterBar">
          <button type="button" class="filter-pill active" onclick="filterStepStream('all', this)">All Steps</button>
          <button type="button" class="filter-pill" onclick="filterStepStream('phase1', this)">1. Papers &amp; Data</button>
          <button type="button" class="filter-pill" onclick="filterStepStream('phase2', this)">2. Physics Rules</button>
          <button type="button" class="filter-pill" onclick="filterStepStream('phase3', this)">3. Safe Fix</button>
          <button type="button" class="filter-pill" onclick="filterStepStream('phase4', this)">4. Blind Test</button>
          <button type="button" class="filter-pill" onclick="filterStepStream('phase5', this)">5. Math Proof</button>
          <button type="button" class="filter-pill" onclick="filterStepStream('phase6', this)">6. Molecule &amp; CAD</button>
        </div>
        <div class="process-stream" id="processStream">
          <div class="empty-state">
            Select a saved run or enter a question above and click <strong>Run Investigation</strong>.<br/>
            Every step the engine takes is explained here in plain English as it happens.
          </div>
        </div>
        <div class="live-progress-footer" id="liveProgressFooter">
          <span id="liveActiveStageLabel">Working on current step...</span>
          <span id="liveElapsedTimerLabel">0.0s</span>
        </div>
      </aside>

      <!-- Right Column: Organized Multi-Tab Results with Collapsible Plain-English Dropdowns -->
      <section class="panel">
        <div class="session-history-bar" id="sessionHistoryBar">
          <span class="session-history-label">Saved &amp; Active Runs:</span>
          <div id="sessionHistoryPills" style="display:flex;gap:6px;flex-wrap:wrap;"></div>
        </div>

        <div class="tabs-bar">
          <button class="tab-btn active" onclick="switchTab('overview', this)">1. Summary &amp; What It Found</button>
          <button class="tab-btn" onclick="switchTab('equations', this)">2. Equations &amp; Math Proofs</button>
          <button class="tab-btn" onclick="switchTab('trajectories', this)">3. Before/After Curves &amp; Safety</button>
          <button class="tab-btn" onclick="switchTab('blueprint', this)">4. Molecule, Recipe &amp; Material Specs</button>
          <button class="tab-btn" onclick="switchTab('hardware', this)">5. 3D CAD, Controller Code &amp; Lab Guide</button>
          <button class="tab-btn" onclick="switchTab('logs', this)">6. All 20 Files &amp; Audit Logs</button>
        </div>

        <!-- ================================================================= -->
        <!-- TAB 1: SUMMARY & WHAT IT FOUND                                    -->
        <!-- ================================================================= -->
        <div id="tab-overview" class="tab-pane active">
          <div class="empty-state" id="overviewEmpty">
            Investigation results, key metrics, and plain-English synthesis will appear here once a run completes.
          </div>
          <div id="overviewContent" style="display:none;">
            <div class="tab-takeaway-banner" id="tab1TakeawayBanner"></div>

            <div class="tab-controls-bar">
              <span><strong>Tip:</strong> Click any dropdown bar below to expand or collapse its details, or click <strong>&#128172; Ask AI</strong> on any section.</span>
              <div style="display:flex;gap:6px;">
                <button type="button" class="tab-control-btn" onclick="window.toggleTabDropdowns('tab-overview', true)">Expand All in Tab</button>
                <button type="button" class="tab-control-btn" onclick="window.toggleTabDropdowns('tab-overview', false)">Collapse All</button>
              </div>
            </div>

            <div class="kpi-grid" id="overviewKpis"></div>

            <!-- Dropdown 1.1: Plain-English Discovery & Safety Brief (Open by default) -->
            <details class="dossier-dropdown" open>
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">1A. What the Engine Discovered &amp; How It Prevents Side Effects</span>
                    <span class="dropdown-sub-caption">Plain-English summary of the problem solved, how much the target improved, and how healthy parts stayed safe</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge1A">PRIMARY RESULT</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('1A. Main Discovery & Side-Effect Shield', 'execSummaryBox')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="summary-prose" id="execSummaryBox"></div>
              </div>
            </details>

            <!-- Dropdown 1.2: 3-Stage Discovery & Realization Cards (Open by default) -->
            <details class="dossier-dropdown" id="pass1Pass2Section" open>
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">1B. How the Solution Works Across All 3 Stages (Physics &rarr; Lab Recipe &rarr; Molecule &amp; 3D Structure)</span>
                    <span class="dropdown-sub-caption">Stage 1 finds which levers to tune &middot; Stage 2 solves the lab manufacturing recipe &middot; Stage 3 designs the molecule &amp; 3D lattice</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge1B">3 STAGES SOLVED</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('1B. 3-Stage Solution Overview (Physics, Lab Recipe, Molecule)', 'pass1Pass2ExecutiveGrid')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout1B"></div>
                <div class="summary-card-grid" id="pass1Pass2ExecutiveGrid"></div>
              </div>
            </details>

            <!-- Dropdown 1.3: 5-Point Verification Scorecard -->
            <details class="dossier-dropdown" id="discoveryProofScorecardSection">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">1C. 5-Point Proof Scorecard: Why Standard Approaches Fail vs. How Cortivra Succeeded</span>
                    <span class="dropdown-sub-caption">Side-by-side comparison against standard curve-fitting and single-lever fixes</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass">5/5 VERIFIED</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('1C. 5-Point Proof Scorecard', 'discoveryProofTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout1C"></div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Test Performed</th>
                      <th>What Question Was Tested?</th>
                      <th>Why Standard / Naive Approach Fails</th>
                      <th>What Cortivra Found &amp; Proved (In Plain Terms)</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody id="discoveryProofTableBody"></tbody>
                </table>
              </div>
            </details>

            <!-- Dropdown 1.4: Real-World Peer-Reviewed Literature & ChEMBL BioAssays -->
            <details class="dossier-dropdown" id="literatureSection">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">1D. Real-World Scientific Papers &amp; Lab BioAssays Pulled Live (`OpenAlex`, `EuropePMC`, `arXiv`, `ChEMBL`)</span>
                    <span class="dropdown-sub-caption">Published papers, DOIs, and real experimental binding/rate numbers used to ground the model</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge1D">LIVE LITERATURE</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('1D. Real-World Scientific Papers & ChEMBL Lab Data', 'literatureTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout1D"></div>
                <div id="literatureConstantsBox" style="margin-bottom:10px;"></div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Database &amp; Year</th>
                      <th>Published Paper or Wet-Lab BioAssay</th>
                      <th>Clickable Link / DOI</th>
                      <th>Match Score</th>
                      <th>Real Lab Numbers Pulled &amp; What They Mean</th>
                    </tr>
                  </thead>
                  <tbody id="literatureTableBody"></tbody>
                </table>
              </div>
            </details>

            <!-- Dropdown 1.5: Competing Hypothesis Tournament -->
            <details class="dossier-dropdown" id="tournamentSection">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">1E. Competing Ideas Tested: Why Other Strategies Were Rejected</span>
                    <span class="dropdown-sub-caption">Every candidate strategy tested against 6 safety and stability rules</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge1E">6-RULE FILTER</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('1E. Competing Hypothesis Tournament', 'tournamentTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout1E"></div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Candidate Strategy</th>
                      <th>Verdict</th>
                      <th>Remaining Target Error</th>
                      <th>Side-Effect Spillover</th>
                      <th>Long-Term Stability</th>
                      <th>Plain-English Reason It Passed or Failed</th>
                    </tr>
                  </thead>
                  <tbody id="tournamentTableBody"></tbody>
                </table>
              </div>
            </details>

            <!-- Dropdown 1.6: 3-Generation Curiosity Chain -->
            <details class="dossier-dropdown" id="curiosityChainSection">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">1F. Self-Improving Question Chain: How the Engine Refined Its Own Equations</span>
                    <span class="dropdown-sub-caption">3 rounds of automatic follow-up questions that improved future prediction accuracy</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge1F">3 ROUNDS</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('1F. 3-Generation Self-Improving Question Chain', 'curiosityChainTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout1F"></div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Round</th>
                      <th>Follow-Up Question the Engine Asked Itself</th>
                      <th>Where It Looked</th>
                      <th>Prediction Accuracy &amp; What Improved</th>
                    </tr>
                  </thead>
                  <tbody id="curiosityChainTableBody"></tbody>
                </table>
                <div id="curiosityNextQuestionsBox" style="margin-top:10px;"></div>
              </div>
            </details>

            <!-- Dropdown 1.7: Council Step-by-Step Decision Ledger -->
            <details class="dossier-dropdown">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">1G. Step-by-Step Scientific Council Decision Log</span>
                    <span class="dropdown-sub-caption">Why each mathematical tool was chosen in sequence and what it found</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge1G">DECISION LOG</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('1G. Scientific Council Step-by-Step Decisions', 'councilStepsTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout1G"></div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Step</th>
                      <th>Agent Role</th>
                      <th>Action Taken</th>
                      <th>Plain-English Reason &amp; Result</th>
                    </tr>
                  </thead>
                  <tbody id="councilStepsTableBody"></tbody>
                </table>
              </div>
            </details>
          </div>
        </div>

        <!-- ================================================================= -->
        <!-- TAB 2: EQUATIONS & MATH PROOFS                                    -->
        <!-- ================================================================= -->
        <div id="tab-equations" class="tab-pane">
          <div class="empty-state" id="equationsEmpty">
            Discovered equations, control laws, conserved quantities, and physics verification checks will appear here.
          </div>
          <div id="equationsContent" style="display:none;">
            <div class="tab-takeaway-banner" id="tab2TakeawayBanner"></div>

            <div class="tab-controls-bar">
              <span><strong>Tip:</strong> Every equation and proof below includes a plain-English explanation of what the symbols mean.</span>
              <div style="display:flex;gap:6px;">
                <button type="button" class="tab-control-btn" onclick="window.toggleTabDropdowns('tab-equations', true)">Expand All in Tab</button>
                <button type="button" class="tab-control-btn" onclick="window.toggleTabDropdowns('tab-equations', false)">Collapse All</button>
              </div>
            </div>

            <!-- Dropdown 2.1: Discovered Equations + Variable Dictionary (Open by default) -->
            <details class="dossier-dropdown" open>
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">2A. The Discovered Equations &amp; Plain-English Variable Guide</span>
                    <span class="dropdown-sub-caption">Exact mathematical formulas governing the system, plus a plain-English translation of every variable</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge2A">EQUATIONS</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('2A. Discovered Equations & Variable Guide', 'equationsBoxList')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout2A"></div>
                <div id="equationsVariableGuideBox" style="margin-bottom:14px;"></div>
                <div id="equationsBoxList"></div>
              </div>
            </details>

            <!-- Dropdown 2.2: Conserved Quantities & Formal Computer Proofs (Lean 4 / SMT-LIB2) -->
            <details class="dossier-dropdown">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">2B. Formal Computer-Checked Math Proofs (`Lean 4` &amp; `SMT-LIB2`)</span>
                    <span class="dropdown-sub-caption">5-step deductive theorem proving the system cannot blow up or violate conservation laws</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge2B">PROVED UNSAT</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('2B. Formal Computer-Checked Math Proofs (Lean 4 & SMT-LIB2)', 'invariantsAndArchetypeBox')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout2B"></div>
                <div id="invariantsAndArchetypeBox"></div>
              </div>
            </details>

            <!-- Dropdown 2.3: 21-Point Physics & Stress-Test Checklist -->
            <details class="dossier-dropdown">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">2C. All 21 Physics, Stability &amp; Stress-Test Checks (Explained Item-by-Item)</span>
                    <span class="dropdown-sub-caption">Units, conservation of energy, noise resilience, 4x time horizon, interval boxes, and 3D spatial diffusion</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge2C">21/21 PASSED</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('2C. 21-Point Physics & Stress-Test Checklist', 'shieldsTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout2C"></div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Check Name</th>
                      <th>Status</th>
                      <th>Measured Value</th>
                      <th>What This Means in Plain English</th>
                    </tr>
                  </thead>
                  <tbody id="shieldsTableBody"></tbody>
                </table>
              </div>
            </details>
          </div>
        </div>

        <!-- ================================================================= -->
        <!-- TAB 3: BEFORE/AFTER CURVES & SAFETY                               -->
        <!-- ================================================================= -->
        <div id="tab-trajectories" class="tab-pane">
          <div class="empty-state" id="trajectoriesEmpty">
            State trajectories over time and neighboring state safety checks will appear here.
          </div>
          <div id="trajectoriesContent" style="display:none;">
            <div class="tab-takeaway-banner" id="tab3TakeawayBanner"></div>

            <div class="tab-controls-bar">
              <span><strong>Tip:</strong> Open any dropdown below to inspect the time-series curves, side-effect shield, ±25% real-world drift test, and blind future forecast.</span>
              <div style="display:flex;gap:6px;">
                <button type="button" class="tab-control-btn" onclick="window.toggleTabDropdowns('tab-trajectories', true)">Expand All in Tab</button>
                <button type="button" class="tab-control-btn" onclick="window.toggleTabDropdowns('tab-trajectories', false)">Collapse All</button>
              </div>
            </div>

            <!-- Dropdown 3.1: Coupled State Trajectories Over Time (Open by default) -->
            <details class="dossier-dropdown" open>
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">3A. How Every Part of the System Changes Over Time (Visual Plot)</span>
                    <span class="dropdown-sub-caption">Time-series curves showing the primary target settling to its goal while healthy parts stay flat and safe</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge3A">TIME CURVES</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('3A. Time-Series Curves & State Trajectories', 'trajectorySvgContainer')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout3A"></div>
                <div id="trajectorySvgContainer"></div>
              </div>
            </details>

            <!-- Dropdown 3.2: Side-Effect & Neighboring State Safety Check (Open by default) -->
            <details class="dossier-dropdown" open>
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">3B. State-by-State Safety &amp; Side-Effect Check (Causal Blast Radius)</span>
                    <span class="dropdown-sub-caption">Start value, finish value, safe limits, and exact plain-English outcome for every compartment</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge3B">100% SAFE</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('3B. State-by-State Safety & Side-Effect Check', 'blastRadiusTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout3B"></div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Part of System (State)</th>
                      <th>Role</th>
                      <th>Start &rarr; Finish</th>
                      <th>Min / Max Reached</th>
                      <th>Safe Corridor</th>
                      <th>Safety Cushion</th>
                      <th>What Happened in Plain English</th>
                    </tr>
                  </thead>
                  <tbody id="blastRadiusTableBody"></tbody>
                </table>
              </div>
            </details>

            <!-- Dropdown 3.3: ±25% Mismatched-Reality Drift Test -->
            <details class="dossier-dropdown" id="mismatchedRealitySection">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">3C. Real-World Stress Test: What If the Real World Drifts by &plusmn;25%?</span>
                    <span class="dropdown-sub-caption">Proves the solution still works even if real-world patient/hardware conditions differ by 25% from the model</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge3C">&plusmn;25% DRIFT SAFE</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('3C. ±25% Real-World Drift Stress Test', 'mismatchedRealityCards')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout3C"></div>
                <div class="summary-card-grid" id="mismatchedRealityCards" style="margin-bottom:12px;"></div>
              </div>
            </details>

            <!-- Dropdown 3.4: Blind Future Prediction Test (30% -> 70%) -->
            <details class="dossier-dropdown" id="blindForecastSection">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">3D. Blind Future Prediction Test (Train on First 30% &rarr; Predict Unseen 70% Future)</span>
                    <span class="dropdown-sub-caption">Proves the discovered equation is a real physical law and not just memorizing past data</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge3D">BLIND FORECAST</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('3D. Blind 30% to 70% Future Prediction Test', 'blindForecastSummaryBox')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout3D"></div>
                <div class="summary-card-grid" id="blindForecastSummaryBox" style="margin-bottom:12px;"></div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Future Time Checkpoint</th>
                      <th>Actual State Values</th>
                      <th>Cortivra Blind Prediction (Error)</th>
                      <th>Standard Curve-Fit Baseline (Error)</th>
                    </tr>
                  </thead>
                  <tbody id="nextSpotForecastTableBody"></tbody>
                </table>
              </div>
            </details>

            <!-- Dropdown 3.5: 3D Spatial Diffusion & Cross-Stream Checks -->
            <details class="dossier-dropdown" id="frontierPillarsSection">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">3E. 3D Tissue/Space Diffusion &amp; New Shock-Event Generalization</span>
                    <span class="dropdown-sub-caption">Simulates how concentrations spread in 3D space (12&times;12&times;12 grid) and under sudden external surges</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass">3D SPATIAL PDE</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('3E. 3D Spatial Diffusion & Multi-Event Generalization', 'frontierPillarsGrid')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout3E"></div>
                <div class="summary-card-grid" id="frontierPillarsGrid"></div>
              </div>
            </details>
          </div>
        </div>

        <!-- ================================================================= -->
        <!-- TAB 4: MOLECULE, RECIPE & MATERIAL SPECS                          -->
        <!-- ================================================================= -->
        <div id="tab-blueprint" class="tab-pane">
          <div class="empty-state" id="blueprintEmpty">
            Computed target specifications for control levers, candidate molecules, lab synthesis recipe, and material properties will appear here.
          </div>
          <div id="blueprintContent" style="display:none;">
            <div class="tab-takeaway-banner" id="tab4TakeawayBanner"></div>

            <div class="tab-controls-bar">
              <span><strong>Tip:</strong> Each dropdown below translates a stage of the physical design (Levers &rarr; Molecule &amp; 3D Lattice &rarr; Lab Recipe &rarr; Material Specs) into plain English.</span>
              <div style="display:flex;gap:6px;">
                <button type="button" class="tab-control-btn" onclick="window.toggleTabDropdowns('tab-blueprint', true)">Expand All in Tab</button>
                <button type="button" class="tab-control-btn" onclick="window.toggleTabDropdowns('tab-blueprint', false)">Collapse All</button>
              </div>
            </div>

            <!-- Dropdown 4.1: Stage 3 Molecule & 3D Lattice (Open by default — users love seeing the molecule first!) -->
            <details class="dossier-dropdown" id="pass3MatterLatticeSection" open>
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">4A. Designed Molecule (`SMILES`), 3D Atomic Structure &amp; 3D Support Lattice (Stage 3)</span>
                    <span class="dropdown-sub-caption">Top-3 molecules selected from 700 candidates + interactive 3D atoms + microscopic 3D scaffold</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge4A">TOP MOLECULE</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('4A. Designed Molecule (SMILES) & 3D Support Lattice', 'pass3MolecularTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout4A"></div>
                <div style="display:grid;grid-template-columns:360px 1fr;gap:14px;align-items:stretch;margin-bottom:16px;">
                  <div class="canvas-box">
                    <div class="cad-toolbar">
                      <span style="color:#38bdf8;font-family:var(--font-mono);font-size:11px;font-weight:700;">3D ATOMIC VIEWPORT (pass3_molecule.xyz)</span>
                      <span style="color:#94a3b8;font-family:var(--font-mono);font-size:10.5px;">Drag: Rotate &middot; Scroll: Zoom</span>
                    </div>
                    <canvas id="mol3dCanvas" width="334" height="210" style="cursor:grab;border-radius:6px;"></canvas>
                    <div class="cad-legend" id="mol3dLegend">
                      <span><i class="cad-legend-dot" style="background:#38bdf8;"></i>C</span>
                      <span><i class="cad-legend-dot" style="background:#818cf8;"></i>N</span>
                      <span><i class="cad-legend-dot" style="background:#f43f5e;"></i>O</span>
                      <span><i class="cad-legend-dot" style="background:#34d399;"></i>F</span>
                      <span><i class="cad-legend-dot" style="background:#fbbf24;"></i>S</span>
                      <span><i class="cad-legend-dot" style="background:#fb923c;"></i>P</span>
                    </div>
                  </div>
                  <div class="svg-blueprint-container" id="pass3StructureSvgBox" style="overflow-x:auto;"></div>
                </div>

                <div style="font-size:12.5px;font-weight:800;color:var(--text-primary);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.03em;">
                  Top-3 Designed Molecules (Out of 700 Candidates Evaluated from First Principles)
                </div>
                <table class="data-table" style="margin-bottom:18px;">
                  <thead>
                    <tr>
                      <th>Rank &amp; Name</th>
                      <th>Chemical Code (`SMILES`) &amp; Formula</th>
                      <th>Fat/Water Balance (`XLogP` &amp; `TPSA`)</th>
                      <th>Lock-and-Key Grips (`HBD/HBA`)</th>
                      <th>Binding Strength (`&Delta;G` &amp; `K_d`)</th>
                      <th>Quality Score (`QED`)</th>
                      <th>What These Numbers Mean in Plain English</th>
                    </tr>
                  </thead>
                  <tbody id="pass3MolecularTableBody"></tbody>
                </table>

                <div style="font-size:12.5px;font-weight:800;color:var(--text-primary);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.03em;">
                  3D Microscopic Support Lattice Comparison (4 Topologies Tested)
                </div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>3D Lattice Shape</th>
                      <th>Solid vs. Open Pore Space</th>
                      <th>Cell Size / Wall / Pore</th>
                      <th>Stiffness &amp; Strength</th>
                      <th>Flex &amp; Heat Flow</th>
                      <th>Crush Safety</th>
                      <th>Plain-English Engineering Meaning</th>
                    </tr>
                  </thead>
                  <tbody id="pass3LatticeTableBody"></tbody>
                </table>
              </div>
            </details>

            <!-- Dropdown 4.2: Stage 1 Primary Control Levers (Open by default) -->
            <details class="dossier-dropdown" open>
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">4B. Which Physical Levers Were Adjusted &amp; How Much Energy Shift They Need (Stage 1)</span>
                    <span class="dropdown-sub-caption">Exact rates/parameters tuned to fix the target without disturbing healthy neighbors</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge4B">LEVERS TUNED</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('4B. Stage 1 Physical Control Levers & Energy Shifts', 'actuatorTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout4B"></div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Lever / Parameter</th>
                      <th>Role in System</th>
                      <th>Old Value &rarr; New Value</th>
                      <th>Multiplier</th>
                      <th>Required Energy Shift (&Delta;&Delta;G)</th>
                      <th>Split: Chemistry / Shape / Control</th>
                      <th>Plain-English Meaning of This Adjustment</th>
                    </tr>
                  </thead>
                  <tbody id="actuatorTableBody"></tbody>
                </table>
              </div>
            </details>

            <!-- Dropdown 4.3: Stage 2 Lab Synthesis & Manufacturing Recipe -->
            <details class="dossier-dropdown" id="pass2SynthesisSection">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">4C. 4-Step Lab Synthesis &amp; Manufacturing Recipe (Stage 2 Reactor Schedule)</span>
                    <span class="dropdown-sub-caption">Exact temperatures, time windows, feed rates, and purity shields to make the molecule/material</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass" id="badge4C">LAB RECIPE</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('4C. Stage 2 Lab Synthesis & Manufacturing Recipe', 'pass2ScheduleTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout4C"></div>
                <div class="summary-card-grid" id="pass2SummaryBanner" style="margin-bottom:12px;"></div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Recipe Step</th>
                      <th>Time Window</th>
                      <th>Temperature</th>
                      <th>Feed, Catalyst &amp; Mixing</th>
                      <th>Expected Yield &amp; Max Impurity</th>
                      <th>Step-by-Step Plain-English Instruction</th>
                    </tr>
                  </thead>
                  <tbody id="pass2ScheduleTableBody"></tbody>
                </table>
              </div>
            </details>

            <!-- Dropdown 4.4: Safe Operating Area & Failure Mode Analysis (FMECA) -->
            <details class="dossier-dropdown" id="fmecaSoaSection">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">4D. Safety Limits, Sensitivity Ranking &amp; Automatic Failure Guards (`FMECA`)</span>
                    <span class="dropdown-sub-caption">Which parameters matter most, safe temperature windows, and automatic safety backups</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass">SAFETY GUARDS</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('4D. Safety Limits & Failure Mode Analysis (FMECA)', 'fmecaTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout4D"></div>
                <div class="summary-card-grid" id="soaSummaryCards" style="margin-bottom:12px;"></div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Check ID</th>
                      <th>Parameter</th>
                      <th>Direct Impact (`S_i`)</th>
                      <th>Total Impact (`S_Ti`)</th>
                      <th>Risk Score (`RPN`)</th>
                      <th>Automatic Safety Guard &amp; Plain-English Meaning</th>
                    </tr>
                  </thead>
                  <tbody id="fmecaTableBody"></tbody>
                </table>
              </div>
            </details>

            <!-- Dropdown 4.5: Required Material & Chemical Properties -->
            <details class="dossier-dropdown">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">4E. Compartment Material &amp; Chemical Property Targets</span>
                    <span class="dropdown-sub-caption">Equilibrium ratios, stiffness, strength limits, and heat conductivity for each compartment</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass">MATERIAL SPECS</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('4E. Material & Chemical Property Targets', 'formulationTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout4E"></div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Compartment</th>
                      <th>Share of System</th>
                      <th>Binding / Balance Ratios (`K_eq` / `K_d`)</th>
                      <th>Stiffness (Modulus)</th>
                      <th>Strength Limit</th>
                      <th>Heat Conductivity &amp; Plain Meaning</th>
                    </tr>
                  </thead>
                  <tbody id="formulationTableBody"></tbody>
                </table>
              </div>
            </details>

            <!-- Dropdown 4.6: Timescales & Dimensionless Numbers -->
            <details class="dossier-dropdown">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">4F. Response Speeds, Natural Frequencies &amp; Scale-Free Ratios</span>
                    <span class="dropdown-sub-caption">How fast each part reacts and whether it settles smoothly or oscillates</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass">TIMESCALE RATIOS</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('4F. Response Speeds & Scale-Free Ratios', 'buckinghamTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout4F"></div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Compartment</th>
                      <th>Reaction Time &tau; (s)</th>
                      <th>Natural Frequency (Hz)</th>
                      <th>Reaction-vs-Transport Speed (`Damk&ouml;hler`)</th>
                      <th>Damping Smoothness (`&zeta;`)</th>
                      <th>Plain-English Behavior</th>
                    </tr>
                  </thead>
                  <tbody id="buckinghamTableBody"></tbody>
                </table>
              </div>
            </details>
          </div>
        </div>

        <!-- ================================================================= -->
        <!-- TAB 5: 3D CAD, CONTROLLER CODE & LAB GUIDE                        -->
        <!-- ================================================================= -->
        <div id="tab-hardware" class="tab-pane">
          <div class="empty-state" id="hardwareEmpty">
            Interactive 3D solid geometry preview, engineering blueprint diagrams, C99 controller code, and runnable Python simulator code will appear here.
          </div>
          <div id="hardwareContent" style="display:none;">
            <div class="tab-takeaway-banner" id="tab5TakeawayBanner"></div>

            <div class="tab-controls-bar">
              <span><strong>Tip:</strong> Drag the 3D viewport to rotate the vessel assembly, or open the dropdowns below to inspect the exact dimensions, C99 code, Python simulator, and Lab SOP.</span>
              <div style="display:flex;gap:6px;">
                <button type="button" class="tab-control-btn" onclick="window.toggleTabDropdowns('tab-hardware', true)">Expand All in Tab</button>
                <button type="button" class="tab-control-btn" onclick="window.toggleTabDropdowns('tab-hardware', false)">Collapse All</button>
              </div>
            </div>

            <!-- Dropdown 5.1: Interactive 3D CAD & Blueprint (Open by default) -->
            <details class="dossier-dropdown" open>
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">5A. Interactive 3D Multi-Compartment CAD Model &amp; Engineering Blueprint</span>
                    <span class="dropdown-sub-caption">3D printable solid geometry (`.stl` / `.obj`) showing primary vessel, regulator, protective shield, and conduits</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass">3D CAD READY</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('5A. 3D Multi-Compartment CAD Model & Blueprint', 'cadSpecsTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout5A"></div>
                <div style="display:grid;grid-template-columns:450px 1fr;gap:16px;align-items:stretch;">
                  <div class="canvas-box">
                    <div class="cad-toolbar">
                      <div style="display:flex;gap:5px;">
                        <button type="button" class="cad-btn active" id="cadBtnSpin" onclick="window.cortivraCadControl && window.cortivraCadControl('spin')">Auto-Spin: ON</button>
                        <button type="button" class="cad-btn" id="cadBtnMode" onclick="window.cortivraCadControl && window.cortivraCadControl('mode')">Mode: Shaded CAD</button>
                      </div>
                      <div style="display:flex;gap:4px;">
                        <button type="button" class="cad-btn" onclick="window.cortivraCadControl && window.cortivraCadControl('zoomIn')">+</button>
                        <button type="button" class="cad-btn" onclick="window.cortivraCadControl && window.cortivraCadControl('zoomOut')">&minus;</button>
                        <button type="button" class="cad-btn" onclick="window.cortivraCadControl && window.cortivraCadControl('reset')">Reset</button>
                      </div>
                    </div>
                    <canvas id="stlCanvas" width="424" height="310" style="cursor:grab;border-radius:6px;"></canvas>
                    <div class="cad-legend">
                      <span><i class="cad-legend-dot" style="background:#38bdf8;"></i>Primary Target Vessel</span>
                      <span><i class="cad-legend-dot" style="background:#10b981;"></i>Regulator</span>
                      <span><i class="cad-legend-dot" style="background:#a855f7;"></i>Protected Neighbor Shield</span>
                      <span><i class="cad-legend-dot" style="background:#f59e0b;"></i>Connecting Channel</span>
                    </div>
                  </div>
                  <div class="svg-blueprint-container" id="blueprintSvgBox" style="overflow-x:auto;"></div>
                </div>
              </div>
            </details>

            <!-- Dropdown 5.2: Exact Vessel Dimensions & Machining Tolerances -->
            <details class="dossier-dropdown" id="cadSpecsTableSection" open>
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">5B. Exact 3D Vessel Dimensions, Wall Thicknesses &amp; Manufacturing Tolerances</span>
                    <span class="dropdown-sub-caption">Physical dimensions computed from diffusion lengths and pressure safety laws</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass">DIMENSIONS</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('5B. 3D Vessel Dimensions & Wall Thicknesses', 'cadSpecsTableBody')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout5B"></div>
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Vessel Name</th>
                      <th>Outer Radius (mm)</th>
                      <th>Diffusion Distance (&mu;m)</th>
                      <th>Wall Thickness (&mu;m)</th>
                      <th>Channel Diameter (&mu;m)</th>
                      <th>Surface-to-Volume Ratio</th>
                      <th>Machining Tolerance &amp; Meaning</th>
                    </tr>
                  </thead>
                  <tbody id="cadSpecsTableBody"></tbody>
                </table>
              </div>
            </details>

            <!-- Dropdown 5.3: 4-Stage Hardware Build Protocol -->
            <details class="dossier-dropdown" id="fabricationStepsSection">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">5C. 4-Step Hardware Build &amp; Testing Checklist</span>
                    <span class="dropdown-sub-caption">How to machine, assemble, calibrate, and test the physical prototype</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass">BUILD STEPS</span>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="summary-card-grid" id="fabricationStepsGrid"></div>
              </div>
            </details>

            <!-- Dropdown 5.4: Real-Time ANSI C99 Controller Code -->
            <details class="dossier-dropdown">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">5D. Embedded Real-Time C99 Controller Firmware (`controller_firmware.h` &amp; `.c`)</span>
                    <span class="dropdown-sub-caption">Zero-allocation C code ready to compile onto a microcontroller or lab pump controller</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass">ANSI C99</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('5D. Embedded ANSI C99 Controller Firmware', 'c99FirmwareBox')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout5D"></div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                  <div>
                    <div style="font-size:11.5px;font-weight:700;color:var(--text-muted);margin-bottom:4px;font-family:var(--font-mono);">deliverables/controller_firmware.h</div>
                    <pre class="code-viewer" id="c99HeaderBox"></pre>
                  </div>
                  <div>
                    <div style="font-size:11.5px;font-weight:700;color:var(--text-muted);margin-bottom:4px;font-family:var(--font-mono);">deliverables/controller_firmware.c</div>
                    <pre class="code-viewer" id="c99FirmwareBox"></pre>
                  </div>
                </div>
              </div>
            </details>

            <!-- Dropdown 5.5: Standalone Python Digital-Twin Simulator -->
            <details class="dossier-dropdown">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">5E. Standalone Python Digital-Twin Simulator (`digital_twin_simulator.py`)</span>
                    <span class="dropdown-sub-caption">Self-contained Python script anyone can run on their laptop to verify the math independently</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass">RUNNABLE PYTHON</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('5E. Standalone Python Digital-Twin Simulator', 'pythonSimBox')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout5E"></div>
                <pre class="code-viewer" id="pythonSimBox"></pre>
              </div>
            </details>

            <!-- Dropdown 5.6: Wet-Lab & Manufacturing SOP -->
            <details class="dossier-dropdown" id="sopMarkdownSection">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">5F. Complete Wet-Lab &amp; Manufacturing Standard Operating Procedure (`wetlab_and_manufacturing_sop.md`)</span>
                    <span class="dropdown-sub-caption">Step-by-step benchtop manual for a scientist or engineer to reproduce the result</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass">LAB MANUAL</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('5F. Wet-Lab & Manufacturing Standard Operating Procedure', 'sopMarkdownBox')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div class="plain-english-callout" id="callout5F"></div>
                <pre class="code-viewer" id="sopMarkdownBox"></pre>
              </div>
            </details>
          </div>
        </div>

        <!-- ================================================================= -->
        <!-- TAB 6: ALL 20 FILES & AUDIT LOGS                                  -->
        <!-- ================================================================= -->
        <div id="tab-logs" class="tab-pane">
          <div class="empty-state" id="logsEmpty">
            Downloadable engineering files and the step-by-step execution flight recorder log will appear here.
          </div>
          <div id="logsContent" style="display:none;">
            <div class="tab-takeaway-banner" id="tab6TakeawayBanner"></div>

            <!-- Dropdown 6.1: All 20 Generated Deliverable Files (Open by default) -->
            <details class="dossier-dropdown" open>
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">6A. All 20 Downloadable Deliverable Files (With Plain-English File Guide)</span>
                    <span class="dropdown-sub-caption">Click "Inspect Inline" on any file to view it right here, or "Open Raw" to download</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass">20 FILES READY</span>
                  <button type="button" class="ask-ai-pill" onclick="event.stopPropagation(); window.askAiAboutSection('6A. 20 Generated Deliverable Files', 'deliverablesGrid')">&#128172; Ask AI</button>
                </div>
              </summary>
              <div class="dropdown-body">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:10px;">
                  <div id="primaryDeliverableQuickBar" style="display:flex;gap:8px;flex-wrap:wrap;"></div>
                </div>
                <div class="filter-pills-bar" id="deliverableCategoryFilterBar" style="margin-bottom:12px;border:1px solid var(--border-light);border-radius:6px;">
                  <button type="button" class="filter-pill active" onclick="filterDeliverablesByCategory('all', this)">All (20)</button>
                  <button type="button" class="filter-pill" onclick="filterDeliverablesByCategory('Report & Data', this)">Report &amp; Data (3)</button>
                  <button type="button" class="filter-pill" onclick="filterDeliverablesByCategory('Formal Proof', this)">Formal Proof (2)</button>
                  <button type="button" class="filter-pill" onclick="filterDeliverablesByCategory('3D CAD & Matter', this)">3D CAD &amp; Matter (6)</button>
                  <button type="button" class="filter-pill" onclick="filterDeliverablesByCategory('Code & Simulator', this)">Code &amp; Simulator (3)</button>
                  <button type="button" class="filter-pill" onclick="filterDeliverablesByCategory('SOP & Audit Ledger', this)">SOP &amp; Audit Ledger (6)</button>
                </div>
                <div class="deliverable-grid" id="deliverablesGrid"></div>
              </div>
            </details>

            <!-- Inline File Inspector -->
            <div class="section-block" id="inlineDeliverablePreviewSection" style="display:none;margin-bottom:14px;">
              <h3 class="section-heading">
                <span id="inlineDeliverablePreviewTitle">Inline Deliverable File Inspector</span>
                <span style="display:inline-flex;gap:10px;align-items:center;">
                  <button type="button" id="inlineDeliverableCopyBtn" class="cad-btn" style="background:var(--bg-subtle);color:var(--text-primary);border-color:var(--border-strong);" onclick="window.copyDeliverableInline && window.copyDeliverableInline()">Copy Content</button>
                  <a id="inlineDeliverableOpenLink" href="#" target="_blank" rel="noopener" style="font-size:12px;color:var(--accent-blue);text-decoration:none;font-family:var(--font-mono);">Open Raw File &rarr;</a>
                </span>
              </h3>
              <div id="inlineDeliverableVisualBox" class="svg-blueprint-container" style="display:none;margin-bottom:12px;padding:12px;background:#ffffff;border:1px solid var(--border-light);border-radius:8px;overflow-x:auto;"></div>
              <pre class="code-viewer" id="inlineDeliverablePreviewBox"></pre>
            </div>

            <!-- Dropdown 6.2: Flight Recorder Log -->
            <details class="dossier-dropdown">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">6B. Complete Step-by-Step Flight Recorder Log (`execution_flight_recorder.log`)</span>
                    <span class="dropdown-sub-caption">Chronological execution trace of all 8 stages</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass">FLIGHT LOG</span>
                </div>
              </summary>
              <div class="dropdown-body">
                <pre class="code-viewer" id="flightRecorderLogBox"></pre>
              </div>
            </details>

            <!-- Dropdown 6.3: Cryptographic Provenance & Forensic Report -->
            <details class="dossier-dropdown">
              <summary class="dropdown-header">
                <div class="dropdown-title-group">
                  <span class="dropdown-chevron">&#9654;</span>
                  <div class="dropdown-title-text">
                    <span class="dropdown-main-title">6C. Full Markdown Scientific Report &amp; SHA-256 Audit Trail (`dossier.md`)</span>
                    <span class="dropdown-sub-caption">Tamper-evident SHA-256 hash chain proving every step was computed live</span>
                  </div>
                </div>
                <div class="dropdown-right-controls">
                  <span class="badge badge-pass">SHA-256 VERIFIED</span>
                </div>
              </summary>
              <div class="dropdown-body">
                <pre class="code-viewer" id="forensicXrayBox"></pre>
              </div>
            </details>
          </div>
        </div>

      </section>
    </div>
  </main>

  <!-- Floating Button + Slide-Out Co-Scientist AI Chat Drawer -->
  <button type="button" class="chat-fab-btn" onclick="window.openCoScientistChat && window.openCoScientistChat('Entire Discovery Run')">
    <span>&#128172; Ask Co-Scientist AI</span>
  </button>

  <aside class="chat-drawer" id="coScientistDrawer">
    <div class="chat-drawer-header">
      <div>
        <div class="chat-drawer-title">&#128172; Cortivra Co-Scientist AI</div>
        <div class="chat-drawer-sub">Ask anything about this run, any molecule, equation, or section in plain English</div>
      </div>
      <button type="button" class="chat-close-btn" onclick="window.closeCoScientistChat && window.closeCoScientistChat()">Close &times;</button>
    </div>
    <div class="chat-context-banner" id="chatContextBanner">
      <span id="chatContextLabel">Focused on: Entire Discovery Run</span>
      <button type="button" style="background:none;border:none;color:#1d4ed8;font-size:11px;font-weight:700;cursor:pointer;" onclick="window.resetChatFocus && window.resetChatFocus()">Reset Focus</button>
    </div>
    <div class="chat-quick-chips">
      <button type="button" class="chat-quick-chip" onclick="window.sendQuickChatQuestion('Explain what this run discovered in 2 simple sentences for a non-scientist.')">Explain in 2 simple sentences</button>
      <button type="button" class="chat-quick-chip" onclick="window.sendQuickChatQuestion('What exact molecule and 3D structure did it design, and what do XLogP, TPSA, and Kd mean here?')">Explain the Molecule &amp; 3D Structure</button>
      <button type="button" class="chat-quick-chip" onclick="window.sendQuickChatQuestion('How did the engine fix the primary target without causing side effects to healthy neighbors?')">How did it prevent side effects?</button>
      <button type="button" class="chat-quick-chip" onclick="window.sendQuickChatQuestion('What real-world scientific papers and ChEMBL lab data did it pull, and how do we know nothing is hardcoded?')">Prove this is real &amp; not hardcoded</button>
    </div>
    <div class="chat-messages" id="chatMessagesBox">
      <div class="chat-bubble assistant">
        <strong>Hi! I'm your Cortivra Co-Scientist.</strong><br/>
        I have the complete mathematical dossier of the run on screen loaded right now. Click <strong>&#128172; Ask AI</strong> on any dropdown section, click a quick question above, or ask me anything about what the engine found!
      </div>
    </div>
    <div class="chat-input-area">
      <textarea
        id="chatInputBox"
        class="chat-input"
        placeholder="Ask any question about this run, molecule, equation, or chart..."
        onkeydown="if(event.key==='Enter' && !event.shiftKey){ event.preventDefault(); window.submitCoScientistChat(); }"
      ></textarea>
      <button type="button" id="chatSendBtn" class="chat-send-btn" onclick="window.submitCoScientistChat()">Ask</button>
    </div>
  </aside>

<!-- __WORKBENCH_FRONTEND_JS__ -->
</body>
</html>
"""
