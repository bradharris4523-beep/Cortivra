# Security, Physical Safety & Intellectual Property Policy

## 1. Zero-Secret & Credential Isolation Architecture
* **Strict Environment Isolation:** Cortivra never stores API keys, tokens, or endpoint credentials in source code. All credentials (`BASETEN_API_KEY`, `FIREWORKS_API_KEY`, `META_API_KEY`, `LOCAL_LLM_BASE_URL`) are loaded strictly at runtime from `.env` (which is blocked by `.gitignore`).
* **Placeholder Blocking:** `cortivra/layer6_agents/meta_clients.py` actively rejects placeholder keys and unconfigured endpoints before any socket request is constructed.

## 2. Web Workbench Hardening (`cortivra/ui_server.py`)
The local HTTP/SSE workbench server enforces strict defense-in-depth controls:
* **Localhost Binding by Default:** Binds strictly to `127.0.0.1` unless explicitly overridden.
* **Path-Traversal & Allowlist Protection:** Every `/api/deliverables/...` and `/api/runs/...` request is canonicalized via `Path.resolve()`, verified with `.is_relative_to(base_dir)`, stripped of null bytes and `..` segments, and restricted to an explicit 14-extension allowlist (`.md`, `.html`, `.json`, `.log`, `.jsonl`, `.stl`, `.obj`, `.svg`, `.c`, `.h`, `.py`, `.lean`, `.smt2`, `.xyz`).
* **Strict Payload & Upload Caps:** Enforces `MAX_POST_BYTES = 2 MB`, `MAX_DATA_BYTES = 1.5 MB`, and `MAX_PROMPT_CHARS = 4000`.
* **HTTP Security Headers:** Every response emits `Content-Security-Policy: default-src 'self' ...`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Referrer-Policy: no-referrer`.

## 3. Physical Safety & Non-Clinical Simulation Boundary
* **Deterministic Physics Guardrails:** Every candidate equation or control intervention proposed by an AI model must pass Layer 1's 7-dimensional SI unit checker, First-Law energy conservation, Second-Law non-negative entropy production, homeostatic corridor bounds, and outward-rounded IEEE-754 interval stability proofs before it is accepted.
* **In-Silico R&D Scope:** Cortivra is an in-silico mathematical physics, materials, and dynamical-systems simulation workbench for engineers and physical scientists. It does not generate clinical or medical advice, and it enforces homeostatic and thermal safety bounds to prevent runaway or hazardous operating regimes.

## 4. Cryptographic Provenance & Anti-Tampering (`cortivra/observability/`)
Every step in a Cortivra run is recorded in a `CausalTraceDAG` where each node's SHA-256 hash chains its parent hashes with the exact symbolic expression, operator spectrum, and verification metrics (`verify_merkle_integrity()`). Any modification to a run log or deliverable immediately invalidates the Merkle root hash.
