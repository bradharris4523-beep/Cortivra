# Cortivra

**Turn a one-sentence engineering question or raw sensor CSV into verified differential equations, zero-side-effect control laws, formal computer proofs, 3D CAD hardware, and runnable C99 firmware.**

> **Built for the Meta Global AI Developer Hackathon (`Push the Frontier` Track | Oct 27 – Nov 4, 2026)**
>
> *Track Goal: "Use Meta's models to tackle challenges in software, robotics, manufacturing, energy, materials and beyond — unlock a better way to understand and control a physical system."*

---

## 1. The Goal (In Plain English)

When engineers and scientists try to fix or control a real physical system—like stopping lithium dendrites in a fast-charging battery, stabilizing plasma in a fusion reactor, optimizing a chemical reactor, or clearing toxic proteins in a biophysical model—they hit **two walls**:

1. **Everything is connected (The Side-Effect Problem):** Pushing a single lever to fix your main target almost always spills over and damages healthy neighboring parts of the system.
2. **Standard AI models guess text instead of obeying physics:** If you ask a standard language model for equations or control settings, it can invent numbers that violate conservation of energy, mix up physical units (like adding meters to seconds), or blow up over time.

**Cortivra was built to solve both problems.**

Instead of letting an AI guess the final answer in plain text, Cortivra pairs **Meta's Muse AI models** with a **6-layer deterministic physics and math compiler**:
* **Meta's AI models** translate your question into a structured physical hypothesis and choose which mathematical tools to run next.
* **Cortivra's math engine** checks every equation against **7-dimensional SI units** ($\text{kg, m, s, A, K, mol, cd}$) and the **First and Second Laws of Thermodynamics**. If the AI makes a unit or energy mistake, the compiler catches the exact error and forces the AI to repair the equation until the math is 100% sound.
* Once the equations are verified, Cortivra solves for a **coordinated combination of levers** that fixes the primary target while mathematically canceling out spillover into neighboring parts—and compiles the solution into **20 real engineering files** (formal proofs, 3D printable `.stl`/`.obj` CAD, `.svg` blueprints, compilable `C99` microcontroller code, a 4-stage reactor recipe, candidate molecular structures, and a standalone Python simulator).

> *Simulation & Safety Scope: Cortivra is an in-silico mathematical physics, materials, and dynamical-systems simulation and verification workbench for R&D engineers and physical scientists. It does not generate or provide clinical or medical advice.*

---

## 2. How One Investigation Works (Step by Step)

```mermaid
flowchart LR
    A["1. Your Question or Sensor CSV"] --> B["2. Meta Muse 4-Agent Council"]
    B --> C["3. 7D SI Unit & Energy Compiler"]
    C --> D["4. Equation Discovery & Blind 30%->70% Test"]
    D --> E["5. Zero-Side-Effect Control Solver"]
    E --> F["6. Computer Proofs & 20 Engineering Files"]
```

1. **You ask a question or upload sensor data:** Type any physical, materials, energy, or engineering question into the Web Workbench or CLI—or upload a `.csv` of time-series measurements.
2. **Equation Compilation & Self-Repair ([`layer3_compiler/`](cortivra/layer3_compiler/symbolic_compiler.py)):** `Muse Wonder` proposes a coupled multi-compartment model. `Muse Spark` and the Layer 3 compiler check that every term has matching SI units, conserves energy, and produces non-negative entropy.
3. **Blind Future Prediction Test ([`universal_telemetry_hunter.py`](cortivra/layer6_agents/universal_telemetry_hunter.py)):** To prove the engine isn't just curve-fitting, Cortivra trains on only the **first 30%** of the time steps and predicts the **unseen 70% future** blind—comparing its discovered equations head-to-head against standard polynomial curve-fitting.
4. **Solving Control Without Side Effects ([`intervention_and_control.py`](cortivra/layer4_engine/intervention_and_control.py)):** Cortivra measures how every tunable lever affects the healthy neighboring parts of the system, then projects its control adjustments into the **nullspace** (the mathematical "blind spot" of the neighbors). The main problem is fixed while neighboring parts stay flat inside their safe operating limits—even when the real world drifts by **$\pm 25\%$**.
5. **Computer-Checked Proofs ([`rigorous_proofs.py`](cortivra/layer1_axioms/rigorous_proofs.py), [`lean4_smt_compiler.py`](cortivra/layer1_axioms/lean4_smt_compiler.py)):** Cortivra proves the controlled system cannot spiral out of control using outward-rounded IEEE-754 interval arithmetic (`math.nextafter`), Sum-of-Squares (SOS) energy decay, and exports machine-checkable **Lean 4** (`formal_theorem.lean`) and **Z3 SMT-LIB2** (`formal_certificate.smt2`) proof scripts.
6. **3-Stage Physical Realization & 20 Exported Files ([`deliverable_exporter.py`](cortivra/layer6_agents/deliverable_exporter.py)):**
   * **Stage 1 (Hardware & Firmware):** 3D printable `.stl`/`.obj` multi-vessel CAD meshes, 3-view `.svg` engineering blueprint, and compilable ANSI C99 controller code (`controller_firmware.c`).
   * **Stage 2 (Manufacturing Recipe):** A 5-compartment reactor simulation that solves for a 4-stage temperature, feed, catalyst, and quench schedule ([`recursive_synthesis_engine.py`](cortivra/layer6_agents/recursive_synthesis_engine.py)).
   * **Stage 3 (Molecule & 3D Porous Lattice):** Screens 700 candidate molecular building-block combinations to output Top-3 `SMILES` structures + 3D atomic `.xyz` coordinates, paired with a 3D cellular metamaterial lattice ([`pass3_matter_and_lattice_compiler.py`](cortivra/layer6_agents/pass3_matter_and_lattice_compiler.py)).

---

## 3. Judge's Proof Map: Click Any Claim to Inspect the Code

Every capability in Cortivra was built from scratch during the hackathon window (**Oct 27 – Nov 4, 2026**) in pure Python + PyTorch. Nothing is a mock or static template—click any module below to inspect the math directly:

| What Cortivra Does | Plain-English Explanation | Exact Code File to Inspect | Verification Test Suite |
| :--- | :--- | :--- | :--- |
| **1. 7D SI Unit & Scaling Engine** | Tracks $[\text{kg, m, s, A, K, mol, cd}]$ on every variable and extracts dimensionless scaling ratios ($\text{Da, Pe, Bi}$) via SVD. | [`cortivra/layer1_axioms/units.py`](cortivra/layer1_axioms/units.py)<br>[`buckingham_pi_scaling.py`](cortivra/layer1_axioms/buckingham_pi_scaling.py) | [`tests/test_layer1_axioms.py`](tests/test_layer1_axioms.py) |
| **2. Physics & Quantum Shields** | Enforces 1st-Law energy balance ($J = -J^\top$), 2nd-Law dissipation ($R \succeq 0$), and GKSL quantum coherence ($T_2 \le 2T_1$). | [`cortivra/layer1_axioms/hamiltonian.py`](cortivra/layer1_axioms/hamiltonian.py)<br>[`cortivra/layer1_axioms/shields.py`](cortivra/layer1_axioms/shields.py) | [`tests/test_layer1_axioms.py`](tests/test_layer1_axioms.py) |
| **3. Interval, SOS & Lean 4 / SMT-LIB2 Prover** | Bounds continuous state boxes with IEEE-754 outward rounding (`math.nextafter`) and emits `.lean` + `.smt2` proof files. | [`cortivra/layer1_axioms/rigorous_proofs.py`](cortivra/layer1_axioms/rigorous_proofs.py)<br>[`lean4_smt_compiler.py`](cortivra/layer1_axioms/lean4_smt_compiler.py) | [`tests/test_live_observatory_and_skeptic_proofs.py`](tests/test_live_observatory_and_skeptic_proofs.py) |
| **4. Symbolic AST & 3D PDE Compiler** | Parses math strings into an AST, takes symbolic derivatives $\nabla H$, checks units, and discretizes 1D/2D/3D PDEs. | [`cortivra/layer3_compiler/ast_nodes.py`](cortivra/layer3_compiler/ast_nodes.py)<br>[`symbolic_compiler.py`](cortivra/layer3_compiler/symbolic_compiler.py) | [`tests/test_layer3_compiler.py`](tests/test_layer3_compiler.py) |
| **5. Zero-Side-Effect Nullspace Control** | Projects parameter updates into the SVD nullspace of bystander sensitivities ($P_\perp = I - V W V^\top$) so neighbors stay safe. | [`cortivra/layer4_engine/intervention_and_control.py`](cortivra/layer4_engine/intervention_and_control.py)<br>[`inverse_solver.py`](cortivra/layer4_engine/inverse_solver.py) | [`tests/test_layer4_engine.py`](tests/test_layer4_engine.py) |
| **6. Noise-Resistant Equation Discovery** | Uses integration by parts with smooth bump functions so noisy sensor data is never differentiated directly, plus rational Padé laws. | [`cortivra/layer5_discovery/sparse_regression.py`](cortivra/layer5_discovery/sparse_regression.py)<br>[`rational_pade_discovery.py`](cortivra/layer5_discovery/rational_pade_discovery.py) | [`tests/test_layer5_discovery.py`](tests/test_layer5_discovery.py)<br>[`test_nobel_prize_blind_rediscoveries.py`](tests/test_nobel_prize_blind_rediscoveries.py) |
| **7. 4-Agent Meta Muse Council** | Orchestrates `Muse Wonder`, `Muse Spark`, `Muse Spark Open` (`meta-models/muse-spark-open-1026`), and `Muse Glimmer` across Baseten, Fireworks, and Meta API. | [`cortivra/layer6_agents/meta_clients.py`](cortivra/layer6_agents/meta_clients.py)<br>[`cortivra/layer6_agents/council.py`](cortivra/layer6_agents/council.py) | [`tests/test_layer6_council_and_deliverables.py`](tests/test_layer6_council_and_deliverables.py) |
| **8. 3-Stage Physical Realization & 20 Files** | Synthesizes 3D `.stl`/`.obj` CAD, `.svg` blueprints, `C99` code, Stage-2 reactor schedules, and Stage-3 `SMILES`/TPMS lattices. | [`blueprint_compiler.py`](cortivra/layer6_agents/blueprint_compiler.py)<br>[`recursive_synthesis_engine.py`](cortivra/layer6_agents/recursive_synthesis_engine.py)<br>[`pass3_matter_and_lattice_compiler.py`](cortivra/layer6_agents/pass3_matter_and_lattice_compiler.py) | [`tests/test_live_end_to_end_verification.py`](tests/test_live_end_to_end_verification.py) |
| **9. Tamper-Evident Merkle Flight Recorder & UI** | Records every math step into a SHA-256 hash-chained DAG and streams it live to the 6-tab Web Workbench. | [`cortivra/observability/flight_recorder.py`](cortivra/observability/flight_recorder.py)<br>[`cortivra/ui_server.py`](cortivra/ui_server.py) | [`tests/test_observability.py`](tests/test_observability.py)<br>[`test_ui_workbench_server.py`](tests/test_ui_workbench_server.py) |

---

## 4. How Cortivra Uses Meta AI Models

Cortivra uses a **4-Agent Epistemic Council** ([`cortivra/layer6_agents/meta_clients.py`](cortivra/layer6_agents/meta_clients.py)) backed by live multi-provider routing across **Baseten** (`meta-models/muse-spark-open-1026`), **Fireworks AI**, and the **Meta Model API**:

| Agent Role | Meta Model / Persona | Exact Job Inside Cortivra |
| :--- | :--- | :--- |
| **`Muse Wonder`** | Theorist & Operator Selector | Formulates coupled N-compartment Port-Hamiltonian world models ($H, J, R, \mathbf{u}$) from natural language and selects the next mathematical operator from Fisher Information spectra. |
| **`Muse Spark`** | Symbolic Compiler & Self-Repair | Inspects compile-time 7D SI unit errors and 2nd-Law thermodynamic violations to algebraically repair symbolic equations ([`compile_with_self_repair`](cortivra/layer6_agents/hypothesis_compiler.py)). |
| **`Muse Spark Open`** | `meta-models/muse-spark-open-1026` | Acts as the Popperian red-team falsifier: stress-tests candidate equations and interventions across 4 falsification gates, $4\times$ extended time horizons, and $\pm 30\%$ parameter drift. |
| **`Muse Glimmer`** | Realization Chronicler & Co-Scientist | Synthesizes the verified math into the executive dossier and powers the interactive **Co-Scientist AI Chat** inside the Web Workbench so anyone can ask questions about any equation, proof, or molecule in plain English. |

---

## 5. Quick Start (Run in 60 Seconds)

### 1. Install Dependencies
```bash
pip install -e ".[dev]"
```

### 2. Configure Your API Key
```bash
cp .env.example .env
# Add your BASETEN_API_KEY, FIREWORKS_API_KEY, or META_API_KEY in .env
```

### 3. Launch the Interactive Web Workbench (`http://127.0.0.1:8765`)
```bash
python -m cortivra --ui --port 8765
```
Open `http://127.0.0.1:8765` in your browser to type any question, upload a `.csv` dataset, watch the 6-phase mathematical trace stream live, rotate the 3D CAD and molecular models, ask the Co-Scientist AI questions about any section, and inspect or download all 20 deliverable files.

### 4. Or Run Directly from the Command Line
```bash
python -m cortivra "Suppress lithium dendrite growth in an argyrodite solid-state battery during fast charging without degrading bulk ionic conductivity or triggering interfacial thermal runaway." --domain electrochemistry --save-logs
```

### 5. Run the Verification Test Suite
```bash
pytest
```

---

## 6. All 20 Files Exported on Every Run

Every investigation saves a SHA-256 hash-chained folder under `logs/runs/<timestamp>_<system_name>/`:

| Category | File Path | What It Contains |
| :--- | :--- | :--- |
| **Interactive Reports** | `dossier.html` | Self-contained HTML5 visual command center with interactive 3D CAD & molecular viewers |
| **Interactive Reports** | `dossier.md` | Complete 9-section scientific discovery dossier, equations, proofs, and scorecard |
| **Interactive Reports** | `dossier.json` | Full machine-readable JSON telemetry across Layers 1–6 |
| **Formal Proofs** | `deliverables/formal_theorem.lean` | **Lean 4** formal proof of Port-Hamiltonian passivity, spectral gap, and interval contraction |
| **Formal Proofs** | `deliverables/formal_certificate.smt2` | **SMT-LIB2 (`QF_NRA`)** nonlinear real arithmetic script for Z3/CVC5 certifying `unsat` (zero counterexamples) |
| **3D CAD & Matter** | `deliverables/cad_assembly.stl` | Watertight 3D-printable multi-compartment solid mesh (`mm` scale) |
| **3D CAD & Matter** | `deliverables/cad_assembly.obj` | Wavefront 3D mesh rendered in the interactive Web Workbench viewport |
| **3D CAD & Matter** | `deliverables/engineering_blueprint.svg` | 3-view orthographic engineering blueprint with dimensional callouts |
| **3D CAD & Matter** | `deliverables/pass3_molecule.xyz` | 3D atomic coordinates (`Å`) for the Top-1 candidate molecule (`SMILES`) |
| **3D CAD & Matter** | `deliverables/pass3_structure.svg` | Dual-panel 2D covalent molecular graph & 3D periodic unit-cell lattice diagram |
| **3D CAD & Matter** | `deliverables/pass3_matter_and_lattice_spec.json` | Top-3 Pareto `SMILES` molecules and 4-topology Gibson-Ashby/TPMS lattice specs |
| **Runnable Code** | `deliverables/controller_firmware.h` | ANSI C99 header with fixed-cycle state structs, safety macros, and bitmask flags |
| **Runnable Code** | `deliverables/controller_firmware.c` | Compilable (`gcc -std=c99`) closed-loop passivity controller with built-in self-test |
| **Runnable Code** | `deliverables/digital_twin_simulator.py` | Standalone Python simulator running RK4 + Langevin noise + $\pm 25\%$ mismatched-reality trial |
| **SOP & Audit Logs** | `deliverables/wetlab_and_manufacturing_sop.md` | Step-by-step Standard Operating Procedure, Bill of Materials (BOM), and 4-stage reactor recipe |
| **SOP & Audit Logs** | `deliverables/fmeca_and_bom.json` | Safe Operating Area (SOA), stoichiometric BOM, and Sobol FMECA risk table |
| **SOP & Audit Logs** | `execution_flight_recorder.log` | Human-readable 8-stage execution walkthrough and causal trace timeline |
| **SOP & Audit Logs** | `execution_ledger.jsonl` | Structured per-stage pass/fail ledger with SHA-256 deliverable digests |
| **SOP & Audit Logs** | `flight_recorder.jsonl` | Cryptographic SHA-256 Merkle-chained causal trace nodes and operator spectra |
| **SOP & Audit Logs** | `deliverables/deliverable_manifest.json` | Master file inventory with byte sizes, SHA-256 hashes, and Merkle root hash |

---

## 7. Roadmap: This Is Just the Beginning

Everything in this repository—all 6 layers, the symbolic compiler, the interval/SOS/Lean 4 provers, the 3-stage realization pipeline, and the Web Workbench—was architected and built during the 9-day hackathon window (**Oct 27 – Nov 4, 2026**).

While Cortivra already closes the loop from a natural-language question to verified differential equations and 20 engineering artifacts in silico, **this is just the foundation**. Here is what comes next:

1. **Direct Hardware-in-the-Loop (HIL) & Lab Robotics Integration:** Connecting the generated ANSI C99 firmware and Stage-2 reactor schedules directly to physical microcontrollers (STM32/ESP32), microfluidic pumps, and automated wet-lab synthesis stations for real-time closed-loop experiments.
2. **Automated External Prover Kernel Execution:** Wiring `formal_theorem.lean` and `formal_certificate.smt2` directly into live Lean 4 (`lake build`) and Z3/dReal solver binaries inside a continuous verification loop for higher-degree non-polynomial PDEs.
3. **Full 3D Finite-Element (FEM) & Navier-Stokes Multi-Physics Meshes:** Expanding Layer 3's 3D voxel reaction-diffusion discretizer (`12×12×12` FVM) to unstructured tetrahedral FEM meshes coupled with compressible fluid flow and Maxwell electrodynamics.
4. **3D De Novo Atomistic Force-Field Relaxation:** Upgrading Stage 3's 700-candidate group-contribution and VSEPR coordinate solver to run GPU-accelerated equivariant neural potentials and DFT single-point energy refinement.

---

## 8. License, Security & Intellectual Property

* **Dual License ([`LICENSE`](LICENSE)):** Released under the **GNU Affero General Public License v3.0 (AGPL-3.0)** for open-source academic and hackathon evaluation (requiring full source disclosure for any network/SaaS deployment under Section 13), with all commercial, closed-source, and enterprise rights reserved by the author (Brad Harris).
* **Security & Provenance ([`SECURITY.md`](SECURITY.md)):** Zero credentials in source code, strict localhost and path-traversal sandboxing on the Web Workbench, and SHA-256 Merkle provenance on all generated artifacts.
