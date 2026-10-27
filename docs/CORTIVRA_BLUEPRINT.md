# Cortivra Architecture Notes

## Overview

Cortivra is a 7-layer differentiable physics and biophysics simulation engine. It connects mathematical operators across thermodynamics, chemistry, cellular electrophysiology, spatial morphology, and network dynamics into a single batched PyTorch tensor pipeline.

Rather than hardcoding domain-specific outcomes, the engine enforces dimensional homogeneity, conservation of mass and charge, and port-Hamiltonian energy balance at the lowest layer (`layer1_axioms`). Higher layers assemble equations from 12 reusable differential operators (`layer2_fabric`), compile them into batched GPU kernels (`layer3_compiler`), filter candidate parameter spaces through a 4-stage elimination and inverse-gradient solver (`layer4_engine`), and stress-test surviving solutions against thermal noise and parameter perturbation (`layer5_proof`). Meta AI models (`Muse Wonder`, `Muse Spark`, `Muse Spark Open`, `Muse Glimmer`, and `SAM 3.1` in `layer6_agents`) formulate hypotheses, steer the simulation loop, and stream telemetry to the web UI (`layer7_observatory`).

---

## Core Physical Principles

Every subsystem in Cortivra is represented through four physical components:

1. **Conservation and Thermodynamics:**
   Subsystems follow open port-Hamiltonian dynamics:
   $$\frac{dx}{dt} = (J(x) - R(x))\nabla H(x) + B(x)u$$
   where $H(x)$ is stored energy, $J(x) = -J(x)^T$ is conservative routing, $R(x) = R(x)^T \succeq 0$ is entropy/heat dissipation, and ports satisfy $\text{Dim}(\text{Effort} \times \text{Flow}) = \text{Watts}$.
2. **Constitutive and Kinetic Laws:**
   Transition rates and binding equilibria follow Arrhenius/Eyring barrier crossing, Boltzmann gating, mass-action stoichiometry, Hill cooperativity, and multi-state Markov graphs.
3. **Spatial Geometry and Graph Topology:**
   Continuous diffusion and multi-compartment cable equations parameterized from segmented morphology (`SAM 3.1`), paired with directed graph connectivity and axonal delays.
4. **Multi-Timescale Control:**
   Fast-slow coupling across microseconds (ion channels), milliseconds (spikes), seconds (calcium and ATP metabolism), and minutes/hours (homeostatic integral feedback and plasticity).

---

## The 12 Differential Primitives (`layer2_fabric`)

1. `ConservationFlux` — Mass and charge continuity across coupled ports.
2. `NernstPlanckDrift` — Electrochemical equilibrium and electrodiffusion ($\frac{RT}{zF}\ln\frac{c_{\text{out}}}{c_{\text{in}}}$).
3. `CapacitiveRC` — Dielectric charge storage and membrane potential integration ($C\frac{dV}{dt} = \sum I$).
4. `BoltzmannBarrier` — Equilibrium activation/inactivation curves ($\frac{1}{1 + \exp((V_{1/2} - V)/k)}$).
5. `EyringTransitionRate` — Voltage- and temperature-dependent barrier crossing rates.
6. `MassActionKinetics` — Reversible stoichiometric binding and catalysis ($A + B \leftrightharpoons AB$).
7. `HillCooperativity` — Cooperative saturating binding ($\frac{[L]^n}{K_d^n + [L]^n}$).
8. `MarkovStateGraph` — Multi-state conformational transition matrices ($\frac{d\mathbf{p}}{dt} = Q(V, [L])\mathbf{p}$).
9. `DiffusiveLaplacian` — Spatial diffusion ($D\nabla^2 c$) and axial compartmental cable coupling.
10. `DelayedSynapticKernel` — Double-exponential impulse response and graph-routed coupling.
11. `HomeostaticIntegralController` — Slow target-tracking feedback and metabolic autoregulation.
12. `StochasticLangevinNoise` — Thermal ($k_B T$) and channel gating fluctuations ($\sqrt{2D}\,dW_t$).

---

## Layer Breakdown

### Layer 1: `layer1_axioms`
- **`units.py`**: 7D SI dimensional algebra `[M, L, T, I, Theta, N, J]`, dynamic unit expression parser, Buckingham $\Pi$ nullspace solver, and CODATA physical constants.
- **`hamiltonian.py`**: `LawBrick` base class, symplectic midpoint and RK4 integrators, and `CoupledPortHamiltonianSystem` with automatic cross-brick timescale scaling.
- **`shields.py`**: Conservation checks for dimensions, mass stoichiometry, charge balance, 1st/2nd laws of thermodynamics, Wegscheider cycle balance, viable state bounds, and gated residual analysis for unmodeled dynamics.
- **`vault.py`**: Two-tier law registry separating read-only core laws (`CORE_IMMUTABLE`) from quarantined discovered laws (`DISCOVERED_SANDBOX`).

### Layer 2: `layer2_fabric`
- **`primitives.py`**: Vectorized PyTorch implementations of the 12 differential primitives.
- **`molecular_pod.py`**: Biochemical reaction networks, ligand binding, and ATP energetics.
- **`cellular_pod.py`**: Conductance-based ion channels and intracellular ion dynamics.
- **`network_pod.py`**: Excitatory/inhibitory spiking circuits, plasticity, and field potentials.
- **`mechanics_pod.py`**: Coupled nonlinear oscillators for cross-domain verification.

### Layer 3: `layer3_compiler`
- **`ast_nodes.py`**: Symbolic equation tree representation.
- **`tensor_compiler.py`**: Compiles symbolic trees into batched PyTorch tensor kernels (`[Batch, StateDim]`).
- **`morphology_bridge.py`**: Maps `SAM 3.1` segmented cell geometry to compartmental capacitance and axial conductance tensors.

### Layer 4: `layer4_engine`
- **`elimination_funnel.py`**: 4-stage pruning pipeline (steady-state algebraic filter, parallel GPU sweep, differentiable inverse gradient homing, and stiff multi-timescale verification).

### Layer 5: `layer5_proof`
- **`falsification.py`**: Thermal noise ($k_B T$), parameter jitter ($\pm 20\%$), and metabolic stress testing.
- **`identifiability.py`**: Complexity penalty and minimal equation selection.
- **`rediscovery.py`**: Blind ground-up rediscovery benchmarks against empirical phenomena.

### Layer 6: `layer6_agents`
- **`council.py`**: Coordinates `SAM 3.1`, `Muse Spark`, `Muse Wonder`, `Muse Spark Open`, and `Muse Glimmer`.

### Layer 7: `layer7_observatory`
- **`server.py`**: FastAPI and WebSocket telemetry server for the web UI.
