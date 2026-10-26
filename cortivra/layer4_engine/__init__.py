"""
Layer 4 (`layer4_engine`) — Parallel GPU Simulation & Inverse Engine.

Goal:
    Test millions of parameter combinations in seconds, quickly filter out what
    fails, and use gradients to lock onto the exact solution.

Modules (Planned):
    - `elimination_funnel.py`: 4-step filter (quick math check -> parallel GPU sweep -> gradient tuning -> full simulation).
    - `inverse_solver.py`:     Runs gradients backwards from a target state to find the right parameters.
"""
