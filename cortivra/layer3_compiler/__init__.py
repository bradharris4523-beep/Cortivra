"""
Layer 3 (`layer3_compiler`) — Math-to-GPU Compiler.

Goal:
    Take equations proposed by the AI (plus real cell shapes from SAM 3.1)
    and turn them into fast GPU tensor operations.

Modules (Planned):
    - `ast_nodes.py`:         Represents equations as a clean math tree.
    - `tensor_compiler.py`:   Compiles equations into batched PyTorch GPU kernels.
    - `morphology_bridge.py`: Turns SAM 3.1 cell images into membrane capacitance and cable values.
"""
