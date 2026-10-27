"""
Layer 3 (`cortivra.layer3_compiler`) — Symbolic AST, GPU tensor compiler, and spatial discretizer.

Provides:
    - `ast_nodes`: Typed symbolic AST (`ExprNode`, `ConstNode`, `SymbolNode`, `UnaryOpNode`,
      `BinaryOpNode`, `PrimitiveCallNode`), `parse_symbolic_expr`, compile-time SI unit inference,
      algebraic `simplify()`, symbolic differentiation `diff()`, and `tree_complexity()`.
    - `symbolic_compiler`: `SymbolicSystemSpec`, `CompiledLawBrick`, `ResidualPatchedLawBrick`,
      `compile_symbolic_system`, and `compile_residual_patch_brick`.
    - `spatial_discretizer`: `CompartmentGeometry`, `DiscretizedCompartmentGraph`,
      `HeterogeneousCompartmentTransportBrick`, `make_sphere_compartment`,
      `make_cylinder_compartment`, `make_cuboid_compartment`, `build_compartment_graph`,
      and `discretize_labeled_mask`.
"""

from cortivra.layer3_compiler.ast_nodes import (
    BinaryOpNode,
    ConstNode,
    ExprNode,
    PrimitiveCallNode,
    SymbolNode,
    SymbolicCompilationError,
    UnaryOpNode,
    parse_symbolic_expr,
)
from cortivra.layer3_compiler.spatial_discretizer import (
    CompartmentGeometry,
    DiscretizedCompartmentGraph,
    HeterogeneousCompartmentTransportBrick,
    build_compartment_graph,
    discretize_labeled_mask,
    make_cuboid_compartment,
    make_cylinder_compartment,
    make_sphere_compartment,
)
from cortivra.layer3_compiler.symbolic_compiler import (
    CompiledLawBrick,
    ResidualPatchedLawBrick,
    SymbolicSystemSpec,
    build_symbol_table,
    compile_residual_patch_brick,
    compile_symbolic_system,
)

__all__ = [
    "SymbolicCompilationError",
    "ExprNode",
    "ConstNode",
    "SymbolNode",
    "UnaryOpNode",
    "BinaryOpNode",
    "PrimitiveCallNode",
    "parse_symbolic_expr",
    "SymbolicSystemSpec",
    "CompiledLawBrick",
    "ResidualPatchedLawBrick",
    "build_symbol_table",
    "compile_symbolic_system",
    "compile_residual_patch_brick",
    "CompartmentGeometry",
    "DiscretizedCompartmentGraph",
    "HeterogeneousCompartmentTransportBrick",
    "make_sphere_compartment",
    "make_cylinder_compartment",
    "make_cuboid_compartment",
    "build_compartment_graph",
    "discretize_labeled_mask",
]
