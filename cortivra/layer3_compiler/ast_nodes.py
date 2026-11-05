"""
Typed symbolic expression AST, compile-time SI unit inference, symbolic differentiation,
and algebraic simplification (`cortivra.layer3_compiler.ast_nodes`).

Layer 2 primitive calls (`PrimitiveCallNode`) and safe AST string parsing (`parse_symbolic_expr`)
are implemented in `cortivra.layer3_compiler.symbolic_parser` and re-exported here.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import torch

from cortivra.layer1_axioms.units import (
    DIM_DIMENSIONLESS,
    DIMENSIONLESS,
    Dimension,
    DimensionError,
    Unit,
)
from cortivra.layer2_fabric.primitives import _clamp_straight_through

if TYPE_CHECKING:
    from cortivra.layer1_axioms.rigorous_proofs import ScalarInterval


class SymbolicCompilationError(ValueError):
    """Raised when a symbolic expression contains syntax, arity, or symbol errors."""


class ExprNode(ABC):
    """Base class for typed symbolic expression trees."""

    @abstractmethod
    def infer_unit(self) -> Unit:
        """Infer and validate the physical SI `Unit` (dimension and scale) of this sub-tree."""

    @property
    def dimension(self) -> Dimension:
        return self.infer_unit().dimension

    @abstractmethod
    def eval_tensor(self, env: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate this expression on a dictionary of batched tensors `[Batch]` in declared units."""

    @abstractmethod
    def diff(self, var_name: str) -> ExprNode:
        """Return the exact symbolic partial derivative `d(self) / d(var_name)`."""

    @abstractmethod
    def simplify(self) -> ExprNode:
        """Return an algebraically simplified copy of this expression tree."""

    @abstractmethod
    def tree_complexity(self) -> int:
        """Return structural node complexity (used for parsimony / Occam's razor scoring)."""

    @abstractmethod
    def free_symbols(self) -> set[str]:
        """Return the set of symbol names referenced in this expression."""

    @abstractmethod
    def to_infix(self) -> str:
        """Format expression as canonical mathematical infix string."""

    @abstractmethod
    def eval_interval(self, env: Mapping[str, Any]) -> Any:
        """Evaluate this expression over outward-rounded IEEE-754 `ScalarInterval` bounds."""

    def evaluate_interval_in_unit(
        self,
        env: Mapping[str, Any],
        target_unit: Unit | None = None,
    ) -> Any:
        """Evaluate this expression on `ScalarInterval` bounds and convert into `target_unit`."""
        inferred = self.infer_unit()
        raw = self.eval_interval(env)
        if target_unit is None:
            return raw
        factor = inferred.conversion_factor_to(target_unit)
        if abs(factor - 1.0) < 1e-12:
            return raw
        return raw * factor

    def to_python_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        """Compile this AST node to a standalone Python (`math.*`) expression string."""
        return self.to_infix()

    def to_c99_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        """Compile this AST node to a standalone ANSI C99 (`<math.h>`) expression string."""
        return self.to_infix()

    def to_smt2_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        """Compile this AST node to an SMT-LIB2 (`QF_NRA`) prefix S-expression."""
        return self.to_infix()

    def to_lean4_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        """Compile this AST node to a Lean 4 (`Mathlib`) real arithmetic expression."""
        return self.to_infix()

    def to_dimensionless_scaled_ast(self) -> ExprNode:
        """Return an equivalent `DIMENSIONLESS` AST with all SI unit scale ratios baked in."""
        return self

    def evaluate_in_unit(
        self,
        env: Mapping[str, torch.Tensor],
        target_unit: Unit | None = None,
    ) -> torch.Tensor:
        """
        Evaluate this expression across `env` and convert the result into `target_unit`
        (verifying dimensional compatibility at compile/eval time).
        """
        inferred = self.infer_unit()
        raw = self.eval_tensor(env)
        if target_unit is None:
            return raw
        factor = inferred.conversion_factor_to(target_unit)
        if abs(factor - 1.0) < 1e-12:
            return raw
        return raw * factor


@dataclass(frozen=True)
class ConstNode(ExprNode):
    """Numeric constant bound to a physical `Unit`."""

    value: float
    unit: Unit = DIMENSIONLESS

    def __post_init__(self) -> None:
        if math.isnan(self.value) or math.isinf(self.value):
            raise SymbolicCompilationError(f"ConstNode value must be finite, got {self.value}.")
        object.__setattr__(self, "value", float(self.value))

    def infer_unit(self) -> Unit:
        return self.unit

    def eval_tensor(self, env: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if env:
            sample = next(iter(env.values()))
            return torch.full_like(sample, self.value)
        return torch.tensor([self.value], dtype=torch.float64)

    def eval_interval(self, env: Mapping[str, Any]) -> Any:
        from cortivra.layer1_axioms.rigorous_proofs import ScalarInterval

        return ScalarInterval(self.value, self.value)

    def diff(self, var_name: str) -> ExprNode:
        return ConstNode(0.0, self.unit)

    def simplify(self) -> ExprNode:
        return self

    def tree_complexity(self) -> int:
        return 1

    def free_symbols(self) -> set[str]:
        return set()

    def to_infix(self) -> str:
        return f"{self.value:g}"

    def to_python_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        return f"{float(self.value):.12g}"

    def to_c99_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        s_val = f"{float(self.value):.12g}"
        if "." not in s_val and "e" not in s_val and "E" not in s_val:
            s_val += ".0"
        return f"({s_val})" if self.value < 0.0 else s_val

    def to_smt2_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        if self.value < 0.0:
            return f"(- {abs(float(self.value)):.8f})"
        return f"{float(self.value):.8f}"

    def to_lean4_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        return f"({float(self.value):.8f})" if self.value < 0.0 else f"{float(self.value):.8f}"

    def to_dimensionless_scaled_ast(self) -> ExprNode:
        return ConstNode(self.value, DIMENSIONLESS)


@dataclass(frozen=True)
class SymbolNode(ExprNode):
    """Named state variable, parameter, or external input bound to a physical `Unit`."""

    name: str
    unit: Unit = DIMENSIONLESS
    role: str = "state"  # "state", "param", or "input"

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise SymbolicCompilationError("SymbolNode name cannot be empty.")

    def infer_unit(self) -> Unit:
        return self.unit

    def eval_tensor(self, env: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.name not in env:
            raise KeyError(f"Missing symbol '{self.name}' in evaluation environment.")
        return env[self.name]

    def eval_interval(self, env: Mapping[str, Any]) -> Any:
        from cortivra.layer1_axioms.rigorous_proofs import ScalarInterval

        if self.name not in env:
            raise KeyError(f"Missing symbol '{self.name}' in interval evaluation environment.")
        val = env[self.name]
        if isinstance(val, ScalarInterval):
            return val
        f_val = float(val)
        return ScalarInterval(f_val, f_val)

    def diff(self, var_name: str) -> ExprNode:
        if self.name == var_name:
            return ConstNode(1.0, DIMENSIONLESS)
        return ConstNode(0.0, DIMENSIONLESS)

    def simplify(self) -> ExprNode:
        return self

    def tree_complexity(self) -> int:
        return 1

    def free_symbols(self) -> set[str]:
        return {self.name}

    def to_infix(self) -> str:
        return self.name

    def to_python_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        if symbol_map and self.name in symbol_map:
            return symbol_map[self.name]
        return self.name

    def to_c99_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        if symbol_map and self.name in symbol_map:
            return symbol_map[self.name]
        return self.name

    def to_smt2_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        if symbol_map and self.name in symbol_map:
            return symbol_map[self.name]
        return self.name

    def to_lean4_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        if symbol_map and self.name in symbol_map:
            return symbol_map[self.name]
        return self.name

    def to_dimensionless_scaled_ast(self) -> ExprNode:
        return SymbolNode(self.name, DIMENSIONLESS, role=self.role)


@dataclass(frozen=True)
class UnaryOpNode(ExprNode):
    """Unary mathematical operation (`neg`, `exp`, `log`, `sin`, `cos`, `tanh`, `sqrt`, `abs`)."""

    op: str
    operand: ExprNode

    _TRANSCENDENTAL_OPS = ("exp", "log", "sin", "cos", "tanh")
    _VALID_OPS = ("neg", "exp", "log", "sin", "cos", "tanh", "sqrt", "abs")

    def __post_init__(self) -> None:
        if self.op not in self._VALID_OPS:
            raise SymbolicCompilationError(f"Unsupported unary operator '{self.op}'.")

    def infer_unit(self) -> Unit:
        cached = getattr(self, "_cached_unit", None)
        if cached is not None:
            return cached
        op_unit = self.operand.infer_unit()
        if self.op in ("neg", "abs"):
            res = op_unit
        elif self.op == "sqrt":
            res = op_unit**0.5
        elif self.op in self._TRANSCENDENTAL_OPS:
            if op_unit.dimension != DIM_DIMENSIONLESS:
                raise DimensionError(
                    f"Transcendental function '{self.op}(...)' requires a dimensionless argument, "
                    f"got '{self.operand.to_infix()}' with unit '{op_unit.symbol}' ({op_unit.dimension.format_si()})."
                )
            res = DIMENSIONLESS
        else:
            raise SymbolicCompilationError(f"Unhandled unary op '{self.op}'.")
        object.__setattr__(self, "_cached_unit", res)
        return res

    def eval_tensor(self, env: Mapping[str, torch.Tensor]) -> torch.Tensor:
        val = self.operand.eval_tensor(env)
        if self.op == "neg":
            return -val
        if self.op == "abs":
            return torch.abs(val)
        if self.op == "sqrt":
            return torch.sqrt(torch.clamp(val, min=1e-18))

        op_unit = self.operand.infer_unit()
        si_val = val * op_unit.scale if abs(op_unit.scale - 1.0) > 1e-12 else val
        if self.op == "exp":
            return torch.exp(_clamp_straight_through(si_val, -80.0, 80.0))
        if self.op == "log":
            return torch.log(torch.clamp(si_val, min=1e-18))
        if self.op == "sin":
            return torch.sin(si_val)
        if self.op == "cos":
            return torch.cos(si_val)
        if self.op == "tanh":
            return torch.tanh(si_val)
        raise SymbolicCompilationError(f"Unhandled unary op '{self.op}'.")

    def diff(self, var_name: str) -> ExprNode:
        du = self.operand.diff(var_name).simplify()
        if isinstance(du, ConstNode) and du.value == 0.0:
            return ConstNode(0.0, DIMENSIONLESS)

        op_unit = self.operand.infer_unit()
        du_scaled = (
            BinaryOpNode("*", ConstNode(op_unit.scale, DIMENSIONLESS), du).simplify()
            if (self.op in ("exp", "sin", "cos", "tanh") and abs(op_unit.scale - 1.0) > 1e-12)
            else du
        )

        if self.op == "neg":
            return UnaryOpNode("neg", du).simplify()
        if self.op == "exp":
            return BinaryOpNode("*", self, du_scaled).simplify()
        if self.op == "log":
            return BinaryOpNode("/", du, self.operand).simplify()
        if self.op == "sin":
            return BinaryOpNode("*", UnaryOpNode("cos", self.operand), du_scaled).simplify()
        if self.op == "cos":
            return UnaryOpNode("neg", BinaryOpNode("*", UnaryOpNode("sin", self.operand), du_scaled)).simplify()
        if self.op == "tanh":
            sech_sq = BinaryOpNode(
                "-",
                ConstNode(1.0, DIMENSIONLESS),
                BinaryOpNode("**", self, ConstNode(2.0, DIMENSIONLESS)),
            )
            return BinaryOpNode("*", sech_sq, du_scaled).simplify()
        if self.op == "sqrt":
            denom = BinaryOpNode("*", ConstNode(2.0, DIMENSIONLESS), self)
            return BinaryOpNode("/", du, denom).simplify()
        if self.op == "abs":
            sign_approx = BinaryOpNode("/", self.operand, self)
            return BinaryOpNode("*", sign_approx, du).simplify()
        raise SymbolicCompilationError(f"Cannot differentiate unary op '{self.op}'.")

    def simplify(self) -> ExprNode:
        sub = self.operand.simplify()
        if isinstance(sub, ConstNode):
            u = self.infer_unit()
            if self.op == "neg":
                return ConstNode(-sub.value, u)
            if self.op == "abs":
                return ConstNode(abs(sub.value), u)
            if self.op == "sqrt" and sub.value >= 0.0:
                return ConstNode(math.sqrt(sub.value), u)
            si_v = sub.value * sub.unit.scale
            if self.op == "exp":
                return ConstNode(math.exp(max(-80.0, min(80.0, si_v))), DIMENSIONLESS)
            if self.op == "log" and si_v > 0.0:
                return ConstNode(math.log(si_v), DIMENSIONLESS)
            if self.op == "sin":
                return ConstNode(math.sin(si_v), DIMENSIONLESS)
            if self.op == "cos":
                return ConstNode(math.cos(si_v), DIMENSIONLESS)
            if self.op == "tanh":
                return ConstNode(math.tanh(si_v), DIMENSIONLESS)
        if self.op == "neg" and isinstance(sub, UnaryOpNode) and sub.op == "neg":
            return sub.operand
        return UnaryOpNode(self.op, sub)

    def tree_complexity(self) -> int:
        weight = 1 if self.op in ("neg", "abs") else 2
        return weight + self.operand.tree_complexity()

    def free_symbols(self) -> set[str]:
        return self.operand.free_symbols()

    def to_infix(self) -> str:
        if self.op == "neg":
            return f"(-{self.operand.to_infix()})"
        return f"{self.op}({self.operand.to_infix()})"

    def eval_interval(self, env: Mapping[str, Any]) -> Any:
        val = self.operand.eval_interval(env)
        if self.op == "neg":
            return -val
        if self.op == "abs":
            return val.abs()
        if self.op == "sqrt":
            return val.sqrt()
        op_unit = self.operand.infer_unit()
        si_val = val * op_unit.scale if abs(op_unit.scale - 1.0) > 1e-12 else val
        if self.op == "exp":
            return si_val.exp()
        if self.op == "log":
            return si_val.log()
        if self.op == "sin":
            return si_val.sin()
        if self.op == "cos":
            return si_val.cos()
        if self.op == "tanh":
            return si_val.tanh()
        raise SymbolicCompilationError(f"Unhandled unary op '{self.op}' in eval_interval.")

    def to_python_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        inner = self.operand.to_python_expr(symbol_map)
        if self.op == "neg":
            return f"(-({inner}))"
        if self.op == "abs":
            return f"abs({inner})"
        if self.op == "sqrt":
            return f"math.sqrt(max(1e-18, {inner}))"
        op_unit = self.operand.infer_unit()
        if abs(op_unit.scale - 1.0) > 1e-12:
            inner = f"({inner} * {op_unit.scale:.12g})"
        if self.op == "exp":
            return f"math.exp(max(-80.0, min(80.0, {inner})))"
        if self.op == "log":
            return f"math.log(max(1e-18, {inner}))"
        return f"math.{self.op}({inner})"

    def to_c99_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        inner = self.operand.to_c99_expr(symbol_map)
        if self.op == "neg":
            return f"(-({inner}))"
        if self.op == "abs":
            return f"fabs({inner})"
        if self.op == "sqrt":
            return f"sqrt(fmax(1e-18, {inner}))"
        op_unit = self.operand.infer_unit()
        if abs(op_unit.scale - 1.0) > 1e-12:
            inner = f"(({inner}) * {op_unit.scale:.12g})"
        if self.op == "exp":
            return f"exp(fmax(-80.0, fmin(80.0, {inner})))"
        if self.op == "log":
            return f"log(fmax(1e-18, {inner}))"
        return f"{self.op}({inner})"

    def to_smt2_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        inner = self.operand.to_smt2_expr(symbol_map)
        if self.op == "neg":
            return f"(- {inner})"
        if self.op == "abs":
            return f"(ite (>= {inner} 0.0) {inner} (- {inner}))"
        return f"({self.op} {inner})"

    def to_lean4_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        inner = self.operand.to_lean4_expr(symbol_map)
        if self.op == "neg":
            return f"(-({inner}))"
        if self.op == "abs":
            return f"|{inner}|"
        return f"Real.{self.op} ({inner})"

    def to_dimensionless_scaled_ast(self) -> ExprNode:
        sub = self.operand.to_dimensionless_scaled_ast()
        if self.op in self._TRANSCENDENTAL_OPS:
            op_unit = self.operand.infer_unit()
            if abs(op_unit.scale - 1.0) > 1e-12:
                sub = BinaryOpNode("*", ConstNode(op_unit.scale, DIMENSIONLESS), sub).simplify()
        return UnaryOpNode(self.op, sub)


@dataclass(frozen=True)
class BinaryOpNode(ExprNode):
    """Binary arithmetic operation (`+`, `-`, `*`, `/`, `**`)."""

    op: str
    left: ExprNode
    right: ExprNode

    _VALID_OPS = ("+", "-", "*", "/", "**")

    def __post_init__(self) -> None:
        if self.op not in self._VALID_OPS:
            raise SymbolicCompilationError(f"Unsupported binary operator '{self.op}'.")

    def infer_unit(self) -> Unit:
        cached = getattr(self, "_cached_unit", None)
        if cached is not None:
            return cached
        lu = self.left.infer_unit()
        ru = self.right.infer_unit()

        if self.op in ("+", "-"):
            if isinstance(self.left, ConstNode) and self.left.value == 0.0:
                res = ru
            elif isinstance(self.right, ConstNode) and self.right.value == 0.0:
                res = lu
            else:
                if lu.dimension != ru.dimension:
                    raise DimensionError(
                        f"Dimensional mismatch in '{self.to_infix()}': cannot apply '{self.op}' between "
                        f"'{self.left.to_infix()}' ({lu.symbol}, {lu.dimension.format_si()}) and "
                        f"'{self.right.to_infix()}' ({ru.symbol}, {ru.dimension.format_si()})."
                    )
                res = lu
        elif self.op == "*":
            res = lu * ru
        elif self.op == "/":
            res = lu / ru
        elif self.op == "**":
            if ru.dimension != DIM_DIMENSIONLESS:
                raise DimensionError(
                    f"Exponent in '{self.to_infix()}' must be dimensionless, "
                    f"got '{ru.symbol}' ({ru.dimension.format_si()})."
                )
            if isinstance(self.right, ConstNode):
                exp_val = self.right.value * ru.scale
                res = lu**exp_val
            else:
                if not lu.dimension.is_dimensionless:
                    raise DimensionError(
                        f"Base '{self.left.to_infix()}' raised to non-constant symbolic exponent "
                        f"'{self.right.to_infix()}' must be dimensionless, got {lu.dimension.format_si()}."
                    )
                res = DIMENSIONLESS
        else:
            raise SymbolicCompilationError(f"Unhandled binary op '{self.op}'.")
        object.__setattr__(self, "_cached_unit", res)
        return res

    def eval_tensor(self, env: Mapping[str, torch.Tensor]) -> torch.Tensor:
        scales = getattr(self, "_cached_scales", None)
        if scales is None:
            lu = self.left.infer_unit()
            ru = self.right.infer_unit()
            l_const = type(self.left) is ConstNode
            r_const = type(self.right) is ConstNode
            if self.op in ("+", "-"):
                if (l_const and self.left.value == 0.0) or (
                    r_const and self.right.value == 0.0
                ):
                    sr = 1.0
                else:
                    sr = ru.scale / lu.scale
            else:
                sr = 1.0
            scales = (lu.scale, ru.scale, sr, l_const, r_const)
            object.__setattr__(self, "_cached_scales", scales)
        lu_scale, ru_scale, scale_ratio, left_is_const, right_is_const = scales

        if self.op in ("+", "-"):
            if left_is_const and not right_is_const:
                rv = self.right.eval_tensor(env)
                rhs_scaled = rv * scale_ratio if abs(scale_ratio - 1.0) > 1e-12 else rv
                c_l = self.left.value
                if c_l == 0.0:
                    return rhs_scaled if self.op == "+" else -rhs_scaled
                return (c_l + rhs_scaled) if self.op == "+" else (c_l - rhs_scaled)
            if right_is_const and not left_is_const:
                lv = self.left.eval_tensor(env)
                c_r = self.right.value * scale_ratio if abs(scale_ratio - 1.0) > 1e-12 else self.right.value
                if c_r == 0.0:
                    return lv
                return (lv + c_r) if self.op == "+" else (lv - c_r)
            lv = self.left.eval_tensor(env)
            rv = self.right.eval_tensor(env)
            rhs_scaled = rv * scale_ratio if abs(scale_ratio - 1.0) > 1e-12 else rv
            return (lv + rhs_scaled) if self.op == "+" else (lv - rhs_scaled)

        if self.op == "*":
            if left_is_const and not right_is_const:
                rv = self.right.eval_tensor(env)
                return rv * self.left.value
            if right_is_const and not left_is_const:
                lv = self.left.eval_tensor(env)
                return lv * self.right.value
            return self.left.eval_tensor(env) * self.right.eval_tensor(env)

        if self.op == "/":
            if right_is_const and not left_is_const:
                lv = self.left.eval_tensor(env)
                denom_c = self.right.value if abs(self.right.value) >= 1e-15 else (1e-15 if self.right.value >= 0.0 else -1e-15)
                return lv * (1.0 / denom_c)
            lv = self.left.eval_tensor(env) if not left_is_const else self.left.value
            rv = self.right.eval_tensor(env)
            rv_safe = torch.where(
                torch.abs(rv) >= 1e-15,
                rv,
                torch.where(rv >= 0.0, 1e-15, -1e-15),
            )
            return lv / rv_safe

        if self.op == "**":
            lv = self.left.eval_tensor(env)
            if right_is_const:
                exp_val = self.right.value * ru_scale
                if float(exp_val).is_integer():
                    i_exp = int(round(exp_val))
                    if i_exp == 2:
                        return lv * lv
                    if i_exp == 3:
                        return lv * lv * lv
                    if i_exp == 4:
                        lv2 = lv * lv
                        return lv2 * lv2
                    return lv ** i_exp
                return torch.clamp(lv, min=1e-15) ** exp_val
            rv = self.right.eval_tensor(env)
            rv_si = rv * ru_scale if abs(ru_scale - 1.0) > 1e-12 else rv
            lv_si = lv * lu_scale if abs(lu_scale - 1.0) > 1e-12 else lv
            return torch.clamp(lv_si, min=1e-15) ** rv_si

        raise SymbolicCompilationError(f"Unhandled binary op '{self.op}'.")

    def diff(self, var_name: str) -> ExprNode:
        dl = self.left.diff(var_name).simplify()
        dr = self.right.diff(var_name).simplify()

        if self.op in ("+", "-"):
            lu = self.left.infer_unit()
            ru = self.right.infer_unit()
            left_is_zero = isinstance(self.left, ConstNode) and self.left.value == 0.0
            right_is_zero = isinstance(self.right, ConstNode) and self.right.value == 0.0
            target_ratio = 1.0 if (left_is_zero or right_is_zero) else (ru.scale / lu.scale)

            dl_is_zero = isinstance(dl, ConstNode) and dl.value == 0.0
            dr_is_zero = isinstance(dr, ConstNode) and dr.value == 0.0

            if not dr_is_zero:
                if dl_is_zero:
                    corr = target_ratio
                else:
                    dl_u = dl.infer_unit()
                    dr_u = dr.infer_unit()
                    corr = target_ratio / (dr_u.scale / dl_u.scale)
                if abs(corr - 1.0) > 1e-12:
                    dr = BinaryOpNode("*", ConstNode(corr, DIMENSIONLESS), dr).simplify()
            return BinaryOpNode(self.op, dl, dr).simplify()

        if self.op == "*":
            term1 = BinaryOpNode("*", dl, self.right).simplify()
            term2 = BinaryOpNode("*", self.left, dr).simplify()
            return BinaryOpNode("+", term1, term2).simplify()

        if self.op == "/":
            term1 = BinaryOpNode("*", dl, self.right).simplify()
            term2 = BinaryOpNode("*", self.left, dr).simplify()
            num = BinaryOpNode("-", term1, term2).simplify()
            den = BinaryOpNode("**", self.right, ConstNode(2.0, DIMENSIONLESS)).simplify()
            return BinaryOpNode("/", num, den).simplify()

        if self.op == "**":
            if isinstance(self.right, ConstNode):
                n = self.right.value * self.right.unit.scale
                if n == 0.0:
                    return ConstNode(0.0, DIMENSIONLESS)
                coeff = ConstNode(n, DIMENSIONLESS)
                new_pow = BinaryOpNode("**", self.left, ConstNode(n - 1.0, DIMENSIONLESS)).simplify()
                return BinaryOpNode("*", BinaryOpNode("*", coeff, new_pow), dl).simplify()
            term1 = BinaryOpNode("*", dr, UnaryOpNode("log", self.left)).simplify()
            term2 = BinaryOpNode("*", self.right, BinaryOpNode("/", dl, self.left)).simplify()
            return BinaryOpNode("*", self, BinaryOpNode("+", term1, term2)).simplify()

        raise SymbolicCompilationError(f"Cannot differentiate binary op '{self.op}'.")

    def simplify(self) -> ExprNode:
        l_sim = self.left.simplify()
        r_sim = self.right.simplify()

        if isinstance(l_sim, ConstNode) and isinstance(r_sim, ConstNode):
            u = BinaryOpNode(self.op, l_sim, r_sim).infer_unit()
            lu = l_sim.unit
            ru = r_sim.unit
            if self.op in ("+", "-"):
                if l_sim.value == 0.0:
                    return r_sim if self.op == "+" else ConstNode(-r_sim.value, ru)
                if r_sim.value == 0.0:
                    return l_sim
                ratio = ru.scale / lu.scale
                val = (l_sim.value + r_sim.value * ratio) if self.op == "+" else (l_sim.value - r_sim.value * ratio)
                return ConstNode(val, u)
            if self.op == "*":
                return ConstNode(l_sim.value * r_sim.value, u)
            if self.op == "/" and abs(r_sim.value) > 1e-15:
                return ConstNode(l_sim.value / r_sim.value, u)
            if self.op == "**":
                exp_v = r_sim.value * ru.scale
                return ConstNode(l_sim.value**exp_v, u)

        if self.op == "+":
            if isinstance(l_sim, ConstNode) and l_sim.value == 0.0:
                return r_sim
            if isinstance(r_sim, ConstNode) and r_sim.value == 0.0:
                return l_sim
        if self.op == "-":
            if isinstance(r_sim, ConstNode) and r_sim.value == 0.0:
                return l_sim
            if isinstance(l_sim, ConstNode) and l_sim.value == 0.0:
                return UnaryOpNode("neg", r_sim).simplify()
            if l_sim == r_sim:
                return ConstNode(0.0, l_sim.infer_unit())
        if self.op == "*":
            if isinstance(l_sim, ConstNode) and l_sim.value == 0.0:
                return ConstNode(0.0, BinaryOpNode("*", l_sim, r_sim).infer_unit())
            if isinstance(r_sim, ConstNode) and r_sim.value == 0.0:
                return ConstNode(0.0, BinaryOpNode("*", l_sim, r_sim).infer_unit())
            if isinstance(l_sim, ConstNode) and l_sim.value == 1.0 and l_sim.unit == DIMENSIONLESS:
                return r_sim
            if isinstance(r_sim, ConstNode) and r_sim.value == 1.0 and r_sim.unit == DIMENSIONLESS:
                return l_sim
        if self.op == "/":
            if isinstance(l_sim, ConstNode) and l_sim.value == 0.0:
                return ConstNode(0.0, BinaryOpNode("/", l_sim, r_sim).infer_unit())
            if isinstance(r_sim, ConstNode) and r_sim.value == 1.0 and r_sim.unit == DIMENSIONLESS:
                return l_sim
            if l_sim == r_sim:
                return ConstNode(1.0, DIMENSIONLESS)
        if self.op == "**":
            if isinstance(r_sim, ConstNode) and r_sim.value == 0.0:
                return ConstNode(1.0, DIMENSIONLESS)
            if isinstance(r_sim, ConstNode) and r_sim.value == 1.0 and r_sim.unit == DIMENSIONLESS:
                return l_sim

        return BinaryOpNode(self.op, l_sim, r_sim)

    def tree_complexity(self) -> int:
        weight = 1 if self.op in ("+", "-", "*") else 2
        return weight + self.left.tree_complexity() + self.right.tree_complexity()

    def free_symbols(self) -> set[str]:
        return self.left.free_symbols() | self.right.free_symbols()

    def to_infix(self) -> str:
        return f"({self.left.to_infix()} {self.op} {self.right.to_infix()})"

    def eval_interval(self, env: Mapping[str, Any]) -> Any:
        lu = self.left.infer_unit()
        ru = self.right.infer_unit()
        lv = self.left.eval_interval(env)
        rv = self.right.eval_interval(env)

        if self.op in ("+", "-"):
            if isinstance(self.left, ConstNode) and self.left.value == 0.0:
                scale_ratio = 1.0
            elif isinstance(self.right, ConstNode) and self.right.value == 0.0:
                scale_ratio = 1.0
            else:
                scale_ratio = ru.scale / lu.scale
            rhs_scaled = rv * scale_ratio if abs(scale_ratio - 1.0) > 1e-12 else rv
            return (lv + rhs_scaled) if self.op == "+" else (lv - rhs_scaled)

        if self.op == "*":
            return lv * rv

        if self.op == "/":
            return lv / rv

        if self.op == "**":
            if isinstance(self.right, ConstNode):
                exp_val = self.right.value * ru.scale
                return lv**exp_val
            rv_si = rv * ru.scale if abs(ru.scale - 1.0) > 1e-12 else rv
            lv_si = lv * lu.scale if abs(lu.scale - 1.0) > 1e-12 else lv
            return lv_si**rv_si

        raise SymbolicCompilationError(f"Unhandled binary op '{self.op}' in eval_interval.")

    def to_python_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        lu = self.left.infer_unit()
        ru = self.right.infer_unit()
        l_str = self.left.to_python_expr(symbol_map)
        r_str = self.right.to_python_expr(symbol_map)

        if self.op in ("+", "-"):
            if isinstance(self.left, ConstNode) and self.left.value == 0.0:
                scale_ratio = 1.0
            elif isinstance(self.right, ConstNode) and self.right.value == 0.0:
                scale_ratio = 1.0
            else:
                scale_ratio = ru.scale / lu.scale
            if abs(scale_ratio - 1.0) > 1e-12:
                r_str = f"({r_str} * {scale_ratio:.12g})"
            return f"({l_str} {self.op} {r_str})"

        if self.op == "*":
            return f"({l_str} * {r_str})"

        if self.op == "/":
            return f"({l_str} / ({r_str} if abs({r_str}) > 1e-15 else (1e-15 if ({r_str}) >= 0.0 else -1e-15)))"

        if self.op == "**":
            if isinstance(self.right, ConstNode):
                exp_val = self.right.value * ru.scale
                if float(exp_val).is_integer():
                    return f"(({l_str}) ** {int(round(exp_val))})"
                return f"(max(1e-15, {l_str}) ** {exp_val:.12g})"
            if abs(ru.scale - 1.0) > 1e-12:
                r_str = f"({r_str} * {ru.scale:.12g})"
            if abs(lu.scale - 1.0) > 1e-12:
                l_str = f"({l_str} * {lu.scale:.12g})"
            return f"(max(1e-15, {l_str}) ** ({r_str}))"

        return f"({l_str} {self.op} {r_str})"

    def to_c99_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        lu = self.left.infer_unit()
        ru = self.right.infer_unit()
        l_str = self.left.to_c99_expr(symbol_map)
        r_str = self.right.to_c99_expr(symbol_map)

        if self.op in ("+", "-"):
            if isinstance(self.left, ConstNode) and self.left.value == 0.0:
                scale_ratio = 1.0
            elif isinstance(self.right, ConstNode) and self.right.value == 0.0:
                scale_ratio = 1.0
            else:
                scale_ratio = ru.scale / lu.scale
            if abs(scale_ratio - 1.0) > 1e-12:
                r_str = f"(({r_str}) * {scale_ratio:.12g})"
            return f"(({l_str}) {self.op} ({r_str}))"

        if self.op == "*":
            return f"(({l_str}) * ({r_str}))"

        if self.op == "/":
            return f"(({l_str}) / (fabs({r_str}) > 1e-15 ? ({r_str}) : (({r_str}) >= 0.0 ? 1e-15 : -1e-15)))"

        if self.op == "**":
            if isinstance(self.right, ConstNode):
                exp_val = self.right.value * ru.scale
                if exp_val == 2.0:
                    return f"(({l_str}) * ({l_str}))"
                if exp_val == 3.0:
                    return f"(({l_str}) * ({l_str}) * ({l_str}))"
                if float(exp_val).is_integer():
                    return f"pow({l_str}, {int(round(exp_val))}.0)"
                return f"pow(fmax(1e-15, {l_str}), {exp_val:.12g})"
            if abs(ru.scale - 1.0) > 1e-12:
                r_str = f"(({r_str}) * {ru.scale:.12g})"
            if abs(lu.scale - 1.0) > 1e-12:
                l_str = f"(({l_str}) * {lu.scale:.12g})"
            return f"pow(fmax(1e-15, {l_str}), {r_str})"

        return f"(({l_str}) {self.op} ({r_str}))"

    def to_smt2_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        l_str = self.left.to_smt2_expr(symbol_map)
        r_str = self.right.to_smt2_expr(symbol_map)
        if self.op == "**":
            if isinstance(self.right, ConstNode) and float(self.right.value).is_integer():
                n = int(round(self.right.value))
                if n == 2:
                    return f"(* {l_str} {l_str})"
                if n == 3:
                    return f"(* {l_str} (* {l_str} {l_str}))"
                if n == 4:
                    return f"(* (* {l_str} {l_str}) (* {l_str} {l_str}))"
            return f"(^ {l_str} {r_str})"
        return f"({self.op} {l_str} {r_str})"

    def to_lean4_expr(self, symbol_map: Mapping[str, str] | None = None) -> str:
        l_str = self.left.to_lean4_expr(symbol_map)
        r_str = self.right.to_lean4_expr(symbol_map)
        if self.op == "**":
            if isinstance(self.right, ConstNode) and float(self.right.value).is_integer():
                return f"(({l_str}) ^ {int(round(self.right.value))})"
            return f"(({l_str}) ^ ({r_str}))"
        return f"({l_str} {self.op} {r_str})"

    def to_dimensionless_scaled_ast(self) -> ExprNode:
        lu = self.left.infer_unit()
        ru = self.right.infer_unit()
        l_dimless = self.left.to_dimensionless_scaled_ast()
        r_dimless = self.right.to_dimensionless_scaled_ast()
        if self.op in ("+", "-"):
            if isinstance(self.left, ConstNode) and self.left.value == 0.0:
                scale_ratio = 1.0
            elif isinstance(self.right, ConstNode) and self.right.value == 0.0:
                scale_ratio = 1.0
            else:
                scale_ratio = ru.scale / lu.scale
            if abs(scale_ratio - 1.0) > 1e-12:
                r_dimless = BinaryOpNode("*", ConstNode(scale_ratio, DIMENSIONLESS), r_dimless).simplify()
            return BinaryOpNode(self.op, l_dimless, r_dimless).simplify()
        if self.op == "**":
            if isinstance(self.right, ConstNode):
                exp_val = self.right.value * ru.scale
                return BinaryOpNode("**", l_dimless, ConstNode(exp_val, DIMENSIONLESS)).simplify()
            if abs(ru.scale - 1.0) > 1e-12:
                r_dimless = BinaryOpNode("*", ConstNode(ru.scale, DIMENSIONLESS), r_dimless).simplify()
            if abs(lu.scale - 1.0) > 1e-12:
                l_dimless = BinaryOpNode("*", ConstNode(lu.scale, DIMENSIONLESS), l_dimless).simplify()
            return BinaryOpNode("**", l_dimless, r_dimless).simplify()
        return BinaryOpNode(self.op, l_dimless, r_dimless).simplify()


from cortivra.layer3_compiler.symbolic_parser import (  # noqa: E402
    _PRIMITIVE_ALIASES,
    PrimitiveCallNode,
    _smooth_bernoulli_derivative,
    parse_symbolic_expr,
)

__all__ = [
    "SymbolicCompilationError",
    "_smooth_bernoulli_derivative",
    "ExprNode",
    "ConstNode",
    "SymbolNode",
    "UnaryOpNode",
    "BinaryOpNode",
    "_PRIMITIVE_ALIASES",
    "PrimitiveCallNode",
    "parse_symbolic_expr",
]
