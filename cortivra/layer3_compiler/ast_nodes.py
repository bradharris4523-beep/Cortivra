"""
Typed symbolic expression AST, safe parser, compile-time SI unit inference,
symbolic differentiation, and algebraic simplification (`cortivra.layer3_compiler.ast_nodes`).
"""

from __future__ import annotations

import ast
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import torch

from cortivra.layer1_axioms.units import (
    BODY_TEMPERATURE_K,
    DIM_CHEMICAL_POTENTIAL,
    DIM_DIMENSIONLESS,
    DIM_VELOCITY,
    DIM_VOLTAGE,
    DIMENSIONLESS,
    FARADAY,
    R_GAS,
    C,
    Dimension,
    DimensionError,
    Unit,
    m_per_s,
    mol,
    mV,
)
from cortivra.layer2_fabric.primitives import (
    _clamp_straight_through,
    boltzmann_partition_switch,
    cooperative_hill_saturation,
    cubic_bistable_feedback,
    euclidean_distance_3d,
    eyring_barrier_rate,
    ghk_electrodiffusion_flux,
    inverse_distance_power_3d,
    linear_relaxation_rate,
    nernst_equilibrium_potential_mv,
    ohmic_dissipative_driving_force,
    quadratic_energy_storage,
    smooth_bernoulli_quotient,
    surrogate_event_impulse,
)


class SymbolicCompilationError(ValueError):
    """Raised when a symbolic expression contains syntax, arity, or symbol errors."""


def _smooth_bernoulli_derivative(u: torch.Tensor) -> torch.Tensor:
    """
    Exact 1st derivative `B'(u) = d/du [u / (exp(u) - 1)]` with 4th-order Taylor
    continuation across the removable singularity at `u = 0`:
        `B'(u) = -1/2 + u/6 - u^3/180` for `|u| < 1e-4`
        `B'(u) = ((1 - u) * exp(u) - 1) / (exp(u) - 1)^2` for `|u| >= 1e-4`
    """
    u_clamped = _clamp_straight_through(u, -80.0, 80.0)
    u2 = u_clamped * u_clamped
    u3 = u2 * u_clamped
    taylor_deriv = -0.5 + (u_clamped / 6.0) - (u3 / 180.0)

    small_mask = torch.abs(u_clamped) < 1e-4
    safe_u = torch.where(small_mask, torch.ones_like(u_clamped), u_clamped)
    # For large positive u > 40, B'(u) -> 0 exponentially; for large negative u < -40, B'(u) -> -1
    exp_neg_abs = torch.exp(-torch.abs(safe_u))
    pos_branch = ((1.0 - safe_u) * exp_neg_abs - exp_neg_abs * exp_neg_abs) / (
        (1.0 - exp_neg_abs) ** 2
    )
    neg_branch = ((1.0 - safe_u) * exp_neg_abs - 1.0) / ((exp_neg_abs - 1.0) ** 2)
    exact_deriv = torch.where(safe_u >= 0.0, pos_branch, neg_branch)
    return torch.where(small_mask, taylor_deriv, exact_deriv)


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
        op_unit = self.operand.infer_unit()
        if self.op in ("neg", "abs"):
            return op_unit
        if self.op == "sqrt":
            return op_unit**0.5
        if self.op in self._TRANSCENDENTAL_OPS:
            if op_unit.dimension != DIM_DIMENSIONLESS:
                raise DimensionError(
                    f"Transcendental function '{self.op}(...)' requires a dimensionless argument, "
                    f"got '{self.operand.to_infix()}' with unit '{op_unit.symbol}' ({op_unit.dimension.format_si()})."
                )
            return DIMENSIONLESS
        raise SymbolicCompilationError(f"Unhandled unary op '{self.op}'.")

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
        lu = self.left.infer_unit()
        ru = self.right.infer_unit()

        if self.op in ("+", "-"):
            if isinstance(self.left, ConstNode) and self.left.value == 0.0:
                return ru
            if isinstance(self.right, ConstNode) and self.right.value == 0.0:
                return lu
            if lu.dimension != ru.dimension:
                raise DimensionError(
                    f"Dimensional mismatch in '{self.to_infix()}': cannot apply '{self.op}' between "
                    f"'{self.left.to_infix()}' ({lu.symbol}, {lu.dimension.format_si()}) and "
                    f"'{self.right.to_infix()}' ({ru.symbol}, {ru.dimension.format_si()})."
                )
            return lu

        if self.op == "*":
            return lu * ru

        if self.op == "/":
            return lu / ru

        if self.op == "**":
            if ru.dimension != DIM_DIMENSIONLESS:
                raise DimensionError(
                    f"Exponent in '{self.to_infix()}' must be dimensionless, "
                    f"got '{ru.symbol}' ({ru.dimension.format_si()})."
                )
            if isinstance(self.right, ConstNode):
                exp_val = self.right.value * ru.scale
                return lu**exp_val
            if not lu.dimension.is_dimensionless:
                raise DimensionError(
                    f"Base '{self.left.to_infix()}' raised to non-constant symbolic exponent "
                    f"'{self.right.to_infix()}' must be dimensionless, got {lu.dimension.format_si()}."
                )
            return DIMENSIONLESS

        raise SymbolicCompilationError(f"Unhandled binary op '{self.op}'.")

    def eval_tensor(self, env: Mapping[str, torch.Tensor]) -> torch.Tensor:
        lu = self.left.infer_unit()
        ru = self.right.infer_unit()
        lv = self.left.eval_tensor(env)
        rv = self.right.eval_tensor(env)

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
            rv_sign = torch.where(rv >= 0.0, torch.ones_like(rv), -torch.ones_like(rv))
            rv_safe = rv_sign * torch.clamp(torch.abs(rv), min=1e-15)
            return lv / rv_safe

        if self.op == "**":
            rv_si = rv * ru.scale if abs(ru.scale - 1.0) > 1e-12 else rv
            if isinstance(self.right, ConstNode):
                exp_val = self.right.value * ru.scale
                if float(exp_val).is_integer():
                    return lv ** int(round(exp_val))
                return torch.clamp(lv, min=1e-15) ** exp_val
            lv_si = lv * lu.scale if abs(lu.scale - 1.0) > 1e-12 else lv
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


# Canonical alias mapping so both concise and full primitive names work in symbolic strings
_PRIMITIVE_ALIASES: dict[str, str] = {
    "boltzmann": "boltzmann",
    "boltzmann_switch": "boltzmann",
    "hill": "hill",
    "hill_inhib": "hill_inhib",
    "michaelis_menten": "michaelis_menten",
    "mm_flux": "michaelis_menten",
    "bernoulli": "bernoulli",
    "bernoulli_deriv": "bernoulli_deriv",
    "ghk_flux": "ghk_flux",
    "nernst_mv": "nernst_mv",
    "ohmic": "ohmic",
    "relaxation": "relaxation",
    "quadratic_storage": "quadratic_storage",
    "eyring": "eyring",
    "eyring_rate": "eyring",
    "cubic": "cubic",
    "cubic_bistable": "cubic",
    "power_drag": "power_drag",
    "saturating_drive": "saturating_drive",
    "impulse": "impulse",
    "surrogate_impulse": "impulse",
    "dist3d": "dist3d",
    "euclidean_dist": "dist3d",
    "inv_dist_pow": "inv_dist_pow",
    "diffusive_flux": "diffusive_flux",
}


@dataclass(frozen=True)
class PrimitiveCallNode(ExprNode):
    """
    Symbolic invocation of a Layer 2 mathematical primitive with compile-time dimensional checking,
    automatic cross-prefix SI conversion, and symbolic differentiation.

    Supported primitives (and aliases):
        - `boltzmann(x, x_half, k)` / `boltzmann_switch` -> `1 / (1 + exp((x_half - x) / k))` (`DIMENSIONLESS`)
        - `hill(c, kd[, n])`                             -> `c^n / (kd^n + c^n)` (`DIMENSIONLESS`)
        - `hill_inhib(c, ki[, n])`                       -> `ki^n / (ki^n + c^n)` (`DIMENSIONLESS`)
        - `michaelis_menten(s, vmax, km)` / `mm_flux`    -> `vmax * s / (km + s)` (`vmax.unit`)
        - `bernoulli(u)`                                 -> `u / (exp(u) - 1)` (`DIMENSIONLESS`)
        - `bernoulli_deriv(u)`                           -> `d/du [u / (exp(u) - 1)]` (`DIMENSIONLESS`)
        - `ghk_flux(v, c_in, c_out, perm[, z])`          -> GHK electrodiffusion flux (`perm.unit * (C/mol) * c_in.unit`)
        - `nernst_mv(c_out, c_in[, z])`                  -> `(RT / zF) * ln(c_out / c_in)` (`mV`)
        - `ohmic(e, e_eq, g)`                            -> `g * (e - e_eq)` (`g.unit * e.unit`)
        - `relaxation(x, x_inf, tau)`                    -> `(x_inf - x) / tau` (`x.unit / tau.unit`)
        - `quadratic_storage(x, cap)`                    -> `0.5 * x^2 / cap` (`x.unit^2 / cap.unit`)
        - `eyring(dG, k0[, delta])` / `eyring_rate`      -> `k0 * exp(delta * dG / RT)` (`k0.unit`)
        - `cubic(x, a, b[, bias])` / `cubic_bistable`    -> `a * x - b * x^3 + bias` (`x.unit`)
        - `power_drag(v, gamma[, p])`                    -> `gamma * |v|^(p-1) * v` (`gamma.unit * v.unit^p`)
        - `saturating_drive(u, u_max, u_half)`           -> `u_max * tanh(relu(u) / u_half)` (`u_max.unit`)
        - `impulse(x, x_th[, beta])` / `surrogate_impulse` -> `beta * sig * (1 - sig)` (`DIMENSIONLESS`)
    """

    primitive_name: str
    args: tuple[ExprNode, ...]

    def __post_init__(self) -> None:
        canonical = _PRIMITIVE_ALIASES.get(self.primitive_name, self.primitive_name)
        object.__setattr__(self, "primitive_name", canonical)

    def infer_unit(self) -> Unit:
        name = self.primitive_name
        a = self.args

        if name == "boltzmann":
            if len(a) != 3:
                raise SymbolicCompilationError("boltzmann(x, x_half, k) requires 3 arguments.")
            u_x, u_xh, u_k = a[0].infer_unit(), a[1].infer_unit(), a[2].infer_unit()
            if u_x.dimension != u_xh.dimension or u_x.dimension != u_k.dimension:
                raise DimensionError(
                    f"boltzmann(x, x_half, k) arguments must share the same dimension, got "
                    f"{u_x.dimension.format_si()}, {u_xh.dimension.format_si()}, {u_k.dimension.format_si()}."
                )
            return DIMENSIONLESS

        if name in ("hill", "hill_inhib"):
            if len(a) not in (2, 3):
                raise SymbolicCompilationError(f"{name}(c, kd[, n]) requires 2 or 3 arguments.")
            u_c, u_k = a[0].infer_unit(), a[1].infer_unit()
            if u_c.dimension != u_k.dimension:
                raise DimensionError(
                    f"{name}(c, kd) requires c and kd to have matching dimensions, got "
                    f"{u_c.dimension.format_si()} vs {u_k.dimension.format_si()}."
                )
            if len(a) == 3 and a[2].infer_unit().dimension != DIM_DIMENSIONLESS:
                raise DimensionError(f"{name} cooperativity exponent 'n' must be dimensionless.")
            return DIMENSIONLESS

        if name == "michaelis_menten":
            if len(a) != 3:
                raise SymbolicCompilationError("michaelis_menten(s, vmax, km) requires 3 arguments.")
            u_s, u_vmax, u_km = a[0].infer_unit(), a[1].infer_unit(), a[2].infer_unit()
            if u_s.dimension != u_km.dimension:
                raise DimensionError(
                    f"michaelis_menten(s, vmax, km) requires s and km to share dimensions, got "
                    f"{u_s.dimension.format_si()} vs {u_km.dimension.format_si()}."
                )
            return u_vmax

        if name in ("bernoulli", "bernoulli_deriv"):
            if len(a) != 1:
                raise SymbolicCompilationError(f"{name}(u) requires 1 argument.")
            if a[0].infer_unit().dimension != DIM_DIMENSIONLESS:
                raise DimensionError(f"{name}(u) argument 'u' must be dimensionless.")
            return DIMENSIONLESS

        if name == "ghk_flux":
            if len(a) not in (4, 5):
                raise SymbolicCompilationError("ghk_flux(v, c_in, c_out, perm[, z]) requires 4 or 5 arguments.")
            u_v, u_cin, u_cout, u_perm = (
                a[0].infer_unit(),
                a[1].infer_unit(),
                a[2].infer_unit(),
                a[3].infer_unit(),
            )
            if u_v.dimension != DIM_VOLTAGE:
                raise DimensionError(
                    f"ghk_flux voltage argument must have electric potential dimension, got {u_v.dimension.format_si()}."
                )
            if u_cin.dimension != u_cout.dimension:
                raise DimensionError(
                    f"ghk_flux c_in and c_out must share concentration dimension, got "
                    f"{u_cin.dimension.format_si()} vs {u_cout.dimension.format_si()}."
                )
            if u_perm.dimension != DIM_VELOCITY:
                raise DimensionError(
                    f"ghk_flux permeability must have velocity dimension (L/T), got {u_perm.dimension.format_si()}."
                )
            if len(a) == 5 and a[4].infer_unit().dimension != DIM_DIMENSIONLESS:
                raise DimensionError("ghk_flux valence 'z' must be dimensionless.")
            return m_per_s * (C / mol) * u_cin

        if name == "ohmic":
            if len(a) != 3:
                raise SymbolicCompilationError("ohmic(e, e_eq, g) requires 3 arguments.")
            u_e, u_eq, u_g = a[0].infer_unit(), a[1].infer_unit(), a[2].infer_unit()
            if u_e.dimension != u_eq.dimension:
                raise DimensionError(
                    f"ohmic(e, e_eq, g) requires e and e_eq to share dimensions, got "
                    f"{u_e.dimension.format_si()} vs {u_eq.dimension.format_si()}."
                )
            return u_g * u_e

        if name == "relaxation":
            if len(a) != 3:
                raise SymbolicCompilationError("relaxation(x, x_inf, tau) requires 3 arguments.")
            u_x, u_inf, u_tau = a[0].infer_unit(), a[1].infer_unit(), a[2].infer_unit()
            if u_x.dimension != u_inf.dimension:
                raise DimensionError(
                    f"relaxation(x, x_inf, tau) requires x and x_inf to share dimensions, got "
                    f"{u_x.dimension.format_si()} vs {u_inf.dimension.format_si()}."
                )
            return u_x / u_tau

        if name == "quadratic_storage":
            if len(a) != 2:
                raise SymbolicCompilationError("quadratic_storage(x, cap) requires 2 arguments.")
            return (a[0].infer_unit() ** 2) / a[1].infer_unit()

        if name == "nernst_mv":
            if len(a) not in (2, 3):
                raise SymbolicCompilationError("nernst_mv(c_out, c_in[, z]) requires 2 or 3 arguments.")
            if a[0].infer_unit().dimension != a[1].infer_unit().dimension:
                raise DimensionError("nernst_mv(c_out, c_in) concentrations must share dimensions.")
            if len(a) == 3 and a[2].infer_unit().dimension != DIM_DIMENSIONLESS:
                raise DimensionError("nernst_mv valence 'z' must be dimensionless.")
            return mV

        if name == "eyring":
            if len(a) not in (2, 3):
                raise SymbolicCompilationError("eyring(dG, k0[, delta]) requires 2 or 3 arguments.")
            u_dg = a[0].infer_unit()
            if u_dg.dimension not in (DIM_CHEMICAL_POTENTIAL, DIM_DIMENSIONLESS):
                raise DimensionError(
                    f"eyring(dG, k0) barrier dG must have molar energy (J/mol) or dimensionless units, "
                    f"got {u_dg.dimension.format_si()}."
                )
            if len(a) == 3 and a[2].infer_unit().dimension != DIM_DIMENSIONLESS:
                raise DimensionError("eyring asymmetry factor 'delta' must be dimensionless.")
            return a[1].infer_unit()

        if name == "cubic":
            if len(a) not in (3, 4):
                raise SymbolicCompilationError("cubic(x, a, b[, bias]) requires 3 or 4 arguments.")
            return a[0].infer_unit()

        if name == "power_drag":
            if len(a) not in (2, 3):
                raise SymbolicCompilationError("power_drag(v, gamma[, p]) requires 2 or 3 arguments.")
            u_v, u_g = a[0].infer_unit(), a[1].infer_unit()
            p_val = 1.0
            if len(a) == 3:
                if a[2].infer_unit().dimension != DIM_DIMENSIONLESS:
                    raise DimensionError("power_drag exponent 'p' must be dimensionless.")
                if isinstance(a[2], ConstNode):
                    p_val = a[2].value * a[2].unit.scale
            return u_g * (u_v**p_val)

        if name == "saturating_drive":
            if len(a) != 3:
                raise SymbolicCompilationError("saturating_drive(u, u_max, u_half) requires 3 arguments.")
            u_in, u_max, u_half = a[0].infer_unit(), a[1].infer_unit(), a[2].infer_unit()
            if u_in.dimension != u_half.dimension:
                raise DimensionError("saturating_drive(u, u_max, u_half) requires u and u_half to share dimensions.")
            return u_max

        if name == "impulse":
            if len(a) not in (2, 3):
                raise SymbolicCompilationError("impulse(x, x_th[, beta]) requires 2 or 3 arguments.")
            if a[0].infer_unit().dimension != a[1].infer_unit().dimension:
                raise DimensionError("impulse(x, x_th) state and threshold must share dimensions.")
            return DIMENSIONLESS

        if name == "dist3d":
            if len(a) not in (2, 3):
                raise SymbolicCompilationError("dist3d(dx, dy[, dz]) requires 2 or 3 coordinate difference arguments.")
            u0 = a[0].infer_unit()
            for arg in a[1:]:
                if arg.infer_unit().dimension != u0.dimension:
                    raise DimensionError(
                        f"dist3d arguments must share spatial dimension {u0.dimension.format_si()}, "
                        f"got {arg.infer_unit().dimension.format_si()}."
                    )
            return u0

        if name == "inv_dist_pow":
            if len(a) not in (3, 4):
                raise SymbolicCompilationError(
                    "inv_dist_pow(dx, dy[, dz], p) requires 3 (2D + p) or 4 (3D + p) arguments."
                )
            coord_args = a[:-1]
            p_node = a[-1]
            if p_node.infer_unit().dimension != DIM_DIMENSIONLESS:
                raise DimensionError("inv_dist_pow exponent 'p' must be dimensionless.")
            u0 = coord_args[0].infer_unit()
            for arg in coord_args[1:]:
                if arg.infer_unit().dimension != u0.dimension:
                    raise DimensionError(
                        f"inv_dist_pow coordinate arguments must share dimension {u0.dimension.format_si()}."
                    )
            p_val = p_node.value * p_node.unit.scale if isinstance(p_node, ConstNode) else 1.0
            return u0 ** (-p_val)

        if name == "diffusive_flux":
            if len(a) != 3:
                raise SymbolicCompilationError("diffusive_flux(u_i, u_j, k_diff) requires 3 arguments.")
            u_i, u_j, u_k = a[0].infer_unit(), a[1].infer_unit(), a[2].infer_unit()
            if u_i.dimension != u_j.dimension:
                raise DimensionError(
                    f"diffusive_flux(u_i, u_j, k_diff) requires u_i and u_j to share dimensions, "
                    f"got {u_i.dimension.format_si()} vs {u_j.dimension.format_si()}."
                )
            return u_k * u_i

        raise SymbolicCompilationError(f"Unknown primitive call '{name}'.")

    def eval_tensor(self, env: Mapping[str, torch.Tensor]) -> torch.Tensor:
        name = self.primitive_name
        a = self.args

        if name == "boltzmann":
            u_x = a[0].infer_unit()
            x = a[0].eval_tensor(env)
            xh = a[1].evaluate_in_unit(env, u_x)
            k = a[2].evaluate_in_unit(env, u_x)
            return boltzmann_partition_switch(x, xh, k)

        if name in ("hill", "hill_inhib"):
            u_c = a[0].infer_unit()
            c = a[0].eval_tensor(env)
            kd = a[1].evaluate_in_unit(env, u_c)
            n = a[2].evaluate_in_unit(env, DIMENSIONLESS) if len(a) == 3 else 1.0
            return cooperative_hill_saturation(c, kd, n, inhibitory=(name == "hill_inhib"))

        if name == "michaelis_menten":
            u_s = a[0].infer_unit()
            s_val = a[0].eval_tensor(env)
            vmax = a[1].eval_tensor(env)
            km = a[2].evaluate_in_unit(env, u_s)
            return vmax * cooperative_hill_saturation(s_val, km, hill_n=1.0)

        if name == "bernoulli":
            u = a[0].evaluate_in_unit(env, DIMENSIONLESS)
            return smooth_bernoulli_quotient(u)

        if name == "bernoulli_deriv":
            u = a[0].evaluate_in_unit(env, DIMENSIONLESS)
            return _smooth_bernoulli_derivative(u)

        if name == "ghk_flux":
            u_cin = a[1].infer_unit()
            v_mv = a[0].evaluate_in_unit(env, mV)
            c_in = a[1].eval_tensor(env)
            c_out = a[2].evaluate_in_unit(env, u_cin)
            perm = a[3].evaluate_in_unit(env, m_per_s)
            z = float(a[4].value) if (len(a) == 5 and isinstance(a[4], ConstNode)) else 1
            return ghk_electrodiffusion_flux(v_mv, c_in, c_out, perm, valence_z=z)

        if name == "ohmic":
            u_e = a[0].infer_unit()
            e = a[0].eval_tensor(env)
            e_eq = a[1].evaluate_in_unit(env, u_e)
            g = a[2].eval_tensor(env)
            return ohmic_dissipative_driving_force(e, e_eq, g)

        if name == "relaxation":
            u_x = a[0].infer_unit()
            x = a[0].eval_tensor(env)
            x_inf = a[1].evaluate_in_unit(env, u_x)
            tau = a[2].eval_tensor(env)
            return linear_relaxation_rate(x, x_inf, tau)

        if name == "quadratic_storage":
            x = a[0].eval_tensor(env)
            cap = a[1].eval_tensor(env)
            return quadratic_energy_storage(x, cap)

        if name == "nernst_mv":
            u_c = a[0].infer_unit()
            c_out = a[0].eval_tensor(env)
            c_in = a[1].evaluate_in_unit(env, u_c)
            z = float(a[2].value) if (len(a) == 3 and isinstance(a[2], ConstNode)) else 1.0
            return nernst_equilibrium_potential_mv(c_out, c_in, valence_z=z)

        if name == "eyring":
            dg = a[0].eval_tensor(env)
            k0 = a[1].eval_tensor(env)
            delta = a[2].evaluate_in_unit(env, DIMENSIONLESS) if len(a) == 3 else 0.5
            return eyring_barrier_rate(dg, k0, asymmetry_delta=delta)

        if name == "cubic":
            x = a[0].eval_tensor(env)
            lin_a = a[1].eval_tensor(env)
            cub_b = a[2].eval_tensor(env)
            bias = a[3].eval_tensor(env) if len(a) == 4 else 0.0
            return cubic_bistable_feedback(x, lin_a, cub_b, bias)

        if name == "power_drag":
            v = a[0].eval_tensor(env)
            gamma = a[1].eval_tensor(env)
            p = float(a[2].value * a[2].unit.scale) if (len(a) == 3 and isinstance(a[2], ConstNode)) else 1.0
            return ohmic_dissipative_driving_force(v, 0.0, gamma, power_exponent=p)

        if name == "saturating_drive":
            u_in = a[0].infer_unit()
            u_val = a[0].eval_tensor(env)
            u_max = a[1].eval_tensor(env)
            u_half = torch.clamp(a[2].evaluate_in_unit(env, u_in), min=1e-12)
            return u_max * torch.tanh(torch.relu(u_val) / u_half)

        if name == "impulse":
            u_x = a[0].infer_unit()
            x = a[0].eval_tensor(env)
            x_th = a[1].evaluate_in_unit(env, u_x)
            beta = float(a[2].value) if (len(a) == 3 and isinstance(a[2], ConstNode)) else 1.0
            return surrogate_event_impulse(x, x_th, sharpness_beta=beta)

        if name == "dist3d":
            u0 = a[0].infer_unit()
            dx = a[0].eval_tensor(env)
            dy = a[1].evaluate_in_unit(env, u0)
            dz = a[2].evaluate_in_unit(env, u0) if len(a) == 3 else 0.0
            return euclidean_distance_3d(dx, dy, dz)

        if name == "inv_dist_pow":
            coord_args = a[:-1]
            p_node = a[-1]
            u0 = coord_args[0].infer_unit()
            dx = coord_args[0].eval_tensor(env)
            dy = coord_args[1].evaluate_in_unit(env, u0)
            dz = coord_args[2].evaluate_in_unit(env, u0) if len(coord_args) == 3 else 0.0
            p_val = float(p_node.value * p_node.unit.scale) if isinstance(p_node, ConstNode) else 1.0
            return inverse_distance_power_3d(dx, dy, dz, power_p=p_val)

        if name == "diffusive_flux":
            u_i = a[0].infer_unit()
            val_i = a[0].eval_tensor(env)
            val_j = a[1].evaluate_in_unit(env, u_i)
            k_diff = a[2].eval_tensor(env)
            return k_diff * (val_j - val_i)

        raise SymbolicCompilationError(f"Unknown primitive call '{name}'.")

    def diff(self, var_name: str) -> ExprNode:
        name = self.primitive_name
        a = self.args

        if name == "boltzmann":
            z = BinaryOpNode("/", BinaryOpNode("-", a[1], a[0]), a[2])
            expanded = BinaryOpNode(
                "/",
                ConstNode(1.0, DIMENSIONLESS),
                BinaryOpNode("+", ConstNode(1.0, DIMENSIONLESS), UnaryOpNode("exp", z)),
            )
            return expanded.diff(var_name)

        if name == "ohmic":
            expanded = BinaryOpNode("*", a[2], BinaryOpNode("-", a[0], a[1]))
            return expanded.diff(var_name)

        if name == "relaxation":
            expanded = BinaryOpNode("/", BinaryOpNode("-", a[1], a[0]), a[2])
            return expanded.diff(var_name)

        if name == "quadratic_storage":
            expanded = BinaryOpNode(
                "/",
                BinaryOpNode(
                    "*",
                    ConstNode(0.5, DIMENSIONLESS),
                    BinaryOpNode("**", a[0], ConstNode(2.0, DIMENSIONLESS)),
                ),
                a[1],
            )
            return expanded.diff(var_name)

        if name in ("hill", "hill_inhib"):
            n_node = a[2] if len(a) == 3 else ConstNode(1.0, DIMENSIONLESS)
            c_n = BinaryOpNode("**", a[0], n_node)
            k_n = BinaryOpNode("**", a[1], n_node)
            num = k_n if name == "hill_inhib" else c_n
            expanded = BinaryOpNode("/", num, BinaryOpNode("+", k_n, c_n))
            return expanded.diff(var_name)

        if name == "michaelis_menten":
            expanded = BinaryOpNode("*", a[1], BinaryOpNode("/", a[0], BinaryOpNode("+", a[2], a[0])))
            return expanded.diff(var_name)

        if name == "bernoulli":
            du = a[0].diff(var_name).simplify()
            op_u = a[0].infer_unit()
            if abs(op_u.scale - 1.0) > 1e-12:
                du = BinaryOpNode("*", ConstNode(op_u.scale, DIMENSIONLESS), du).simplify()
            return BinaryOpNode("*", PrimitiveCallNode("bernoulli_deriv", (a[0],)), du).simplify()

        if name == "nernst_mv":
            z_val = float(a[2].value) if (len(a) == 3 and isinstance(a[2], ConstNode)) else 1.0
            vt_mv = (float(R_GAS.value) * float(BODY_TEMPERATURE_K.value) / (z_val * float(FARADAY.value))) * 1e3
            expanded = BinaryOpNode(
                "*",
                ConstNode(vt_mv, mV),
                UnaryOpNode("log", BinaryOpNode("/", a[0], a[1])),
            )
            return expanded.diff(var_name)

        if name == "eyring":
            rt_val = float(R_GAS.value) * float(BODY_TEMPERATURE_K.value)
            delta_node = a[2] if len(a) == 3 else ConstNode(0.5, DIMENSIONLESS)
            rt_node = ConstNode(rt_val, a[0].infer_unit())
            arg = BinaryOpNode("*", delta_node, BinaryOpNode("/", a[0], rt_node))
            expanded = BinaryOpNode("*", a[1], UnaryOpNode("exp", arg))
            return expanded.diff(var_name)

        if name == "cubic":
            u_x = a[0].infer_unit()
            lin_term = BinaryOpNode("*", a[1], a[0])
            cub_term = BinaryOpNode("*", a[2], BinaryOpNode("**", a[0], ConstNode(3.0, DIMENSIONLESS)))
            # Keep unit homogeneous with x by scaling derivative directly
            dx = a[0].diff(var_name)
            da = a[1].diff(var_name)
            db = a[2].diff(var_name)
            d_lin = BinaryOpNode("+", BinaryOpNode("*", da, a[0]), BinaryOpNode("*", a[1], dx))
            x2 = BinaryOpNode("**", a[0], ConstNode(2.0, DIMENSIONLESS))
            x3 = BinaryOpNode("**", a[0], ConstNode(3.0, DIMENSIONLESS))
            d_cub = BinaryOpNode(
                "+",
                BinaryOpNode("*", db, x3),
                BinaryOpNode("*", BinaryOpNode("*", ConstNode(3.0, DIMENSIONLESS), a[2]), BinaryOpNode("*", x2, dx)),
            )
            res = BinaryOpNode("-", d_lin, d_cub)
            if len(a) == 4:
                res = BinaryOpNode("+", res, a[3].diff(var_name))
            return res.simplify()

        if name == "power_drag":
            p_node = a[2] if len(a) == 3 else ConstNode(1.0, DIMENSIONLESS)
            if isinstance(p_node, ConstNode) and p_node.value == 1.0:
                return BinaryOpNode("*", a[1], a[0]).diff(var_name)
            p_val = p_node.value if isinstance(p_node, ConstNode) else 1.0
            abs_v_pow = BinaryOpNode("**", UnaryOpNode("abs", a[0]), ConstNode(p_val - 1.0, DIMENSIONLESS))
            dv = a[0].diff(var_name)
            dg = a[1].diff(var_name)
            term1 = BinaryOpNode("*", dg, BinaryOpNode("*", abs_v_pow, a[0]))
            term2 = BinaryOpNode(
                "*",
                BinaryOpNode("*", ConstNode(p_val, DIMENSIONLESS), a[1]),
                BinaryOpNode("*", abs_v_pow, dv),
            )
            return BinaryOpNode("+", term1, term2).simplify()

        if name == "impulse":
            beta_val = float(a[2].value) if (len(a) == 3 and isinstance(a[2], ConstNode)) else 1.0
            k_node = ConstNode(1.0 / beta_val, a[0].infer_unit())
            sig = PrimitiveCallNode("boltzmann", (a[0], a[1], k_node))
            one_minus_sig = BinaryOpNode("-", ConstNode(1.0, DIMENSIONLESS), sig)
            expanded = BinaryOpNode("*", ConstNode(beta_val, DIMENSIONLESS), BinaryOpNode("*", sig, one_minus_sig))
            return expanded.diff(var_name)

        if name == "dist3d":
            u0 = a[0].infer_unit()
            eps_sq = ConstNode(1e-12, u0 ** 2)
            sum_sq = BinaryOpNode(
                "+",
                BinaryOpNode("**", a[0], ConstNode(2.0, DIMENSIONLESS)),
                BinaryOpNode("**", a[1], ConstNode(2.0, DIMENSIONLESS)),
            )
            if len(a) == 3:
                sum_sq = BinaryOpNode("+", sum_sq, BinaryOpNode("**", a[2], ConstNode(2.0, DIMENSIONLESS)))
            sum_sq = BinaryOpNode("+", sum_sq, eps_sq)
            return UnaryOpNode("sqrt", sum_sq).diff(var_name)

        if name == "inv_dist_pow":
            coord_args = a[:-1]
            p_node = a[-1]
            p_val = float(p_node.value * p_node.unit.scale) if isinstance(p_node, ConstNode) else 1.0
            dist_node = PrimitiveCallNode("dist3d", coord_args)
            expanded = BinaryOpNode("**", dist_node, ConstNode(-p_val, DIMENSIONLESS))
            return expanded.diff(var_name)

        if name == "diffusive_flux":
            expanded = BinaryOpNode("*", a[2], BinaryOpNode("-", a[1], a[0]))
            return expanded.diff(var_name)

        raise SymbolicCompilationError(f"Symbolic diff not defined for primitive '{name}'; use PyTorch autograd.")

    def simplify(self) -> ExprNode:
        return PrimitiveCallNode(self.primitive_name, tuple(arg.simplify() for arg in self.args))

    def tree_complexity(self) -> int:
        return 2 + sum(arg.tree_complexity() for arg in self.args)

    def free_symbols(self) -> set[str]:
        out: set[str] = set()
        for arg in self.args:
            out |= arg.free_symbols()
        return out

    def to_infix(self) -> str:
        inner = ", ".join(arg.to_infix() for arg in self.args)
        return f"{self.primitive_name}({inner})"


def parse_symbolic_expr(
    expr_str: str,
    symbol_table: Mapping[str, SymbolNode | ConstNode | Unit] | None = None,
) -> ExprNode:
    """
    Safely parse a mathematical string expression into a typed `ExprNode` tree using
    Python's `ast.parse(..., mode='eval')` (zero `eval()` execution).

    `symbol_table` maps variable/parameter names to either a `SymbolNode`, a `ConstNode`,
    or a Layer 1 `Unit` (which is automatically wrapped in `SymbolNode(name, unit)`).
    """
    if not isinstance(expr_str, str) or not expr_str.strip():
        raise SymbolicCompilationError("Expression string cannot be empty.")

    sym_map: dict[str, ExprNode] = {}
    if symbol_table:
        for k, v in symbol_table.items():
            if isinstance(v, (SymbolNode, ConstNode)):
                sym_map[k] = v
            elif isinstance(v, Unit):
                sym_map[k] = SymbolNode(name=k, unit=v)
            else:
                raise TypeError(f"Invalid symbol_table entry for '{k}': {type(v).__name__}.")

    normalized = expr_str.strip().replace("^", "**")
    try:
        parsed = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise SymbolicCompilationError(f"Syntax error in symbolic expression '{expr_str}': {exc}") from exc

    unary_funcs = {"exp", "log", "sin", "cos", "tanh", "sqrt", "abs"}
    primitive_funcs = set(_PRIMITIVE_ALIASES.keys())

    def _convert(node: ast.AST) -> ExprNode:
        if isinstance(node, ast.Expression):
            return _convert(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return ConstNode(float(node.value), DIMENSIONLESS)
            raise SymbolicCompilationError(f"Unsupported constant literal {node.value!r} in '{expr_str}'.")
        if isinstance(node, ast.Name):
            if node.id in sym_map:
                return sym_map[node.id]
            raise SymbolicCompilationError(
                f"Unknown symbol '{node.id}' in expression '{expr_str}'. "
                f"Declared symbols: {sorted(sym_map.keys())}."
            )
        if isinstance(node, ast.UnaryOp):
            operand = _convert(node.operand)
            if isinstance(node.op, ast.USub):
                if isinstance(operand, ConstNode):
                    return ConstNode(-operand.value, operand.unit)
                return UnaryOpNode("neg", operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            raise SymbolicCompilationError(f"Unsupported unary operator {type(node.op).__name__}.")
        if isinstance(node, ast.BinOp):
            left = _convert(node.left)
            right = _convert(node.right)
            op_map = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
                ast.Pow: "**",
            }
            for ast_type, op_sym in op_map.items():
                if isinstance(node.op, ast_type):
                    return BinaryOpNode(op_sym, left, right)
            raise SymbolicCompilationError(f"Unsupported binary operator {type(node.op).__name__}.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise SymbolicCompilationError("Only direct function calls are allowed in symbolic expressions.")
            fn_name = node.func.id
            if fn_name not in unary_funcs and fn_name not in primitive_funcs:
                raise SymbolicCompilationError(f"Unrecognized function or primitive '{fn_name}' in '{expr_str}'.")
            if node.keywords:
                raise SymbolicCompilationError(
                    f"Keyword arguments are not supported in symbolic call '{fn_name}'; pass positional arguments."
                )
            args = tuple(_convert(arg) for arg in node.args)
            if fn_name in unary_funcs:
                if len(args) != 1:
                    raise SymbolicCompilationError(f"Function '{fn_name}' expects 1 argument, got {len(args)}.")
                return UnaryOpNode(fn_name, args[0])
            return PrimitiveCallNode(fn_name, args)

        raise SymbolicCompilationError(f"Unsupported AST construct '{type(node).__name__}' in '{expr_str}'.")

    tree = _convert(parsed)
    # Trigger compile-time dimensional inference immediately so invalid units fail at parse time
    tree.infer_unit()
    return tree
