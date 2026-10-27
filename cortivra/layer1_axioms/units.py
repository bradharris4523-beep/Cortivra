"""
SI dimensional algebra, unit conversion, Buckingham Pi dimensional analysis,
and physical constants.

Represents physical dimensions in the 7-element SI basis:
    [M (kg), L (m), T (s), I (A), Theta (K), N (mol), J (cd)]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import torch

NumericValue = Union[int, float, np.ndarray, torch.Tensor]
_EXP_TOL = 1e-7


class DimensionError(TypeError):
    """Raised when an operation violates SI dimensional homogeneity."""


def _clean_exp(val: float) -> float:
    rounded = round(val)
    if abs(val - rounded) < _EXP_TOL:
        return float(rounded)
    return float(round(val, 6))


@dataclass(frozen=True)
class Dimension:
    """7-dimensional SI exponent vector: kg^M * m^L * s^T * A^I * K^Theta * mol^N * cd^J."""

    M: float = 0.0
    L: float = 0.0
    T: float = 0.0
    I: float = 0.0
    Theta: float = 0.0
    N: float = 0.0
    J: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("M", "L", "T", "I", "Theta", "N", "J"):
            val = getattr(self, field_name)
            if math.isnan(val) or math.isinf(val):
                raise DimensionError(f"Dimension exponent '{field_name}' must be finite, got {val}.")
            object.__setattr__(self, field_name, _clean_exp(val))

    @property
    def is_dimensionless(self) -> bool:
        return (
            self.M == 0.0
            and self.L == 0.0
            and self.T == 0.0
            and self.I == 0.0
            and self.Theta == 0.0
            and self.N == 0.0
            and self.J == 0.0
        )

    @property
    def as_tuple(self) -> tuple[float, float, float, float, float, float, float]:
        return (self.M, self.L, self.T, self.I, self.Theta, self.N, self.J)

    @property
    def as_vector(self) -> np.ndarray:
        return np.array(self.as_tuple, dtype=np.float64)

    def __mul__(self, other: Dimension) -> Dimension:
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(
            M=self.M + other.M,
            L=self.L + other.L,
            T=self.T + other.T,
            I=self.I + other.I,
            Theta=self.Theta + other.Theta,
            N=self.N + other.N,
            J=self.J + other.J,
        )

    def __truediv__(self, other: Dimension) -> Dimension:
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(
            M=self.M - other.M,
            L=self.L - other.L,
            T=self.T - other.T,
            I=self.I - other.I,
            Theta=self.Theta - other.Theta,
            N=self.N - other.N,
            J=self.J - other.J,
        )

    def __pow__(self, power: float) -> Dimension:
        if math.isnan(power) or math.isinf(power):
            raise DimensionError(f"Exponent must be finite, got {power}.")
        return Dimension(
            M=self.M * power,
            L=self.L * power,
            T=self.T * power,
            I=self.I * power,
            Theta=self.Theta * power,
            N=self.N * power,
            J=self.J * power,
        )

    def d_dt(self, order: int = 1) -> Dimension:
        """Dimension of the n-th time derivative (d^n / dt^n)."""
        return Dimension(
            M=self.M,
            L=self.L,
            T=self.T - order,
            I=self.I,
            Theta=self.Theta,
            N=self.N,
            J=self.J,
        )

    def int_dt(self, order: int = 1) -> Dimension:
        """Dimension of the n-th time integral."""
        return Dimension(
            M=self.M,
            L=self.L,
            T=self.T + order,
            I=self.I,
            Theta=self.Theta,
            N=self.N,
            J=self.J,
        )

    def d_dx(self, order: int = 1) -> Dimension:
        """Dimension of the n-th spatial derivative (d^n / dx^n)."""
        return Dimension(
            M=self.M,
            L=self.L - order,
            T=self.T,
            I=self.I,
            Theta=self.Theta,
            N=self.N,
            J=self.J,
        )

    def format_si(self) -> str:
        if self.is_dimensionless:
            return "1"
        symbols = [
            ("kg", self.M),
            ("m", self.L),
            ("s", self.T),
            ("A", self.I),
            ("K", self.Theta),
            ("mol", self.N),
            ("cd", self.J),
        ]
        parts: list[str] = []
        for sym, exp in symbols:
            if exp == 0.0:
                continue
            exp_str = str(int(exp)) if exp.is_integer() else f"{exp:g}"
            parts.append(sym if exp == 1.0 else f"{sym}^{exp_str}")
        return "*".join(parts)

    def __repr__(self) -> str:
        return f"Dimension({self.format_si()})"


# Base and derived SI dimensions
DIM_DIMENSIONLESS = Dimension()
DIM_MASS = Dimension(M=1)
DIM_LENGTH = Dimension(L=1)
DIM_TIME = Dimension(T=1)
DIM_CURRENT = Dimension(I=1)
DIM_TEMPERATURE = Dimension(Theta=1)
DIM_AMOUNT = Dimension(N=1)
DIM_LUMINOUS = Dimension(J=1)

DIM_AREA = DIM_LENGTH**2
DIM_VOLUME = DIM_LENGTH**3
DIM_WAVENUMBER = DIM_LENGTH**-1
DIM_FREQUENCY = DIM_TIME**-1
DIM_VELOCITY = DIM_LENGTH / DIM_TIME
DIM_ACCELERATION = DIM_VELOCITY / DIM_TIME
DIM_DIFFUSIVITY = DIM_AREA / DIM_TIME

DIM_MOMENTUM = DIM_MASS * DIM_VELOCITY
DIM_FORCE = DIM_MASS * DIM_ACCELERATION
DIM_PRESSURE = DIM_FORCE / DIM_AREA
DIM_ENERGY = DIM_FORCE * DIM_LENGTH
DIM_POWER = DIM_ENERGY / DIM_TIME
DIM_ACTION = DIM_ENERGY * DIM_TIME
DIM_VISCOSITY = DIM_PRESSURE * DIM_TIME

DIM_ENTROPY = DIM_ENERGY / DIM_TEMPERATURE
DIM_CHEMICAL_POTENTIAL = DIM_ENERGY / DIM_AMOUNT
DIM_MOLAR_ENTROPY = DIM_ENTROPY / DIM_AMOUNT
DIM_CONCENTRATION = DIM_AMOUNT / DIM_VOLUME
DIM_MOLAR_FLUX = DIM_AMOUNT / DIM_TIME
DIM_MOLAR_FLUX_DENSITY = DIM_AMOUNT / (DIM_AREA * DIM_TIME)
DIM_REACTION_RATE = DIM_CONCENTRATION / DIM_TIME

DIM_CHARGE = DIM_CURRENT * DIM_TIME
DIM_MOLAR_CHARGE = DIM_CHARGE / DIM_AMOUNT
DIM_VOLTAGE = DIM_POWER / DIM_CURRENT
DIM_ELECTRIC_FIELD = DIM_VOLTAGE / DIM_LENGTH
DIM_RESISTANCE = DIM_VOLTAGE / DIM_CURRENT
DIM_CONDUCTANCE = DIM_CURRENT / DIM_VOLTAGE
DIM_CAPACITANCE = DIM_CHARGE / DIM_VOLTAGE
DIM_INDUCTANCE = DIM_VOLTAGE * DIM_TIME / DIM_CURRENT
DIM_MAGNETIC_FLUX = DIM_VOLTAGE * DIM_TIME
DIM_MAGNETIC_FIELD = DIM_MAGNETIC_FLUX / DIM_AREA
DIM_PERMITTIVITY = DIM_CAPACITANCE / DIM_LENGTH
DIM_PERMEABILITY = DIM_INDUCTANCE / DIM_LENGTH
DIM_CURRENT_DENSITY = DIM_CURRENT / DIM_AREA
DIM_SPECIFIC_CAPACITANCE = DIM_CAPACITANCE / DIM_AREA
DIM_SPECIFIC_CONDUCTANCE = DIM_CONDUCTANCE / DIM_AREA


@dataclass(frozen=True)
class Unit:
    """Physical unit with an SI `Dimension` and linear scale factor to base SI (`si_val = val * scale`)."""

    symbol: str
    dimension: Dimension
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.scale <= 0.0 or math.isnan(self.scale) or math.isinf(self.scale):
            raise ValueError(f"Unit scale must be finite and strictly positive, got {self.scale}.")

    def is_compatible(self, other: Unit) -> bool:
        return self.dimension == other.dimension

    def conversion_factor_to(self, target: Unit) -> float:
        if not self.is_compatible(target):
            raise DimensionError(
                f"Cannot convert '{self.symbol}' ({self.dimension.format_si()}) "
                f"to '{target.symbol}' ({target.dimension.format_si()})."
            )
        return self.scale / target.scale

    def __mul__(self, other: Unit) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        new_dim = self.dimension * other.dimension
        new_scale = self.scale * other.scale
        sym = (
            f"{self.symbol}*{other.symbol}"
            if self.symbol != "1" and other.symbol != "1"
            else (other.symbol if self.symbol == "1" else self.symbol)
        )
        return Unit(symbol=sym, dimension=new_dim, scale=new_scale)

    def __truediv__(self, other: Unit) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        new_dim = self.dimension / other.dimension
        new_scale = self.scale / other.scale
        if other.symbol == "1":
            sym = self.symbol
        elif self.symbol == "1":
            sym = f"1/{other.symbol}"
        else:
            sym = f"{self.symbol}/{other.symbol}"
        return Unit(symbol=sym, dimension=new_dim, scale=new_scale)

    def __pow__(self, power: float) -> Unit:
        new_dim = self.dimension**power
        new_scale = self.scale**power
        if self.symbol == "1" or power == 0:
            sym = "1"
        else:
            p_str = str(int(power)) if float(power).is_integer() else f"{power:g}"
            sym = (
                f"({self.symbol})^{p_str}"
                if "*" in self.symbol or "/" in self.symbol
                else f"{self.symbol}^{p_str}"
            )
        return Unit(symbol=sym, dimension=new_dim, scale=new_scale)

    def __call__(self, value: NumericValue) -> Quantity:
        return Quantity(value=value, unit=self)


class Quantity:
    """Numeric value (scalar, NumPy array, or PyTorch Tensor) bound to a physical `Unit`."""

    __slots__ = ("_value", "_unit")

    def __init__(self, value: NumericValue, unit: Unit) -> None:
        if isinstance(value, Quantity):
            raise TypeError("Cannot wrap a Quantity inside another Quantity; use `.to(unit)`.")
        if not isinstance(unit, Unit):
            raise TypeError(f"Expected a `Unit` instance, got {type(unit).__name__}.")
        self._value: NumericValue = value
        self._unit: Unit = unit

    @property
    def value(self) -> NumericValue:
        return self._value

    @property
    def unit(self) -> Unit:
        return self._unit

    @property
    def dimension(self) -> Dimension:
        return self._unit.dimension

    @property
    def si_value(self) -> NumericValue:
        if self._unit.scale == 1.0:
            return self._value
        return self._value * self._unit.scale

    def to(self, target_unit: Unit) -> Quantity:
        factor = self._unit.conversion_factor_to(target_unit)
        return Quantity(value=self._value * factor, unit=target_unit)

    def to_si(self) -> Quantity:
        base_unit = Unit(symbol=self.dimension.format_si(), dimension=self.dimension, scale=1.0)
        return Quantity(value=self.si_value, unit=base_unit)

    def to_tensor(
        self,
        target_unit: Unit | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        q = self.to(target_unit) if target_unit is not None else self
        if isinstance(q._value, torch.Tensor):
            return q._value.to(dtype=dtype, device=device)
        return torch.as_tensor(q._value, dtype=dtype, device=device)

    def assert_dimension(self, expected: Dimension, context: str = "") -> None:
        if self.dimension != expected:
            prefix = f"[{context}] " if context else ""
            raise DimensionError(
                f"{prefix}Expected dimension {expected.format_si()}, "
                f"got '{self._unit.symbol}' ({self.dimension.format_si()})."
            )

    def _check_compatible_quantity(self, other: object, op_name: str) -> float:
        if not isinstance(other, Quantity):
            raise DimensionError(
                f"Cannot perform {op_name} between Quantity ({self._unit.symbol}) "
                f"and raw {type(other).__name__}."
            )
        if self.dimension != other.dimension:
            raise DimensionError(
                f"Dimensional mismatch in {op_name}: '{self._unit.symbol}' "
                f"({self.dimension.format_si()}) vs '{other._unit.symbol}' ({other.dimension.format_si()})."
            )
        return other._unit.scale / self._unit.scale

    def __add__(self, other: Quantity) -> Quantity:
        scale_ratio = self._check_compatible_quantity(other, "addition")
        return Quantity(value=self._value + other._value * scale_ratio, unit=self._unit)

    def __sub__(self, other: Quantity) -> Quantity:
        scale_ratio = self._check_compatible_quantity(other, "subtraction")
        return Quantity(value=self._value - other._value * scale_ratio, unit=self._unit)

    def __neg__(self) -> Quantity:
        return Quantity(value=-self._value, unit=self._unit)

    def __abs__(self) -> Quantity:
        if isinstance(self._value, torch.Tensor):
            return Quantity(value=torch.abs(self._value), unit=self._unit)
        if isinstance(self._value, np.ndarray):
            return Quantity(value=np.abs(self._value), unit=self._unit)
        return Quantity(value=abs(self._value), unit=self._unit)

    def __lt__(self, other: Quantity) -> bool | np.ndarray | torch.Tensor:
        scale_ratio = self._check_compatible_quantity(other, "comparison '<'")
        return self._value < (other._value * scale_ratio)

    def __le__(self, other: Quantity) -> bool | np.ndarray | torch.Tensor:
        scale_ratio = self._check_compatible_quantity(other, "comparison '<='")
        return self._value <= (other._value * scale_ratio)

    def __gt__(self, other: Quantity) -> bool | np.ndarray | torch.Tensor:
        scale_ratio = self._check_compatible_quantity(other, "comparison '>'")
        return self._value > (other._value * scale_ratio)

    def __ge__(self, other: Quantity) -> bool | np.ndarray | torch.Tensor:
        scale_ratio = self._check_compatible_quantity(other, "comparison '>='")
        return self._value >= (other._value * scale_ratio)

    def is_close(self, other: Quantity, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Return True if `other` has the same dimension and its SI value is within tolerance."""
        if not isinstance(other, Quantity) or self.dimension != other.dimension:
            return False
        scale_ratio = other._unit.scale / self._unit.scale
        other_scaled = other._value * scale_ratio
        if isinstance(self._value, torch.Tensor) or isinstance(other_scaled, torch.Tensor):
            t_self = torch.as_tensor(self._value, dtype=torch.float64)
            t_other = torch.as_tensor(other_scaled, dtype=torch.float64, device=t_self.device)
            return bool(torch.allclose(t_self, t_other, rtol=rtol, atol=atol))
        if isinstance(self._value, np.ndarray) or isinstance(other_scaled, np.ndarray):
            return bool(np.allclose(self._value, other_scaled, rtol=rtol, atol=atol))
        return math.isclose(float(self._value), float(other_scaled), rel_tol=rtol, abs_tol=atol)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return False
        return self.is_close(other, rtol=1e-7, atol=1e-12)

    def __mul__(self, other: Quantity | NumericValue) -> Quantity:
        if isinstance(other, Quantity):
            new_dim = self.dimension * other.dimension
            if new_dim.is_dimensionless:
                return Quantity(
                    value=self._value * other._value * (self._unit.scale * other._unit.scale),
                    unit=DIMENSIONLESS,
                )
            return Quantity(value=self._value * other._value, unit=self._unit * other._unit)
        if isinstance(other, (int, float, np.ndarray, torch.Tensor)):
            return Quantity(value=self._value * other, unit=self._unit)
        return NotImplemented

    def __rmul__(self, other: NumericValue) -> Quantity:
        if isinstance(other, (int, float, np.ndarray, torch.Tensor)):
            return Quantity(value=other * self._value, unit=self._unit)
        return NotImplemented

    def __truediv__(self, other: Quantity | NumericValue) -> Quantity:
        if isinstance(other, Quantity):
            new_dim = self.dimension / other.dimension
            if new_dim.is_dimensionless:
                return Quantity(
                    value=(self._value / other._value) * (self._unit.scale / other._unit.scale),
                    unit=DIMENSIONLESS,
                )
            return Quantity(value=self._value / other._value, unit=self._unit / other._unit)
        if isinstance(other, (int, float, np.ndarray, torch.Tensor)):
            return Quantity(value=self._value / other, unit=self._unit)
        return NotImplemented

    def __rtruediv__(self, other: NumericValue) -> Quantity:
        if isinstance(other, (int, float, np.ndarray, torch.Tensor)):
            inv_unit = Unit(symbol="1", dimension=DIM_DIMENSIONLESS, scale=1.0) / self._unit
            return Quantity(value=other / self._value, unit=inv_unit)
        return NotImplemented

    def __pow__(self, power: float) -> Quantity:
        return Quantity(value=self._value**power, unit=self._unit**power)

    def sqrt(self) -> Quantity:
        return self**0.5

    def _apply_dimensionless_transcendental(self, fn_name: str, torch_fn, np_fn, math_fn) -> Quantity:
        self.assert_dimension(DIM_DIMENSIONLESS, context=f"{fn_name}(x)")
        val = self.si_value
        if isinstance(val, torch.Tensor):
            res = torch_fn(val)
        elif isinstance(val, np.ndarray):
            res = np_fn(val)
        else:
            res = math_fn(val)
        return Quantity(value=res, unit=DIMENSIONLESS)

    def exp(self) -> Quantity:
        return self._apply_dimensionless_transcendental("exp", torch.exp, np.exp, math.exp)

    def log(self) -> Quantity:
        return self._apply_dimensionless_transcendental("log", torch.log, np.log, math.log)

    def sin(self) -> Quantity:
        return self._apply_dimensionless_transcendental("sin", torch.sin, np.sin, math.sin)

    def cos(self) -> Quantity:
        return self._apply_dimensionless_transcendental("cos", torch.cos, np.cos, math.cos)

    def tanh(self) -> Quantity:
        return self._apply_dimensionless_transcendental("tanh", torch.tanh, np.tanh, math.tanh)

    def __repr__(self) -> str:
        return f"Quantity({self._value!r}, unit='{self._unit.symbol}')"


@dataclass(frozen=True)
class DimensionlessGroup:
    """Dimensionless Pi group `prod_i (var_i ^ exponent_i)` in the nullspace of the SI dimension matrix."""

    variable_names: tuple[str, ...]
    exponents: tuple[float, ...]

    def format_expression(self) -> str:
        num_parts: list[str] = []
        den_parts: list[str] = []
        for name, exp in zip(self.variable_names, self.exponents):
            if abs(exp) < 1e-6:
                continue
            abs_exp = abs(exp)
            exp_str = (
                ""
                if abs(abs_exp - 1.0) < 1e-6
                else (
                    f"^{int(round(abs_exp))}"
                    if abs(abs_exp - round(abs_exp)) < 1e-5
                    else f"^{abs_exp:.2g}"
                )
            )
            term = f"{name}{exp_str}"
            if exp > 0:
                num_parts.append(term)
            else:
                den_parts.append(term)
        num = " * ".join(num_parts) if num_parts else "1"
        if not den_parts:
            return num
        den = " * ".join(den_parts)
        return f"({num}) / ({den})"


def discover_dimensionless_groups(
    variables: Sequence[tuple[str, Unit | Dimension | Quantity]],
    tol: float = 1e-9,
) -> list[DimensionlessGroup]:
    """
    Compute the basis of dimensionless groups for `variables` via the right nullspace
    of the 7 x k SI dimensional exponent matrix `D` (`D * alpha = 0`).
    """
    if not variables:
        return []

    names = tuple(v[0] for v in variables)
    if len(set(names)) != len(names):
        raise ValueError("Variable names passed to discover_dimensionless_groups must be unique.")

    dims: list[Dimension] = []
    for _, obj in variables:
        if isinstance(obj, Dimension):
            dims.append(obj)
        elif isinstance(obj, (Unit, Quantity)):
            dims.append(obj.dimension)
        else:
            raise TypeError(f"Expected Unit, Dimension, or Quantity, got {type(obj).__name__}.")

    k = len(dims)
    d_mat = np.column_stack([d.as_vector for d in dims])  # (7, k)

    _, s_vals, vt = np.linalg.svd(d_mat, full_matrices=True)
    rank = int(np.sum(s_vals > tol))
    null_dim = k - rank
    if null_dim <= 0:
        return []

    null_basis = vt[rank:, :].copy()  # (null_dim, k)

    # Row-reduce null_basis to canonical rational form
    pivot_row = 0
    for col in range(k):
        if pivot_row >= null_dim:
            break
        max_row = pivot_row + int(np.argmax(np.abs(null_basis[pivot_row:, col])))
        if abs(null_basis[max_row, col]) <= 1e-7:
            continue
        null_basis[[pivot_row, max_row]] = null_basis[[max_row, pivot_row]]
        null_basis[pivot_row] = null_basis[pivot_row] / null_basis[pivot_row, col]
        for r in range(null_dim):
            if r != pivot_row and abs(null_basis[r, col]) > 1e-7:
                null_basis[r] = null_basis[r] - null_basis[r, col] * null_basis[pivot_row]
        pivot_row += 1

    groups: list[DimensionlessGroup] = []
    for row in null_basis:
        cleaned = tuple(_clean_exp(float(val)) for val in row)
        residual = np.max(np.abs(d_mat @ np.array(cleaned, dtype=np.float64)))
        if residual <= 1e-5 and any(abs(x) > 1e-6 for x in cleaned):
            groups.append(DimensionlessGroup(variable_names=names, exponents=cleaned))
    return groups


# Standard units
DIMENSIONLESS = Unit("1", DIM_DIMENSIONLESS, 1.0)
PERCENT = Unit("%", DIM_DIMENSIONLESS, 1e-2)

kg = Unit("kg", DIM_MASS, 1.0)
g = Unit("g", DIM_MASS, 1e-3)
mg = Unit("mg", DIM_MASS, 1e-6)
ug = Unit("ug", DIM_MASS, 1e-9)
ng = Unit("ng", DIM_MASS, 1e-12)
Da = Unit("Da", DIM_MASS, 1.66053906660e-27)

m = Unit("m", DIM_LENGTH, 1.0)
cm = Unit("cm", DIM_LENGTH, 1e-2)
mm = Unit("mm", DIM_LENGTH, 1e-3)
um = Unit("um", DIM_LENGTH, 1e-6)
nm = Unit("nm", DIM_LENGTH, 1e-9)
Angstrom = Unit("A_len", DIM_LENGTH, 1e-10)

s = Unit("s", DIM_TIME, 1.0)
ms = Unit("ms", DIM_TIME, 1e-3)
us = Unit("us", DIM_TIME, 1e-6)
ns = Unit("ns", DIM_TIME, 1e-9)
minute = Unit("min", DIM_TIME, 60.0)
hour = Unit("hr", DIM_TIME, 3600.0)

A = Unit("A", DIM_CURRENT, 1.0)
mA = Unit("mA", DIM_CURRENT, 1e-3)
uA = Unit("uA", DIM_CURRENT, 1e-6)
nA = Unit("nA", DIM_CURRENT, 1e-9)
pA = Unit("pA", DIM_CURRENT, 1e-12)

K = Unit("K", DIM_TEMPERATURE, 1.0)

mol = Unit("mol", DIM_AMOUNT, 1.0)
mmol = Unit("mmol", DIM_AMOUNT, 1e-3)
umol = Unit("umol", DIM_AMOUNT, 1e-6)
nmol = Unit("nmol", DIM_AMOUNT, 1e-9)
pmol = Unit("pmol", DIM_AMOUNT, 1e-12)
fmol = Unit("fmol", DIM_AMOUNT, 1e-15)

cd = Unit("cd", DIM_LUMINOUS, 1.0)

m2 = Unit("m^2", DIM_AREA, 1.0)
cm2 = Unit("cm^2", DIM_AREA, 1e-4)
mm2 = Unit("mm^2", DIM_AREA, 1e-6)
um2 = Unit("um^2", DIM_AREA, 1e-12)
Angstrom2 = Unit("A_len^2", DIM_AREA, 1e-20)

m3 = Unit("m^3", DIM_VOLUME, 1.0)
L = Unit("L", DIM_VOLUME, 1e-3)
mL = Unit("mL", DIM_VOLUME, 1e-6)
uL = Unit("uL", DIM_VOLUME, 1e-9)
nL = Unit("nL", DIM_VOLUME, 1e-12)
pL = Unit("pL", DIM_VOLUME, 1e-15)
fL = Unit("fL", DIM_VOLUME, 1e-18)

Hz = Unit("Hz", DIM_FREQUENCY, 1.0)
kHz = Unit("kHz", DIM_FREQUENCY, 1e3)
m_per_s = Unit("m/s", DIM_VELOCITY, 1.0)
m_per_s2 = Unit("m/s^2", DIM_ACCELERATION, 1.0)
m2_per_s = Unit("m^2/s", DIM_DIFFUSIVITY, 1.0)
um2_per_ms = Unit("um^2/ms", DIM_DIFFUSIVITY, 1e-9)

N_force = Unit("N", DIM_FORCE, 1.0)
pN = Unit("pN", DIM_FORCE, 1e-12)
Pa = Unit("Pa", DIM_PRESSURE, 1.0)
kPa = Unit("kPa", DIM_PRESSURE, 1e3)

J = Unit("J", DIM_ENERGY, 1.0)
kJ = Unit("kJ", DIM_ENERGY, 1e3)
mJ = Unit("mJ", DIM_ENERGY, 1e-3)
uJ = Unit("uJ", DIM_ENERGY, 1e-6)
nJ = Unit("nJ", DIM_ENERGY, 1e-9)
pJ = Unit("pJ", DIM_ENERGY, 1e-12)
fJ = Unit("fJ", DIM_ENERGY, 1e-15)
eV = Unit("eV", DIM_ENERGY, 1.602176634e-19)

W = Unit("W", DIM_POWER, 1.0)
mW = Unit("mW", DIM_POWER, 1e-3)
uW = Unit("uW", DIM_POWER, 1e-6)
nW = Unit("nW", DIM_POWER, 1e-9)
pW = Unit("pW", DIM_POWER, 1e-12)

J_per_K = Unit("J/K", DIM_ENTROPY, 1.0)
J_per_mol = Unit("J/mol", DIM_CHEMICAL_POTENTIAL, 1.0)
kJ_per_mol = Unit("kJ/mol", DIM_CHEMICAL_POTENTIAL, 1e3)
J_per_mol_K = Unit("J/(mol*K)", DIM_MOLAR_ENTROPY, 1.0)

Molar = Unit("M", DIM_CONCENTRATION, 1e3)
mM = Unit("mM", DIM_CONCENTRATION, 1.0)
uM = Unit("uM", DIM_CONCENTRATION, 1e-3)
nM = Unit("nM", DIM_CONCENTRATION, 1e-6)
pM = Unit("pM", DIM_CONCENTRATION, 1e-9)
mol_per_s = Unit("mol/s", DIM_MOLAR_FLUX, 1.0)
mmol_per_s = Unit("mmol/s", DIM_MOLAR_FLUX, 1e-3)
fmol_per_ms = Unit("fmol/ms", DIM_MOLAR_FLUX, 1e-12)
mM_per_ms = Unit("mM/ms", DIM_REACTION_RATE, 1e3)

C = Unit("C", DIM_CHARGE, 1.0)
pC = Unit("pC", DIM_CHARGE, 1e-12)
C_per_mol = Unit("C/mol", DIM_MOLAR_CHARGE, 1.0)

V = Unit("V", DIM_VOLTAGE, 1.0)
mV = Unit("mV", DIM_VOLTAGE, 1e-3)
uV = Unit("uV", DIM_VOLTAGE, 1e-6)
V_per_m = Unit("V/m", DIM_ELECTRIC_FIELD, 1.0)

Ohm = Unit("Ohm", DIM_RESISTANCE, 1.0)
MOhm = Unit("MOhm", DIM_RESISTANCE, 1e6)
GOhm = Unit("GOhm", DIM_RESISTANCE, 1e9)

S = Unit("S", DIM_CONDUCTANCE, 1.0)
mS = Unit("mS", DIM_CONDUCTANCE, 1e-3)
uS = Unit("uS", DIM_CONDUCTANCE, 1e-6)
nS = Unit("nS", DIM_CONDUCTANCE, 1e-9)
pS = Unit("pS", DIM_CONDUCTANCE, 1e-12)

F = Unit("F", DIM_CAPACITANCE, 1.0)
uF = Unit("uF", DIM_CAPACITANCE, 1e-6)
nF = Unit("nF", DIM_CAPACITANCE, 1e-9)
pF = Unit("pF", DIM_CAPACITANCE, 1e-12)

uF_per_cm2 = Unit("uF/cm^2", DIM_SPECIFIC_CAPACITANCE, 1e-2)
mS_per_cm2 = Unit("mS/cm^2", DIM_SPECIFIC_CONDUCTANCE, 10.0)
uA_per_cm2 = Unit("uA/cm^2", DIM_CURRENT_DENSITY, 1e-2)

# CODATA 2019 physical constants
R_GAS = Quantity(8.31446261815324, J_per_mol_K)
FARADAY = Quantity(96485.33212331001, C_per_mol)
BOLTZMANN = Quantity(1.380649e-23, J_per_K)
AVOGADRO = Quantity(6.02214076e23, Unit("1/mol", DIM_AMOUNT**-1, 1.0))
ELEMENTARY_CHARGE = Quantity(1.602176634e-19, C)
PLANCK = Quantity(6.62607015e-34, Unit("J*s", DIM_ACTION, 1.0))
HBAR = Quantity(1.054571817e-34, Unit("J*s", DIM_ACTION, 1.0))
SPEED_OF_LIGHT = Quantity(299792458.0, m_per_s)
VACUUM_PERMITTIVITY = Quantity(8.8541878128e-12, Unit("F/m", DIM_PERMITTIVITY, 1.0))
VACUUM_PERMEABILITY = Quantity(1.25663706212e-6, Unit("H/m", DIM_PERMEABILITY, 1.0))
GRAVITATIONAL_CONSTANT = Quantity(
    6.67430e-11, Unit("m^3/(kg*s^2)", Dimension(M=-1, L=3, T=-2), 1.0)
)
STANDARD_GRAVITY = Quantity(9.80665, m_per_s2)

ROOM_TEMPERATURE_K = Quantity(298.15, K)
BODY_TEMPERATURE_K = Quantity(310.15, K)
THERMAL_VOLTAGE_37C = (R_GAS * BODY_TEMPERATURE_K / FARADAY).to(mV)


# Dynamic compound unit parser
_SI_PREFIXES: dict[str, float] = {
    "Y": 1e24,
    "Z": 1e21,
    "E": 1e18,
    "P": 1e15,
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "h": 1e2,
    "da": 1e1,
    "d": 1e-1,
    "c": 1e-2,
    "m": 1e-3,
    "u": 1e-6,
    "μ": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
    "z": 1e-21,
    "y": 1e-24,
}

_ATOMIC_UNITS: dict[str, Unit] = {
    "1": DIMENSIONLESS,
    "dimensionless": DIMENSIONLESS,
    "%": PERCENT,
    "g": g,
    "Da": Da,
    "m": m,
    "Angstrom": Angstrom,
    "A_len": Angstrom,
    "s": s,
    "min": minute,
    "hr": hour,
    "A": A,
    "K": K,
    "mol": mol,
    "cd": cd,
    "Hz": Hz,
    "N": N_force,
    "Pa": Pa,
    "J": J,
    "eV": eV,
    "W": W,
    "C": C,
    "V": V,
    "Ohm": Ohm,
    "S": S,
    "F": F,
    "L": L,
    "M": Molar,
    "H": Unit("H", DIM_INDUCTANCE, 1.0),
    "Wb": Unit("Wb", DIM_MAGNETIC_FLUX, 1.0),
    "T": Unit("T", DIM_MAGNETIC_FIELD, 1.0),
}


def _resolve_atomic_unit_token(token: str) -> Unit:
    token = token.strip()
    if not token:
        raise DimensionError("Empty unit token.")
    if token == "1":
        return DIMENSIONLESS
    if token in _ATOMIC_UNITS:
        return _ATOMIC_UNITS[token]

    for prefix in sorted(_SI_PREFIXES.keys(), key=len, reverse=True):
        if token.startswith(prefix):
            base_sym = token[len(prefix):]
            if base_sym in _ATOMIC_UNITS and base_sym not in ("1", "dimensionless", "%", "min", "hr"):
                base_u = _ATOMIC_UNITS[base_sym]
                return Unit(
                    symbol=token,
                    dimension=base_u.dimension,
                    scale=base_u.scale * _SI_PREFIXES[prefix],
                )

    raise DimensionError(f"Unrecognized physical unit symbol: '{token}'.")


def parse_unit(expr: str) -> Unit:
    """
    Parse a compound SI unit expression (e.g. 'pA / pF', 'kJ / (mol * K)', 'um^2 / ms')
    into an exact `Unit` instance.
    """
    if not isinstance(expr, str):
        raise DimensionError(f"Expected string unit expression, got {type(expr).__name__}.")
    cleaned = expr.strip()
    if not cleaned:
        raise DimensionError("Unit expression cannot be empty.")
    if cleaned == "1":
        return DIMENSIONLESS

    cleaned = cleaned.replace("**", "^")

    tokens: list[str] = []
    i = 0
    n = len(cleaned)
    while i < n:
        ch = cleaned[i]
        if ch.isspace():
            i += 1
        elif ch in "()*/^":
            tokens.append(ch)
            i += 1
        elif ch in "+-" or ch.isdigit() or ch == ".":
            start = i
            if ch in "+-":
                i += 1
            while i < n and (cleaned[i].isdigit() or cleaned[i] == "."):
                i += 1
            tokens.append(cleaned[start:i])
        else:
            start = i
            while i < n and not cleaned[i].isspace() and cleaned[i] not in "()*/^":
                i += 1
            tokens.append(cleaned[start:i])

    pos = 0

    def _parse_expr() -> Unit:
        nonlocal pos
        unit = _parse_factor()
        while pos < len(tokens) and tokens[pos] in ("*", "/"):
            op = tokens[pos]
            pos += 1
            rhs = _parse_factor()
            unit = (unit * rhs) if op == "*" else (unit / rhs)
        return unit

    def _parse_factor() -> Unit:
        nonlocal pos
        base = _parse_primary()
        if pos < len(tokens) and tokens[pos] == "^":
            pos += 1
            if pos >= len(tokens):
                raise DimensionError(f"Missing exponent after '^' in unit expression '{expr}'.")
            if tokens[pos] == "(":
                pos += 1
                if pos >= len(tokens):
                    raise DimensionError(f"Empty exponent parentheses in '{expr}'.")
                exp_str = tokens[pos]
                pos += 1
                if pos >= len(tokens) or tokens[pos] != ")":
                    raise DimensionError(f"Unclosed parenthesis in exponent of '{expr}'.")
                pos += 1
            else:
                exp_str = tokens[pos]
                pos += 1
            try:
                power = float(exp_str)
            except ValueError as exc:
                raise DimensionError(f"Invalid numeric exponent '{exp_str}' in unit '{expr}'.") from exc
            base = base**power
        return base

    def _parse_primary() -> Unit:
        nonlocal pos
        if pos >= len(tokens):
            raise DimensionError(f"Unexpected end of unit expression '{expr}'.")
        tok = tokens[pos]
        if tok == "(":
            pos += 1
            if pos < len(tokens) and tokens[pos] == ")":
                raise DimensionError(f"Empty parentheses '()' in unit expression '{expr}'.")
            inside = _parse_expr()
            if pos >= len(tokens) or tokens[pos] != ")":
                raise DimensionError(f"Missing closing ')' in unit expression '{expr}'.")
            pos += 1
            return inside
        if tok in (")", "*", "/", "^"):
            raise DimensionError(f"Unexpected operator token '{tok}' in unit expression '{expr}'.")
        pos += 1
        return _resolve_atomic_unit_token(tok)

    result = _parse_expr()
    if pos != len(tokens):
        raise DimensionError(f"Unexpected trailing token '{tokens[pos]}' in unit expression '{expr}'.")
    return Unit(symbol=expr.strip(), dimension=result.dimension, scale=result.scale)
