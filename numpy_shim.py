"""Lightweight numpy compatibility shim for constrained environments."""
from __future__ import annotations

import math
import random as _random
import statistics
from typing import Iterable, List, Sequence


class SimpleArray(list):
    def __init__(self, data: Iterable, dtype=None):
        super().__init__(float(x) if dtype in (float,) else x for x in data)

    @property
    def size(self) -> int:
        return len(self)

    def mean(self) -> float:
        return float(statistics.fmean(self)) if self else 0.0

    def std(self) -> float:
        return float(statistics.pstdev(self)) if len(self) > 1 else 0.0

    def tolist(self) -> List:
        return list(self)

    def __binary_op(self, other, op):
        if isinstance(other, (list, SimpleArray, tuple)):
            length = min(len(self), len(other))
            return SimpleArray(op(self[i], other[i]) for i in range(length))
        return SimpleArray(op(x, other) for x in self)

    def __add__(self, other):
        return self.__binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return SimpleArray(other for _ in self).__sub__(self)

    def __mul__(self, other):
        return self.__binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__binary_op(other, lambda a, b: a / b if b != 0 else 0.0)

    def __rtruediv__(self, other):
        return SimpleArray(other for _ in self).__truediv__(self)


class _RandomModule:
    def seed(self, seed_value: int | None = None) -> None:
        _random.seed(seed_value)

    def randint(self, low: int, high: int) -> int:
        return _random.randint(low, high - 1)

    def uniform(self, low: float, high: float) -> float:
        return _random.uniform(low, high)

    def choice(self, options: Sequence, size: int | None = None, p: Sequence[float] | None = None):
        if size is None:
            return _random.choice(list(options))
        weights = list(p) if p is not None else None
        return SimpleArray(_random.choices(list(options), weights=weights, k=size))

    def randn(self, *shape: int) -> SimpleArray:
        count = 1
        for dim in shape:
            count *= dim
        vals = [self._normal() for _ in range(count)]
        if len(shape) == 2:
            rows, cols = shape
            matrix = [SimpleArray(vals[i * cols : (i + 1) * cols]) for i in range(rows)]
            return SimpleArray(matrix)
        return SimpleArray(vals)

    def _normal(self) -> float:
        return _random.gauss(0, 1)

    def default_rng(self, seed_value: int | None = None):
        return _RandomGenerator(seed_value)


class _RandomGenerator:
    def __init__(self, seed_value: int | None = None) -> None:
        self._rng = _random.Random(seed_value)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: int = 1) -> SimpleArray:
        return SimpleArray(self._rng.gauss(loc, scale) for _ in range(size))

    def choice(self, options: Sequence, size: int | None = None, p: Sequence[float] | None = None):
        choices = list(options)
        if size is None:
            return self._rng.choice(choices)
        weights = list(p) if p is not None else None
        return SimpleArray(self._rng.choices(choices, weights=weights, k=size))


random = _RandomModule()

def array(data: Iterable, dtype=None) -> SimpleArray:
    return SimpleArray(data, dtype=dtype)


def asarray(data: Iterable, dtype=None) -> SimpleArray:
    return SimpleArray(data, dtype=dtype)


def log(x):
    if isinstance(x, (list, SimpleArray, tuple)):
        return SimpleArray(math.log(v) for v in x)
    return math.log(x)


def exp(x):
    if isinstance(x, (list, SimpleArray, tuple)):
        return SimpleArray(math.exp(v) for v in x)
    return math.exp(x)


def sqrt(x):
    return math.sqrt(x)


def sign(x):
    if isinstance(x, (list, SimpleArray, tuple)):
        return SimpleArray(0 if v == 0 else (1 if v > 0 else -1) for v in x)
    return 0 if x == 0 else (1 if x > 0 else -1)


def abs(x):
    if isinstance(x, (list, SimpleArray, tuple)):
        return SimpleArray(math.fabs(v) for v in x)
    return math.fabs(x)


def diff(arr: Iterable, prepend=None) -> SimpleArray:
    vals = list(arr)
    if prepend is not None:
        vals = [prepend] + vals
    return SimpleArray(vals[i] - vals[i - 1] for i in range(1, len(vals)))


def cumprod(arr: Iterable) -> SimpleArray:
    result = []
    prod = 1
    for val in arr:
        prod *= val
        result.append(prod)
    return SimpleArray(result)


def maximum_accumulate(arr: Iterable) -> SimpleArray:
    result = []
    current = -math.inf
    for val in arr:
        current = max(current, val)
        result.append(current)
    return SimpleArray(result)


class _Maximum:
    @staticmethod
    def accumulate(arr: Iterable) -> SimpleArray:
        return maximum_accumulate(arr)


maximum = _Maximum()

def isclose(a, b) -> bool:
    try:
        return math.isclose(a, b)
    except Exception:
        return False


def mean(arr: Iterable) -> float:
    data = list(arr)
    return float(statistics.fmean(data)) if data else 0.0


def std(arr: Iterable) -> float:
    data = list(arr)
    return float(statistics.pstdev(data)) if len(data) > 1 else 0.0


def vstack(arrays: List[Iterable]) -> SimpleArray:
    return SimpleArray([list(a) for a in arrays])


def argmax(arr: Iterable) -> int:
    data = list(arr)
    if not data:
        return 0
    return max(range(len(data)), key=lambda i: data[i])


def clip(arr: Iterable, a_min, a_max):
    return SimpleArray(min(max(x, a_min), a_max) for x in arr)


def zeros_like(arr: Iterable, dtype=float) -> SimpleArray:
    return SimpleArray(dtype() if callable(dtype) else 0 for _ in arr)


def expit(x):
    if isinstance(x, (list, SimpleArray, tuple)):
        return SimpleArray(1 / (1 + math.exp(-v)) for v in x)
    return 1 / (1 + math.exp(-x))

__all__ = [
    "array",
    "asarray",
    "argmax",
    "clip",
    "cumprod",
    "diff",
    "exp",
    "expit",
    "isclose",
    "log",
    "maximum",
    "mean",
    "random",
    "sign",
    "sqrt",
    "std",
    "vstack",
    "zeros_like",
    "SimpleArray",
]
