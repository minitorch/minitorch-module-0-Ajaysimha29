"""
Collection of the core mathematical operators used throughout the code base.
"""
import functools
import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return


def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    return abs(x - y) < 1e-6


def sigmoid(x: float) -> float:
    if x > 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)
    


def relu(x: float) -> float:
    return x if x>0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    return d / (x+EPS)



def inv(x: float) -> float:
    return 1 / x


def inv_back(x: float, d: float) -> float:
    return -d / (x * x) 


def relu_back(x: float, d: float) -> float:
    return d if x>0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def apply_map(ls:Iterable[float]) -> Iterable[float]:
        return [fn(item) for item in ls]
    return apply_map


def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg)(ls)


def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    def apply_zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]
    return apply_zipWith


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(ls1, ls2)


def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Iterable[float]], float]:
    def apply_reduce(ls: Iterable[float]) -> float:
        result = start
        for item in ls:
            result = fn(result, item)
        return result
    return apply_reduce


def sum(ls: Iterable[float]) -> float:
    "Sum of a list using reduce and add"
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using reduce and mul"
    return reduce(mul, 1.0)(ls)