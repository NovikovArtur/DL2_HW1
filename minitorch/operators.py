import math
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def relu(x: float) -> float:
    return x if x > 0 else 0.0


def log(x: float) -> float:
    return math.log(x + 1e-6)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    return d / (x + 1e-6)


def inv_back(x: float, d: float) -> float:
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0


def map(fn: Callable) -> Callable:
    def apply(xs: Iterable) -> list:
        return [fn(x) for x in xs]
    return apply


def zipWith(fn: Callable) -> Callable:
    def apply(xs: Iterable, ys: Iterable) -> list:
        return [fn(x, y) for x, y in zip(xs, ys)]
    return apply


def reduce(fn: Callable, start: float) -> Callable:
    def apply(xs: Iterable) -> float:
        result = start
        for x in xs:
            result = fn(result, x)
        return result
    return apply


negList = map(neg)
addLists = zipWith(add)
sum = reduce(add, 0.0)
prod = reduce(mul, 1.0)
