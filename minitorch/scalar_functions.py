from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:

    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:


    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v, history=None))
                raw_vals.append(float(v))


        ctx = Context(False)


        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))


        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


class Add(ScalarFunction):


    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):


    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        result = a * b
        ctx.save_for_backward(a, b)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float):
        a, b, = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        result = operators.inv(a)
        ctx.save_for_backward(a)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float):
        a, = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        result = -a
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float):
        return -d_output


class Sigmoid(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        result = operators.sigmoid(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float):
        result, = ctx.saved_values
        return result * (1.0 - result) * d_output


class ReLU(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        result = operators.relu(a)
        ctx.save_for_backward(a)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float):
        a, = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        result = operators.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float):
        result, = ctx.saved_values
        return result * d_output


class LT(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        result = float(a < b)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float):
        return 0.0, 0.0


class EQ(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        result = float(a == b)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float):
        return 0.0, 0.0
