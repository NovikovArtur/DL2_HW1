from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides


Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:


        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:

        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:

        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1


            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:


        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())


        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:


    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        if len(out_shape) == len(in_shape) and np.array_equal(out_shape, in_shape) and np.array_equal(out_strides, in_strides):
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
            return

        for i in prange(len(out)):
            oi = np.empty(len(out_shape), dtype=np.int32)
            in_index = np.empty(len(in_shape), dtype=np.int32)
            to_index(i, out_shape, oi)
            broadcast_index(oi, out_shape, in_shape, in_index)
            out[index_to_position(oi, out_strides)] = fn(in_storage[index_to_position(in_index, in_strides)])

    return njit(_map, parallel=True)


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:


    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        if np.array_equal(out_shape, a_shape) and np.array_equal(out_shape, b_shape) and np.array_equal(out_strides, a_strides) and np.array_equal(out_strides, b_strides):
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
            return

        for i in prange(len(out)):
            oi = np.empty(len(out_shape), dtype=np.int32)
            ai = np.empty(len(a_shape), dtype=np.int32)
            bi = np.empty(len(b_shape), dtype=np.int32)
            to_index(i, out_shape, oi)
            broadcast_index(oi, out_shape, a_shape, ai)
            broadcast_index(oi, out_shape, b_shape, bi)
            out[index_to_position(oi, out_strides)] = fn(a_storage[index_to_position(ai, a_strides)], b_storage[index_to_position(bi, b_strides)])

    return njit(_zip, parallel=True)


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:


    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        for i in prange(len(out)):
            index = np.empty(len(out_shape), dtype=np.int32)
            to_index(i, out_shape, index)
            opos = index_to_position(index, out_strides)
            apos = index_to_position(index, a_strides)
            value = out[opos]
            for j in range(a_shape[reduce_dim]):
                value = fn(value, a_storage[apos])
                apos += a_strides[reduce_dim]
            out[opos] = value

    return njit(_reduce, parallel=True)


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:

    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    for ordinal in prange(len(out)):
        batch = ordinal // (out_shape[1] * out_shape[2])
        row = (ordinal // out_shape[2]) % out_shape[1]
        col = ordinal % out_shape[2]
        ap = batch * a_batch_stride + row * a_strides[1]
        bp = batch * b_batch_stride + col * b_strides[2]
        value = 0.0
        for k in range(a_shape[2]):
            value += a_storage[ap] * b_storage[bp]
            ap += a_strides[2]
            bp += b_strides[1]
        out[batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]] = value


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
