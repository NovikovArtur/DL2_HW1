


from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any


Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:

        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)


            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 128 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 128
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return ret(out_a, dim) if out_shape[dim] > 1 else out_a

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

        blockspergrid = ((out.shape[1] + 15) // 16, (out.shape[2] + 15) // 16, out.shape[0])
        threadsperblock = (16, 16, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )


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
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        oi = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            to_index(i, out_shape, oi)
            broadcast_index(oi, out_shape, in_shape, in_index)
            out[index_to_position(oi, out_strides)] = fn(in_storage[index_to_position(in_index, in_strides)])

    return cuda.jit()(_map)


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:


    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        oi = cuda.local.array(MAX_DIMS, numba.int32)
        ai = cuda.local.array(MAX_DIMS, numba.int32)
        bi = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            to_index(i, out_shape, oi)
            broadcast_index(oi, out_shape, a_shape, ai)
            broadcast_index(oi, out_shape, b_shape, bi)
            out[index_to_position(oi, out_strides)] = fn(a_storage[index_to_position(ai, a_strides)], b_storage[index_to_position(bi, b_strides)])

    return cuda.jit()(_zip)


def _sum_practice(out: Storage, a: Storage, size: int) -> None:

    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    cache[pos] = a[i] if i < size else 0.0
    cuda.syncthreads()
    stride = 1
    while stride < BLOCK_DIM:
        if pos % (2 * stride) == 0 and pos + stride < BLOCK_DIM:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride *= 2
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    out = TensorData([0.0] * blockspergrid, (blockspergrid,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:


    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 128
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        oi = cuda.local.array(MAX_DIMS, numba.int32)
        opos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        to_index(opos, out_shape, oi)
        start = oi[reduce_dim] * BLOCK_DIM
        out_address = index_to_position(oi, out_strides)
        oi[reduce_dim] = start + pos
        cache[pos] = reduce_value
        if start + pos < a_shape[reduce_dim]:
            cache[pos] = a_storage[index_to_position(oi, a_strides)]
        cuda.syncthreads()
        stride = 1
        while stride < BLOCK_DIM:
            if pos % (2 * stride) == 0 and pos + stride < BLOCK_DIM:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride *= 2
        if pos == 0:
            out[out_address] = cache[0]

    return jit(_reduce)


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    i, j = cuda.threadIdx.x, cuda.threadIdx.y
    if i < size and j < size:
        a_shared[i, j] = a[i * size + j]
        b_shared[i, j] = b[i * size + j]
    cuda.syncthreads()
    if i < size and j < size:
        value = 0.0
        for k in range(size):
            value += a_shared[i, k] * b_shared[k, j]
        out[i * size + j] = value


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:

    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    batch = cuda.blockIdx.z

    BLOCK_DIM = 16
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)


    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y


    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y


    value = 0.0
    for tile in range((a_shape[2] + BLOCK_DIM - 1) // BLOCK_DIM):
        ak = tile * BLOCK_DIM + pj
        bk = tile * BLOCK_DIM + pi
        a_shared[pi, pj] = 0.0
        b_shared[pi, pj] = 0.0
        if i < a_shape[1] and ak < a_shape[2]:
            a_shared[pi, pj] = a_storage[batch * a_batch_stride + i * a_strides[1] + ak * a_strides[2]]
        if bk < b_shape[1] and j < b_shape[2]:
            b_shared[pi, pj] = b_storage[batch * b_batch_stride + bk * b_strides[1] + j * b_strides[2]]
        cuda.syncthreads()
        for k in range(BLOCK_DIM):
            value += a_shared[pi, k] * b_shared[k, pj]
        cuda.syncthreads()
    if i < out_shape[1] and j < out_shape[2]:
        out[batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = value


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
