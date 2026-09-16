import numpy as np
import numba
from numba import cuda
import pytest
import minitorch as mt

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason='CUDA or CUDASIM required')


def numpy(t):
    storage = t._tensor._storage
    if hasattr(storage, 'copy_to_host'):
        storage = storage.copy_to_host()
    return np.asarray(storage).reshape(t.shape)


def test_map_zip_and_broadcast():
    backend = mt.TensorBackend(mt.CudaOps)
    a = mt.tensor([[1., 2., 3.], [4., 5., 6.]], backend=backend)
    b = mt.tensor([[2.], [3.]], backend=backend)
    np.testing.assert_allclose(numpy((a * b).permute(1, 0).relu()), [[2, 12], [4, 15], [6, 18]])


def test_reduction_noncontiguous_and_multiblock():
    backend = mt.TensorBackend(mt.CudaOps)
    a = mt.tensor([[1., 2., 3.], [4., 5., 6.]], backend=backend)
    np.testing.assert_allclose(numpy(a.permute(1, 0).sum(1)), [[5], [7], [9]])
    b = mt.tensor([1.] * 1031, backend=backend)
    assert b.sum()[0] == 1031


def test_practice_kernels():
    from minitorch.cuda_ops import jit_sum_practice, jit_mm_practice
    x = cuda.to_device(np.arange(37, dtype=np.float64))
    out = cuda.to_device(np.zeros(2))
    jit_sum_practice[2, 32](out, x, 37)
    np.testing.assert_allclose(out.copy_to_host(), [sum(range(32)), sum(range(32, 37))])
    a = cuda.to_device(np.arange(9, dtype=np.float64))
    b = cuda.to_device(np.arange(9, dtype=np.float64))
    out = cuda.to_device(np.zeros(9))
    jit_mm_practice[1, (3, 3)](out, a, b, 3)
    np.testing.assert_allclose(out.copy_to_host().reshape(3, 3), np.arange(9).reshape(3, 3) @ np.arange(9).reshape(3, 3))


def test_tiled_matmul_broadcast_and_strides():
    from minitorch.cuda_ops import tensor_matrix_multiply
    a_np = np.arange(2 * 3 * 35, dtype=np.float64).reshape(2, 3, 35) / 100
    b_np = np.arange(1 * 4 * 35, dtype=np.float64).reshape(1, 4, 35) / 100
    a = cuda.to_device(a_np.ravel())
    b = cuda.to_device(b_np.ravel())
    out = cuda.to_device(np.zeros(24))
    tensor_matrix_multiply[(1, 1, 2), (16, 16, 1)](
        out, np.array([2, 3, 4]), np.array([12, 4, 1]), 24,
        a, np.array([2, 3, 35]), np.array([105, 35, 1]),
        b, np.array([1, 35, 4]), np.array([140, 1, 35]))
    np.testing.assert_allclose(out.copy_to_host().reshape(2, 3, 4), a_np @ b_np.transpose(0, 2, 1))


def test_backward():
    backend = mt.TensorBackend(mt.CudaOps)
    a = mt.tensor([[0.2, 0.4], [0.6, 0.8]], backend=backend)
    b = mt.tensor([[0.3], [0.5]], backend=backend)
    mt.grad_check(lambda x, y: (x * y).sigmoid().sum(0), a, b)
