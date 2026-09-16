

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData


from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:


    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:


    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend
        if backend.cuda:
            self._tensor.to_cuda_()

    def requires_grad_(self, x: bool) -> None:
        self.history = History() if x else None

    def requires_grad(self) -> bool:
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:

        storage = self.contiguous()._tensor._storage
        if hasattr(storage, "copy_to_host"):
            storage = storage.copy_to_host()
        return storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:

        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:

        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:

        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)


    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        self.f = backend
        if backend.cuda:
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:

        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:


        if self.shape == other.shape:
            return other


        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf


        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"

        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)


    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:

        return self._tensor.tuple()

    def detach(self) -> Tensor:

        return Tensor(self._tensor, backend=self.backend)


    def accumulate_derivative(self, x: Any) -> None:

        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:

        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:

        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:

        return self._tensor.shape


    @property
    def size(self):
        return self._tensor.size

    @property
    def dims(self):
        return self._tensor.dims

    def zero_grad_(self):
        self.grad = None

    def __add__(self, b):
        return Add.apply(self, self._ensure_tensor(b))

    def __radd__(self, b):
        return self + b

    def __mul__(self, b):
        return Mul.apply(self, self._ensure_tensor(b))

    def __rmul__(self, b):
        return self * b

    def __neg__(self):
        return Neg.apply(self)

    def __sub__(self, b):
        return self + (-self._ensure_tensor(b))

    def __rsub__(self, b):
        return self._ensure_tensor(b) + (-self)

    def __lt__(self, b):
        return LT.apply(self, self._ensure_tensor(b))

    def __gt__(self, b):
        return LT.apply(self._ensure_tensor(b), self)

    def __eq__(self, b):
        return EQ.apply(self, self._ensure_tensor(b))

    def is_close(self, b):
        return IsClose.apply(self, self._ensure_tensor(b))

    def sigmoid(self):
        return Sigmoid.apply(self)

    def relu(self):
        return ReLU.apply(self)

    def log(self):
        return Log.apply(self)

    def exp(self):
        return Exp.apply(self)

    def sum(self, dim=None):
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim=None):
        return self.sum(dim) / (self.size if dim is None else self.shape[dim])

    def all(self, dim=None):
        if dim is None:
            return All.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        return All.apply(self, self._ensure_tensor(dim))

    def view(self, *shape):
        return View.apply(self, tensor(list(shape), backend=self.backend))

    def permute(self, *order):
        return Permute.apply(self, tensor(list(order), backend=self.backend))
