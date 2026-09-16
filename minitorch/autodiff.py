from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:

    left, right = list(vals), list(vals)
    left[arg] = left[arg] - epsilon
    right[arg] = right[arg] + epsilon
    return (f(*right) - f(*left)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None: ...

    @property
    def unique_id(self) -> int: ...

    def is_leaf(self) -> bool: ...

    def is_constant(self) -> bool: ...

    @property
    def parents(self) -> Iterable["Variable"]: ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]: ...


def topological_sort(variable: Variable) -> Iterable[Variable]:

    seen, result = set(), []
    stack = [(variable, False)]
    while stack:
        node, expanded = stack.pop()
        if node.is_constant():
            continue
        if expanded:
            result.append(node)
        elif node.unique_id not in seen:
            seen.add(node.unique_id)
            stack.append((node, True))
            if not node.is_leaf():
                stack.extend((parent, False) for parent in node.parents)
    return reversed(result)


def backpropagate(variable: Variable, deriv: Any) -> None:

    derivatives = {variable.unique_id: deriv}
    for node in topological_sort(variable):
        d = derivatives[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(d)
        else:
            for parent, value in node.chain_rule(d):
                if not parent.is_constant():
                    derivatives[parent.unique_id] = derivatives.get(parent.unique_id, 0.0) + value


@dataclass
class Context:


    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:

        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
