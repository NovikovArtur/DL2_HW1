from hypothesis import settings
from hypothesis.strategies import floats, integers

import minitorch


settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


small_ints = integers(min_value=1, max_value=3)
small_floats = floats(min_value=-100, max_value=100, allow_nan=False)
med_ints = integers(min_value=1, max_value=20)


def assert_close(a: float, b: float) -> None:
    assert minitorch.operators.is_close(a, b), "Failure x=%f y=%f" % (a, b)


def checked_grad(fn, *values):
    from hypothesis import assume
    import numpy as np
    if fn.__name__ in ('lt2', 'gt2', 'eq2'):
        a, b = (x.to_numpy() for x in values)
        gap = a - b - 5.5 if fn.__name__ == 'eq2' else a + 1.2 - b
        assume(np.all(np.abs(gap) > 1e-5))
    minitorch.grad_check(fn, *values)
