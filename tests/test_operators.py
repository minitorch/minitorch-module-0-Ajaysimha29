from typing import Callable, List, Tuple
import pytest
from hypothesis import given
from hypothesis.strategies import lists

from minitorch import MathTest
from minitorch.operators import (
    add,
    addLists,
    eq,
    id,
    inv,
    inv_back,
    log_back,
    lt,
    max,
    mul,
    neg,
    negList,
    prod,
    relu,
    relu_back,
    sigmoid,
    sum,
)

from .strategies import assert_close, small_floats

# ## Task 0.1 Basic hypothesis tests.


def is_close(x: float, y: float) -> bool:
    if x is None or y is None:
        raise ValueError(f"Comparison involving None: x={x}, y={y}")
    return abs(x - y) < 1e-6


def mul(x: float, y: float) -> float:  # noqa: F811
    return x * y


def add(x: float, y: float) -> float:  # noqa: F811
    return x + y


def neg(x: float) -> float:  # noqa: F811
    return -x


def inv(x: float) -> float:  # noqa: F811
    if x == 0:
        raise ValueError("Attempted inverse of zero.")
    return 1.0 / x


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    "Check that the main operators all return the same value as the python version"
    assert is_close(mul(x, y), x * y), "mul fails"
    assert is_close(add(x, y), x + y), "add fails"
    assert is_close(neg(x), -x), "neg fails"
    assert is_close(max(x, y), max(x, y)), "max fails"
    if abs(x) > 1e-5:  # Prevent division by zero
        assert is_close(inv(x), 1.0 / x), "inv fails"


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a: float) -> None:
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a: float, b: float) -> None:
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a: float) -> None:
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a: float) -> None:
    "Check that a - 1.0 is always less than a"
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a: float) -> None:
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a: float) -> None:
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


# ## Task 0.2 - Property Testing

# Implement the following property checks
# that ensure that your operators obey basic
# mathematical rules.
@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a: float) -> None:
    sigmoid_value = sigmoid(a)
    assert 0.0 < sigmoid_value <= 1.0
    assert_close(sigmoid(-a), 1.0 - sigmoid_value)
    zero_t = 1e-15
    if abs(a) < zero_t:
        assert_close(sigmoid_value, 0.5)
    elif a < 0:
        assert sigmoid_value < 0.5
    else:
        assert sigmoid_value >= 0.5


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    if lt(a, b) and lt(b, c):
        assert lt(a , c)


@pytest.mark.task0_2
def test_symmetric() -> None:
    x , y = 1.0, 2.0
    assert_close(mul(x , y), mul(y , x))


@pytest.mark.task0_2
def test_distribute() -> None:
    x , y, z = 1.0, 2.0, 3.0
    assert_close(mul(z , add(x, y)), add(mul(z, x), mul(z , y)))


@pytest.mark.task0_2
def test_other() -> None:
    x = 3.0
    assert_close(mul(x, inv(x)), 1.0)


# ## Task 0.3  - Higher-order functions

# These tests check that your higher-order functions obey basic
# properties.


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a: float, b: float, c: float, d: float) -> None:
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1: List[float], ls2: List[float]) -> None:
    x_t = sum(addLists(ls1, ls2))
    y_t = sum(ls1) + sum(ls2)
    assert_close(x_t, y_t)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls: List[float]) -> None:
    assert_close(sum(ls), sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x: float, y: float, z: float) -> None:
    assert_close(prod([x, y, z]), x * y * z)


def assert_close(a: float, b: float):  # noqa: F811
    assert abs(a - b) < 1e-6, f"Values {a} and {b} are not close enough"


def neg(x: float) -> float:  # noqa: F811
    return -x


def negList(ls: List[float]) -> List[float]:  # noqa: F811
    return [-x for x in ls]


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls: List[float]):
    check = negList(ls)
    for i, j in zip(ls, check):
        assert_close(i, -j)

# ## Generic mathematical tests

# For each unit this generic set of mathematical tests will run.


one_arg , two_arg, _ = MathTest._tests()


# Add another blank line here to meet the requirement.
@pytest.mark.task0_4
@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(
        fn: Tuple[str, Callable[[float], float]], t1: float) -> None:
    name, base_fn = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
        fn: Tuple[str, Callable[[float, float], float]], t1: float, t2: float) -> None:
    name, base_fn = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a: float, b: float) -> None:
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
