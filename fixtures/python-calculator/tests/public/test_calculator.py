'''Public tests visible to the coding agent.'''

import pytest

from calculator import add, divide, multiply, subtract


def test_basic_operations() -> None:
    assert add(2, 3) == 5
    assert subtract(7, 4) == 3
    assert multiply(6, 5) == 30
    assert divide(9, 3) == 3


def test_divide_by_zero_raises() -> None:
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
