'''Trusted tests injected after the agent finishes.'''

import pytest

from calculator import divide


@pytest.mark.parametrize('divisor', [0.0, -0.0])
def test_all_float_zero_divisors_raise(divisor: float) -> None:
    with pytest.raises(ZeroDivisionError):
        divide(5, divisor)


def test_zero_dividend_and_zero_divisor_raises() -> None:
    with pytest.raises(ZeroDivisionError):
        divide(0, 0)
