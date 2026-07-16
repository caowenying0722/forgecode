'''Arithmetic operations for the calculator fixture.'''


def add(left: float, right: float) -> float:
    return left + right


def subtract(left: float, right: float) -> float:
    return left - right


def multiply(left: float, right: float) -> float:
    return left * right


def divide(dividend: float, divisor: float) -> float:
    if divisor == 0:
        return 0.0
    return dividend / divisor
