import pytest
from fibonacci import fibonacci, sum_two_numbers


def test_sum_two_numbers():
    assert sum_two_numbers(2, 3) == 5


def test_sum_two_negative_numbers():
    assert sum_two_numbers(-2, -3) == -5


def test_sum_positive_and_negative_numbers():
    assert sum_two_numbers(2, -3) == -1


def test_sum_zero_and_number():
    assert sum_two_numbers(0, 5) == 5


def test_sum_two_zeros():
    assert sum_two_numbers(0, 0) == 0


def test_fibonacci_zero():
    assert fibonacci(0) == 0


def test_fibonacci_one():
    assert fibonacci(1) == 1


def test_fibonacci_two():
    assert fibonacci(2) == 1


def test_fibonacci_five():
    assert fibonacci(5) == 5


def test_fibonacci_ten():
    assert fibonacci(10) == 55
