import pytest

def test_sum_two_numbers_positive():
    assert sum_two_numbers(5, 7) == 12

def test_sum_two_numbers_negative():
    assert sum_two_numbers(-3, -7) == -10

def test_sum_two_numbers_mixed():
    assert sum_two_numbers(-5, 7) == 2
