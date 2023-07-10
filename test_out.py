
import pytest

def test_sum_positive_numbers():
    assert sum(2, 3) == 5

def test_sum_negative_numbers():
    assert sum(-2, -3) == -5

def test_sum_zero():
    assert sum(0, 0) == 0

def test_sum_positive_and_negative_numbers():
    assert sum(2, -3) == -1

def test_sum_float_numbers():
    assert sum(2.5, 3.5) == 6.0

def test_sum_string_numbers():
    assert sum("2", "3") == "23"

def test_sum_invalid_input():
    with pytest.raises(TypeError):
        sum("2", 3)

