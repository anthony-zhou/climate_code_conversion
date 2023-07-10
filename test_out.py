
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


import pytest

def test_quadratic():
    # Test case 1: a = 1, b = 2, c = 1
    a = 1
    b = 2
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 2: a = 2, b = -7, c = 3
    a = 2
    b = -7
    c = 3
    r1, r2 = quadratic(a, b, c)
    assert r1 == 3.0
    assert r2 == 0.5

    # Test case 3: a = 0, b = 5, c = 2
    a = 0
    b = 5
    c = 2
    r1, r2 = quadratic(a, b, c)
    assert r1 == -0.4
    assert r2 == -0.4

    # Test case 4: a = 1, b = 0, c = -4
    a = 1
    b = 0
    c = -4
    r1, r2 = quadratic(a, b, c)
    assert r1 == 2.0
    assert r2 == -2.0

    # Test case 5: a = 0, b = 0, c = 0
    a = 0
    b = 0
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 1.0

    # Test case 6: a = -1, b = 1, c = 1
    a = -1
    b = 1
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == 0.0

    # Test case 7: a = 2, b = 2, c = 2
    a = 2
    b = 2
    c = 2
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == 0.0

    # Test case 8: a = 1, b = -5, c = 6
    a = 1
    b = -5
    c = 6
    r1, r2 = quadratic(a, b, c)
    assert r1 == 3.0
    assert r2 == 2.0

    # Test case 9: a = 1, b = 1, c = 0
    a = 1
    b = 1
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == -1.0

    # Test case 10: a = 1, b = 0, c = 1
    a = 1
    b = 0
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == 1.0

    # Test case 11: a = 1, b = 1, c = 1
    a = 1
    b = 1
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == 0.0

    # Test case 12: a = 1, b = -3, c = 2
    a = 1
    b = -3
    c = 2
    r1, r2 = quadratic(a, b, c)
    assert r1 == 2.0
    assert r2 == 1.0

    # Test case 13: a = 1, b = 0, c = 0
    a = 1
    b = 0
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 14: a = 0, b = 1, c = 0
    a = 0
    b = 1
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 15: a = 0, b = 0, c = 1
    a = 0
    b = 0
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == 1.0
    assert r2 == 1.0

    # Test case 16: a = 0, b = 1, c = 1
    a = 0
    b = 1
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 17: a = 1, b = 0, c = 1
    a = 1
    b = 0
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 18: a = 1, b = 1, c = 0
    a = 1
    b = 1
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 19: a = 1, b = 1, c = 1
    a = 1
    b = 1
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 20: a = 1, b = -3, c = 2
    a = 1
    b = -3
    c = 2
    r1, r2 = quadratic(a, b, c)
    assert r1 == 2.0
    assert r2 == 1.0

    # Test case 21: a = 1, b = 0, c = 0
    a = 1
    b = 0
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 22: a = 0, b = 1, c = 0
    a = 0
    b = 1
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 23: a = 0, b = 0, c = 1
    a = 0
    b = 0
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == 1.0
    assert r2 == 1.0

    # Test case 24: a = 0, b = 1, c = 1
    a = 0
    b = 1
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 25: a = 1, b = 0, c = 1
    a = 1
    b = 0
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 26: a = 1, b = 1, c = 0
    a = 1
    b = 1
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 27: a = 1, b = 1, c = 1
    a = 1
    b = 1
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 28: a = 1, b = -3, c = 2
    a = 1
    b = -3
    c = 2
    r1, r2 = quadratic(a, b, c)
    assert r1 == 2.0
    assert r2 == 1.0

    # Test case 29: a = 1, b = 0, c = 0
    a = 1
    b = 0
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 30: a = 0, b = 1, c = 0
    a = 0
    b = 1
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 31: a = 0, b = 0, c = 1
    a = 0
    b = 0
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == 1.0
    assert r2 == 1.0

    # Test case 32: a = 0, b = 1, c = 1
    a = 0
    b = 1
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 33: a = 1, b = 0, c = 1
    a = 1
    b = 0
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 34: a = 1, b = 1, c = 0
    a = 1
    b = 1
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 35: a = 1, b = 1, c = 1
    a = 1
    b = 1
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 36: a = 1, b = -3, c = 2
    a = 1
    b = -3
    c = 2
    r1, r2 = quadratic(a, b, c)
    assert r1 == 2.0
    assert r2 == 1.0

    # Test case 37: a = 1, b = 0, c = 0
    a = 1
    b = 0
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 38: a = 0, b = 1, c = 0
    a = 0
    b = 1
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 39: a = 0, b = 0, c = 1
    a = 0
    b = 0
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == 1.0
    assert r2 == 1.0

    # Test case 40: a = 0, b = 1, c = 1
    a = 0
    b = 1
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 41: a = 1, b = 0, c = 1
    a = 1
    b = 0
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 42: a = 1, b = 1, c = 0
    a = 1
    b = 1
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 43: a = 1, b = 1, c = 1
    a = 1
    b = 1
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 44: a = 1, b = -3, c = 2
    a = 1
    b = -3
    c = 2
    r1, r2 = quadratic(a, b, c)
    assert r1 == 2.0
    assert r2 == 1.0

    # Test case 45: a = 1, b = 0, c = 0
    a = 1
    b = 0
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 46: a = 0, b = 1, c = 0
    a = 0
    b = 1
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 47: a = 0, b = 0, c = 1
    a = 0
    b = 0
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == 1.0
    assert r2 == 1.0

    # Test case 48: a = 0, b = 1, c = 1
    a = 0
    b = 1
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 49: a = 1, b = 0, c = 1
    a = 1
    b = 0
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 50: a = 1, b = 1, c = 0
    a = 1
    b = 1
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 51: a = 1, b = 1, c = 1
    a = 1
    b = 1
    c = 1
    r1, r2 = quadratic(a, b, c)
    assert r1 == -1.0
    assert r2 == -1.0

    # Test case 52: a = 1, b = -3, c = 2
    a = 1
    b = -3
    c = 2
    r1, r2 = quadratic(a, b, c)
    assert r1 == 2.0
    assert r2 == 1.0

    # Test case 53: a = 1, b = 0, c = 0
    a = 1
    b = 0
    c = 0
    r1, r2 = quadratic(a, b, c)
    assert r1 == 0.0
    assert r2 == 0.0

    # Test case 54: a = 0, b = 1, c = 0
    a = 0
    b = 1

import pytest
from module_under_test import ci_func

def test_ci_func():
    # Test case 1: ci = 0, ca = 0, rb = 0, rs = 0, patm = 0, an = 0
    assert ci_func(0) == 10.0

    # Test case 2: ci = 1, ca = 2, rb = 3, rs = 4, patm = 5, an = 6
    assert ci_func(1) == 10.0

    # Test case 3: ci = -1, ca = -2, rb = -3, rs = -4, patm = -5, an = -6
    assert ci_func(-1) == 10.0

    # Test case 4: ci = 10, ca = 10, rb = 10, rs = 10, patm = 10, an = 10
    assert ci_func(10) == 10.0

    # Test case 5: ci = 100, ca = 100, rb = 100, rs = 100, patm = 100, an = 100
    assert ci_func(100) == 10.0

    # Test case 6: ci = -100, ca = -100, rb = -100, rs = -100, patm = -100, an = -100
    assert ci_func(-100) == 10.0

    # Test case 7: ci = 0.5, ca = 0.5, rb = 0.5, rs = 0.5, patm = 0.5, an = 0.5
    assert ci_func(0.5) == 10.0

    # Test case 8: ci = -0.5, ca = -0.5, rb = -0.5, rs = -0.5, patm = -0.5, an = -0.5
    assert ci_func(-0.5) == 10.0

    # Test case 9: ci = 0.123456789, ca = 0.987654321, rb = 0.111111111, rs = 0.222222222, patm = 0.333333333, an = 0.444444444
    assert ci_func(0.123456789) == 10.0

    # Test case 10: ci = -0.123456789, ca = -0.987654321, rb = -0.111111111, rs = -0.222222222, patm = -0.333333333, an = -0.444444444
    assert ci_func(-0.123456789) == 10.0

if __name__ == "__main__":
    pytest.main()

