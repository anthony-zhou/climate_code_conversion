
import pytest

@pytest.mark.parametrize("n, expected", [
    (0, 0),
    (1, 1),
    (2, 1),
    (3, 2),
    (4, 3),
    (5, 5),
    (6, 8),
    (7, 13),
    (8, 21),
    (9, 34),
    (10, 55),
])
def test_fibonacci(n, expected):
    assert fibonacci(n) == expected

def test_fibonacci_negative_input():
    with pytest.raises(IndexError):
        fibonacci(-1)

def test_fibonacci_large_input():
    assert fibonacci(100) == 354224848179261915075



import pytest

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

