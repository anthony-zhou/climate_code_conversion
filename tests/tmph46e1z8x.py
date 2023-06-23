import math
import numpy as np

def quadratic_roots(a, b, c):
    roots = np.empty(2)
    discriminant = b * b - 4.0 * a * c

    if discriminant >= 0.0:
        roots[0] = (-b + math.sqrt(discriminant)) / (2.0 * a)
        roots[1] = (-b - math.sqrt(discriminant)) / (2.0 * a)
    else:
        roots[0] = np.nan
        roots[1] = np.nan

    return roots

import pytest
import numpy as np

def test_quadratic_roots_real():
    roots = quadratic_roots(1, -3, 2)
    assert np.allclose(roots, [2.0, 1.0])

def test_quadratic_roots_imaginary():
    roots = quadratic_roots(1, 0, 1)
    assert np.isnan(roots[0])
    assert np.isnan(roots[1])

def test_quadratic_roots_zero_discriminant():
    roots = quadratic_roots(1, -2, 1)
    assert np.allclose(roots, [1.0, 1.0])

def test_quadratic_roots_zero_a():
    with pytest.raises(ZeroDivisionError):
        quadratic_roots(0, -2, 1)
