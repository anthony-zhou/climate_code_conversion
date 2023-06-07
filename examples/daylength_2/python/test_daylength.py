import numpy as np
import pytest
from chatgpt_daylength import daylength

# tolerance
tol = 1e-3


def test_standard_points():
    assert np.allclose(daylength(np.array([-1.4, -1.3]), 0.1), 
                       np.array([26125.331269192659, 33030.159082987258]), 
                       atol=tol)

def test_near_poles():
    assert np.allclose(daylength(np.array([-1.5, 1.5]), 0.1), 
                       np.array([0.0, 86400.0]), 
                       atol=tol)

def test_north_pole():
    assert abs(daylength(np.pi/2.0, 0.1) - 86400.0) < tol
    assert abs(daylength(np.pi/1.999999999999999, 0.1) - 86400.0) < tol

def test_south_pole():
    assert abs(daylength(-1.0 * np.pi/2.0, 0.1)) < tol
    assert abs(daylength(-1.0 * np.pi/1.999999999999999, 0.1)) < tol

def test_error_in_decl():
    assert np.isnan(daylength(-1.0, -3.0))

def test_error_in_lat_scalar():
    assert np.isnan(daylength(3.0, 0.1))

def test_error_in_lat_array():
    my_result = daylength(np.array([1.0, 3.0]), 0.1)
    assert np.isfinite(my_result[0])
    assert np.isnan(my_result[1])