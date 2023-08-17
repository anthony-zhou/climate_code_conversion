import pytest
import numpy as np

def test_ci_func_c3flag_true():
    ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c = 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)
    assert np.isclose(fval, 0.0, atol=1e-6)
    assert gs_mol is None

def test_ci_func_c3flag_false():
    ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c = 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)
    assert np.isclose(fval, 0.0, atol=1e-6)
    assert gs_mol is None

def test_ci_func_an_negative():
    ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c = 0.5, 0.6, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)
    assert np.isclose(fval, 0.0, atol=1e-6)
    assert gs_mol is None