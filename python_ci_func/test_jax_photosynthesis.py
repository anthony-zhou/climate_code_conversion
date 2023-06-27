import pytest
from jax_photosynthesis import ci_func, find_root

def test_ci_func():
    ci = 40
    lmr_z = 6
    par_z = 500
    gb_mol = 500
    je = 40
    cair = 40
    oair = 21000
    rh_can = 40
    p = 1
    iv = 1
    c = 1

    expected_fval = 1995.7688742784451
    expected_gs_mol = 153.26317840161104

    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)

    assert fval == pytest.approx(expected_fval)
    assert gs_mol == pytest.approx(expected_gs_mol)


def test_find_root():
    ci = 40
    lmr_z = 6
    par_z = 500
    gb_mol = 500
    je = 40
    cair = 40
    oair = 21000
    rh_can = 40
    p = 1
    iv = 1
    c = 1

    ci_val = find_root(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)

    assert ci_val == pytest.approx(40.0)