import pytest
from comparisons.vnumba import ci_func

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

    expected_fval = 441.5118
    expected_gs_mol = 12411.9808

    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)

    assert fval == pytest.approx(expected_fval)
    assert gs_mol == pytest.approx(expected_gs_mol)


def test_ci_func_different_input():
    ci = 60
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

    expected_fval = 736.7780
    expected_gs_mol = 12306.8789

    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)

    assert fval == pytest.approx(expected_fval)
    assert gs_mol == pytest.approx(expected_gs_mol)