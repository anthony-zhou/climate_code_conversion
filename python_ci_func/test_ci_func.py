import pytest
import numpy as np
from python_ci_func.comparisons.voriginal import ci_func


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


# def test_ci_func_c3_photosynthesis():
#     ci = 100
#     lmr_z = 10
#     par_z = 20
#     gb_mol = 30
#     je = 40
#     cair = 50
#     oair = 60
#     rh_can = 70
#     p = 80
#     iv = 90
#     c = 100
#     fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)
#     assert isinstance(fval, float)
#     assert isinstance(gs_mol, float)

# def test_ci_func_c4_photosynthesis():
#     ci = 100
#     lmr_z = 10
#     par_z = 20
#     gb_mol = 30
#     je = 40
#     cair = 50
#     oair = 60
#     rh_can = 70
#     p = 80
#     iv = 90
#     c = 100
#     fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c, c3flag=False)
#     assert isinstance(fval, float)
#     assert isinstance(gs_mol, float)

# def test_ci_func_negative_net_photosynthesis():
#     ci = 100
#     lmr_z = 200
#     par_z = 20
#     gb_mol = 30
#     je = 40
#     cair = 50
#     oair = 60
#     rh_can = 70
#     p = 80
#     iv = 90
#     c = 100
#     fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)
#     assert fval == 0.0
#     assert gs_mol is None

# def test_ci_func_stomatalcond_mtd_medlyn2011():
#     ci = 100
#     lmr_z = 10
#     par_z = 20
#     gb_mol = 30
#     je = 40
#     cair = 50
#     oair = 60
#     rh_can = 70
#     p = 80
#     iv = 90
#     c = 100
#     fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c, stomatalcond_mtd=1)
#     assert isinstance(fval, float)
#     assert isinstance(gs_mol, float)

# def test_ci_func_stomatalcond_mtd_bb1987():
#     ci = 100
#     lmr_z = 10
#     par_z = 20
#     gb_mol = 30
#     je = 40
#     cair = 50
#     oair = 60
#     rh_can = 70
#     p = 80
#     iv = 90
#     c = 100
#     fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c, stomatalcond_mtd=2)
#     assert isinstance(fval, float)
#     assert isinstance(gs_mol, float)
