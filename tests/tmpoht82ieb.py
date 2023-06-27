import numpy as np

def ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c, c3flag=True, stomatalcond_mtd=1):
    # Constants
    forc_pbot = 121000.0
    medlynslope = 6.0
    medlynintercept = 100.0
    vcmax_z = 62.5
    cp = 4.275
    kc = 40.49
    ko = 27840.0
    qe = 1.0
    tpu_z = 31.5
    kp_z = 1.0
    bbb = 100.0
    mbb = 9.0
    theta_cj = 0.98
    theta_ip = 0.95
    stomatalcond_mtd_medlyn2011 = 1
    stomatalcond_mtd_bb1987 = 2

    if c3flag:
        ac = vcmax_z * max(ci-cp, 0.0) / (ci+kc*(1.0+oair/ko))
        aj = je * max(ci-cp, 0.0) / (4.0*ci+8.0*cp)
        ap = 3.0 * tpu_z
    else:
        ac = vcmax_z
        aj = qe * par_z * 4.6
        ap = kp_z * max(ci, 0.0) / forc_pbot

    aquad = theta_cj
    bquad = -(ac + aj)
    cquad = ac * aj
    r1, r2 = np.roots([aquad, bquad, cquad])
    ai = min(r1,r2)

    aquad = theta_ip
    bquad = -(ai + ap)
    cquad = ai * ap
    r1, r2 = np.roots([aquad, bquad, cquad])
    ag = max(0.0,min(r1,r2))

    an = ag - lmr_z
    if an < 0.0:
        fval = 0.0
        return fval, None

    cs = cair - 1.4/gb_mol * an * forc_pbot
    if stomatalcond_mtd == stomatalcond_mtd_medlyn2011:
        term = 1.6 * an / (cs / forc_pbot * 1.e06)
        aquad = 1.0
        bquad = -(2.0 * (medlynintercept*1.e-06 + term) + (medlynslope * term)**2 / (gb_mol*1.e-06 * rh_can))
        cquad = medlynintercept**2*1.e-12 + (2.0*medlynintercept*1.e-06 + term * (1.0 - medlynslope**2 / rh_can)) * term
        r1, r2 = np.roots([aquad, bquad, cquad])
        gs_mol = max(r1,r2) * 1.e06
    elif stomatalcond_mtd == stomatalcond_mtd_bb1987:
        aquad = cs
        bquad = cs*(gb_mol - bbb) - mbb*an*forc_pbot
        cquad = -gb_mol*(cs*bbb + mbb*an*forc_pbot*rh_can)
        r1, r2 = np.roots([aquad, bquad, cquad])
        gs_mol = max(r1,r2)
    else:
        raise ValueError("Invalid stomatalcond_mtd")

    fval = ci - cair + an * forc_pbot * (1.4*gs_mol+1.6*gb_mol) / (gb_mol*gs_mol)
    return fval, gs_mol


UNIT TESTS:
import pytest
import numpy as np

def test_ci_func_c3flag_true():
    ci, gs_mol = ci_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert isinstance(ci, float)
    assert isinstance(gs_mol, float)

def test_ci_func_c3flag_false():
    ci, gs_mol = ci_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, c3flag=False)
    assert isinstance(ci, float)
    assert isinstance(gs_mol, float)

def test_ci_func_an_negative():
    ci, gs_mol = ci_func(1, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert ci == 0.0
    assert gs_mol is None

def test_ci_func_stomatalcond_mtd_medlyn2011():
    ci, gs_mol = ci_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, stomatalcond_mtd=1)
    assert isinstance(ci, float)
    assert isinstance(gs_mol, float)

def test_ci_func_stomatalcond_mtd_bb1987():
    ci, gs_mol = ci_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, stomatalcond_mtd=2)
    assert isinstance(ci, float)
    assert isinstance(gs_mol, float)

def test_ci_func_invalid_stomatalcond_mtd():
    with pytest.raises(ValueError):
        ci_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, stomatalcond_mtd=3)
import pytest
import numpy as np

def test_ci_func_c3flag_true():
    ci, gs_mol = ci_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert isinstance(ci, float)
    assert isinstance(gs_mol, float)

def test_ci_func_c3flag_false():
    ci, gs_mol = ci_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, c3flag=False)
    assert isinstance(ci, float)
    assert isinstance(gs_mol, float)

def test_ci_func_an_negative():
    ci, gs_mol = ci_func(1, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert ci == 0.0
    assert gs_mol is None

def test_ci_func_stomatalcond_mtd_medlyn2011():
    ci, gs_mol = ci_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, stomatalcond_mtd=1)
    assert isinstance(ci, float)
    assert isinstance(gs_mol, float)

def test_ci_func_stomatalcond_mtd_bb1987():
    ci, gs_mol = ci_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, stomatalcond_mtd=2)
    assert isinstance(ci, float)
    assert isinstance(gs_mol, float)

def test_ci_func_invalid_stomatalcond_mtd():
    with pytest.raises(ValueError):
        ci_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, stomatalcond_mtd=3)