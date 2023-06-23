import numpy as np

def ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c, atm2lnd_inst, photosyns_inst):
    # local variables
    ai = 0.0
    cs = 0.0
    term = 0.0
    aquad = 0.0
    bquad = 0.0
    cquad = 0.0
    r1 = 0.0
    r2 = 0.0

    # associate
    forc_pbot = atm2lnd_inst['forc_pbot_downscaled_col']
    c3flag = photosyns_inst['c3flag_patch']
    ivt = photosyns_inst['itype']
    medlynslope = photosyns_inst['medlynslope']
    medlynintercept = photosyns_inst['medlynintercept']
    stomatalcond_mtd = photosyns_inst['stomatalcond_mtd']
    ac = photosyns_inst['ac_patch']
    aj = photosyns_inst['aj_patch']
    ap = photosyns_inst['ap_patch']
    ag = photosyns_inst['ag_patch']
    an = photosyns_inst['an_patch']
    vcmax_z = photosyns_inst['vcmax_z_patch']
    cp = photosyns_inst['cp_patch']
    kc = photosyns_inst['kc_patch']
    ko = photosyns_inst['ko_patch']
    qe = photosyns_inst['qe_patch']
    tpu_z = photosyns_inst['tpu_z_patch']
    kp_z = photosyns_inst['kp_z_patch']
    bbb = photosyns_inst['bbb_patch']
    mbb = photosyns_inst['mbb_patch']

    if c3flag[p]:
        # C3: Rubisco-limited photosynthesis
        ac[p, iv] = vcmax_z[p, iv] * max(ci - cp[p], 0.0) / (ci + kc[p] * (1.0 + oair / ko[p]))

        # C3: RuBP-limited photosynthesis
        aj[p, iv] = je * max(ci - cp[p], 0.0) / (4.0 * ci + 8.0 * cp[p])

        # C3: Product-limited photosynthesis
        ap[p, iv] = 3.0 * tpu_z[p, iv]
    else:
        # C4: Rubisco-limited photosynthesis
        ac[p, iv] = vcmax_z[p, iv]

        # C4: RuBP-limited photosynthesis
        aj[p, iv] = qe[p] * par_z * 4.6

        # C4: PEP carboxylase-limited (CO2-limited)
        ap[p, iv] = kp_z[p, iv] * max(ci, 0.0) / forc_pbot[c]

    # Gross photosynthesis. First co-limit ac and aj. Then co-limit ap
    aquad = photosyns_inst['theta_cj'][ivt[p]]
    bquad = -(ac[p, iv] + aj[p, iv])
    cquad = ac[p, iv] * aj[p, iv]
    r1, r2 = np.roots([aquad, bquad, cquad])
    ai = min(r1, r2)

    aquad = photosyns_inst['theta_ip']
    bquad = -(ai + ap[p, iv])
    cquad = ai * ap[p, iv]
    r1, r2 = np.roots([aquad, bquad, cquad])
    ag[p, iv] = max(0.0, min(r1, r2))

    # Net photosynthesis. Exit iteration if an < 0
    an[p, iv] = ag[p, iv] - lmr_z
    if an[p, iv] < 0.0:
        fval = 0.0
        return fval, gs_mol

    # Quadratic gs_mol calculation with an known. Valid for an >= 0.
    # With an <= 0, then gs_mol = bbb or medlyn intercept
    cs = cair - 1.4 / gb_mol * an[p, iv] * forc_pbot[c]
    cs = max(cs, max_cs)
    if stomatalcond_mtd == stomatalcond_mtd_medlyn2011:
        term = 1.6 * an[p, iv] / (cs / forc_pbot[c] * 1.e06)
        aquad = 1.0
        bquad = -(2.0 * (medlynintercept[ivt[p]] * 1.e-06 + term) + (medlynslope[ivt[p]] * term) ** 2 /
                  (gb_mol * 1.e-06 * rh_can))
        cquad = medlynintercept[ivt[p]] * medlynintercept[ivt[p]] * 1.e-12 + \
                (2.0 * medlynintercept[ivt[p]] * 1.e-06 + term *
                 (1.0 - medlynslope[ivt[p]] * medlynslope[ivt[p]] / rh_can)) * term

        r1, r2 = np.roots([aquad, bquad, cquad])
        gs_mol = max(r1, r2) * 1.e06
    elif stomatalcond_mtd == stomatalcond_mtd_bb1987:
        aquad = cs
        bquad = cs * (gb_mol - bbb[p]) - mbb[p] * an[p, iv] * forc_pbot[c]
        cquad = -gb_mol * (cs * bbb[p] + mbb[p] * an[p, iv] * forc_pbot[c] * rh_can)
        r1, r2 = np.roots([aquad, bquad, cquad])
        gs_mol = max(r1, r2)

    # Derive new estimate for ci
    fval = ci - cair + an[p, iv] * forc_pbot[c] * (1.4 * gs_mol + 1.6 * gb_mol) / (gb_mol * gs_mol)

    return fval, gs_mol

import pytest
import numpy as np

# Mock data
atm2lnd_inst = {'forc_pbot_downscaled_col': np.array([1.0])}
photosyns_inst = {
    'c3flag_patch': np.array([True]),
    'itype': np.array([1]),
    'medlynslope': np.array([1.0]),
    'medlynintercept': np.array([1.0]),
    'stomatalcond_mtd': np.array([1]),
    'ac_patch': np.zeros((1, 1)),
    'aj_patch': np.zeros((1, 1)),
    'ap_patch': np.zeros((1, 1)),
    'ag_patch': np.zeros((1, 1)),
    'an_patch': np.zeros((1, 1)),
    'vcmax_z_patch': np.array([[1.0]]),
    'cp_patch': np.array([1.0]),
    'kc_patch': np.array([1.0]),
    'ko_patch': np.array([1.0]),
    'qe_patch': np.array([1.0]),
    'tpu_z_patch': np.array([[1.0]]),
    'kp_z_patch': np.array([[1.0]]),
    'bbb_patch': np.array([1.0]),
    'mbb_patch': np.array([1.0]),
    'theta_cj': np.array([1.0]),
    'theta_ip': 1.0
}

def test_ci_func_c3():
    ci = 1.0
    lmr_z = 1.0
    par_z = 1.0
    gb_mol = 1.0
    je = 1.0
    cair = 1.0
    oair = 1.0
    rh_can = 1.0
    p = 0
    iv = 0
    c = 0
    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c, atm2lnd_inst, photosyns_inst)
    assert fval == pytest.approx(0.0, 0.01)
    assert gs_mol == pytest.approx(1.0, 0.01)

def test_ci_func_c4():
    photosyns_inst['c3flag_patch'] = np.array([False])
    ci = 1.0
    lmr_z = 1.0
    par_z = 1.0
    gb_mol = 1.0
    je = 1.0
    cair = 1.0
    oair = 1.0
    rh_can = 1.0
    p = 0
    iv = 0
    c = 0
    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c, atm2lnd_inst, photosyns_inst)
    assert fval == pytest.approx(0.0, 0.01)
    assert gs_mol == pytest.approx(1.0, 0.01)