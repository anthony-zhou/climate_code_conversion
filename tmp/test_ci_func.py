import numpy as np

def ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c, atm2lnd_inst, photosyns_inst):
    # local variables
    ai = None                  # intermediate co-limited photosynthesis (umol CO2/m**2/s)
    cs = None                  # CO2 partial pressure at leaf surface (Pa)
    term = None                # intermediate in Medlyn stomatal model
    aquad, bquad, cquad = None, None, None  # terms for quadratic equations
    r1, r2 = None, None        # roots of quadratic equation

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
        ac[p, iv] = vcmax_z[p, iv] * max(ci - cp[p], 0) / (ci + kc[p] * (1 + oair / ko[p]))

        # C3: RuBP-limited photosynthesis
        aj[p, iv] = je * max(ci - cp[p], 0) / (4 * ci + 8 * cp[p])

        # C3: Product-limited photosynthesis
        ap[p, iv] = 3 * tpu_z[p, iv]

    else:
        # C4: Rubisco-limited photosynthesis
        ac[p, iv] = vcmax_z[p, iv]

        # C4: RuBP-limited photosynthesis
        aj[p, iv] = qe[p] * par_z * 4.6

        # C4: PEP carboxylase-limited (CO2-limited)
        ap[p, iv] = kp_z[p, iv] * max(ci, 0) / forc_pbot[c]

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
    ag[p, iv] = max(0, min(r1, r2))

    # Net photosynthesis. Exit iteration if an < 0
    an[p, iv] = ag[p, iv] - lmr_z
    if an[p, iv] < 0:
        fval = 0
        return fval, None

    # Quadratic gs_mol calculation with an known. Valid for an >= 0.
    # With an <= 0, then gs_mol = bbb or medlyn intercept
    cs = cair - 1.4 / gb_mol * an[p, iv] * forc_pbot[c]
    cs = max(cs, photosyns_inst['max_cs'])
    if stomatalcond_mtd == photosyns_inst['stomatalcond_mtd_medlyn2011']:
        term = 1.6 * an[p, iv] / (cs / forc_pbot[c] * 1.e06)
        aquad = 1.0
        bquad = -(2.0 * (medlynintercept[ivt[p]] * 1.e-06 + term) + (medlynslope[ivt[p]] * term) ** 2 /
                  (gb_mol * 1.e-06 * rh_can))
        cquad = medlynintercept[ivt[p]] * medlynintercept[ivt[p]] * 1.e-12 + \
                (2.0 * medlynintercept[ivt[p]] * 1.e-06 + term *
                 (1.0 - medlynslope[ivt[p]] * medlynslope[ivt[p]] / rh_can)) * term

        r1, r2 = np.roots([aquad, bquad, cquad])
        gs_mol = max(r1, r2) * 1.e06
    elif stomatalcond_mtd == photosyns_inst['stomatalcond_mtd_bb1987']:
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

def test_ci_func():
    # Define a mock for atm2lnd_inst
    atm2lnd_inst = {
        'forc_pbot_downscaled_col': np.array([1.0])
    }

    # Define a mock for photosyns_inst
    photosyns_inst = {
        'c3flag_patch': np.array([True]),
        'itype': np.array([0]),
        'medlynslope': np.array([1.0]),
        'medlynintercept': np.array([1.0]),
        'stomatalcond_mtd': 'stomatalcond_mtd_medlyn2011',
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
        'theta_ip': 1.0,
        'max_cs': 1.0,
        'stomatalcond_mtd_medlyn2011': 'stomatalcond_mtd_medlyn2011',
        'stomatalcond_mtd_bb1987': 'stomatalcond_mtd_bb1987'
    }

    # Call the function with test inputs
    fval, gs_mol = ci_func(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, atm2lnd_inst, photosyns_inst)

    # Assert the expected outputs
    assert fval is not None
    assert gs_mol is not None

def test_ci_func_with_c4_photosynthesis():
    # Similar to the previous test, but with 'c3flag_patch' set to False to test the C4 photosynthesis path
    # Define the mocks for atm2lnd_inst and photosyns_inst similar to the previous test
    # but with 'c3flag_patch' set to False
    atm2lnd_inst = {
        'forc_pbot_downscaled_col': np.array([1.0])
    }

    photosyns_inst = {
        'c3flag_patch': np.array([False]),
        'itype': np.array([0]),
        'medlynslope': np.array([1.0]),
        'medlynintercept': np.array([1.0]),
        'stomatalcond_mtd': 'stomatalcond_mtd_medlyn2011',
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
        'theta_ip': 1.0,
        'max_cs': 1.0,
        'stomatalcond_mtd_medlyn2011': 'stomatalcond_mtd_medlyn2011',
        'stomatalcond_mtd_bb1987': 'stomatalcond_mtd_bb1987'
    }

    # Call the function with test inputs
    fval, gs_mol = ci_func(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, atm2lnd_inst, photosyns_inst)

    # Assert the expected outputs
    assert fval is not None
    assert gs_mol is not None

def test_ci_func_with_negative_net_photosynthesis():
    # Similar to the previous tests, but with inputs that result in negative net photosynthesis
    # Define the mocks for atm2lnd_inst and photosyns_inst similar to the previous tests
    # but with inputs that result in negative net photosynthesis
    atm2lnd_inst = {
        'forc_pbot_downscaled_col': np.array([1.0])
    }

    photosyns_inst = {
        'c3flag_patch': np.array([True]),
        'itype': np.array([0]),
        'medlynslope': np.array([1.0]),
        'medlynintercept': np.array([1.0]),
        'stomatalcond_mtd': 'stomatalcond_mtd_medlyn2011',
        'ac_patch': np.zeros((1, 1)),
        'aj_patch': np.zeros((1, 1)),
        'ap_patch': np.zeros((1, 1)),
        'ag_patch': np.zeros((1, 1)),
        'an_patch': np.zeros((1, 1)),
        'vcmax_z_patch': np.array([[0.0]]),  # Set to 0 to result in negative net photosynthesis
        'cp_patch': np.array([1.0]),
        'kc_patch': np.array([1.0]),
        'ko_patch': np.array([1.0]),
        'qe_patch': np.array([1.0]),
        'tpu_z_patch': np.array([[1.0]]),
        'kp_z_patch': np.array([[1.0]]),
        'bbb_patch': np.array([1.0]),
        'mbb_patch': np.array([1.0]),
        'theta_cj': np.array([1.0]),
        'theta_ip': 1.0,
        'max_cs': 1.0,
        'stomatalcond_mtd_medlyn2011': 'stomatalcond_mtd_medlyn2011',
        'stomatalcond_mtd_bb1987': 'stomatalcond_mtd_bb1987'
    }

    # Call the function with test inputs
    fval, gs_mol = ci_func(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, atm2lnd_inst, photosyns_inst)

    # Assert the expected outputs
    assert fval == 0
    assert gs_mol is None