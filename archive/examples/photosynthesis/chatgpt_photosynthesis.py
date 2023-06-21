import numpy as np

# define the quadratic function used in the original Fortran code
def quadratic(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant >= 0:
        r1 = (-b + np.sqrt(discriminant)) / (2 * a)
        r2 = (-b - np.sqrt(discriminant)) / (2 * a)
    else:
        r1 = r2 = np.nan
    return r1, r2

def ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c, atm2lnd_inst, photosyns_inst):
    forc_pbot = atm2lnd_inst['forc_pbot_downscaled_col']
    c3flag = photosyns_inst['c3flag_patch']
    ivt = patch['itype']
    medlynslope = pftcon['medlynslope']
    medlynintercept = pftcon['medlynintercept']
    stomatalcond_mtd = photosyns_inst['stomatalcond_mtd']
    vcmax_z = photosyns_inst['vcmax_z_patch']
    cp = photosyns_inst['cp_patch']
    kc = photosyns_inst['kc_patch']
    ko = photosyns_inst['ko_patch']
    qe = photosyns_inst['qe_patch']
    tpu_z = photosyns_inst['tpu_z_patch']
    kp_z = photosyns_inst['kp_z_patch']
    bbb = photosyns_inst['bbb_patch']
    mbb = photosyns_inst['mbb_patch']
    ac = aj = ap = ag = an = np.zeros_like(vcmax_z)
    
    if c3flag[p]:
        ac[p, iv] = vcmax_z[p, iv] * max(ci - cp[p], 0) / (ci + kc[p] * (1 + oair / ko[p]))
        aj[p, iv] = je * max(ci - cp[p], 0) / (4 * ci + 8 * cp[p])
        ap[p, iv] = 3 * tpu_z[p, iv]
    else:
        ac[p, iv] = vcmax_z[p, iv]
        aj[p, iv] = qe[p] * par_z * 4.6
        ap[p, iv] = kp_z[p, iv] * max(ci, 0) / forc_pbot[c]

    aquad = params_inst['theta_cj'][ivt[p]]
    bquad = -(ac[p, iv] + aj[p, iv])
    cquad = ac[p, iv] * aj[p, iv]
    r1, r2 = quadratic(aquad, bquad, cquad)
    ai = min(r1, r2)

    aquad = params_inst['theta_ip']
    bquad = -(ai + ap[p, iv])
    cquad = ai * ap[p, iv]
    r1, r2 = quadratic(aquad, bquad, cquad)
    ag[p, iv] = max(0, min(r1, r2))

    an[p, iv] = ag[p, iv] - lmr_z
    if an[p, iv] < 0:
        fval = 0
        return fval, gs_mol

    cs = cair - 1.4 / gb_mol * an[p, iv] * forc_pbot[c]
    cs = max(cs, max_cs)
    if stomatalcond_mtd == stomatalcond_mtd_medlyn2011:
        term = 1.6 * an[p, iv] / (cs / forc_pbot[c] * 1e6)
        aquad = 1.0
        bquad = -(2.0 * (medlynintercept[patch['itype'][p]] * 1e-6 + term) +
                  (medlynslope[patch['itype'][p]] * term)**2 /
                  (gb_mol * 1e-6 * rh_can))
        cquad = (medlynintercept[patch['itype'][p]]**2 * 1e-12 +
                 (2.0 * medlynintercept[patch['itype'][p]] * 1e-6 + term *
                  (1.0 - medlynslope[patch['itype'][p]] * medlynslope[patch['itype'][p]] / rh_can)) * term)
        r1, r2 = quadratic(aquad, bquad, cquad)
        gs_mol = max(r1, r2) * 1e6
    elif stomatalcond_mtd == stomatalcond_mtd_bb1987:
        aquad = cs
        bquad = cs * (gb_mol - bbb[p]) - mbb[p] * an[p, iv] * forc_pbot[c]
        cquad = -gb_mol * (cs * bbb[p] + mbb[p] * an[p, iv] * forc_pbot[c] * rh_can)
        r1, r2 = quadratic(aquad, bquad, cquad)
        gs_mol = max(r1, r2)

    fval = ci - cair + an[p, iv] * forc_pbot[c] * (1.4 * gs_mol + 1.6 * gb_mol) / (gb_mol * gs_mol)
    return fval, gs_mol
