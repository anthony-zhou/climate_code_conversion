# Complete conversion to JAX

import jax.numpy as np

def ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c, c3flag=True, stomatalcond_mtd=1):
    # Constants
    forc_pbot = 121000.0
    medlynslope = 6.0
    medlynintercept = 10000.0
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

    # C3 or C4 photosynthesis
    if c3flag:
        ac = vcmax_z * max(ci-cp, 0.0) / (ci+kc*(1.0+oair/ko))
        aj = je * max(ci-cp, 0.0) / (4.0*ci+8.0*cp)
        ap = 3.0 * tpu_z
    else:
        ac = vcmax_z
        aj = qe * par_z * 4.6
        ap = kp_z * max(ci, 0.0) / forc_pbot

    # Gross photosynthesis
    aquad = theta_cj
    bquad = -(ac + aj)
    cquad = ac * aj
    r1, r2 = np.roots(np.array([aquad, bquad, cquad]))
    ai = min(r1,r2)

    aquad = theta_ip
    bquad = -(ai + ap)
    cquad = ai * ap
    r1, r2 = np.roots(np.array([aquad, bquad, cquad]))
    ag = max(0.0, min(r1,r2))

    # Net photosynthesis
    an = ag - lmr_z
    if an < 0.0:
        fval = 0.0
        return fval, None

    # Quadratic gs_mol calculation
    cs = cair - 1.4/gb_mol * an * forc_pbot
    if stomatalcond_mtd == stomatalcond_mtd_medlyn2011:
        term = 1.6 * an / (cs / forc_pbot * 1.e06)
        aquad = 1.0
        bquad = -(2.0 * (medlynintercept*1.e-06 + term) + (medlynslope * term)**2 / (gb_mol*1.e-06 * rh_can))
        cquad = medlynintercept**2*1.e-12 + (2.0*medlynintercept*1.e-06 + term * (1.0 - medlynslope**2 / rh_can)) * term
        r1, r2 = np.roots(np.array([aquad, bquad, cquad]))
        gs_mol = max(r1,r2) * 1.e06
    elif stomatalcond_mtd == stomatalcond_mtd_bb1987:
        aquad = cs
        bquad = cs*(gb_mol - bbb) - mbb*an*forc_pbot
        cquad = -gb_mol*(cs*bbb + mbb*an*forc_pbot*rh_can)
        r1, r2 = np.roots(np.array([aquad, bquad, cquad]))
        gs_mol = max(r1,r2)
    else:
        gs_mol = 0.0

    # Derive new estimate for ci
    fval = ci - cair + an * forc_pbot * (1.4*gs_mol+1.6*gb_mol) / (gb_mol*gs_mol)

    return fval.real, gs_mol.real


def find_root(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c, c3flag=True, stomatalcond_mtd=1):

    # Define a function to find the root of my_function
    def find_root(ci):
        return ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c, c3flag=True, stomatalcond_mtd=1)[0]

    # Use SciPy to find the root of my_function
    from scipy.optimize import root_scalar
    sol = root_scalar(find_root, bracket=[500.0, 20000.0], method='brentq')

    # Graph value of ci_func from 1 to 100 using plotly
    import plotly.graph_objects as go
    import numpy as np
    ci = np.linspace(0, 8000, 100)
    fval = np.zeros(100)
    for i in range(100):
        fval[i] = find_root(ci[i])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ci, y=fval))
    fig.update_layout(title='ci_func', xaxis_title='ci', yaxis_title='fval')
    fig.write_image('./fig4.png')

    gs_mol = ci_func(sol.root, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c, c3flag=True, stomatalcond_mtd=1)[1]
    print('ci = ', sol.root, 'gs_mol = ', gs_mol)



    # Return the root of my_function
    return sol.root


if __name__ == '__main__':
    ci = 40
    lmr_z = 4
    par_z = 500
    gb_mol = 500
    je = 60
    cair = 5000
    oair = 21000
    rh_can = 0.40
    p = 1
    iv = 1
    c = 1

    ci_val = find_root(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)

    # assert ci_val == pytest.approx(40.0)