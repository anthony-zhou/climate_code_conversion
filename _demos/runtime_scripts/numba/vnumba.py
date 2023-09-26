import math
import numpy as np
from numba import jit


@jit(nopython=True)
def hybrid(x0, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c):
    eps = 1e-2
    eps1 = 1e-4
    itmax = 40
    iter = 0
    tol, minx, minf = 0.0, 0.0, 0.0

    f0, gs_mol = ci_func(x0, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)

    if f0 == 0.0:
        return x0, gs_mol, iter

    minx = x0
    minf = f0
    x1 = x0 * 0.99

    f1, gs_mol = ci_func(x1, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)

    if f1 == 0.0:
        return x1, gs_mol, iter

    if f1 < minf:
        minx = x1
        minf = f1

    while True:
        iter += 1
        dx = -f1 * (x1 - x0) / (f1 - f0)
        x = x1 + dx
        tol = abs(x) * eps

        if abs(dx) < tol:
            return x, gs_mol, iter

        x0 = x1
        f0 = f1
        x1 = x

        f1, gs_mol = ci_func(x1, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)

        if f1 < minf:
            minx = x1
            minf = f1

        if abs(f1) <= eps1:
            return x1, gs_mol, iter

        if f1 * f0 < 0.0:
            x, gs_mol = brent(
                x0,
                x1,
                f0,
                f1,
                tol,
                p,
                iv,
                c,
                gb_mol,
                je,
                cair,
                oair,
                lmr_z,
                par_z,
                rh_can,
                gs_mol,
            )
            return x, gs_mol, iter

        if iter > itmax:
            f1, gs_mol = ci_func(
                minx, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c
            )
            break

    return x0, gs_mol, iter


@jit(nopython=True)
def brent(
    x1,
    x2,
    f1: float,
    f2: float,
    tol,
    ip,
    iv,
    ic,
    gb_mol,
    je,
    cair,
    oair,
    lmr_z,
    par_z,
    rh_can,
    gs_mol,
):
    itmax = 20
    eps = 1e-2
    iter = 0
    a = x1
    b = x2
    fa = f1
    fb = f2

    if (fa > 0 and fb > 0) or (fa < 0 and fb < 0):
        print("root must be bracketed for brent")
        raise ValueError("f(a) and f(b) must have opposite signs for Brent's method.")

    c = b
    fc = fb

    while iter != itmax:
        iter += 1
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c = a
            fc = fa
            d = b - a
            e = d

        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol1 = 2 * eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)

        if abs(xm) <= tol1 or fb == 0:
            x = b
            return x, gs_mol

        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa

            if a == c:
                p = 2 * xm * s
                q = 1 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)

            if p > 0:
                q = -q

            p = abs(p)

            if 2 * p < min(3 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d

        a = b
        fa = fb

        if abs(d) > tol1:
            b = b + d
        else:
            b = b + math.copysign(tol1, xm)

        fb, gs_mol = ci_func(
            b, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, ip, iv, ic
        )

        if fb == 0:
            break

    if iter == itmax:
        print("brent exceeding maximum iterations", b, fb)

    x = b
    return x, gs_mol


@jit(nopython=True)
def quadratic_roots(a, b, c):
    sqrt_discriminant = math.sqrt(b**2 - 4 * a * c)
    root1 = (-b - sqrt_discriminant) / (2 * a)
    root2 = (-b + sqrt_discriminant) / (2 * a)
    return root1, root2


@jit(nopython=True)
def ci_func(
    ci,
    lmr_z,
    par_z,
    gb_mol,
    je,
    cair,
    oair,
    rh_can,
    p,
    iv,
    c,
    c3flag=True,
    stomatalcond_mtd=1,
):
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
        ac = vcmax_z * max(ci - cp, 0.0) / (ci + kc * (1.0 + oair / ko))
        aj = je * max(ci - cp, 0.0) / (4.0 * ci + 8.0 * cp)
        ap = 3.0 * tpu_z
    else:
        ac = vcmax_z
        aj = qe * par_z * 4.6
        ap = kp_z * max(ci, 0.0) / forc_pbot

    # Gross photosynthesis
    aquad = theta_cj
    bquad = -(ac + aj)
    cquad = ac * aj
    r1, r2 = quadratic_roots(aquad, bquad, cquad)
    ai = min(r1, r2)

    aquad = theta_ip
    bquad = -(ai + ap)
    cquad = ai * ap
    r1, r2 = quadratic_roots(aquad, bquad, cquad)
    ag = max(0.0, min(r1, r2))

    # Net photosynthesis
    an = ag - lmr_z
    if an < 0.0:
        # print("NEGATIVE PHOTOSYNTHESIS")
        fval = 0.0
        return fval, None

    # Quadratic gs_mol calculation
    cs = cair - 1.4 / gb_mol * an * forc_pbot
    if stomatalcond_mtd == stomatalcond_mtd_medlyn2011:
        term = 1.6 * an / (cs / forc_pbot * 1.0e06)
        aquad = 1.0
        bquad = -(
            2.0 * (medlynintercept * 1.0e-06 + term)
            + (medlynslope * term) ** 2 / (gb_mol * 1.0e-06 * rh_can)
        )
        cquad = (
            medlynintercept**2 * 1.0e-12
            + (
                2.0 * medlynintercept * 1.0e-06
                + term * (1.0 - medlynslope**2 / rh_can)
            )
            * term
        )
        r1, r2 = quadratic_roots(aquad, bquad, cquad)
        gs_mol = max(r1, r2) * 1.0e06
    elif stomatalcond_mtd == stomatalcond_mtd_bb1987:
        aquad = cs
        bquad = cs * (gb_mol - bbb) - mbb * an * forc_pbot
        cquad = -gb_mol * (cs * bbb + mbb * an * forc_pbot * rh_can)
        r1, r2 = quadratic_roots(aquad, bquad, cquad)
        gs_mol = max(r1, r2)
    else:
        gs_mol = 0.0

    # Derive new estimate for ci
    fval = ci - cair + an * forc_pbot * (1.4 / gb_mol + 1.6 / gs_mol)

    return fval, gs_mol


@jit(nopython=True)
def main(
    ci,
    lmr_z,
    par_z,
    gb_mol,
    je,
    cair,
    oair,
    rh_can,
    p,
    iv,
    c,
    c3flag=True,
    stomatalcond_mtd=1,
):
    ci_val, gs_mol, _ = hybrid(
        ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c
    )

    return ci_val, gs_mol
