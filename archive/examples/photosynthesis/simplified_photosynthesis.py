import numpy as np
import matplotlib.pyplot as plt

class Atm2Land(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update(kwargs)

class Photosynthesis(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update(kwargs)


def smaller_root(a: float, b: float, c: float) -> float:
    """
    Finds the smaller root of a quadratic equation of the form ax^2 + bx + c = 0.

    :param a: Coefficient of x^2
    :param b: Coefficient of x
    :param c: Constant term
    :return: Smaller root of the quadratic equation
    """
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("Quadratic equation has no real roots")
    elif discriminant == 0:
        return -b / (2*a)
    else:
        root1 = (-b - discriminant**0.5) / (2*a)
        root2 = (-b + discriminant**0.5) / (2*a)
        return min(root1, root2)
    

def co_limit(A_c: float, A_j: float, A_p: float, theta_cj: float, theta_ip: float) -> float:
    """
    Co-limits A_c, A_j, and A_p using a single variable theta_cj and another constant theta_ip.

    :param A_c: Rubisco-limited photosynthesis rate (umol CO2 per square meter per second)
    :param A_j: Electron transport-limited photosynthesis rate (umol CO2 per square meter per second)
    :param A_p: Triose phosphate utilization rate (umol CO2 per square meter per second)
    :param theta_cj: Weighting factor for A_c and A_j
    :param theta_ip: Weighting factor for A_tot and A_p
    :return: Co-limited photosynthesis rate (umol CO2 per square meter per second)
    """

    A_tot = smaller_root(theta_cj, -(A_c + A_j), A_c * A_j)
    A_tot = smaller_root(theta_ip, -(A_tot + A_p), A_tot * A_p)
    return A_tot

def medlyn2011(an: float, cs, forc_pbot: float, gb_mol: float, rh_can: float, medlyn_intercept: float, medlyn_slope: float) -> float:
    term = 1.6 * an / (cs / forc_pbot * 1e6)
    a = 1.0
    b = -(2.0 * (medlyn_intercept * 1e-6 + term) + (medlyn_slope * term)**2 / (gb_mol * 1e-6 * rh_can))
    c = medlyn_intercept**2 * 1e-12 + (2.0 * medlyn_intercept * 1e-6 + term * (1.0 - medlyn_slope**2 / rh_can)) * term
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("Quadratic equation has no real roots")
    elif discriminant == 0:
        gs_mol = -b / (2*a)
    else:
        root1 = (-b - discriminant**0.5) / (2*a)
        root2 = (-b + discriminant**0.5) / (2*a)
        gs_mol = max(root1, root2)
    return gs_mol * 1e6

def ci_func(
        ci: float, # Internal CO2 concentration (umol CO2 per mol air)
        gb_mol: float, # Boundary layer conductance (mol air per square meter per second)
        je: float, # Electron transport rate (umol electrons per square meter per second)
        cair: float, # Atmospheric CO2 concentration (umol CO2 per mol air)
        oair: float, # Atmospheric O2 concentration (umol O2 per mol air)
        lmr_z: float, # Leaf to air vapor pressure difference (kPa)
        par_z: float, # Photosynthetically active radiation (umol photons per square meter per second)
        rh_can: float, # Relative humidity (%)
        atm2land: Atm2Land,
        photosynthesis: Photosynthesis,
):
    """
    Simplified photosynthesis model.

    :param ci: Internal CO2 partial pressure (Pa)
    :param gb_mol: Boundary layer conductance (mol air per square meter per second)
    :param je: Electron transport rate (umol electrons per square meter per second)
    :param cair: Atmospheric CO2 partial pressure (Pa)
    :param oair: Atmospheric O2 partial pressure (Pa)
    :param lmr_z: Leaf to air vapor pressure difference (kPa)
    :param par_z: Photosynthetically active radiation (umol photons per square meter per second)
    :param rh_can: Relative humidity (%)
    :param atm2land: Atmosphere to land parameters
    :param photosynthesis: Photosynthesis parameters
    :return: Photosynthesis rate (umol CO2 per square meter per second), stomatal conductance (mol air per square meter per second), internal CO2 partial pressure (umol CO2 per mol air)
    """

    # These are defined as constants for simplicity
    theta_cj = 0.9
    theta_ip = 0.9
    # Michelis-Menten constants for oxygen and carbon dioxide (come back to this)
    ko = 1e4
    kc = 1000
    vcmax = 50 # Maximum Rubisco-limited photosynthesis rate (umol CO2 per square meter per second)
    gamma = 20 # CO2 compensation point (umol CO2 per mol air)
    tpu_z = 1 # Triose phosphate utilization rate (umol CO2 per square meter per second)
    max_cs = 2000 # Max CO2 partial pressure at leaf surface
    medlyn_slope = 6
    medlyn_intercept = 100


    # Atmospheric pressure (Pa)
    forc_pbot = 121000

    # Assume c3 photosynthesis
    A_c = vcmax * max(ci - gamma, 0) / (ci + kc * (1 + oair / ko))
    A_j = je * max(ci - gamma, 0) / (4 * ci + 8 * gamma)
    A_p = 3 * tpu_z

    print("Photosynthesis rates")
    print(f"A_c: {A_c}")
    print(f"A_j: {A_j}")
    print(f"A_p: {A_p}")

    # Co-limit A_c, A_j, and A_p using a single variable theta_cj and another constant theta_ip
    A_tot = co_limit(A_c, A_j, A_p, theta_cj, theta_ip)

    print(f"A_tot: {A_tot}")

    # Calculate stomatal conductance
    cs = cair - 1.4 / gb_mol * A_tot * forc_pbot
    cs = max(cs, max_cs)
    # Assume the medlyn method
    gs_mol = medlyn2011(A_tot, cs, forc_pbot, gb_mol, rh_can, medlyn_intercept, medlyn_slope)
    fval = ci - (cair + A_tot * forc_pbot * (1.4 / gb_mol + 1.6 / gs_mol))


    return A_tot, gs_mol, fval


if __name__ == '__main__':
    ci = 40
    gb_mol = 500
    
    # Electron transport rate (depends on photon flux)
    theta_psii = 0.8
    

    cair = 40
    oair = 21000
    lmr_z = 6
    par_z = 500
    rh_can = 0.40
    atm2land = Atm2Land()
    photosynth = Photosynthesis()

    a, gs_mol, fval = ci_func(ci,
                  gb_mol,
                  je,
                  cair,
                  oair, lmr_z, par_z, rh_can, atm2land, photosynth)
    # Plot fval as a function of ci
    cis = np.linspace(30, 1200, 200)
    fvals = []
    for ci in cis:
        a, gs_mol, fval = ci_func(ci,
                      gb_mol,
                      je,
                      cair,
                      oair, lmr_z, par_z, rh_can, atm2land, photosynth)
        fvals.append(fval)
    plt.plot(cis, fvals)
    plt.savefig("fval.png")
    







