# Basic script to make sure GPU and CPU are both working. 

import jax
import jax.numpy as jnp
from jax import jit, vmap
import time

import math
import numpy as np
from functools import partial


@jax.jit
def ci_func(
    ci,
    lmr_z,
    par_z,
    gb_mol,
    je,
    cair,
    oair,
    rh_can,
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
        ac = vcmax_z * jnp.maximum(ci - cp, 0.0) / (ci + kc * (1.0 + oair / ko))
        aj = je * jnp.maximum(ci - cp, 0.0) / (4.0 * ci + 8.0 * cp)
        ap = 3.0 * tpu_z
    else:
        ac = vcmax_z
        aj = qe * par_z * 4.6
        ap = kp_z * jnp.maximum(ci, 0.0) / forc_pbot

    # Gross photosynthesis
    aquad = theta_cj
    bquad = -(ac + aj)
    cquad = ac * aj
    r1, r2 = quadratic_roots(aquad, bquad, cquad)
    ai = jnp.minimum(r1, r2)

    aquad = theta_ip
    bquad = -(ai + ap)
    cquad = ai * ap
    r1, r2 = quadratic_roots(aquad, bquad, cquad)
    ag = jnp.maximum(0.0, jnp.minimum(r1, r2))

    # Net photosynthesis
    an = ag - lmr_z

    def compute_conductance(a): # for a single photosynthesis value
        # Quadratic gs_mol calculation
        cs = cair - 1.4 / gb_mol * a * forc_pbot
        if stomatalcond_mtd == stomatalcond_mtd_medlyn2011:
            term = 1.6 * a / (cs / forc_pbot * 1.0e06)
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
            gs_mol = jnp.maximum(r1, r2) * 1.0e06
        elif stomatalcond_mtd == stomatalcond_mtd_bb1987:
            aquad = cs
            bquad = cs * (gb_mol - bbb) - mbb * a * forc_pbot
            cquad = -gb_mol * (cs * bbb + mbb * a * forc_pbot * rh_can)
            r1, r2 = quadratic_roots(aquad, bquad, cquad)
            gs_mol = jnp.maximum(r1, r2)
        else:
            gs_mol = 0.0

        # Derive new estimate for ci
        fval = ci - cair + a * forc_pbot * (1.4 / gb_mol + 1.6 / gs_mol)

        return fval

    return jnp.where(an < 0.0, 0.0, compute_conductance(an))


# Just use brent now
lmr_z = 4
par_z = 500
gb_mol = 50_000
je = 40
cair = 45
oair = 21000
rh_can = 0.40

@jax.jit
def f(x):
    return partial(ci_func, lmr_z=4, par_z=500, gb_mol=50_000, je=40, cair=45, oair=21000, rh_can=0.40)(x)

eps = 1e-6

@jax.jit
def sign(x):
    return x < 0


@jax.jit
def between(a, b, x):
    return jax.lax.cond(
        a <= b,
        lambda: (a <= x) & (x <= b),
        lambda: (b <= x) & (x <= a)
    )


# Inverse quadratic interpolation
@jax.jit
def iqi(a, b, c):
    """
    Arguments: a, b, and c are three x coordinates.
    Returns: the x-intercept of the "sideways" parabola spanning (a, f(a)), (b, f(b)) and (c, f(c)).
    """
    fa = f(a)
    fb = f(b)
    fc = f(c)

    return c * (fa*fb) / ((fc - fb) * (fc - fa)) \
            + b * (fc * fa) / ((fb - fc) * (fb - fa)) \
            + a * (fc * fb) / ((fa - fc) * (fa - fb))


@jax.jit
def element_brent(a, b):
    def condition(state):
        a, b, c, d, bflag = state

        return abs(f(b)) >= eps
    
    def body(state):
        a, b, c, d, bflag = state
        d = c
        a, b = jax.lax.cond(
            abs(f(a)) < abs(f(b)),
            lambda: (b, a),
            lambda: (a, b)
        )        

        # Compute midpoint and interpolation
        m = (a+b)/2.0
        s = jax.lax.cond(
            (f(a) != f(c)) & (f(b) != f(c)),
            lambda: iqi(a, b, c), # IQI
            lambda: b - f(b)/((f(b)-f(c))/(b-c)) # Secant
        )

        # Take a step
        k = (3*a + b) / 4
        can_interpolate = jax.lax.cond(
            bflag,
            lambda: abs(s - b) < (1/2)*abs(b - c),
            lambda: (c == d) | (abs(s - b) < (1/2)*abs(c - d))
        )

        a, b, c, bflag = jax.lax.cond(
            can_interpolate & between(k, b, s),
            # Use interpolation
            lambda: jax.lax.cond(
                sign(f(a)) == sign(f(s)),
                lambda: (b, s, b, False),
                lambda: (a, s, b, False)
            ),
            # Use bisection
            lambda: jax.lax.cond(
                sign(f(a)) == sign(f(m)),
                lambda: (b, m, b, True),
                lambda: (a, m, b,True)
            )
        )
        
        return a, b, c, d, bflag
    

    bflag = False
    a, b, c = jax.lax.cond(
        abs(f(a)) < abs(f(b)),
        lambda: (b, a, b),
        lambda: (a, b, a)
    )
    d = c

    a, b, c, d, bflag = jax.lax.while_loop(condition, body, (a, b, c, d, bflag))

    return b


@jax.jit
def sign_change(x):
    # Keep checking outward points until you have a sign change on f

    diff = jax.lax.cond(
        x != 0,
        lambda: x / 50,
        lambda: 1/50
    )
    a, b = x - diff, x + diff 
    
    def condition(state):
        a, b, diff = state
        return sign(f(a)) == sign(f(b))
    
    def body(state):
        a, b, diff = state

        diff *= math.sqrt(2)
        a, b = x - diff, x + diff

        return a, b, diff

    a, b, diff = jax.lax.while_loop(condition, body, (a, b, diff))
    return a, b


@jax.jit
def element_brent_with_sign_change(x):
    a, b = sign_change(x)
    return element_brent(a, b)

def quadratic_roots(a, b, c):
    sqrt_discriminant = jnp.sqrt(b**2 - 4 * a * c)
    root1 = (-b - sqrt_discriminant) / (2 * a)
    root2 = (-b + sqrt_discriminant) / (2 * a)
    return root1, root2






def measure_time(device, fn, inputs):
    # Set the JAX device
    jax.devices(device)

    # Warm-up
    def process_item(item):
        if callable(item):
            return item
        else:
            return jnp.array([item[0]])
    
    _ = fn(*tuple(process_item(item) for item in inputs))
    
    # Run the function
    start_time = time.time()

    _ = fn(*inputs)

    return time.time() - start_time

def compare_cpu_gpu(fn, *args):
    vectorized_fn = jax.vmap(fn)
    cpu_time = measure_time("cpu", vectorized_fn, args)
    print(f"Time on CPU: {cpu_time:.6f} seconds")

    # Only measure GPU time if a GPU is available
    if jax.lib.xla_bridge.get_backend().platform == "gpu":
        gpu_time = measure_time("gpu", vectorized_fn, args)
        print(f"Time on GPU: {gpu_time:.6f} seconds")
    else:
        print("No GPU backend detected.")


if __name__ == "__main__":
    num_samples = 1000000
    ci = jnp.linspace(35, 70, num=num_samples)

    print("Runtime for ci_func with JAX:")
    compare_cpu_gpu(element_brent_with_sign_change, ci)
