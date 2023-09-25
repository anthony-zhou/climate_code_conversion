## Testing different methods of root finding in JAX

# Methods adapted from Haskell: https://github.com/osveliz/numerical-veliz/blob/master/src/rootfinding/BrentDekker.hs

import jax
from jax import numpy as jnp
from collections import namedtuple
from jax import jit
import time

from functools import partial

def f(x):
    return x ** 3 - x ** 2 - x - 1

eps = 1e-6


Result = namedtuple('Result', ['root', 'steps'])

@jax.jit
def element_bisect(a, b):
    def condition(state):
        a, b, c = state
        return jnp.abs(b - a) >= eps

    def body(state):
        a, b, c = state

        return jax.lax.cond(
            f(a) * f(c) > 0,
            lambda: (c, b, (a+b)/2),
            lambda: (a, c, (a+b)/2)
        )

    init_state = (a, b, (a + b) / 2.0)
    final_a, final_b, final_c = jax.lax.while_loop(condition, body, init_state)
    
    return final_c

@jax.jit
def element_secant(a, b):
    def condition(state):
        a, b = state
        return jnp.abs(f(a)) >= eps

    def body(state):
        a, b = state
        secant_intercept = a - f(a) / ((f(a) - f(b)) / (a - b))
        b = a
        a = secant_intercept
        return a, b
    
    init_state = (a, b)
    final_a, final_b = jax.lax.while_loop (condition, body, init_state)

    return final_a

@jax.jit
def between(a, b, x):
    return jax.lax.cond(
        a <= b,
        lambda: (a <= x) & (x <= b),
        lambda: (b <= x) & (x <= a)
    )

@jax.jit
def element_dekker(a, b):
    return jax.lax.cond(
        abs(f(a)) < abs(f(b)),
        lambda: dekker_helper(b, a, b),
        lambda: dekker_helper(a, b, a)
    )

@jax.jit
def dekker_helper(a, b, c):
    def condition(state):
        a, b, c = state
        return abs(b - a) >= eps

    def body(state):
        a, b, c = state
        # precondition
        a, b = jax.lax.cond(
            abs(f(a)) < abs(f(b)),
            lambda: (b, a),
            lambda: (a, b)
        )
        
        # compute midpoint and secant
        m = (a+b)/2.0
        s = jax.lax.cond(
            f(b) - f(c) != 0,
            lambda: b - f(b)/((f(b)-f(c))/(b-c)),
            lambda: m
        )

        a, b = jax.lax.cond(
            between(b, m, s),
            # secant
            lambda: jax.lax.cond(
                f(a) * f(s) > 0,
                lambda: (b, s),
                lambda: (a, s)
            ),
            # bisect
            lambda: jax.lax.cond(
                f(a) * f(m) > 0,
                lambda: (b, m),
                lambda: (a, m)
            )
        )

        c = b

        return a, b, c

    init_state = (a, b, c)
    final_a, final_b, final_c = jax.lax.while_loop(condition, body, init_state)    

    return final_b



# Inverse quadratic interpolation
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


def element_brent(a, b):
    fa = f(a)
    fb = f(b)

    bflag = False
    a, b, c = (b, a, b) if abs(fa) < abs(fb) else (a, b, a)


    while abs(fb) >= eps:
        d = c
        a, b = (b, a) if abs(fa) < abs(fb) else (a, b)
        print(a, b, c, d)

        # TODO: maybe check signs of a and b?
        

        # Compute midpoint and interpolation
        m = (a+b)/2.0
        if f(a) != f(c) and f(b) != f(c):
            # IQI
            s = iqi(a, b, c)
        else:
            # secant
            s = b - f(b)/((f(b)-f(c))/(b-c))
        

        # Take a step
        k = (3*a + b) / 4
        can_interpolate = abs(s - b) < (1/2)*abs(b - c) if bflag else c == d or abs(s - b) < (1/2)*abs(c - d)

        if can_interpolate and between(k, b, s):
            # use interpolation
            a, b, c = (b, s, b) if f(a) * f(s) > 0 else (a, s, b)
            bflag = False
        else:
            # use bisection
            a, b, c = (b, m, b) if f(a) * f(m) > 0 else (a, m, b)
            bflag = True

        fb = f(b)


    return b



def measure_time(device, fn, inputs):
    # Set the JAX device
    jax.devices(device)
    
    # (Not sure if this warm-up phase actually does anything).

    # Helper function to keep functions intact 
    # and otherwise select the first array element.
    def process_item(item):
        if callable(item):
            return item
        else:
            return jnp.array([item[0]])
    
    # # Warm-up
    _ = fn(*tuple(process_item(item) for item in inputs))

    start_time = time.time()

    _ = fn(*inputs)

    return time.time() - start_time


# Alternative measure time, using a for loop
# The results here confirm our intuition that 
# GPU is only faster on large vector inputs,
# where we see a benefit from parallelization.
# On sequential operations, GPU is the same as CPU. 
def measure_time_sequential(device, fn, inputs):
    # Set the JAX device
    jax.devices(device)
    
    a, b = inputs
    a = [jnp.array([i]) for i in a]
    b = [jnp.array([i]) for i in b]

    start_time = time.time()

    for a, b in zip(a, b):
        _ = fn(a, b)

    return time.time() - start_time



def compare_cpu_gpu(fn):
    num_samples = 1000
    a = jnp.linspace(-4, 0.5, num=num_samples)
    b = jnp.linspace(2, 5, num=num_samples)

    vectorized_fn = jax.vmap(fn, in_axes=(0, 0))
    cpu_time = measure_time("cpu", vectorized_fn, (a, b))
    print(f"Time on CPU: {cpu_time:.6f} seconds")

    # Only measure GPU time if a GPU is available
    if jax.lib.xla_bridge.get_backend().platform == "gpu":
        gpu_time = measure_time("gpu", vectorized_fn, (a, b))
        print(f"Time on GPU: {gpu_time:.6f} seconds")
    else:
        print("No GPU backend detected.")


if __name__ == '__main__':
    a = 1.0
    b = 2.0
    # a_arr = jnp.array([1])
    # b_arr = jnp.array([2])

    print("Brent's Method")
    root = element_brent(a, b)
    print(root)

    print("Dekker's Method")
    root = element_dekker(a, b)
    print(root)

    print("Bisection Method")
    root = element_bisect(a, b)
    print(root)

    print("Secant Method")
    root = element_secant(a, b)
    print(root)

    # print("Dekker's Method:")
    # compare_cpu_gpu(element_dekker)

    # print("Bisection Method:")
    # compare_cpu_gpu(element_bisect)

    # print("Secant Method:")
    # compare_cpu_gpu(element_secant)