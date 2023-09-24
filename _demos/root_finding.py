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

eps = 10**(-7)


Result = namedtuple('Result', ['root', 'steps'])

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


# def root_dekker(a, b):




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





if __name__ == '__main__':

    num_samples = 100
    a = jnp.linspace(-4, 0.5, num=num_samples)
    b = jnp.linspace(2, 5, num=num_samples)


    vectorized_bisect = jax.vmap(element_bisect, in_axes=(0, 0))


    cpu_time = measure_time("cpu", vectorized_bisect, (a, b))
    print(f"Time on CPU: {cpu_time:.6f} seconds")

    # Only measure GPU time if a GPU is available
    if jax.lib.xla_bridge.get_backend().platform == "gpu":
        gpu_time = measure_time("gpu", vectorized_bisect, (a, b))
        print(f"Time on GPU: {gpu_time:.6f} seconds")
    else:
        print("No GPU backend detected.")



    vectorized_secant = jax.vmap(element_secant, in_axes=(0, 0))


    cpu_time = measure_time("cpu", vectorized_secant, (a, b))
    print(f"Time on CPU: {cpu_time:.6f} seconds")

    # Only measure GPU time if a GPU is available
    if jax.lib.xla_bridge.get_backend().platform == "gpu":
        gpu_time = measure_time("gpu", vectorized_secant, (a, b))
        print(f"Time on GPU: {gpu_time:.6f} seconds")
    else:
        print("No GPU backend detected.")

    # Measure runtime sequentially.
    # This becomes prohibitively slow for n>100 (even on n=100 it takes 8 seconds).

    # cpu_time = measure_time_sequential("cpu", vectorized_bisect, (a, b))
    # print(f"Time on CPU: {cpu_time:.6f} seconds")

    # # Only measure GPU time if a GPU is available
    # if jax.lib.xla_bridge.get_backend().platform == "gpu":
    #     gpu_time = measure_time_sequential("gpu", vectorized_bisect, (a, b))
    #     print(f"Time on GPU: {gpu_time:.6f} seconds")
    # else:
    #     print("No GPU backend detected.")
