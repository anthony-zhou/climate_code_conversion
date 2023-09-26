import jax
from jax import numpy as jnp
import time

from jax_version import element_brent_with_sign_change


# Note that this version is exactly the same as the JAX GPU benchmark
# The difference is just that you use a different library for JAX (jax[cpu])

grid_samples = [1000, 10000, 100000, 1000000, 10000000, 100000000]

print("Runtimes for JAX")

def measure_time(fn, *inputs):
    
    # Warm-up
    _ = fn(*inputs)

    start_time = time.time()

    _ = fn(*inputs)

    return time.time() - start_time


vectorized_fn = jax.vmap(element_brent_with_sign_change)


with open('runtime.txt', 'w') as f:
    

    for grid_sample in grid_samples:
        ci_vals = jnp.linspace(35.0, 70.0, num=grid_sample)

        runtime = measure_time(vectorized_fn, ci_vals)

        f.write(f"{grid_sample}, {runtime:.6f}\n")

        print(f"Grid size: {grid_sample}")
        print(f"Time on GPU: {runtime:.6f} seconds")



