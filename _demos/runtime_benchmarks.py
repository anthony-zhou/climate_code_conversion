import jax
from jax import numpy as jnp
import time


grid_samples = [1000, 10000, 100000, 1000000, 10000000]

# grid_samples = [1]
# jax.config.update('jax_platform_name', 'cpu')

print("Runtimes for scipy")

# Now let's do it for Scipy, Numpy (on CPU)
from runtime_scripts.python.comparisons.vscipy import main
from functools import partial 
import numpy as np

f = partial(main, lmr_z=4, par_z=500, gb_mol=50_000, je=40, cair=45, oair=21000, rh_can=0.40, p=1, iv=1, c=1)
vectorized_fn = np.vectorize(f)


vectorized_fn(np.array([35.0])) # Compile the function by running it

with open('scipy_runtime.txt', 'w') as f:

    for grid_sample in grid_samples:
        ci_vals = np.linspace(35, 70, num=grid_sample)

        start_time = time.time()

        _, _ = vectorized_fn(ci_vals)

        runtime = time.time() - start_time

        f.write(f"{grid_sample}, {runtime:.6f}\n")

        print(f"Grid size: {grid_sample}")
        print(f"Time on CPU: {runtime:.6f} seconds")








# print("Runtimes for JAX")

# def measure_time(fn, *inputs):
    
#     # Warm-up
#     _ = fn(*inputs)

#     start_time = time.time()

#     _ = fn(*inputs)

#     return time.time() - start_time


# from jax_version import element_brent_with_sign_change
# import numpy as np


# vectorized_fn = jax.vmap(element_brent_with_sign_change)


# with open('jax_cpu_runtime.txt', 'w') as f:
    

#     for grid_sample in grid_samples:
#         ci_vals = jnp.linspace(35.0, 70.0, num=grid_sample)

#         runtime = measure_time(vectorized_fn, ci_vals)

#         f.write(f"{grid_sample}, {runtime:.6f}\n")

#         print(f"Grid size: {grid_sample}")
#         print(f"Time on CPU: {runtime:.6f} seconds")



