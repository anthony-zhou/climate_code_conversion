import jax
from jax import numpy as jnp
import time

from vnumba import main
from functools import partial 
import numpy as np

grid_samples = [1000, 10000, 100000, 1000000, 10000000, 100000000]


print("Runtimes for numba")

f = partial(main, lmr_z=4, par_z=500, gb_mol=50_000, je=40, cair=45, oair=21000, rh_can=0.40, p=1, iv=1, c=1)
vectorized_fn = np.vectorize(f)

vectorized_fn(np.array([35.0])) # Compile the function by running it

with open('numba_runtime.txt', 'w') as f:

    for grid_sample in grid_samples:
        ci_vals = np.linspace(35, 70, num=grid_sample)

        start_time = time.time()

        _, _ = vectorized_fn(ci_vals)

        runtime = time.time() - start_time

        f.write(f"{grid_sample}, {runtime:.6f}\n")

        print(f"Grid size: {grid_sample}")
        print(f"Time on CPU: {runtime:.6f} seconds")
