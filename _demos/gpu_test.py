# Basic script to make sure GPU and CPU are both working. 

import jax
import jax.numpy as jnp
from jax import jit
import time

def simple_function(x):
    # Some arbitrary computation
    return jnp.sin(x) * jnp.cos(x) + jnp.tanh(x)

# JIT compile the function for faster execution
jit_function = jit(simple_function)

def measure_time(device):
    # Set the JAX device
    jax.devices(device)
    
    # Warm-up
    _ = jit_function(1.0)

    start_time = time.time()

    x = jnp.ones(10000)
    _ = jit_function(x)

    return time.time() - start_time

if __name__ == "__main__":
    cpu_time = measure_time("cpu")
    print(f"Time on CPU: {cpu_time:.6f} seconds")

    # Only measure GPU time if a GPU is available
    if jax.lib.xla_bridge.get_backend().platform == "gpu":
        gpu_time = measure_time("gpu")
        print(f"Time on GPU: {gpu_time:.6f} seconds")
    else:
        print("No GPU backend detected.")
