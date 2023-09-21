import jax


def is_jax_using_gpu():
    return any(device.device_kind == "gpu" for device in jax.devices())

print(is_jax_using_gpu())
print(jax.devices())
