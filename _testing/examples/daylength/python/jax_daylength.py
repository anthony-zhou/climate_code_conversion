import jax.numpy as jnp
from jax import device_put

def daylength(lat, decl):
    SHR_CONST_PI = jnp.pi
    secs_per_radian = 13750.9871
    lat_epsilon = 10. * jnp.finfo(float).eps
    pole = SHR_CONST_PI / 2.0
    offset_pole = pole - lat_epsilon
    
    # Check if inputs are array-like and convert to numpy arrays if necessary
    lat = device_put(lat)  # Moves the array to the device (GPU/TPU)
    decl = device_put(decl)  # Moves the array to the device (GPU/TPU)
    
    # Broadcast lat and decl to the same shape
    lat, decl = jnp.broadcast_arrays(lat, decl)

    # Create an output array filled with NaN
    result = jnp.full_like(lat, jnp.nan)

    # Apply the calculation where the conditions are met
    condition = (jnp.abs(lat) < (pole + lat_epsilon)) & (jnp.abs(decl) < pole)
    my_lat = jnp.minimum(offset_pole, jnp.maximum(-offset_pole, lat))
    temp = - (jnp.sin(my_lat) * jnp.sin(decl)) / (jnp.cos(my_lat) * jnp.cos(decl))
    temp = jnp.minimum(1., jnp.maximum(-1., temp))
    result = jnp.where(condition, 2.0 * secs_per_radian * jnp.arccos(temp), result)

    return result
