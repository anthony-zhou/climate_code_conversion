import numpy as np

def daylength(lat, decl):
    SHR_CONST_PI = np.pi
    secs_per_radian = 13750.9871
    lat_epsilon = 10. * np.finfo(float).eps
    pole = SHR_CONST_PI / 2.0
    offset_pole = pole - lat_epsilon
    
    # Check if inputs are array-like and convert to numpy arrays if necessary
    lat = np.asarray(lat)
    decl = np.asarray(decl)
    
    # Broadcast lat and decl to the same shape
    lat, decl = np.broadcast_arrays(lat, decl)

    # Create an output array filled with NaN
    result = np.full_like(lat, np.nan)

    # Apply the calculation where the conditions are met
    condition = (np.abs(lat) < (pole + lat_epsilon)) & (np.abs(decl) < pole)
    my_lat = np.minimum(offset_pole, np.maximum(-offset_pole, lat[condition]))
    temp = - (np.sin(my_lat) * np.sin(decl[condition])) / (np.cos(my_lat) * np.cos(decl[condition]))
    temp = np.minimum(1., np.maximum(-1., temp))
    result[condition] = 2.0 * secs_per_radian * np.arccos(temp)

    # If the input was a scalar, return a scalar; otherwise, return a numpy array
    if np.isscalar(lat) and np.isscalar(decl):
        return np.asscalar(result)
    else:
        return result
