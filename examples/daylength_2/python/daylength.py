import numpy as np


def daylength(lat, decl):
    """
    Calculate the length of the day (in hours) given the latitude and the
    declination of the sun.  This is the number of seconds between sunrise
    and sunset. Returns NaN if input arguments are invalid.

    Parameters
    ----------
    lat : float or numpy array
        Latitude of the location in radians.
    decl : float
        Declination of the sun in radians.

    Returns
    -------
    float
        Length of the day in seconds.

    """
    # Number of seconds per radian of hour-angle
    secs_per_radian = 13750.9871

    # Epsilon for defining latitudes "near" the pole
    lat_epsilon = 10.0 * np.finfo(float).eps

    # Define an offset pole as slightly less than pi/2 to avoid problems with cos(lat) being negative
    pole = np.pi / 2
    offset_pole = pole - lat_epsilon

    # Lat must be less than pi/2 within a small tolerance
    # Decl must be strictly less than pi/2
    lat = np.where(abs(lat) >= pole + lat_epsilon, np.NAN, lat)
    decl = np.where(abs(decl) >= pole, np.NAN, decl)

    # Ensure that latitude isn't too close to pole, to avoid problems with cos(lat) being negative
    my_lat = np.clip(lat, -offset_pole, offset_pole)
    temp = -np.tan(my_lat) * np.tan(decl)
    temp = np.clip(temp, -1, 1)
    return 2.0 * secs_per_radian * np.arccos(temp)


class Bounds:
    def __init__(self, begg, endg):
        self.begg = begg
        self.endg = endg


def compute_max_daylength(bounds, lat, obliquity):
    """Compute max daylength for each grid cell"""
    max_daylength = []
    for g in range(bounds.begg, bounds.endg):
        max_decl = obliquity
        if lat[g] < 0.0:
            max_decl = -max_decl
        max_daylength.append(daylength(lat[g], max_decl))
    return max_daylength
