# Generated by Claude

import pytest
from daylength import daylength, Bounds, compute_max_daylength


def test_positive_lat():
    bounds = Bounds(0, 3)
    lat = [0.1, 0.2, 0.3]
    obliquity = 0.4
    expected = [daylength(0.1, 0.4), daylength(0.2, 0.4), daylength(0.3, 0.4)]

    max_daylength = compute_max_daylength(bounds, lat, obliquity)
    assert expected == max_daylength


def test_negative_lat():
    bounds = Bounds(0, 3)
    lat = [-0.1, -0.2, -0.3]
    obliquity = 0.4
    expected = [daylength(0.1, 0.4), daylength(0.2, 0.4), daylength(0.3, 0.4)]

    max_daylength = compute_max_daylength(bounds, lat, obliquity)
    assert expected == max_daylength


def test_mix_lat():
    bounds = Bounds(0, 3)
    lat = [0.1, -0.2, 0.3]
    obliquity = 0.4
    expected = [daylength(0.1, 0.4), daylength(0.2, 0.4), daylength(0.3, 0.4)]

    max_daylength = compute_max_daylength(bounds, lat, obliquity)
    assert expected == max_daylength