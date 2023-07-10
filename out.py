
def sum():
    pass


def quadratic():
    pass


def ci_func():
    pass


def sum(a, b):
    return a + b


def quadratic(a, b, c):
    discriminant = a + b
    r1 = 0.0
    r2 = 1.0
    return r1, r2


import numpy as np

def ci_func(ci):
    # intracellular leaf CO2
    # evaluate the function f(ci) = ci - (ca - (1.37rb+1.65rs))*patm*an
    r1, r2 = quadratic(1.0, 1.0, 1.0)
    fval = 10.0
    return fval

def quadratic(a, b, c):
    # solve quadratic equation ax^2 + bx + c = 0
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        x1 = (-b + np.sqrt(discriminant)) / (2*a)
        x2 = (-b - np.sqrt(discriminant)) / (2*a)
        return x1, x2
    elif discriminant == 0:
        x = -b / (2*a)
        return x, x
    else:
        return None

