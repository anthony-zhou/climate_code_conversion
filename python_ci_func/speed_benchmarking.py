# Run 100 iterations of ci_func

import timeit
from jax_photosynthesis import ci_func


import python_ci_func.numpy_photosynthesis as np_ps

import python_ci_func.numba_simpler_photosynthesis as nb_ps

import python_ci_func.jax_jit_photosynthesis as jax_jit

ci = 40
lmr_z = 4
par_z = 500
gb_mol = 50_000
je = 40
cair = 45
oair = 21000
rh_can = 0.40
p = 1
iv = 1
c = 1

def run_ci_func(n):
    for i in range(0, n):
        ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)


def run_np_ci_func(n):
    for i in range(0, n):
        np_ps.ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)


def run_nb_ci_func(n):
    for i in range(0, n):
        nb_ps.ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)


def run_jax_jit(n):
    for i in range(0, n):
        jax_jit.ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)


if __name__ == '__main__':

    with open("speed_benchmarking_jax_jit.txt", "a") as f:
        for n in range(0, 1000, 100):
            # Write outputs to file
            f.write(f"{n}, ")
            f.write(f"{timeit.timeit(lambda: run_np_ci_func(n), number=1)}\n")