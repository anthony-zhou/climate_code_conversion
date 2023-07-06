from original_photosynthesis import hybrid

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


def test_original_photosynthesis():
    x0, gs_mol, iter = hybrid(
        ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c
    )

    print(f"ci value = {x0}, gs_mol = {gs_mol}, iter = {iter}")

    assert 1 == 2
