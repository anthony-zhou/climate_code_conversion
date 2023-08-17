# Graph an vs. ci by varying cair

import numpy as np

from numba_an import main
from plotly import graph_objects as go

if __name__ == "__main__":
    ci = 35
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

    cair_values = np.linspace(10, 100, 19)
    ci_values = np.zeros_like(cair_values)
    an_values = np.zeros_like(cair_values)
    gs_mol_values = np.zeros_like(cair_values)

    for cair in cair_values:
        ci_val, gs_mol, an = main(
            ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c
        )
        ci_values[cair_values == cair] = ci_val
        an_values[cair_values == cair] = an
        gs_mol_values[cair_values == cair] = gs_mol
        print(f"ci: {ci_val}, an: {an}, gs_mol: {gs_mol}")

    fig = go.Figure(data=go.Scatter(x=ci_values, y=an_values, mode="markers+lines"))
    fig.update(
        layout_title_text="An vs. Ci",
        layout_xaxis_title_text="Ci",
        layout_yaxis_title_text="An",
    )
    fig.show()
