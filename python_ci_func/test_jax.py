import pytest
import numpy as np
from comparisons.vjax import main
from functools import partial
from jax import grad
import plotly.graph_objects as go


def test_ci_func():
    ci = 35.0
    lmr_z = 4.0
    par_z = 500.0
    gb_mol = 50_000.0
    je = 40.0
    cair = 45.0
    oair = 21000.0
    rh_can = 0.40
    p = 1
    iv = 1
    c = 1

    # Define a partial version of main with a fixed set of arguments
    main_partial = partial(
        main,
        ci=ci,
        lmr_z=lmr_z,
        par_z=par_z,
        gb_mol=gb_mol,
        je=je,
        # cair=cair,
        oair=oair,
        rh_can=rh_can,
        p=p,
        iv=iv,
        c=c,
    )

    def gs_mol(x):
        return main_partial(cair=x)[1]

    x = np.linspace(20, 100, 50)
    y = [gs_mol(x) for x in x]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name="gs_mol (umol/m2/s)"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[grad(gs_mol)(x) for x in x],
            name="derivative of gs_mol (umol/m2/s)/Pa",
        )
    )
    fig.update_xaxes(title_text="Atmospheric partial pressure of CO2 (Pa)")
    fig.update_layout(
        title="Change in stomatal conductance with respect to atmospheric partial pressure"
    )
    fig.show()


if __name__ == "__main__":
    test_ci_func()
