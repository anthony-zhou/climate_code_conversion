import translation.utils as utils
import csv
import os

def test_save_to_csv():
    data = [
        {'name': 'John', 'age': 25},
        {'name': 'Jane', 'age': 30},
        {'name': 'Bob', 'age': 35}
    ]
    outfile = 'test.csv'
    utils.save_to_csv(data, outfile)
    
    with open(outfile, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert rows[0]['name'] == 'John'
        assert rows[0]['age'] == '25'
        assert rows[1]['name'] == 'Jane'
        assert rows[1]['age'] == '30'
        assert rows[2]['name'] == 'Bob'
        assert rows[2]['age'] == '35'    

    os.remove(outfile)
    

def test_remove_ansi_escape_codes():
        # ANSI escape code for color red
        input_str = "\033[31mHello, World!\033[0m"
        expected_output = "Hello, World!"
        assert utils.remove_ansi_escape_codes(input_str) == expected_output


def test_extract_source_code():
    message = "Here is some text before the source code.\n\nSOURCE CODE:\n```python\nprint('Hello, world!')\n```\n\nAnd here is some text after the source code."
    expected = "print('Hello, world!')"
    assert utils.extract_source_code(message) == expected

def test_extract_source_code_with_both():
    message = """The unit test is failing because the function `ci_func` is returning `None` for `gs_mol` when it should be returning a float. This happens when the `an` variable is less than 0.0, causing the function to return early. To fix this, we can modify the function to return 0.0 instead of `None` for `gs_mol` when `an` is less than 0.0.

SOURCE CODE:
```python
print("Hello, world!")
```

UNIT TESTS:
```python
import pytest

def test_add():
    assert 1 + 2 == 3
```"""
    expected = """print("Hello, world!")"""
    assert utils.extract_source_code(message) == expected


def test_extract_unit_test_code():
    message = "Here is some text before the unit tests.\n\nUNIT TESTS:\n```python\nassert 1 + 1 == 2\n```\n\nAnd here is some text after the unit tests."
    expected = "assert 1 + 1 == 2"
    assert utils.extract_unit_test_code(message) == expected


def test_extract_unit_test_longer():
    message = """The unit test is failing because the function `ci_func` is returning `None` for `gs_mol` when it should be returning a float. This happens when the `an` variable is less than 0.0, causing the function to return early. To fix this, we can modify the function to return 0.0 instead of `None` for `gs_mol` when `an` is less than 0.0.

SOURCE CODE:
```python
import numpy as np

def ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c):
    # Constants
    forc_pbot = 121000.0
    c3flag = True
    medlynslope = 6.0
    medlynintercept = 100.0
    stomatalcond_mtd = 1
    vcmax_z = 62.5
    cp = 4.275
    kc = 40.49
    ko = 27840.0
    qe = 1.0
    tpu_z = 31.5
    kp_z = 1.0
    bbb = 100.0
    mbb = 9.0
    theta_cj = 0.98
    theta_ip = 0.95
    stomatalcond_mtd_medlyn2011 = 1
    stomatalcond_mtd_bb1987 = 2

    # Calculate photosynthesis rates
    if c3flag:
        ac = vcmax_z * max(ci-cp, 0.0) / (ci+kc*(1.0+oair/ko))
        aj = je * max(ci-cp, 0.0) / (4.0*ci+8.0*cp)
        ap = 3.0 * tpu_z
    else:
        ac = vcmax_z
        aj = qe * par_z * 4.6
        ap = kp_z * max(ci, 0.0) / forc_pbot

    # Gross photosynthesis
    aquad = theta_cj
    bquad = -(ac + aj)
    cquad = ac * aj
    r1, r2 = np.roots([aquad, bquad, cquad])
    ai = min(r1,r2)

    aquad = theta_ip
    bquad = -(ai + ap)
    cquad = ai * ap
    r1, r2 = np.roots([aquad, bquad, cquad])
    ag = max(0.0,min(r1,r2))

    # Net photosynthesis
    an = ag - lmr_z
    if an < 0.0:
        fval = 0.0
        return fval, 0.0

    # Quadratic gs_mol calculation
    cs = cair - 1.4/gb_mol * an * forc_pbot
    if stomatalcond_mtd == stomatalcond_mtd_medlyn2011:
        term = 1.6 * an / (cs / forc_pbot * 1.e06)
        aquad = 1.0
        bquad = -(2.0 * (medlynintercept*1.e-06 + term) + (medlynslope * term)**2 / (gb_mol*1.e-06 * rh_can))
        cquad = medlynintercept*medlynintercept*1.e-12 + (2.0*medlynintercept*1.e-06 + term * (1.0 - medlynslope* medlynslope / rh_can)) * term
        r1, r2 = np.roots([aquad, bquad, cquad])
        gs_mol = max(r1,r2) * 1.e06
    elif stomatalcond_mtd == stomatalcond_mtd_bb1987:
        aquad = cs
        bquad = cs*(gb_mol - bbb) - mbb*an*forc_pbot
        cquad = -gb_mol*(cs*bbb + mbb*an*forc_pbot*rh_can)
        r1, r2 = np.roots([aquad, bquad, cquad])
        gs_mol = max(r1,r2)

    # Derive new estimate for ci
    fval = ci - cair + an * forc_pbot * (1.4*gs_mol+1.6*gb_mol) / (gb_mol*gs_mol)
    return fval, gs_mol
```

UNIT TESTS:
```python
import pytest
import numpy as np

def test_ci_func():
    # Test with normal values
    ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)
    assert isinstance(fval, float)
    assert isinstance(gs_mol, float)

    # Test with ci = 0
    ci = 0.0
    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)
    assert isinstance(fval, float)
    assert isinstance(gs_mol, float)

    # Test with negative ci
    ci = -1.0
    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)
    assert isinstance(fval, float)
    assert isinstance(gs_mol, float)

    # Test with large ci
    ci = 1e6
    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)
    assert isinstance(fval, float)
    assert isinstance(gs_mol, float)

    # Test with ci = NaN
    ci = np.nan
    with pytest.raises(ValueError):
        ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)
```"""
    expected = """import pytest
import numpy as np

def test_ci_func():
    # Test with normal values
    ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)
    assert isinstance(fval, float)
    assert isinstance(gs_mol, float)

    # Test with ci = 0
    ci = 0.0
    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)
    assert isinstance(fval, float)
    assert isinstance(gs_mol, float)

    # Test with negative ci
    ci = -1.0
    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)
    assert isinstance(fval, float)
    assert isinstance(gs_mol, float)

    # Test with large ci
    ci = 1e6
    fval, gs_mol = ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)
    assert isinstance(fval, float)
    assert isinstance(gs_mol, float)

    # Test with ci = NaN
    ci = np.nan
    with pytest.raises(ValueError):
        ci_func(ci, lmr_z, par_z, gb_mol, je, cair, oair, rh_can, p, iv, c)"""

    assert utils.extract_unit_test_code(message) == expected



def test_extract_unit_test_code():
    message = "Here is some text before the unit tests. "
    expected = None
    assert utils.extract_unit_test_code(message) == expected
