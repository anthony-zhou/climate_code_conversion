import pytest

import translation.modules.ast.parser as parser
import os
import pathlib

# Compute filename based on path to current file.
filename = __file__[: __file__.rindex("/")] + "/SampleMod.f90"


def test_find_public_functions():
    public_functions = parser.get_public_functions(filename=filename)
    assert len(public_functions) == 1
    assert public_functions == ["ci_func"]


def test_filter_for_function():
    dag = parser.get_dag(filename)

    expected = [
        ("add", {"source": "source will go here", "calls": []}),
        ("quadratic", {"source": "source will go here", "calls": ["sum"]}),
        ("ci_func", {"source": "source here", "calls": ["quadratic"]}),
    ]

    actual = parser.filter_for_dependencies(dag, "ci_func")

    # Assert that the names match
    assert [x[0] for x in expected] == [x[0] for x in actual]


def test_fib_public_functions():
    fib_filename = __file__[: __file__.rindex("/")] + "/fibmod.f90"
    public_functions = parser.get_public_functions(filename=fib_filename)
    assert len(public_functions) == 1
    assert public_functions == ["fibonacci"]


def test_fib_filter_for_functino():
    dag = parser.get_dag(__file__[: __file__.rindex("/")] + "/fibmod.f90")

    expected = [
        ("sum_two_numbers", {"source": "source will go here", "calls": []}),
        ("fibonacci", {"source": "source here", "calls": ["quadratic"]}),
    ]

    actual = parser.filter_for_dependencies(dag, "fibonacci")

    # Assert that the names match
    assert [x[0] for x in expected] == [x[0] for x in actual]
