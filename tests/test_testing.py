import textwrap
from translation.testing import run_tests, TestResult


def test_run_tests_passed():
    source_code = textwrap.dedent(
        """\
        def add(a, b):
            return a + b
    """
    )
    unit_tests = textwrap.dedent(
        """\
        def test_add():
            assert add(2, 3) == 5
            assert add(0, 0) == 0
    """
    )
    docker_image = "python:3.8"
    result, output = run_tests(source_code, unit_tests, docker_image)
    assert result == TestResult.PASSED


def test_run_tests_failed():
    source_code = textwrap.dedent(
        """\
        def subtract(a, b):
            return a - b
    """
    )
    unit_tests = textwrap.dedent(
        """\
        def test_subtract():
            assert subtract(2, 3) == -1
            assert subtract(0, 0) == 5
    """
    )
    docker_image = "python:3.8"
    result, output = run_tests(source_code, unit_tests, docker_image)
    assert result == TestResult.FAILED
