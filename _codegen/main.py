from testing import run_tests, TestResult, check_if_docker_is_running
from utils import logger
from debug import fix_problem
from generate import generate_unit_tests, generate_python


def translate(fortran_code: str):
    if not check_if_docker_is_running():
        logger.error("Docker is not running. Please start Docker and try again.")
        return ""

    python_code = generate_python(fortran_code)
    unit_tests = generate_unit_tests(python_code)
    status, result = run_tests(python_code, unit_tests)

    while status != TestResult.PASSED:
        python_code, unit_tests = fix_problem(python_code, unit_tests, result)
        status, result = run_tests(python_code, unit_tests)

        if status != TestResult.PASSED:
            response = input("Would you like to continue (Y/n)? ")
            if response.lower() == "n":
                break
        else:
            logger.info("All tests passed!")

    return python_code, unit_tests


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Translate Fortran code to Python code."
    )
    parser.add_argument(
        "--infile", type=str, default="source/input.f90", help="input Fortran file path"
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="source/output.py",
        help="output Python file path",
    )
    parser.add_argument(
        "--testfile",
        type=str,
        default="source/test_output.py",
        help="output unit test file path",
    )
    args = parser.parse_args()

    infile = args.infile
    outfile = args.outfile
    testfile = args.testfile

    with open(infile, "r") as f:
        logger.info(f"Reading from {infile}")
        fortran_function = f.read()
        python_code, unit_tests = translate(fortran_function)

    with open(outfile, "w") as f:
        f.write(python_code)
        logger.info(f"Python translation written to {outfile}")

    with open(testfile, "w") as f:
        f.write(unit_tests)
        logger.info(f"Unit tests written to {testfile}")
