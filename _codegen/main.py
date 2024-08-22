from testing import run_tests, TestResult, check_if_docker_is_running
from utils import logger
from debug import fix_problem
from generate import generate_unit_tests, generate_python
from pathlib import Path

def write_to_file(i: int, filename: str, content: str, outdir: str):
    # Open new file for writing
    with open(f"{outdir}/v{i}_{filename}", "w") as f:
        f.write(content)

def translate(name: str, fortran_code: str, outdir: str):
    if not check_if_docker_is_running():
        logger.error("Docker is not running. Please start Docker and try again.")
        return ""

    i = 0

    python_code = generate_python(fortran_code)
    unit_tests = generate_unit_tests(python_code)
    status, result = run_tests(python_code, unit_tests)

    # Write to file
    write_to_file(i, f"{name}_python.py", python_code, outdir)
    write_to_file(i, f"{name}_tests.py", unit_tests, outdir)
    write_to_file(i, f"{name}_output.txt", result, outdir)


    while status != TestResult.PASSED:
        python_code, unit_tests = fix_problem(python_code, unit_tests, result)
        status, result = run_tests(python_code, unit_tests)

        i += 1
        write_to_file(i, f"{name}_python.py", python_code, outdir)
        write_to_file(i, f"{name}_tests.py", unit_tests, outdir)
        write_to_file(i, f"{name}_output.txt", result, outdir)

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
        "--outdir",
        type=str,
        default="source",
        help="output directory, will get created if it doesn't exist",
    )
    args = parser.parse_args()

    infile = args.infile
    outdir = args.outdir

    # Make output directory if it doesn't exist
    Path(outdir).mkdir(parents=True, exist_ok=True)

    with open(infile, "r") as f:
        logger.info(f"Reading from {infile}")
        fortran_function = f.read()
        python_code, unit_tests = translate(Path(infile).stem, fortran_function, outdir)

    outfile = outdir + '/output.py'
    with open(outfile, "w") as f:
        f.write(python_code)
        logger.info(f"Python translation written to {outfile}")

    testfile = outdir + '/test_output.py'
    with open(testfile, "w") as f:
        f.write(unit_tests)
        logger.info(f"Unit tests written to {testfile}")
