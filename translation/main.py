import typer
import textwrap
from git.repo import Repo
import os
import random

from translation.modules.ast.dag import DAG
from translation.modules.translate import translate, generate_unit_tests, iterate
from translation.modules.testing import run_tests, TestResult

app = typer.Typer()

from translation.utils import logger, options_menu, write_to_file
from yaspin import yaspin

import sys

logger.remove()
logger.add(sys.stderr, level="INFO")


@app.command()
def main(
    input_file: str = "./examples/fibonacci/fortran/fibonacci.f90",
    output_file: str = "./examples/fibonacci/python/fibonacci.py",
    output_test_file: str = "./examples/fibonacci/python/test_fibonacci.py",
):
    dag = DAG(input_file)

    function_name = options_menu(dag.public_functions)

    print(f"Translating the function {function_name}")

    # Classify dependencies as external or internal
    externals, internals = dag.classify_dependencies(function_name)

    # For each external dependency, do the following:
    # Look across the codebase to see if module is defined (do this once I have internet)
    # Otherwise: "We couldn't find quadraticmod.f90. Options are:"
    # - generate function from context
    # - leave a TODO comment and define the function later
    # - supply your own function
    print(externals)
    print(internals)

    if len(externals) > 0:
        print("Translating external dependencies")
        for func, item in externals:
            # TODO: implement this
            continue

    if len(internals) > 0:
        print("Translating internal dependencies")
        for func_name, dependency in internals:
            with yaspin(text=f"Translating {func_name}...") as spinner:
                python_function = translate(dependency["source"])
                write_to_file(python_function, output_file)
                while True:
                    python_unit_tests = generate_unit_tests(python_function)
                    write_to_file(python_unit_tests, output_test_file)

                    test_result, pytest_output = run_tests(
                        python_function, python_unit_tests, docker_image="python:3.8"
                    )

                    print(pytest_output)

                    if test_result == TestResult.PASSED:
                        break
                    else:
                        response = typer.prompt("Would you like to continue? [y/N]: ")
                        if response.lower() == "y":
                            break

            # repo = Repo(os.getcwd())
            # repo.index.add([os.getcwd() + "/" + outfile])
            # repo.index.commit(f"[AI] Created {func} in {outfile}")

            # Let human make edits to make the unit tests pass
            continue


if __name__ == "__main__":
    app()
