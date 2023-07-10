import typer
import textwrap
from git.repo import Repo
import os
import random

import translation.ast.dag
from translation.llm import translate, generate_unit_tests, iterate
from translation.testing import run_tests, TestResult

app = typer.Typer()

from translation.utils import logger
from yaspin import yaspin

import sys

logger.remove()
logger.add(sys.stderr, level="INFO")


def options_menu(options: list[str]):
    response = ""

    print("Please select one of the following options:")
    for i, option in enumerate(options):
        print(f"{i+1}. {option}")

    while True:
        response = input(f"Select an option [1-{len(options)}]: ")
        try:
            choice = int(response)
            if choice >= 1 and choice <= len(options):
                return options[choice - 1]
        except:
            pass


def write_to_file(source_code: str, outfile: str):
    with open(outfile, "a") as f:
        f.write("\n")
        f.write(source_code)
        f.write("\n")


@app.command()
def main(
    input_file: str = "./examples/fibonacci/fortran/fibonacci.f90",
    output_file: str = "./examples/fibonacci/python/fibonacci.py",
    output_test_file: str = "./examples/fibonacci/python/test_fibonacci.py",
):
    dag = translation.ast.dag.DAG(input_file)

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

        # python_translations = [
        #     translate_internal(func, item["source"], "./out.py")
        #     for func, item in internals
        # ]
        # for translation in python_translations:
        #     generate_unit_tests(translation, "./test_out.py")

    # Generate unit test, write to test/test_photosynthesis.py
    # iterate code until it passes or converges on unit tests, commiting to git each time
    # Write to file
    # Let human make updates before continuing

    # Would you like to translate another function?


if __name__ == "__main__":
    app()
