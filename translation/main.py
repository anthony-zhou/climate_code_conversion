import typer
import textwrap
from git.repo import Repo
import os
import random

import translation.ast.dag
import translation.llm as llm
import translation.testing as testing

app = typer.Typer()


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


def translate_internal(func: str, source: str, outfile):
    print(f"Translating {func}")
    # TODO: start the translation loop for this function.
    func_body = textwrap.dedent(
        f"""\
    def {func}():
        pass
    """
    )

    # Write function to the appropriate location (top of the open file)
    with open(outfile, "a") as f:
        f.write("\n")
        f.write(func_body)
        f.write("\n")

    repo = Repo(os.getcwd())
    repo.index.add([outfile])
    repo.index.commit(f"[AI] Created {func} in {outfile}")


def generate_unit_tests(translation: str, outfile: str):
    unit_tests = textwrap.dedent(
        f"""\
    def test_{random.randint(0, 1000)}():
        assert 1 == 1
    """
    )


@app.command()
def main(filename: str = "./translation/ast/tests/SampleMod.f90"):
    dag = translation.ast.dag.DAG(filename)

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

    if len(externals) > 0:
        print("Translating external dependencies")
        for func, item in externals:
            # TODO: implement this
            continue

    if len(internals) > 0:
        print("Translating internal dependencies")
        # For each internal dependency, translate with a unit test.
        for func, item in internals:
            # Translate the function using LLM, writing to file with each iteration
            python_function = llm._translate_function_to_python(item["source"])
            python_unit_tests = llm.generate_unit_tests(python_function)
            pytest_output = testing.run_tests(
                python_function, python_unit_tests, "python:3.8"
            )
            print(pytest_output)

            # Ignore iterations for now, just write to a file
            outfile = "out.py"
            with open(outfile, "a") as f:
                f.write("\n")
                f.write(python_function)
                f.write("\n")

            repo = Repo(os.getcwd())
            repo.index.add([os.getcwd() + "/" + outfile])
            repo.index.commit(f"[AI] Created {func} in {outfile}")

            testfile = "test_out.py"
            with open(testfile, "a") as f:
                f.write("\n")
                f.write(python_unit_tests)
                f.write("\n")

            repo = Repo(os.getcwd())
            repo.index.add([os.getcwd() + "/" + testfile])
            repo.index.commit(f"[AI] Created test_{func} in {testfile}")
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
