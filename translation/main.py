import translation.ast.dag
import typer
import textwrap

app = typer.Typer()


def get_input_args():
    # TODO: read input from stdin or CLI args
    return "./ast/tests/SampleMod.f90"


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


@app.command()
def main(filename: str = "./ast/tests/SampleMod.f90"):
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
            translate_internal(func, item["source"], "./out.py")
            continue

    # Generate unit test, write to test/test_photosynthesis.py
    # iterate code until it passes or converges on unit tests, commiting to git each time
    # Write to file
    # Let human make updates before continuing

    # Would you like to translate another function?


if __name__ == "__main__":
    app()
