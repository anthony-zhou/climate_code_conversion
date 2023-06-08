import docker
import tempfile
import os
import re

def _extract_pytest_output(output):
    """
    Get everything after ============================= test session starts ============================== (including that line)
    """

    result = re.split(r"={2,}\stest session starts\s={2,}", output)[1]

    return f"============================= test session starts ==============================\n{result}"


def _run_tests_in_docker(source_code):
    """
    Run unit tests on the given source code and return the output.
    Uses Docker in case we need to install dependencies.

    """
    # Initialize Docker client
    client = docker.from_env()

    # Pull the Python Docker image
    client.images.pull("python:3.8")

    test_dir = os.getcwd() + "/tests"

    # Create a temporary file and write the test code to it
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, dir=test_dir) as temp:
        temp.write(source_code.encode("utf-8"))
        temp_filename = temp.name

    # Define the commands to run
    commands = f"""
    pip install pytest
    pytest {os.path.basename(temp_filename)} 
    """

    # Create and run the container, capturing the output
    container = client.containers.create(
        "python:3.8",
        command='/bin/bash -c "{}"'.format(commands),
        tty=True,
        volumes={test_dir: {"bind": f"/tests", "mode": "rw"}},
        working_dir="/tests",
    )
    container.start()

    # Wait for the container to finish and capture the output
    result = container.wait()
    output = container.logs()

    # Clean up
    container.remove()
    os.remove(temp_filename)

    return output.decode("utf-8")


def run_tests(source_code):
    output = _run_tests_in_docker(source_code)

    result = _extract_pytest_output(output)

    return result
