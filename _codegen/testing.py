import docker
import tempfile
import os
import re
from utils import logger, remove_ansi_escape_codes
import textwrap

from enum import Enum


class TestResult(Enum):
    PASSED = 1
    FAILED = 2


def _extract_pytest_output(output):
    """
    Get everything after ============================= test session starts ============================== (including that line)
    """

    result = re.split(r"={2,}\stest session starts\s={2,}", output)[1]

    return f"============================= test session starts ==============================\n{result}"


def _run_tests_in_docker(source_code, docker_image):
    """
    Run unit tests on the given source code and return the output.
    Uses Docker in case we need to install dependencies.

    """
    # Initialize Docker client
    client = docker.from_env()

    # Pull the Python Docker image
    logger.debug(f"Pulling Docker image {docker_image}...")
    client.images.pull(docker_image)

    # Create a temporary directory to store the test code
    test_dir = os.getcwd() + "/tests"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Create a temporary file and write the test code to it
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, dir=test_dir) as temp:
        temp.write(source_code.encode("utf-8"))
        temp_filename = temp.name

    # Define the commands to run
    # commands = f"""
    # pip install pytest numpy jaxlib
    # pip install --upgrade "jax[cpu]"
    # pytest {os.path.basename(temp_filename)}
    # """
    commands = f"""
    pip install pytest numpy
    pytest {os.path.basename(temp_filename)} 
    """

    # Create and run the container, capturing the output
    container = client.containers.create(
        image=docker_image,
        command='/bin/bash -c "{}"'.format(commands),
        tty=True,
        volumes={test_dir: {"bind": f"/tests", "mode": "rw"}},
        working_dir="/tests",
    )
    logger.debug(f"Running tests in Docker container {container.id}...")
    container.start()  # type: ignore

    # Wait for the container to finish and capture the output
    result = container.wait()  # type: ignore
    output = container.logs()  # type: ignore

    # Clean up
    container.remove()  # type: ignore
    os.remove(temp_filename)

    return output.decode("utf-8")


def run_tests(source_code, unit_tests, docker_image="python:3.8"):
    logger.info(f"Running tests using docker image {docker_image}")
    source_code = source_code + "\n" + unit_tests
    output = _run_tests_in_docker(source_code, docker_image=docker_image)

    result = _extract_pytest_output(output)

    logger.trace(result)

    if "FAILED" in result:
        return TestResult.FAILED, remove_ansi_escape_codes(result)
    else:
        return TestResult.PASSED, remove_ansi_escape_codes(result)


if __name__ == "__main__":
    source_code = textwrap.dedent(
        """\
    import numpy as np

    def average(x):
        return np.mean(x)
    """
    )
    unit_tests = textwrap.dedent(
        """\
    import pytest

    def test_average():
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        assert average(x) == np.mean(x)
    
    def test_average2():
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        assert average(x) == 0
    """
    )

    status, result = run_tests(source_code, unit_tests, "python:3.8")
    logger.debug(status)
    logger.debug(result)


def check_if_docker_is_running():
    """
    Check if Docker is running.
    """
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception as e:
        print(e)
        return False
