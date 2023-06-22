import docker
import tempfile
import os
import re
from translation.utils import logger


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

    test_dir = os.getcwd() + "/tests"

    # Create a temporary file and write the test code to it
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, dir=test_dir) as temp:
        temp.write(source_code.encode("utf-8"))
        temp_filename = temp.name


    # Define the commands to run
    commands = f"""
    pip install pytest numpy jaxlib
    pip install --upgrade "jax[cpu]"
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
    container.start() # type: ignore

    # Wait for the container to finish and capture the output
    result = container.wait() # type: ignore
    output = container.logs() # type: ignore

    # Clean up
    container.remove() # type: ignore
    os.remove(temp_filename)

    return output.decode("utf-8")


def run_tests(source_code, unit_tests, docker_image):
    logger.info(f"Running tests using docker image {docker_image}")
    source_code = source_code + "\n" + unit_tests
    output = _run_tests_in_docker(source_code, docker_image=docker_image)

    result = _extract_pytest_output(output)

    return result


if __name__ == '__main__':
    source_code = """
import jax
import jax.numpy as jnp

def make_numbers(n=10):
    x = jnp.arange(n)
    return x
"""
    unit_tests = """
import pytest

def test_make_numbers():
    x = make_numbers(10)

    assert len(x) == 10
"""

    result = run_tests(source_code, unit_tests, "python:3.8")
    logger.debug(result)