import testing as testing
import utils as utils
from utils import logger, completion_with_backoff
import messages
from config import model_name
import random
import string

import debug


def generate_unit_tests(python_function: str):
    logger.debug("Generating unit tests based on python code...")

    completion = completion_with_backoff(
        model=model_name,
        messages=messages.generate_python_test_messages(python_function),
        temperature=0.0,
    )

    logger.trace(f'COMPLETION: {completion.choices[0].message["content"]}')  # type: ignore

    unit_tests = utils.extract_code_block(completion)

    return unit_tests


def generate_python(fortran_code: str):
    logger.debug("Translating function to Python...")

    completion = completion_with_backoff(
        model=model_name,
        messages=messages.translate_to_python_messages(fortran_code),
        temperature=0,
    )

    logger.trace(f'COMPLETION: {completion.choices[0].message["content"]}')

    # Extract the code block from the completion
    python_function = utils.extract_code_block(completion)

    return python_function
