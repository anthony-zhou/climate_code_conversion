# Given source code, unit tests, and failing test output, make a GPT-4 API call that attempts to fix it
import textwrap
import testing as testing
import utils as utils
from utils import logger, completion_with_backoff

from config import model_name


def fix_problem(source_code: str, unit_tests: str, test_output: str):
    messages = [
        {
            "role": "system",
            "content": """You're a programmer proficient in Fortran and Python. You can write and execute Python code by enclosing it in triple backticks, e.g. ```code goes here```.
            When prompted to fix source code and unit tests, always return a response of the form:
            SOURCE CODE: ```<python source code>```
            UNIT TESTS: ```<python unit tests>```. Do not return any additional context.
            """,
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
            Function being tested:
            ```python\n{source_code}\n
            Here are some unit tests for the above code and the corresponding output.
            Unit tests:
            ```python
            {unit_tests}
            ```
            Output from `pytest`:
            ```
            {test_output}
            ```

            Modify the source code to pass the failing unit tests. Return a response of the following form:
            SOURCE CODE: ```<python source code>```
            UNIT TESTS: ```<python unit tests>```
            """
            ),
        },
    ]

    logger.trace(messages)
    completion = completion_with_backoff(
        model=model_name,
        messages=messages,
        temperature=0.3,
    )

    response = completion.choices[0].message["content"]
    logger.trace(f"RESPONSE:\n{response}")

    new_source_code = utils.extract_source_code(response)
    new_unit_tests = utils.extract_unit_test_code(response)

    if new_source_code is None:
        new_source_code = source_code
    if new_unit_tests is None:
        new_unit_tests = unit_tests

    return str(new_source_code), str(new_unit_tests)
