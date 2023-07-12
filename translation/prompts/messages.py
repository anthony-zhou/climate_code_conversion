from translation.utils import logger


def fortran_unit_test_messages(fortran_code: str):
    prompt = f"""
    Given fortran code, write unit tests using funit.

    Example:
    FORTRAN CODE:
    ```
    module fac
        implicit none
        
        contains

        recursive function factorial(n) result(fact)
            integer, intent(in) :: n
            integer :: fact

            if (n == 0) then
            fact = 1
            else
            fact = n * factorial(n - 1)
            end if
        end function factorial
    end module fac
    ```

    FORTRAN TESTS:
    ```
    @test
    subroutine test_fac()
        use funit

        @assertEqual(120, factorial(5), 'factorial(5)')
        @assertEqual(1, factorial(1), 'factorial(1)')
        @assertEqual(1, factorial(0), 'factorial(0)')

    end subroutine test_fac
    ```

    Your turn:
    FORTRAN CODE:\n```\n{fortran_code}\n```\n
    FORTRAN TESTS:
    """

    logger.debug(f"PROMPT: {prompt}")

    return [
        {"role": "system", "content": "You're a proficient Fortran programmer."},
        {
            "role": "user",
            "content": prompt,
        },
    ]


def iterate_messages(
    python_function: str, python_unit_tests: str, python_test_results: str
):
    return [
        {
            "role": "system",
            "content": """You're a programmer proficient in Fortran and Python. You can write and execute Python code by enclosing it in triple backticks, e.g. ```code goes here```.
            When prompted to fix source code and unit tests, always return a response of the form:
            SOURCE CODE: ```<python source code>```
            UNIT TESTS: ```<python unit tests>```. Do not return any additional context.
            """,
        },
        # {
        #     "role": "user",
        #     "content": f"""Convert the following Fortran function to Python. ```\n{fortran_function}\n```"""
        # },
        # {
        #     "role": "assistant",
        #     "content": f"""Here's the converted Python function:\n```python\n{python_function}\n```"""
        # },
        {
            "role": "user",
            "content": f"""
Function being tested:
```python\n{python_function}\n
Here are some unit tests for the above code and the corresponding output.
Unit tests:
```python
{python_unit_tests}
```
Output from `pytest`:
```
{python_test_results}
```

Modify the source code to pass the failing unit tests. Return a response of the following form:
SOURCE CODE: ```<python source code>```
UNIT TESTS: ```<python unit tests>```
""",
        },
    ]


def generate_python_test_messages(python_function: str):
    prompt = f"""
Generate 5 unit tests for the following Python function using pytest. No need to import the module under test. ```python\n{python_function}\n```
    """

    logger.debug(f"PROMPT: {prompt}")

    return [
        {
            "role": "system",
            "content": """You're a programmer proficient in Python and unit testing. You can write and execute Python code by enclosing it in triple backticks, e.g. ```code goes here```""",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def translate_to_python_messages(fortran_code: str):
    prompt = f"""
    Convert the following Fortran function to Python. ```\n{fortran_code}```\n
    """
    logger.debug(f"PROMPT: {prompt}")

    return [
        {
            "role": "system",
            "content": "You're a programmer proficient in Fortran and Python.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def translate_tests_to_python_messages(unit_tests: str):
    prompt = f"""
    Convert the following unit tests from Fortran to Python using pytest. No need to import the module under test. ```\n{unit_tests}```\n
    """
    logger.debug(f"PROMPT: {prompt}")
    return [
        {
            "role": "system",
            "content": "You're a programmer proficient in Fortran and Python.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
