from utils import logger


def generate_python_test_messages(python_function: str):
    prompt = f"""
Generate 3 unit tests for the following Python function using pytest. No need to import the module under test. ```python\n{python_function}\n```
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
