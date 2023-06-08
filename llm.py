import openai
import os
import dotenv

import dag

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


example_file = "./examples/daylength_2/fortran/DaylengthMod.f90"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def _generate_fortran_unit_tests(source_code):
    print("Generating unit tests in Fortran...")

    prompt = f"""
    Write a test suite in Fortran using `pfTest` for the following code. Make the code runnable and return nothing but the code.  \n{source_code}\n
    """
    print(f"PROMPT: {prompt}")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You're a Fortran programmer."
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    print(f'COMPLETION: {completion.choices[0].message["content"]}')

    return completion.choices[0].message["content"]


def _translate_tests_to_python(unit_tests):
    print("Translating unit tests to Python...")

    prompt = f"""
    Convert the following unit tests from Fortran to Python: \n{unit_tests}\n
    """
    print(f"PROMPT: {prompt}")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You're a programmer proficient in Fortran and Python."
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    print(f'COMPLETION: {completion.choices[0].message["content"]}')

    return completion.choices[0].message["content"]


def generate_unit_tests(source_code):
    unit_tests = _generate_fortran_unit_tests(source_code)
    python_tests = _translate_tests_to_python(unit_tests)
    print("Done!")

    return python_tests


# def run_tests(source_code):
#     test_output = testing.run_tests(source_code)
#     print(test_output)


if __name__ == "__main__":
    sorted_functions = dag.get_sorted_functions(example_file)

    func_name, func = sorted_functions[0]
    source_code = func["source"]

    unit_tests = generate_unit_tests(source_code)

    # run_tests(source_code)