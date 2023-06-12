import openai
import os
import dotenv

import dag
import testing

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
    FORTRAN CODE:\n```\n{source_code}\n```\n
    FORTRAN TESTS:
    """
    print(f"PROMPT: {prompt}")

    completion = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You're a proficient Fortran programmer."
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )

    print(f'COMPLETION: {completion.choices[0].message["content"]}')

    # Extract the code block from the completion
    unit_tests = completion.choices[0].message["content"].split("```")[1]

    return unit_tests


def _translate_tests_to_python(unit_tests):
    print("Translating unit tests to Python...")

    prompt = f"""
    Convert the following unit tests from Fortran to Python using pytest: ```\n{unit_tests}```\n
    """
    print(f"PROMPT: {prompt}")

    completion = completion_with_backoff(
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
        ]
    )

    print(f'COMPLETION: {completion.choices[0].message["content"]}')

    # Extract the code block from the completion
    unit_tests = completion.choices[0].message["content"].split("```")[1]

    return unit_tests


def generate_unit_tests(source_code):
    unit_tests = _generate_fortran_unit_tests(source_code)
    python_tests = _translate_tests_to_python(unit_tests)
    print("Done!")

    return python_tests


# def run_tests(source_code):
#     test_output = testing.run_tests(source_code)
#     print(test_output)


if __name__ == "__main__":
    generated_code = """
import numpy as np

def daylength(lat, decl):
    SHR_CONST_PI = np.pi
    secs_per_radian = 13750.9871
    lat_epsilon = 10. * np.finfo(float).eps
    pole = SHR_CONST_PI / 2.0
    offset_pole = pole - lat_epsilon
    
    # Check if inputs are array-like and convert to numpy arrays if necessary
    lat = np.asarray(lat)
    decl = np.asarray(decl)
    
    # Broadcast lat and decl to the same shape
    lat, decl = np.broadcast_arrays(lat, decl)

    # Create an output array filled with NaN
    result = np.full_like(lat, np.nan)

    # Apply the calculation where the conditions are met
    condition = (np.abs(lat) < (pole + lat_epsilon)) & (np.abs(decl) < pole)
    my_lat = np.minimum(offset_pole, np.maximum(-offset_pole, lat[condition]))
    temp = - (np.sin(my_lat) * np.sin(decl[condition])) / (np.cos(my_lat) * np.cos(decl[condition]))
    temp = np.minimum(1., np.maximum(-1., temp))
    result[condition] = 2.0 * secs_per_radian * np.arccos(temp)

    # If the input was a scalar, return a scalar; otherwise, return a numpy array
    if np.isscalar(lat) and np.isscalar(decl):
        return np.asscalar(result)
    else:
        return result
    """
    sorted_functions = dag.get_sorted_functions(example_file)

    func_name, func = sorted_functions[0]
    source_code = func["source"]

    unit_tests = generate_unit_tests(source_code)

    # Run the unit tests
    test_output = testing.run_tests(generated_code + '\n' + unit_tests, docker_image="slothai/numpy")
    print(test_output)
    # run_tests(source_code)