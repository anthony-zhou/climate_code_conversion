import random
import string
# import openai
import os
import dotenv

import testing as testing
import utils as utils
from utils import logger

logger.add("llm_outputs.log")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

import promptlayer

dotenv.load_dotenv()

promptlayer.api_key = os.environ.get("PROMPTLAYER_API_KEY")

# Swap out your 'import openai'
openai = promptlayer.openai
openai.api_key = os.environ.get("OPENAI_API_KEY")

model_name = "gpt-4-0613"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def _generate_fortran_unit_tests(source_code):
    logger.info("Generating unit tests in Fortran...")

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
    logger.debug(f"PROMPT: {prompt}")

    completion = completion_with_backoff(
        model=model_name,
        messages=[
            {"role": "system", "content": "You're a proficient Fortran programmer."},
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    logger.debug(f'COMPLETION: {completion.choices[0].message["content"]}')

    # Extract the code block from the completion
    unit_tests = completion.choices[0].message["content"].split("```")[1]

    return unit_tests



def _generate_python_tests(python_function: str):
    logger.info("Generating unit tests based on python code...")

    prompt = f"""
    Generate unit tests for the following Python function using pytest. No need to import the module under test. ```python\n{python_function}\n```
    """

    logger.debug(f"PROMPT: {prompt}")


    completion = completion_with_backoff(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": """You're a programmer proficient in Python and unit testing. You can write and execute Python code by enclosing it in triple backticks, e.g. ```code goes here```"""
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
    )

    logger.debug(f'COMPLETION: {completion.choices[0].message["content"]}')

    unit_tests = completion.choices[0].message["content"].split("```")[1]
    unit_tests = unit_tests.replace("python\n", "")


    return unit_tests



def _translate_tests_to_python(unit_tests):
    logger.info("Translating unit tests to Python...")

    prompt = f"""
    Convert the following unit tests from Fortran to Python using pytest. No need to import the module under test. ```\n{unit_tests}```\n
    """
    logger.debug(f"PROMPT: {prompt}")

    completion = completion_with_backoff(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You're a programmer proficient in Fortran and Python.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    logger.debug(f'COMPLETION: {completion.choices[0].message["content"]}')

    # Extract the code block from the completion
    unit_tests = completion.choices[0].message["content"].split("```")[1]
    # Remove `python` from the first line
    unit_tests = unit_tests.replace("python\n", "")

    return unit_tests


def generate_unit_tests(source_code):
    unit_tests = _generate_fortran_unit_tests(source_code)
    python_tests = _translate_tests_to_python(unit_tests)

    return python_tests


def _translate_function_to_python(source_code):
    logger.info("Translating function to Python...")
    prompt = f"""
    Convert the following Fortran function to Python. ```\n{source_code}```\n
    """
    logger.debug(f"PROMPT: {prompt}")

    completion = completion_with_backoff(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You're a programmer proficient in Fortran and Python.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    logger.debug(f'COMPLETION: {completion.choices[0].message["content"]}')

    # Extract the code block from the completion
    python_function = completion.choices[0].message["content"].split("```")[1]
    # Remove `python` from the first line
    python_function = python_function.replace("python\n", "")

    return python_function


def iterate(fortran_function, fortran_unit_tests, python_function, python_unit_tests, python_test_results, temperature=0.0):
    messages = [
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
            """
        }
    ]

    logger.debug(messages)
    completion = completion_with_backoff(
        model="gpt-4-0613",
        messages=messages,
        temperature=temperature,
    )

    response = completion.choices[0].message["content"]
    logger.debug(f"RESPONSE:\n{response}")


    start_index = response.find("```")
    end_index = response.find("```", start_index + 3)
    source_code = response[start_index + 3:end_index]
    source_code = source_code.replace("python\n", "")

    # Find the second instance of code included in triple backticks
    start_index = response.find("```", end_index + 3)
    end_index = response.find("```", start_index + 3)
    unit_tests = response[start_index + 3:end_index]
    unit_tests = unit_tests.replace("python\n", "")

    return source_code, unit_tests


def generate_python_code(fortran_function, function_name=""):
    # Given a Fortran function, translate it into Python, with unit tests for each

    filename = ''.join(random.choices(string.ascii_lowercase, k=10))
    filename = f'./output/translations/{function_name}_{filename}.csv'
    logger.info(f"Saving outputs to {filename}")

    fortran_unit_tests = _generate_fortran_unit_tests(fortran_function)
    # python_unit_tests = _translate_tests_to_python(fortran_unit_tests)
    python_function = _translate_function_to_python(fortran_function)
    python_unit_tests = _generate_python_tests(python_function)

    # TODO: determine what packages we need in the docker image (basic static analysis)
    docker_image = "faizanbashir/python-datascience:3.6"
    python_test_results = testing.run_tests(python_function, python_unit_tests, docker_image=docker_image)


    i = 0
    dict = [{
        'fortran_function': fortran_function,
        'fortran_unit_tests': fortran_unit_tests, 
        'python_function': python_function,
        'python_unit_tests': python_unit_tests,
        'python_test_results': python_test_results,
        'code_diffs': '',
        'test_diffs': ''
    }]

    logger.debug(f"Test results for iteration {i}")
    logger.debug(python_test_results)

    utils.save_to_csv(dict, outfile=filename)

    response = input("Would you like to continue (Y/n)? ")
    while response.lower() != "n":
        i += 1
        new_python_function, new_python_unit_tests = iterate(fortran_function=fortran_function,
                                          fortran_unit_tests=fortran_unit_tests,
                                          python_function=python_function,
                                          python_unit_tests=python_unit_tests,
                                          python_test_results=utils.remove_ansi_escape_codes(python_test_results),
                                          temperature=0.5)
        
        code_diffs = utils.find_diffs(new_python_function, python_function)
        test_diffs = utils.find_diffs(new_python_unit_tests, python_unit_tests)
        
        python_function = new_python_function
        python_unit_tests = new_python_unit_tests
        
        python_test_results = testing.run_tests(python_function, python_unit_tests, docker_image=docker_image)
        logger.debug(f"Test results for iteration {i}")
        logger.debug(python_test_results)

        dict.append({
            'fortran_function': fortran_function,
            'fortran_unit_tests': fortran_unit_tests, 
            'python_function': new_python_function,
            'python_unit_tests': new_python_unit_tests,
            'python_test_results': python_test_results,
            'code_diffs': code_diffs,
            'test_diffs': test_diffs
        })

        utils.save_to_csv(dict, filename)

        response = input("Would you like to continue (Y/n)? ")

    logger.info(f"Done. Output saved to {filename}.")


if __name__ == "__main__":

    fortran_function = """
!-----------------------------------------------------------------------
elemental real(r8) function daylength(lat, decl)
    !
    ! !DESCRIPTION:
    ! Computes daylength (in seconds)
    !
    ! Latitude and solar declination angle should both be specified in radians. decl must
    ! be strictly less than pi/2; lat must be less than pi/2 within a small tolerance.
    !
    ! !USES:
    use shr_infnan_mod, only : nan => shr_infnan_nan, &
                            assignment(=)
    use shr_const_mod , only : SHR_CONST_PI
    !
    ! !ARGUMENTS:
    real(r8), intent(in) :: lat    ! latitude (radians)
    real(r8), intent(in) :: decl   ! solar declination angle (radians)
    !
    ! !LOCAL VARIABLES:
    real(r8) :: my_lat             ! local version of lat, possibly adjusted slightly
    real(r8) :: temp               ! temporary variable

    ! number of seconds per radian of hour-angle
    real(r8), parameter :: secs_per_radian = 13750.9871_r8

    ! epsilon for defining latitudes "near" the pole
    real(r8), parameter :: lat_epsilon = 10._r8 * epsilon(1._r8)

    ! Define an offset pole as slightly less than pi/2 to avoid problems with cos(lat) being negative
    real(r8), parameter :: pole = SHR_CONST_PI/2.0_r8
    real(r8), parameter :: offset_pole = pole - lat_epsilon
    !-----------------------------------------------------------------------

    ! Can't SHR_ASSERT in an elemental function; instead, return a bad value if any
    ! preconditions are violated

    ! lat must be less than pi/2 within a small tolerance
    if (abs(lat) >= (pole + lat_epsilon)) then
    daylength = nan

    ! decl must be strictly less than pi/2
    else if (abs(decl) >= pole) then
    daylength = nan

    ! normal case
    else    
    ! Ensure that latitude isn't too close to pole, to avoid problems with cos(lat) being negative
    my_lat = min(offset_pole, max(-1._r8 * offset_pole, lat))

    temp = -(sin(my_lat)*sin(decl))/(cos(my_lat) * cos(decl))
    temp = min(1._r8,max(-1._r8,temp))
    daylength = 2.0_r8 * secs_per_radian * acos(temp) 
    end if

end function daylength"""
#     fortran_function = """
# recursive function factorial(n) result(fact)
#     integer, intent(in) :: n
#     integer :: fact

#     if (n == 0) then
#       fact = 1
#     else
#       fact = n * factorial(n - 1)
#     end if
# end function factorial
#     """
    # fortran_function = """!N order matrix A, return det(A)
    # real*8 function determinant(A, N)
    #     integer,intent(in)::N
    #     real*8,dimension(N,N),intent(in)::A
    #     integer::i; integer,dimension(N)::ipiv; real*8::sign
    #     real*8,dimension(N,N)::Acopy
    #     Acopy=A; call dgetrf(N,N,Acopy,N,ipiv,i)
    #     if(ipiv(1)==1) then; sign=1d0; else; sign=-1d0; end if
    #     determinant=Acopy(1,1)
    #     do i=2,N
    #         if(ipiv(i)/=i) sign=-sign
    #         determinant=determinant*Acopy(i,i)
    #     end do
    #     determinant=determinant*sign
    # end function determinant"""
    
    generate_python_code(fortran_function, function_name="daylength")
