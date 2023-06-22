import random
import string
# import openai
import os
import dotenv

import translation.testing as testing
import translation.utils as utils
from translation.utils import logger

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


def iterate(python_function, python_unit_tests, python_test_results, temperature=0.0):
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


    source_code = utils.extract_source_code(response)
    unit_tests = utils.extract_unit_test_code(response)

    return source_code, unit_tests


def generate_python_code(fortran_function: str, function_name=""):
    # Given a Fortran function, translate it into Python, with unit tests for each

    filename = ''.join(random.choices(string.ascii_lowercase, k=10))
    filename = f'./output/translations/{function_name}_{filename}.csv'
    logger.info(f"Saving outputs to {filename}")

    # fortran_unit_tests = _generate_fortran_unit_tests(fortran_function)
    # python_unit_tests = _translate_tests_to_python(fortran_unit_tests)
    python_function = _translate_function_to_python(fortran_function)
    python_unit_tests = _generate_python_tests(python_function)

    # TODO: determine what packages we need in the docker image (basic static analysis)
    docker_image = "faizanbashir/python-datascience:3.6"
    python_test_results = testing.run_tests(python_function, python_unit_tests, docker_image=docker_image)


    i = 0
    dict = [{
        'fortran_function': fortran_function,
        # 'fortran_unit_tests': fortran_unit_tests, 
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
        new_python_function, new_python_unit_tests = iterate(
                                        # fortran_function=fortran_function,
                                        #   fortran_unit_tests=fortran_unit_tests,
                                          python_function=python_function,
                                          python_unit_tests=python_unit_tests,
                                          python_test_results=utils.remove_ansi_escape_codes(python_test_results),
                                          temperature=0)
        

        if new_python_function:
            code_diffs = utils.find_diffs(new_python_function, python_function)
            python_function = new_python_function
        else:
            code_diffs = ""

        if new_python_unit_tests:
            test_diffs = utils.find_diffs(new_python_unit_tests, python_unit_tests)
            python_unit_tests = new_python_unit_tests
        else:
            code_diffs = ""
        
        python_test_results = testing.run_tests(python_function, python_unit_tests, docker_image=docker_image)
        logger.debug(f"Test results for iteration {i}")
        logger.debug(python_test_results)

        dict.append({
            'fortran_function': fortran_function,
            # 'fortran_unit_tests': fortran_unit_tests, 
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
!------------------------------------------------------------------------------
subroutine ci_func(ci, fval, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z,&
    rh_can, gs_mol, atm2lnd_inst, photosyns_inst)
!
!! DESCRIPTION:
! evaluate the function
! f(ci)=ci - (ca - (1.37rb+1.65rs))*patm*an
!
! remark:  I am attempting to maintain the original code structure, also
! considering one may be interested to output relevant variables for the
! photosynthesis model, I have decided to add these relevant variables to
! the relevant data types.
!
!!ARGUMENTS:
real(r8)             , intent(in)    :: ci       ! intracellular leaf CO2 (Pa)
real(r8)             , intent(in)    :: lmr_z    ! canopy layer: leaf maintenance respiration rate (umol CO2/m**2/s)
real(r8)             , intent(in)    :: par_z    ! par absorbed per unit lai for canopy layer (w/m**2)
real(r8)             , intent(in)    :: gb_mol   ! leaf boundary layer conductance (umol H2O/m**2/s)
real(r8)             , intent(in)    :: je       ! electron transport rate (umol electrons/m**2/s)
real(r8)             , intent(in)    :: cair     ! Atmospheric CO2 partial pressure (Pa)
real(r8)             , intent(in)    :: oair     ! Atmospheric O2 partial pressure (Pa)
real(r8)             , intent(in)    :: rh_can   ! canopy air realtive humidity
integer              , intent(in)    :: p, iv, c ! pft, vegetation type and column indexes
real(r8)             , intent(out)   :: fval     ! return function of the value f(ci)
real(r8)             , intent(out)   :: gs_mol   ! leaf stomatal conductance (umol H2O/m**2/s)
type(atm2lnd_type)   , intent(in)    :: atm2lnd_inst
type(photosyns_type) , intent(inout) :: photosyns_inst
!
!local variables
real(r8) :: ai                  ! intermediate co-limited photosynthesis (umol CO2/m**2/s)
real(r8) :: cs                  ! CO2 partial pressure at leaf surface (Pa)
real(r8) :: term                 ! intermediate in Medlyn stomatal model
real(r8) :: aquad, bquad, cquad  ! terms for quadratic equations
real(r8) :: r1, r2               ! roots of quadratic equation
!------------------------------------------------------------------------------

associate(&
        forc_pbot  => atm2lnd_inst%forc_pbot_downscaled_col   , & ! Output: [real(r8) (:)   ]  atmospheric pressure (Pa)
        c3flag     => photosyns_inst%c3flag_patch             , & ! Output: [logical  (:)   ]  true if C3 and false if C4
        ivt        => patch%itype                             , & ! Input:  [integer  (:)   ]  patch vegetation type
        medlynslope      => pftcon%medlynslope                , & ! Input:  [real(r8) (:)   ]  Slope for Medlyn stomatal conductance model method
        medlynintercept  => pftcon%medlynintercept            , & ! Input:  [real(r8) (:)   ]  Intercept for Medlyn stomatal conductance model method
        stomatalcond_mtd => photosyns_inst%stomatalcond_mtd   , & ! Input:  [integer        ]  method type to use for stomatal conductance
        ac         => photosyns_inst%ac_patch                 , & ! Output: [real(r8) (:,:) ]  Rubisco-limited gross photosynthesis (umol CO2/m**2/s)
        aj         => photosyns_inst%aj_patch                 , & ! Output: [real(r8) (:,:) ]  RuBP-limited gross photosynthesis (umol CO2/m**2/s)
        ap         => photosyns_inst%ap_patch                 , & ! Output: [real(r8) (:,:) ]  product-limited (C3) or CO2-limited (C4) gross photosynthesis (umol CO2/m**2/s)
        ag         => photosyns_inst%ag_patch                 , & ! Output: [real(r8) (:,:) ]  co-limited gross leaf photosynthesis (umol CO2/m**2/s)
        an         => photosyns_inst%an_patch                 , & ! Output: [real(r8) (:,:) ]  net leaf photosynthesis (umol CO2/m**2/s)
        vcmax_z    => photosyns_inst%vcmax_z_patch            , & ! Input:  [real(r8) (:,:) ]  maximum rate of carboxylation (umol co2/m**2/s)
        cp         => photosyns_inst%cp_patch                 , & ! Output: [real(r8) (:)   ]  CO2 compensation point (Pa)
        kc         => photosyns_inst%kc_patch                 , & ! Output: [real(r8) (:)   ]  Michaelis-Menten constant for CO2 (Pa)
        ko         => photosyns_inst%ko_patch                 , & ! Output: [real(r8) (:)   ]  Michaelis-Menten constant for O2 (Pa)
        qe         => photosyns_inst%qe_patch                 , & ! Output: [real(r8) (:)   ]  quantum efficiency, used only for C4 (mol CO2 / mol photons)
        tpu_z      => photosyns_inst%tpu_z_patch              , & ! Output: [real(r8) (:,:) ]  triose phosphate utilization rate (umol CO2/m**2/s)
        kp_z       => photosyns_inst%kp_z_patch               , & ! Output: [real(r8) (:,:) ]  initial slope of CO2 response curve (C4 plants)
        bbb        => photosyns_inst%bbb_patch                , & ! Output: [real(r8) (:)   ]  Ball-Berry minimum leaf conductance (umol H2O/m**2/s)
        mbb        => photosyns_inst%mbb_patch                  & ! Output: [real(r8) (:)   ]  Ball-Berry slope of conductance-photosynthesis relationship
        )

    if (c3flag(p)) then
        ! C3: Rubisco-limited photosynthesis
        ac(p,iv) = vcmax_z(p,iv) * max(ci-cp(p), 0._r8) / (ci+kc(p)*(1._r8+oair/ko(p)))

        ! C3: RuBP-limited photosynthesis
        aj(p,iv) = je * max(ci-cp(p), 0._r8) / (4._r8*ci+8._r8*cp(p))

        ! C3: Product-limited photosynthesis
        ap(p,iv) = 3._r8 * tpu_z(p,iv)

    else

        ! C4: Rubisco-limited photosynthesis
        ac(p,iv) = vcmax_z(p,iv)

        ! C4: RuBP-limited photosynthesis
        aj(p,iv) = qe(p) * par_z * 4.6_r8

        ! C4: PEP carboxylase-limited (CO2-limited)
        ap(p,iv) = kp_z(p,iv) * max(ci, 0._r8) / forc_pbot(c)

    end if

    ! Gross photosynthesis. First co-limit ac and aj. Then co-limit ap

    aquad = params_inst%theta_cj(ivt(p))
    bquad = -(ac(p,iv) + aj(p,iv))
    cquad = ac(p,iv) * aj(p,iv)
    call quadratic (aquad, bquad, cquad, r1, r2)
    ai = min(r1,r2)

    aquad = params_inst%theta_ip
    bquad = -(ai + ap(p,iv))
    cquad = ai * ap(p,iv)
    call quadratic (aquad, bquad, cquad, r1, r2)
    ag(p,iv) = max(0._r8,min(r1,r2))

    ! Net photosynthesis. Exit iteration if an < 0

    an(p,iv) = ag(p,iv) - lmr_z
    if (an(p,iv) < 0._r8) then
        fval = 0._r8
        return
    endif
    ! Quadratic gs_mol calculation with an known. Valid for an >= 0.
    ! With an <= 0, then gs_mol = bbb or medlyn intercept
    cs = cair - 1.4_r8/gb_mol * an(p,iv) * forc_pbot(c)
    cs = max(cs,max_cs)
    if ( stomatalcond_mtd == stomatalcond_mtd_medlyn2011 )then
        term = 1.6_r8 * an(p,iv) / (cs / forc_pbot(c) * 1.e06_r8)
        aquad = 1.0_r8
        bquad = -(2.0 * (medlynintercept(patch%itype(p))*1.e-06_r8 + term) + (medlynslope(patch%itype(p)) * term)**2 / &
            (gb_mol*1.e-06_r8 * rh_can))
        cquad = medlynintercept(patch%itype(p))*medlynintercept(patch%itype(p))*1.e-12_r8 + &
            (2.0*medlynintercept(patch%itype(p))*1.e-06_r8 + term * &
            (1.0 - medlynslope(patch%itype(p))* medlynslope(patch%itype(p)) / rh_can)) * term

        call quadratic (aquad, bquad, cquad, r1, r2)
        gs_mol = max(r1,r2) * 1.e06_r8
    else if ( stomatalcond_mtd == stomatalcond_mtd_bb1987 )then
        aquad = cs
        bquad = cs*(gb_mol - bbb(p)) - mbb(p)*an(p,iv)*forc_pbot(c)
        cquad = -gb_mol*(cs*bbb(p) + mbb(p)*an(p,iv)*forc_pbot(c)*rh_can)
        call quadratic (aquad, bquad, cquad, r1, r2)
        gs_mol = max(r1,r2)
    end if

    ! Derive new estimate for ci

    fval =ci - cair + an(p,iv) * forc_pbot(c) * (1.4_r8*gs_mol+1.6_r8*gb_mol) / (gb_mol*gs_mol)

end associate

end subroutine ci_func
    """

#     fortran_function = """
# !-----------------------------------------------------------------------
# elemental real(r8) function daylength(lat, decl)
#     !
#     ! !DESCRIPTION:
#     ! Computes daylength (in seconds)
#     !
#     ! Latitude and solar declination angle should both be specified in radians. decl must
#     ! be strictly less than pi/2; lat must be less than pi/2 within a small tolerance.
#     !
#     ! !USES:
#     use shr_infnan_mod, only : nan => shr_infnan_nan, &
#                             assignment(=)
#     use shr_const_mod , only : SHR_CONST_PI
#     !
#     ! !ARGUMENTS:
#     real(r8), intent(in) :: lat    ! latitude (radians)
#     real(r8), intent(in) :: decl   ! solar declination angle (radians)
#     !
#     ! !LOCAL VARIABLES:
#     real(r8) :: my_lat             ! local version of lat, possibly adjusted slightly
#     real(r8) :: temp               ! temporary variable

#     ! number of seconds per radian of hour-angle
#     real(r8), parameter :: secs_per_radian = 13750.9871_r8

#     ! epsilon for defining latitudes "near" the pole
#     real(r8), parameter :: lat_epsilon = 10._r8 * epsilon(1._r8)

#     ! Define an offset pole as slightly less than pi/2 to avoid problems with cos(lat) being negative
#     real(r8), parameter :: pole = SHR_CONST_PI/2.0_r8
#     real(r8), parameter :: offset_pole = pole - lat_epsilon
#     !-----------------------------------------------------------------------

#     ! Can't SHR_ASSERT in an elemental function; instead, return a bad value if any
#     ! preconditions are violated

#     ! lat must be less than pi/2 within a small tolerance
#     if (abs(lat) >= (pole + lat_epsilon)) then
#     daylength = nan

#     ! decl must be strictly less than pi/2
#     else if (abs(decl) >= pole) then
#     daylength = nan

#     ! normal case
#     else    
#     ! Ensure that latitude isn't too close to pole, to avoid problems with cos(lat) being negative
#     my_lat = min(offset_pole, max(-1._r8 * offset_pole, lat))

#     temp = -(sin(my_lat)*sin(decl))/(cos(my_lat) * cos(decl))
#     temp = min(1._r8,max(-1._r8,temp))
#     daylength = 2.0_r8 * secs_per_radian * acos(temp) 
#     end if

# end function daylength"""
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
