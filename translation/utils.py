from typing import List, Dict
import csv
import difflib
import re
from loguru import logger



def save_to_csv(dict: List[Dict], outfile: str):
    with open(outfile, 'w') as f:  
        w = csv.DictWriter(f, fieldnames=dict[0].keys())
        w.writeheader()
        w.writerows(dict)



def find_diffs(source_code: str, previous_source_code: str) -> str:
    """
    Finds the diffs between the newly generated source_code and unit_tests and the corresponding python_function and python_unit_tests.
    """

    # Find the diff between the unit tests
    diff = difflib.unified_diff(source_code.splitlines(), previous_source_code.splitlines())
    
    # Convert the diff to a string
    diff_str = '\n'.join(diff)

    return diff_str


def remove_ansi_escape_codes(s):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', s)


def extract_code_block(completion):
    # Extract the code block from the completion
    code = completion.choices[0].message["content"].split("```")[1]
    code = code.replace("python\n", "")

    return code


def extract_unit_test_code(message):
    start_marker = "UNIT TESTS:"
    end_marker = "```"

    if message.find(start_marker) < 0:
        return None
    
    # find the position of start and end markers
    start = message.find(start_marker) + len(start_marker)
    end = message.rfind(end_marker)
    
    # extract the unit test code
    unit_test_code = message[start:end].strip()

    # Remove ``` and python
    unit_test_code = unit_test_code.replace("```", "")
    unit_test_code = unit_test_code.replace("python\n", "")
    
    return unit_test_code

def extract_source_code(message: str):
    start_marker = "SOURCE CODE:"
    end_marker = "```"

    if message.find(start_marker) < 0:
        return None


    # find the position of start and end markers
    start = message.find(start_marker) + len(start_marker)
    end = message.rfind(end_marker)

    # extract the source code
    source_code = message[start:end].strip()

    # Remove ``` and python
    source_code = source_code.replace("```", "")
    source_code = source_code.replace("python\n", "")

    return source_code