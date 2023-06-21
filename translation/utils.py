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