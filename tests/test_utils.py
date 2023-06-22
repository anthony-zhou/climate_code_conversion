import translation.utils as utils
import csv
import os

def test_save_to_csv():
    data = [
        {'name': 'John', 'age': 25},
        {'name': 'Jane', 'age': 30},
        {'name': 'Bob', 'age': 35}
    ]
    outfile = 'test.csv'
    utils.save_to_csv(data, outfile)
    
    with open(outfile, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert rows[0]['name'] == 'John'
        assert rows[0]['age'] == '25'
        assert rows[1]['name'] == 'Jane'
        assert rows[1]['age'] == '30'
        assert rows[2]['name'] == 'Bob'
        assert rows[2]['age'] == '35'    

    os.remove(outfile)
    

def test_remove_ansi_escape_codes():
        # ANSI escape code for color red
        input_str = "\033[31mHello, World!\033[0m"
        expected_output = "Hello, World!"
        assert utils.remove_ansi_escape_codes(input_str) == expected_output


def test_extract_source_code():
    message = "Here is some text before the source code.\n\nSOURCE CODE:\n```python\nprint('Hello, world!')\n```\n\nAnd here is some text after the source code."
    expected = "print('Hello, world!')"
    assert utils.extract_source_code(message) == expected

def test_extract_unit_test_code():
    message = "Here is some text before the unit tests.\n\nUNIT TESTS:\n```python\nassert 1 + 1 == 2\n```\n\nAnd here is some text after the unit tests."
    expected = "assert 1 + 1 == 2"
    assert utils.extract_unit_test_code(message) == expected


def test_extract_unit_test_code():
    message = "Here is some text before the unit tests. "
    expected = None
    assert utils.extract_unit_test_code(message) == expected
