from fparser.two.parser import ParserFactory
from fparser.common.readfortran import FortranFileReader
from fparser.two import Fortran2003

# create the Fortran code parser
f2003_parser = ParserFactory().create(std="f2003")

# path to the Fortran file to parse
file_path = 'preprocessed.f90'

# parse the code
reader = FortranFileReader(file_path)
parse_tree = f2003_parser(reader)

# function to walk the parse tree and print function, type, and subroutine names
def print_public_symbols(node):
    
    if isinstance(node, (Fortran2003.Function_Subprogram, Fortran2003.Subroutine_Subprogram)):
        print(node)
        print()
    elif type(node) is not str and type(node) is not int and type(node) is not float and node is not None:
        for child in node.children:
            print_public_symbols(child)

# print the names of all public functions, types, and subroutines
print_public_symbols(parse_tree)
