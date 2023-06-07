from fparser.two.parser import ParserFactory
from fparser.common.readfortran import FortranFileReader
from fparser.two import Fortran2003



def get_subroutines_and_functions(file_path='DaylengthMod.f90'):
    # create the Fortran code parser
    f2003_parser = ParserFactory().create(std="f2003")

    # parse the code
    reader = FortranFileReader(file_path)
    parse_tree = f2003_parser(reader)

    # function to walk the parse tree and print function, type, and subroutine names
    def print_public_symbols(node):
        result = []
        if isinstance(node, (Fortran2003.Function_Subprogram, Fortran2003.Subroutine_Subprogram)):
            result.append(node)
        elif type(node) is not str and type(node) is not int and type(node) is not float and node is not None:
            for child in node.children:
                result.extend(print_public_symbols(child))
        
        return result

    # print the names of all public functions, types, and subroutines
    nodes = print_public_symbols(parse_tree)

    print(f"Found {len(nodes)} subroutines and functions in {file_path}")

    return [node.tostr() for node in nodes]

nodes = get_subroutines_and_functions()
for node in nodes:
    print(node)
    print()
