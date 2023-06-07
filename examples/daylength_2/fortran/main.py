from fparser.two.parser import ParserFactory
from fparser.common.readfortran import FortranFileReader
from fparser.two import Fortran2003

from fparser.two import Fortran2003, parser
from fparser.common.readfortran import FortranStringReader
from collections import defaultdict
import pprint

import networkx as nx
import matplotlib.pyplot as plt

import openai
import os
import dotenv

dotenv.load_dotenv()

def parse_source(file_path):
    f2003_parser = ParserFactory().create(std="f2003")
    reader = FortranFileReader(file_path)
    parse_tree = f2003_parser(reader)
    return parse_tree


def find_calls(node, func_name=None):
    if isinstance(
        node, (Fortran2003.Function_Subprogram, Fortran2003.Subroutine_Subprogram)
    ):
        func_name = str(node.content[0].get_name())
        print(f"Found function or subroutine {func_name}")
        source_code = str(node)
        yield (func_name, source_code, None)
    if isinstance(node, Fortran2003.Call_Stmt):
        yield (func_name, "", str(node.items[0]))
    elif (
        type(node) is not str
        and type(node) is not int
        and type(node) is not float
        and node is not None
        and type(node) is not tuple
    ):
        for child in node.children:
            yield from find_calls(child, func_name)


def find_dependencies(source):
    ast = parse_source(source)
    dependencies = defaultdict(lambda: {"source": "", "calls": []})
    for func_name, source_code, call in find_calls(ast):
        if source_code != "":
            dependencies[func_name]["source"] = source_code
        if call is not None:
            dependencies[func_name]["calls"].append(call)
    return dict(dependencies)


def dependencies_to_dag(dependencies):
    """
    In this DAG, the nodes are functions and the edges are calls.
    An arrow means that the function at the tail of the arrow calls the function at the head of the arrow.

    Thus, a topological sort of this DAG will give us an order in which we can compile the functions.
    """
    dag = nx.DiGraph()

    for func_name, info in dependencies.items():
        dag.add_node(func_name, source=info["source"])
        for call in info["calls"]:
            dag.add_edge(call, func_name)

    return dag


def topological_sort(dependencies):
    dag = dependencies_to_dag(dependencies)
    return list(nx.topological_sort(dag))


def draw_dag_and_save(dag, filename):
    pos = nx.drawing.nx_agraph.graphviz_layout(dag, prog='dot')
    nx.draw(dag, pos, with_labels=True, arrows=True, node_color="skyblue")
    plt.margins(0.20)
    plt.savefig(filename)


def generate_unit_tests(source_code):
    openai.api_key = os.environ["OPENAI_API_KEY"]

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Generate unit tests for the following code: \n" + source_code + "\n"},
    ]
    )

    print(completion.choices[0].message)
    return completion.choices[0].message['content']



# Generate a DAG from the dependencies

pp = pprint.PrettyPrinter(indent=4)

dependencies = find_dependencies("DaylengthMod.f90")
sorted_functions = topological_sort(dependencies)

for func_name in sorted_functions:
    print(func_name)

draw_dag_and_save(dependencies_to_dag(dependencies), "dag.png")

unit_tests = generate_unit_tests(dependencies["daylength"]["source"])

print(unit_tests)

# nodes = get_subroutines_and_functions()
# for node in nodes:
#     print(node)
#     print()
