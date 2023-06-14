from fparser.two.parser import ParserFactory
from fparser.common.readfortran import FortranFileReader
from fparser.two import Fortran2003

from fparser.two import Fortran2003, parser
from fparser.common.readfortran import FortranStringReader
from collections import defaultdict
import pprint

from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv

def _parse_source(file_path):
    f2003_parser = ParserFactory().create(std="f2003")
    reader = FortranFileReader(file_path)
    parse_tree = f2003_parser(reader)
    return parse_tree


def _find_calls(node, func_name=None):
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
            yield from _find_calls(child, func_name)


def _find_dependencies(source):
    ast = _parse_source(source)
    dependencies = defaultdict(lambda: {"source": "", "calls": []})
    for func_name, source_code, call in _find_calls(ast):
        if source_code != "":
            dependencies[func_name]["source"] = source_code
        if call is not None:
            dependencies[func_name]["calls"].append(call) # type: ignore
    return dict(dependencies)


def _dependencies_to_dag(dependencies):
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


def _topological_sort(dependencies):
    dag = _dependencies_to_dag(dependencies)
    return list(nx.topological_sort(dag)) # type: ignore


def draw_dag_and_save(dag, filename):
    A = nx.nx_agraph.to_agraph(dag)
    A.graph_attr['ranksep'] = '10.0'
    A.graph_attr['nodesep'] = '1.0'
    G = nx.DiGraph(A)
    layout = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")

    # layout = nx.spring_layout(dag, k=0.7)
    plt.figure(figsize=(30, 30))
    nx.draw(dag, pos=layout, with_labels=True, arrows=True, node_color="skyblue") # type: ignore
    plt.margins(0.20)
    plt.savefig(filename)


def draw_dag_interactive(dag, outfile):
    net = Network(notebook=True, directed=True)
    net.from_nx(dag)
    net.show_buttons(filter_=["physics"])
    net.toggle_physics(True)
    net.show(outfile)


def get_sorted_functions(source_file):
    dependencies = _find_dependencies(source_file)
    sorted_keys = _topological_sort(dependencies)
    return [(key, dependencies.get(key, {"source": "not_found", "calls": []})) for key in sorted_keys]


if __name__ == "__main__":
    # Generate a DAG from the dependencies

    pp = pprint.PrettyPrinter(indent=4)

    sorted_functions = get_sorted_functions(
        # "./examples/daylength_2/fortran/DaylengthMod.f90"
        "./examples/photosynthesis/PhotosynthesisMod.f90"
    )

    for func_name, func in sorted_functions:
        print(func_name)
        print(func["source"])

    dag = _dependencies_to_dag(
            _find_dependencies("./examples/photosynthesis/PhotosynthesisMod.f90")
        )

    draw_dag_and_save(
        dag,
        "dag.png",
    )

    draw_dag_interactive(dag, "./output/dag2.html")
