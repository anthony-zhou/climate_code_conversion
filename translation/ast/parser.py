# Parse individual Fortran files to get functions and dependencies
from collections import defaultdict

from fparser.two.parser import ParserFactory
from fparser.common.readfortran import FortranFileReader
from fparser.two import Fortran2003
from fparser.two import Fortran2003
import networkx as nx


def _get_parse_tree(file_path):
    f2003_parser = ParserFactory().create(std="f2003")
    reader = FortranFileReader(file_path, ignore_comments=False)
    parse_tree = f2003_parser(reader)
    return parse_tree


def _find_public_functions(node, func_name=None):
    if isinstance(node, (Fortran2003.Access_Stmt)):
        # TODO: catch more edge cases when finding public functions
        items = node.items
        if items[0].lower() == "public":
            yield str(items[1])
    elif (
        type(node) is not str
        and type(node) is not int
        and type(node) is not float
        and node is not None
        and type(node) is not tuple
    ):
        for child in node.children:
            yield from _find_public_functions(child, func_name)


def _find_calls(node, func_name=None):
    if isinstance(
        node, (Fortran2003.Function_Subprogram, Fortran2003.Subroutine_Subprogram)
    ):
        # Get first non-comment item in the node
        index = 0
        while isinstance(node.content[index], Fortran2003.Comment):
            index += 1
        func_name = str(node.content[index].get_name())
        print(f"Found function or subroutine {func_name}")
        source_code = str(node)
        # print(f"Source code: {source_code}")
        yield (func_name, source_code, None)
    if isinstance(
        node, (Fortran2003.Call_Stmt, Fortran2003.Intrinsic_Function_Reference)
    ):
        called = str(node.items[0]).lower()
        # print(f"{called} called from {func_name}")
        if func_name == None:
            # `called` is being called from the top level. This must be an external
            yield ("EXTERNAL", "", called)
        else:
            yield (func_name, "", called)
    elif (
        type(node) is not str
        and type(node) is not int
        and type(node) is not float
        and node is not None
        and type(node) is not tuple
    ):
        # print(f"Found statement of type{type(node)}")
        for child in node.children:
            yield from _find_calls(child, func_name)


def get_dag(filename: str):
    """
    In this dependency graph, the nodes are functions and the edges are calls.
    An arrow means that the function at the tail of the arrow calls the function at the head of the arrow.

    Thus, a topological sort of this graph will give us an order in which we can compile the functions.
    """
    ast = _get_parse_tree(filename)
    dependencies = defaultdict(lambda: {"source": "", "calls": []})
    for func_name, source_code, call in _find_calls(ast):
        if source_code != "":
            dependencies[func_name]["source"] = source_code
        if call is not None:
            dependencies[func_name]["calls"].append(call)  # type: ignore
    dependencies = dict(dependencies)

    dag = nx.DiGraph()

    for func_name, info in dependencies.items():
        dag.add_node(func_name, source=info["source"], calls=info["calls"])
        for call in info["calls"]:
            dag.add_edge(call, func_name)

    return dag


def get_public_functions(filename: str):
    """
    Get all public functions in `filename` by name
    """
    ast = _get_parse_tree(filename)
    funcs: list[str] = []
    for func_name in _find_public_functions(ast):
        funcs.append(func_name)

    return funcs


def get_sorted_functions(source_file):
    dag = get_dag(source_file)
    sorted_keys = list(nx.topological_sort(dag))
    return [
        (key, dag.nodes.get(key, {"source": "not_found", "calls": []}))
        for key in sorted_keys
    ]


def filter_for_dependencies(dag: nx.DiGraph, function_name: str):
    """
    Given a function name, return a topologically sorted list of the function's dependencies.

    Returns a list of topologically sorted tuples `[(func_name, {source, calls}), ...]` that includes the function itself.
    """

    def _get_parents(node_key: str):
        node_dict = dag.nodes.get(node_key)
        if node_dict is not None:
            parents = node_dict["calls"]
            if len(parents) > 0:
                for parent in parents:
                    yield from _get_parents(parent)
        yield node_key

    node_list = list(_get_parents(function_name))

    sorted_keys: list[str] = list(nx.topological_sort(dag))

    result = [
        (key, dag.nodes.get(key, {"source": "not_found", "calls": []}))
        for key in sorted_keys
        if key in node_list
    ]

    return result
