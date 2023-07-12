# Parse individual Fortran files to get functions and dependencies
from collections import defaultdict

from fparser.two.parser import ParserFactory
from fparser.common.readfortran import FortranFileReader
from fparser.two import Fortran2003
from fparser.two import Fortran2003
import networkx as nx

from . import parser_types


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
    elif "children" in dir(node):
        for child in node.children:
            yield from _find_public_functions(child, func_name)


def is_function_or_subroutine(node):
    return isinstance(
        node, (Fortran2003.Function_Subprogram, Fortran2003.Subroutine_Subprogram)
    )


def get_function_name(node):
    index = 0
    while isinstance(node.content[index], Fortran2003.Comment):
        index += 1
    func_name = str(node.content[index].get_name())
    return func_name


def get_source_code(node):
    return str(node)


def is_function_call(node, functions):
    return (
        isinstance(
            node, (Fortran2003.Call_Stmt, Fortran2003.Intrinsic_Function_Reference)
        )
        or isinstance(node, (Fortran2003.Part_Ref))
        and str(node.items[0]).lower() in functions
    )


def traverse_function_definitions(node):
    if is_function_or_subroutine(node):
        func_name = get_function_name(node)
        source_code = get_source_code(node)
        print("found function", func_name)
        yield (func_name, source_code)
    if "children" in dir(node):
        for child in node.children:
            yield from traverse_function_definitions(child)


def traverse_function_calls(node, functions, func_name=None):
    # NOTE: this will not find function calls defined outside of the module, if they are called inside a function.
    if is_function_or_subroutine(node):
        func_name = get_function_name(node)
    if is_function_call(node, functions):
        called = str(node.items[0]).lower()
        print("Called function ", called, " of type ", type(node))
        if func_name is not None:
            yield (func_name, called)
        else:
            yield ("EXTERNAL", called)
    if "children" in dir(node):
        for child in node.children:
            yield from traverse_function_calls(child, functions, func_name)


def get_dag(filename: str):
    """
    In this dependency graph, the nodes are functions and the edges are calls.
    An arrow means that the function at the tail of the arrow calls the function at the head of the arrow.

    Thus, a topological sort of this graph will give us an order in which we can compile the functions.
    """
    ast = _get_parse_tree(filename)
    dependencies = defaultdict(lambda: {"source": "", "calls": []})
    for func_name, source_code in traverse_function_definitions(ast):
        if source_code != "":
            dependencies[func_name]["source"] = source_code
    for func_name, call in traverse_function_calls(ast, dependencies):
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


def filter_for_dependencies(
    dag: nx.DiGraph, function_name: str
) -> list[tuple[str, parser_types.Dependency]]:
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

    result: list[tuple[str, parser_types.Dependency]] = [
        (key, dag.nodes.get(key, {"source": "not_found", "calls": []}))
        for key in sorted_keys
        if key in node_list
    ]

    return result
