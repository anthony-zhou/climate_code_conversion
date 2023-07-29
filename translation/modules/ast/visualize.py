import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network


def draw_dag_and_save(dag, filename):
    A = nx.nx_agraph.to_agraph(dag)
    G = nx.DiGraph(A)
    layout = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")

    nx.draw(dag, pos=layout, with_labels=True, arrows=True, node_color="skyblue")  # type: ignore
    plt.margins(0.20)
    plt.savefig(filename)


def draw_dag_interactive(dag, outfile):
    net = Network(notebook=True, directed=True)
    net.from_nx(dag)
    net.show_buttons(filter_=["physics"])
    net.toggle_physics(True)
    net.show(outfile)


if __name__ == "__main__":
    pass
    # Generate a DAG from the dependencies

    # pp = pprint.PrettyPrinter(indent=4)

    # sorted_functions = get_sorted_functions(
    #     # "./examples/daylength_2/fortran/DaylengthMod.f90"
    #     "./ast/tests/SampleMod.f90"
    # )

    # for func_name, func in sorted_functions:
    #     print(func_name)
    #     if func_name == "ci_func":
    #         print(func["source"])
    #     # print(func["source"])

    # dag = _dependencies_to_dag(_find_dependencies("./tests/SampleMod.f90"))

    # draw_dag_and_save(
    #     dag,
    #     "dag.png",
    # )

    # draw_dag_interactive(dag, "./output/dag2.html")
