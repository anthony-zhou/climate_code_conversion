import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import pprint

import translation.modules.ast.parser as parser


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

    # sorted_functions = parser.get_sorted_functions(
    #     # "./examples/daylength_2/fortran/DaylengthMod.f90"
    #     "../../../archive/examples/photosynthesis/PhotosynthesisMod.f90"
    # )

    # for func_name, func in sorted_functions:
    #     print(func_name)
    #     # print(func["source"])

    # dag = parser.get_dag("../../../archive/examples/photosynthesis/PhotosynthesisMod.f90")

    # BEGIN: 6j8d5k3h7f4g
    node = "hybrid"

    # Get the DAG
    dag = parser.get_dag(
        "../../../archive/examples/photosynthesis/PhotosynthesisMod.f90"
    )

    # Create a new graph with only the predecessors of node
    predecessors = list(dag.predecessors(node))
    predecessors.append(node)
    current_predecessors = predecessors.copy()
    while current_predecessors:
        pred = current_predecessors.pop()
        new_predecessors = dag.predecessors(pred)
        predecessors.extend(new_predecessors)
        current_predecessors.extend(new_predecessors)

    new_dag = dag.subgraph(predecessors)

    # Draw the new DAG
    draw_dag_and_save(new_dag, "node_predecessors.png")
    # END: 6j8d5k3h7f4g

    # dag = p.get_dag()

    # draw_dag_and_save(
    #     dag,
    #     "dag.png",
    # )

    # draw_dag_interactive(dag, "./output/dag2.html")
