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
    # In this example, we're getting all the predecessors to the function "hybrid" in the graph, using fparser.

    node = "hybrid"

    # Get the DAG
    dag = parser.get_dag("./tests/PhotosynthesisMod.f90")

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

    draw_dag_and_save(new_dag, "node_predecessors.png")
