# Source: https://github.com/benedekrozemberczki/karateclub/blob/master/karateclub/estimator.py#L12
import warnings
from typing import List
import re
import random
import numpy as np
import networkx as nx
from tqdm.auto import trange


class Estimator(object):
    """Estimator base class with constructor and public methods."""

    seed: int

    def __init__(self):
        """Creating an estimator."""

    def fit(self):
        """Fitting a model."""

    def get_embedding(self):
        """Getting the embeddings (graph or node level)."""

    def get_memberships(self):
        """Getting the membership dictionary."""

    def get_cluster_centers(self):
        """Getting the cluster centers."""

    def get_params(self):
        """Get parameter dictionary for this estimator.."""
        rx = re.compile(r"^\_")
        params = self.__dict__
        params = {key: params[key] for key in params if not rx.search(key)}
        return params

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _set_seed(self):
        """Creating the initial random seed."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def _ensure_walk_traversal_conditions(
        graph: nx.classes.graph.Graph,
    ) -> nx.classes.graph.Graph:
        """Ensure walk traversal conditions."""
        for node_index in trange(
            graph.number_of_nodes(),
            leave=False,
            disable=graph.number_of_nodes() < 10_000,
            desc="Checking main diagonal existance",
            dynamic_ncols=True,
        ):
            if not graph.has_edge(node_index, node_index):
                warnings.warn(
                    "Please do be advised that "
                    "the graph you have provided does not "
                    "contain (some) edges in the main "
                    "diagonal, for instance the self-loop "
                    f"constitued of ({node_index}, {node_index}). These selfloops "
                    "are necessary to ensure that the graph "
                    "is traversable, and for this reason we "
                    "create a copy of the graph and add therein "
                    "the missing edges. Since we are creating "
                    "a copy, this will immediately duplicate "
                    "the memory requirements. To avoid this double "
                    "allocation, you can provide the graph with the selfloops."
                )

                graph = graph.copy()
                graph.add_edges_from(
                    (
                        (index, index)
                        for index in range(graph.number_of_nodes())
                        if not graph.has_edge(index, index)
                    )
                )
                break

        return graph

    @staticmethod
    def _check_indexing(graph: nx.classes.graph.Graph):
        """Checking the consecutive numeric indexing."""
        numeric_indices = [index for index in range(graph.number_of_nodes())]
        node_indices = sorted([node for node in graph.nodes()])

        assert numeric_indices == node_indices, "The node indexing is wrong."

    def _check_graph(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """Check the Karate Club assumptions about the graph."""
        self._check_indexing(graph)
        graph = self._ensure_walk_traversal_conditions(graph)

        return graph

    def _check_graphs(self, graphs: List[nx.classes.graph.Graph]):
        """Check the Karate Club assumptions for a list of graphs."""
        graphs = [self._check_graph(graph) for graph in graphs]

        return graphs