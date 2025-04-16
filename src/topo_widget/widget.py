import importlib.metadata
import pathlib
import json
import numpy as np

import anywidget
import traitlets

import networkx as nx
from .embedding import FeatureWalk
from .core import EmbeddingManager, LayoutManager

try:
    __version__ = importlib.metadata.version("topo_widget")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


class Widget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "static" / "widget.js"
    _css = pathlib.Path(__file__).parent / "static" / "widget.css"
    
    value = traitlets.Int(0).tag(sync=True)
    alpha = traitlets.Float(0.5).tag(sync=True)
    selected_nodes = traitlets.List([]).tag(sync=True)
    node_positions = traitlets.Dict({}).tag(sync=True)
    selected_node_attributes = traitlets.Dict({}).tag(sync=True)
    all_node_attributes = traitlets.Dict({}).tag(sync=True)
    node_travel_distances = traitlets.Dict({}).tag(sync=True)
    graph_edges = traitlets.List([]).tag(sync=True)  # New traitlet for edges

    def __init__(self, graph: nx.Graph, **kwargs):
        super().__init__(**kwargs)
        self._graph = graph
        self.feature_walk = FeatureWalk(embedding_dim=16, compute_split=True)
        self.feature_walk.fit(self._graph)
        self.embedding_manager = EmbeddingManager(self.feature_walk)
        self.embedding_manager.precompute_embeddings()
        self.layout_manager = LayoutManager(self._graph, self.embedding_manager)
        
        # Initialize node positions and attributes immediately
        self._process_graph_attributes()
        self._update_node_positions(self.alpha)
        self._calculate_travel_distances()
        self._extract_graph_edges()  # Add this method call

    def _process_graph_attributes(self):
        all_attrs = {}
        for node_id, attrs in self._graph.nodes(data=True):
            str_id = str(node_id)
            all_attrs[str_id] = {}
            for key, value in attrs.items():
                try:
                    # Force numeric conversion for specific attributes
                    if key in ["age", "completion_p", "days_since_registration", "days_since_login", "gender", "public", "group"]:
                        try:
                            all_attrs[str_id][key] = float(value)
                            continue
                        except (ValueError, TypeError):
                            pass
                    
                    # Test JSON serialization for other attributes
                    json.dumps(value)
                    all_attrs[str_id][key] = value
                except (TypeError, OverflowError):
                    # If not serializable, convert to string
                    all_attrs[str_id][key] = str(value)
        
        self.all_node_attributes = all_attrs
    
    def _update_node_positions(self, alpha):
        """Compute node positions based on embeddings and alpha value"""
        # Get the layout for the current alpha
        positions = self.layout_manager.get_layout(alpha)
        
        # Convert positions to a dictionary for sending to frontend
        pos_dict = {}
        for node_id, pos in positions.items():
            pos_dict[str(node_id)] = {"x": float(pos[0]), "y": float(pos[1])}
        
        # Update the node_positions trait
        self.node_positions = pos_dict

    def _calculate_travel_distances(self):
        """Calculate how far each node travels from alpha=0 to alpha=1"""
        # Get positions at extreme values
        topo_positions = self.layout_manager.get_layout(0.0)  # alpha = 0
        feature_positions = self.layout_manager.get_layout(1.0)  # alpha = 1
        
        # Calculate Euclidean distance between positions for each node
        travel_dict = {}
        for node_id in self._graph.nodes():
            str_id = str(node_id)
            if str_id in topo_positions and str_id in feature_positions:
                pos1 = topo_positions[str_id]
                pos2 = feature_positions[str_id]
                
                # Calculate Euclidean distance
                distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                travel_dict[str_id] = float(distance)
        
        # Update the traitlet
        self.node_travel_distances = travel_dict

    def _extract_graph_edges(self):
        """Extract edges from the graph and store them in the graph_edges trait"""
        # Convert edges to a list of string pairs
        edge_list = []
        for u, v in self._graph.edges():
            edge_list.append([str(u), str(v)])
        
        self.graph_edges = edge_list

    @traitlets.observe('alpha')
    def _on_alpha_change(self, change):
        """Handle changes to the alpha value"""
        new_alpha = change['new']
        self._update_node_positions(new_alpha)
    
    @traitlets.observe('selected_nodes')
    def _on_selected_nodes_change(self, change):
        """Handle changes to the selected nodes"""
        selected = change.new
        print(f"Selected nodes: {selected}")
        
        # Extract node attributes for selected nodes
        attributes = {}
        for node_id in selected:
            # Skip if node doesn't exist in graph
            if node_id not in self._graph:
                continue
                
            # Get all attributes for this node
            node_attrs = self._graph.nodes[node_id]
            
            # Store attributes, converting any non-JSON serializable types to strings
            attributes[node_id] = {}
            for key, value in node_attrs.items():
                try:
                    # Force numeric conversion for specific attributes
                    if key in ["age", "completion_p", "days_since_registration", "days_since_login", "gender", "public"]:
                        try:
                            attributes[node_id][key] = float(value)
                            continue
                        except (ValueError, TypeError):
                            pass
                    
                    # Test JSON serialization for other attributes
                    import json
                    json.dumps(value)
                    attributes[node_id][key] = value
                except (TypeError, OverflowError):
                    # If not serializable, convert to string
                    attributes[node_id][key] = str(value)
        
        # Update the attributes trait
        self.selected_node_attributes = attributes
        print(f"Attribute count: {len(attributes)}")