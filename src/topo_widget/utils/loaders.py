# data/loaders.py
import os
import networkx as nx
import pandas as pd
import json

def load_json_graph(file_path):
    """Load a graph from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        NetworkX graph
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Load the JSON data from the file
    with open(file_path, 'r') as f:
        graph_data = json.load(f)
    
    # Create a new graph
    graph = nx.Graph()
    
    # Add nodes with attributes
    for index, node in enumerate(graph_data['nodes']):
        # Use the index as the node ID and add attributes
        node_attrs = {k: v for k, v in node.items() if k != 'id'}
        graph.add_node(index, **node_attrs)
    
    # Add edges
    for edge in graph_data['edges']:
        # Use the index of the source and target nodes
        graph.add_edge(edge['source'], edge['target'])
        
    return graph

def load_edgelist_with_attributes(edge_path, attr_path=None):
    """Load a graph from an edge list file with optional attributes.
    
    Args:
        edge_path: Path to the edge list file
        attr_path: Optional path to the attributes file
        
    Returns:
        NetworkX graph
    """
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Edge file not found: {edge_path}")
    
    # Create an empty graph
    graph = nx.Graph()
    
    # Read the edge list from the file
    with open(edge_path, 'r') as f:
        for line in f:
            edge = line.strip().split()
            if len(edge) >= 2:  # Ensure that the line contains at least two nodes
                node1, node2 = edge[0], edge[1]
                graph.add_edge(node1, node2)
    
    # Add attributes if provided
    if attr_path and os.path.exists(attr_path):
        try:
            # Try to load attributes as CSV
            attributes_df = pd.read_csv(attr_path, header=0)
            
            # Ensure ID column is string for compatibility
            id_col = attributes_df.columns[0]
            attributes_df[id_col] = attributes_df[id_col].astype(str)
            
            # Convert to dictionary and add to graph
            attributes_dict = attributes_df.set_index(id_col).T.to_dict()
            nx.set_node_attributes(graph, attributes_dict)
        except Exception as e:
            print(f"Warning: Could not load attributes: {e}")
    
    return graph