# utils/helpers.py
import time
import numpy as np
import altair as alt
import pandas as pd
from IPython.display import display
import networkx as nx
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def preprocess_graph_attributes(graph: nx.Graph) -> np.ndarray:
    """Preprocesses node attributes in the graph, encoding and normalizing them."""
    attributes_dict = {node: graph.nodes[node] for node in graph.nodes()}
    attributes_df = pd.DataFrame.from_dict(attributes_dict, orient='index')

    # Separate attribute types
    numerical_attributes = attributes_df.select_dtypes(include=np.number)
    categorical_attributes = attributes_df.select_dtypes(include='object')
    boolean_attributes = attributes_df.select_dtypes(include='bool')

    # Normalize numerical attributes
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_numerical = scaler.fit_transform(numerical_attributes)

    # Encode categorical attributes using OneHotEncoder
    if not categorical_attributes.empty:
        encoder = OneHotEncoder()
        encoded_categorical = encoder.fit_transform(categorical_attributes)
    else:
        encoded_categorical = np.empty((attributes_df.shape[0], 0))

    # Convert boolean attributes to integers (0 for False, 1 for True)
    if not boolean_attributes.empty:
        boolean_attributes = boolean_attributes.astype(int).values
    else:
        boolean_attributes = np.empty((attributes_df.shape[0], 0))

    # Combine all processed attributes
    processed_attributes = np.hstack((scaled_numerical, encoded_categorical, boolean_attributes))

    return processed_attributes

def print_graph_summary(graph):
    """Print a summary of the graph structure and attributes.
    
    Args:
        graph: NetworkX graph
    """
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Check for node attributes
    if graph.nodes():
        sample_node = list(graph.nodes())[0]
        attrs = list(graph.nodes[sample_node].keys())
        print(f"Node attributes: {attrs}")
    
    # Get graph type and density
    print(f"Graph type: {type(graph).__name__}")
    if graph.number_of_nodes() > 0:
        density = graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2)
        print(f"Graph density: {density:.4f}")
    
    # Check for connected components
    import networkx as nx
    if not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        print(f"Graph has {len(components)} connected components")
        print(f"Largest component size: {len(max(components, key=len))}")

def timer(func):
    """Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that prints execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def test_altair():
    """Test if Altair is properly configured.
    
    Creates and displays a simple chart to verify Altair is working.
    """
    print(f"Altair version: {alt.__version__}")
    
    # Create test data
    test_df = pd.DataFrame({
        'x': range(10),
        'y': range(10),
        'category': ['A', 'B'] * 5
    })
    
    # Create and display a simple chart
    chart = alt.Chart(test_df).mark_circle().encode(
        x='x',
        y='y',
        color='category'
    )
    
    display(chart)
    print("If you can see a chart above, Altair is working correctly.")