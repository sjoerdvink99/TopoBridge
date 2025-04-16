# core/data_processing.py
import pandas as pd

def prepare_node_data(graph, positions):
    """Prepare node data for visualization.
    
    Args:
        graph: NetworkX graph
        positions: Dictionary mapping node IDs to positions
        
    Returns:
        DataFrame containing node data with attributes and positions
    """
    nodes = list(graph.nodes())
    
    # Gather node data with attributes
    node_data = []
    for n in nodes:
        entry = {'node': str(n), 'x': positions[n][0], 'y': positions[n][1]}
        
        # Add node attributes, converting complex types to strings
        entry.update({
            k: str(v) if isinstance(v, (list, dict)) else v 
            for k, v in graph.nodes[n].items()
        })
        
        node_data.append(entry)
        
    return pd.DataFrame(node_data)

def get_attribute_columns(df, exclude_cols=None):
    """Get categorical and numerical attribute columns from DataFrame.
    
    Args:
        df: DataFrame with node data
        exclude_cols: Columns to exclude from attribute analysis
        
    Returns:
        Tuple of (categorical_columns, numerical_columns)
    """
    if exclude_cols is None:
        exclude_cols = ['node', 'x', 'y', 'selected']
        
    # Get categorical attributes (object dtype or few unique values)
    cat_cols = [col for col in df.columns 
               if col not in exclude_cols
               and (pd.api.types.is_object_dtype(df[col]) or df[col].nunique() < 10)]
    
    # Get numeric attributes (excluding those already identified as categorical)
    num_cols = [col for col in df.columns 
               if col not in exclude_cols + cat_cols
               and pd.api.types.is_numeric_dtype(df[col])]
    
    return cat_cols, num_cols