import numpy as np
import networkx as nx
import logging
from typing import Optional, List, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .node2vec import Node2Vec
from ..utils.helpers import preprocess_graph_attributes

class FeatureWalk:
    """
    FeatureWalk: Simple yet powerful embedding algorithm that combines topology and attribute information.
    
    Creates separate embeddings for graph structure and node attributes, then combines them
    with a user-controllable mixing parameter (alpha).
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        random_seed: int = 42,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize FeatureWalk with basic parameters.
        
        Args:
            embedding_dim: Target dimensionality for all embeddings
            combine_method: How to combine structure and attribute embeddings
            random_seed: For reproducibility
            verbose: Whether to log detailed information
            logger: Optional custom logger
        """
        self.embedding_dim = embedding_dim
        self.random_seed = random_seed
        
        self.logger = logger or logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        
        # Will be populated during fit
        self.graph = None
        self.structure_embedding = None
        self.attribute_embedding = None
        
    def fit(self, graph: nx.Graph) -> 'FeatureWalk':
        """Compute both topology and attribute embeddings.
        
        Args:
            graph: Input graph with node attributes
            
        Returns:
            Self, for method chaining
        """
        self.logger.info(f"Fitting FeatureWalk on graph with {graph.number_of_nodes()} nodes")
        self.graph = graph
        self._compute_topology_embedding()
        self._compute_attribute_embedding()
        return self
        
    def _compute_topology_embedding(self) -> None:
        """Compute embedding based on graph topology using Node2Vec."""
        self.logger.info("Computing topology embedding with Node2Vec")
        
        try:
            model = Node2Vec(
                dimensions=self.embedding_dim, 
                walk_length=80, 
                walk_number=10,
                p=1.0, 
                q=1.0, 
                seed=self.random_seed
            )
            model.fit(self.graph)
            self.structure_embedding = model.get_embedding()
            
            # Check for invalid values
            if np.isnan(self.structure_embedding).any() or np.isinf(self.structure_embedding).any():
                raise ValueError("Node2Vec produced invalid embedding values")
                
        except Exception as e:
            self.logger.error(f"Error computing topology embedding: {e}")
    
    def _compute_attribute_embedding(self) -> None:
        """Compute embedding based on node attributes using PCA."""
        self.logger.info("Computing attribute embedding with PCA")

        node_features = preprocess_graph_attributes(self.graph)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(node_features)
        
        # Handle dimensionality
        if scaled_features.shape[1] > self.embedding_dim:
            # Reduce dimensions with PCA
            pca = PCA(n_components=self.embedding_dim, random_state=self.random_seed)
            self.attribute_embedding = pca.fit_transform(scaled_features)
            explained = sum(pca.explained_variance_ratio_)
            self.logger.info(f"Reduced {scaled_features.shape[1]} features to {self.embedding_dim} dimensions "
                          f"(explained variance: {explained:.2f})")
        else:
            # Pad with zeros if needed
            self.attribute_embedding = np.zeros((scaled_features.shape[0], self.embedding_dim))
            self.attribute_embedding[:, :scaled_features.shape[1]] = scaled_features
            
        # Normalize to similar scale as structure embedding
        self.attribute_embedding = self._normalize_embedding(self.attribute_embedding)
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding values to [0,1] range for each dimension."""
        # Handle edge case of constant values
        if np.all(embedding == embedding[0, 0]):
            return np.zeros_like(embedding)
            
        # Min-max scaling by column
        min_vals = np.min(embedding, axis=0, keepdims=True)
        max_vals = np.max(embedding, axis=0, keepdims=True)
        range_vals = np.maximum(max_vals - min_vals, 1e-8)  # Avoid division by zero
        
        return (embedding - min_vals) / range_vals
        
    def get_embedding(self, alpha: float = 0.5) -> np.ndarray:
        """Get combined embedding with user-controlled mixing parameter.
        
        Args:
            alpha: Weight for topology embedding (0-1)
                  0 = pure attribute embedding
                  1 = pure topology embedding
        
        Returns:
            Combined embedding matrix
        """
        if self.structure_embedding is None or self.attribute_embedding is None:
            raise ValueError("Must call fit() before get_embedding()")
            
        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
        
        # Normalize both embeddings
        topology_norm = self._normalize_embedding(self.structure_embedding)
        attribute_norm = self._normalize_embedding(self.attribute_embedding)
        
        topology_part = topology_norm * alpha
        attribute_part = attribute_norm * (1 - alpha)
        return np.hstack((topology_part, attribute_part))
    
    def fit_transform(self, graph: nx.Graph, alpha: float = 0.5) -> np.ndarray:
        """Convenience method to fit and get embedding in one call.
        
        Args:
            graph: Input graph with node attributes
            alpha: Weight for topology embedding (0-1)
            
        Returns:
            Combined embedding matrix
        """
        return self.fit(graph).get_embedding(alpha)