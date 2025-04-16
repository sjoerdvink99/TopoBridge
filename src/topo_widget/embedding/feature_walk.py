import math
import numpy as np
import pandas as pd
import networkx as nx
import logging
from sklearn.neighbors import BallTree
from typing import Optional, List, Dict, Any
from .walklets import Walklets
from .node2vec import Node2Vec
from ..utils.helpers import preprocess_graph_attributes

class FeatureWalk:
    """
    FeatureWalk computes graph embeddings by generating feature graphs and walking on them.
    
    Attributes:
        embedding_dim (int): Dimensionality of the embeddings.
        random_seed (int): Random seed for reproducibility.
        structure_embedding (np.ndarray): Embedding based on the graph structure.
        feature_embedding (np.ndarray): Embedding based on the attributes.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        compute_split: bool = True,
        verbose: bool = False,
        walklets_params: Optional[Dict[str, Any]] = None,
        node2vec_params: Optional[Dict[str, Any]] = None,
        combine_method: str = "concatenate",
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initializes the FeatureWalk class with data, dimensionality, and other configurations.
        
        Args:
            embedding_dim: Dimensionality of the embeddings.
            compute_split: Whether to compute structure and feature embeddings separately.
            verbose: Whether to print verbose logging information.
            walklets_params: Parameters for the Walklets model.
            node2vec_params: Parameters for the Node2Vec model (alternative embedding method).
            combine_method: Method to combine structure and feature embeddings.
            logger: Logger instance.
        """
        self.compute_split = compute_split
        self.embedding_dim = embedding_dim
        self.random_seed = 42
        self.combine_method = combine_method
        
        # Default parameters for Walklets
        self.walklets_params = {
            "dimensions": self.embedding_dim,
            "seed": self.random_seed,
            "window_size": 5,
            "epochs": 10,
            "walk_number": 10,
            "walk_length": 80
        }
        
        # Default parameters for Node2Vec
        self.node2vec_params = {
            "dimensions": self.embedding_dim,
            "seed": self.random_seed,
            "walk_number": 10,
            "walk_length": 80,
            "p": 1.0,
            "q": 1.0
        }
        
        # Override defaults with user-provided parameters
        if walklets_params:
            self.walklets_params.update(walklets_params)
        if node2vec_params:
            self.node2vec_params.update(node2vec_params)
            
        self.logger = logger or logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        
        # Initialize embeddings
        self.structure_embedding = None
        self.feature_embedding = None
        self.graph = None
        self.scaled_attributes = None

    def fit(self, graph: nx.Graph, **kwargs) -> None:
        """Runs the embedding process.
        
        Args:
            graph: Input graph with node attributes.
            **kwargs: Additional keyword arguments.
        """
        self.graph = graph
        
        # Check if graph has node attributes - fixed to handle empty attribute case
        node_attrs = list(graph.nodes(data=True))
        if not node_attrs or not any(attr for _, attr in node_attrs):
            self.logger.warning("Graph has no node attributes. Feature embedding will be ineffective.")
        
        self.structure_embedding = self.generate_graph_embeddings(self.graph)
        self.feature_embedding = self.generate_feature_embedding()
        
        self.logger.info(f"Generated embeddings: structure {self.structure_embedding.shape}, "
                         f"feature {self.feature_embedding.shape}")

    def fit_transform(self, graph: nx.Graph, alpha: float = 0.5, 
                    selected_features: Optional[List[int]] = None) -> np.ndarray:
        """Fits the model and returns the combined embedding.
        
        Args:
            graph: Input graph with node attributes.
            alpha: Weight for structure embedding (0-1).
            selected_features: Specific feature indices to use.
            
        Returns:
            Combined embedding matrix.
        """
        self.fit(graph)
        return self.get_embedding(alpha, selected_features)

    def generate_feature_embedding(self) -> np.ndarray:
        """Generates embeddings based on feature similarity using BallTree for nearest neighbors search.
        
        Returns:
            Feature embedding matrix.
        """
        if self.graph is None:
            raise ValueError("Graph must be set before generating feature embeddings.")
            
        self.scaled_attributes = preprocess_graph_attributes(self.graph)
        
        if self.scaled_attributes.shape[1] == 0:
            self.logger.warning("No node attributes found or all attributes were filtered out.")
            # Return zero embedding matrix as fallback
            return np.zeros((self.graph.number_of_nodes(), self.embedding_dim))

        # Build a BallTree for the scaled attributes
        tree = BallTree(self.scaled_attributes, leaf_size=20)

        # Adaptive k-neighbors selection based on graph size
        k_neighbors = min(
            max(5, round(math.sqrt(self.scaled_attributes.shape[0]) / 3)), 
            self.scaled_attributes.shape[0] // 2
        )
        
        distances, indices = tree.query(self.scaled_attributes, k=k_neighbors)

        # Apply distance transformation to create meaningful edge weights
        # Keep first column as is (self-connections), transform others
        if distances.shape[1] > 1:
            max_dist = np.max(distances[:, 1:])
            min_dist = np.min(distances[:, 1:])
            if max_dist > min_dist:
                distances[:, 1:] = 1 - ((distances[:, 1:] - min_dist) / (max_dist - min_dist))
            else:
                distances[:, 1:] = 1.0
                
        # Create weighted kNN graph
        knn_graph = np.zeros((self.scaled_attributes.shape[0], self.scaled_attributes.shape[0]))
        for i in range(self.scaled_attributes.shape[0]):
            knn_graph[i, indices[i]] = distances[i]

        feature_graph = nx.from_numpy_array(knn_graph)
        feature_graph = feature_graph.to_directed()  # Ensure graph is directed for asymmetric distances
        
        # Add self-loops to ensure walk traversability
        for i in range(feature_graph.number_of_nodes()):
            feature_graph.add_edge(i, i, weight=1.0)
            
        feature_embedding = self.generate_graph_embeddings(feature_graph)
        return feature_embedding

    def generate_graph_embeddings(self, graph: nx.Graph) -> np.ndarray:
        """Generates embeddings using the selected embedding model for a given graph.
        
        Args:
            graph: Input graph.
            
        Returns:
            Embedding matrix.
        """
        # Try using Walklets first
        try:
            model = Walklets(**self.walklets_params)
            model.fit(graph)
            embedding = model.get_embedding()
            
            # Check if embedding has reasonable values
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                raise ValueError("Walklets produced NaN or Inf values")
                
            return embedding
            
        except Exception as e:
            self.logger.warning(f"Walklets embedding failed: {e}. Trying Node2Vec as fallback.")
            
            # Fallback to Node2Vec
            try:
                model = Node2Vec(**self.node2vec_params)
                model.fit(graph)
                embedding = model.get_embedding()
                
                if np.isnan(embedding).any() or np.isinf(embedding).any():
                    raise ValueError("Node2Vec produced NaN or Inf values")
                    
                return embedding
                
            except Exception as e2:
                self.logger.error(f"Node2Vec embedding also failed: {e2}")
                # Return random embedding as last resort
                return np.random.normal(size=(graph.number_of_nodes(), self.embedding_dim))

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Safely normalize an embedding matrix to [0,1] range.
        
        Args:
            embedding: Input embedding matrix.
            
        Returns:
            Normalized embedding matrix.
        """
        if embedding is None:
            return None
            
        # Handle edge cases
        if np.all(embedding == embedding[0, 0]):
            # All values are the same, return zeros
            return np.zeros_like(embedding)
            
        min_vals = np.min(embedding, axis=0, keepdims=True)
        max_vals = np.max(embedding, axis=0, keepdims=True)
        
        # Prevent division by zero
        denom = np.maximum(max_vals - min_vals, 1e-10)
        normalized = (embedding - min_vals) / denom
        
        return normalized

    def get_embedding(self, alpha: float = 0.5, selected_features: Optional[List[int]] = None) -> np.ndarray:
        """Get combined embedding with controllable balance between structure and features.
        
        Args:
            alpha: Weight for structure embedding (0-1).
            selected_features: Specific feature indices to use.
            
        Returns:
            Combined embedding matrix.
        """
        if self.structure_embedding is None or self.feature_embedding is None:
            raise ValueError("Both structure_embedding and feature_embedding must be computed before combining.")
            
        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

        # Normalize embeddings
        structure_norm = self.normalize_embedding(self.structure_embedding)
        feature_norm = self.normalize_embedding(self.feature_embedding)

        # Filter selected features if specified
        if selected_features is not None:
            selected_embeddings = feature_norm[:, selected_features] if selected_features else feature_norm
            feature_norm = selected_embeddings

        if self.combine_method == "concatenate":
            # Concatenate and scale by alpha
            structure_scaled = structure_norm * alpha
            feature_scaled = feature_norm * (1 - alpha)
            combined = np.hstack((structure_scaled, feature_scaled))
            
        elif self.combine_method == "hadamard":
            # Element-wise product (requires same dimensionality)
            if structure_norm.shape[1] == feature_norm.shape[1]:
                # Weighted geometric mean
                combined = np.power(structure_norm, alpha) * np.power(feature_norm, (1-alpha))
            else:
                # Default to weighted sum if dimensions don't match
                self.logger.warning("Hadamard product requires same dimensionality. Using weighted sum instead.")
                from sklearn.decomposition import PCA
                pca = PCA(n_components=structure_norm.shape[1])
                feature_resized = pca.fit_transform(feature_norm)
                feature_resized = self.normalize_embedding(feature_resized)
                combined = alpha * structure_norm + (1 - alpha) * feature_resized
        else:
            self.logger.warning(f"Unknown combine method: {self.combine_method}, using weighted_sum")
            # Default to weighted sum
            if structure_norm.shape[1] == feature_norm.shape[1]:
                combined = alpha * structure_norm + (1 - alpha) * feature_norm
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=structure_norm.shape[1])
                feature_resized = pca.fit_transform(feature_norm)
                feature_resized = self.normalize_embedding(feature_resized)
                combined = alpha * structure_norm + (1 - alpha) * feature_resized
        
        return combined

    def save_embeddings(self, save_path: str, name: str = "") -> None:
        """Saves structure and feature embeddings to CSV files.
        
        Args:
            save_path: Directory to save embeddings.
            name: Optional name prefix for the saved files.
        """
        if self.structure_embedding is not None:
            pd.DataFrame(self.structure_embedding).to_csv(f"{save_path}/structure_embedding_{name}.csv")
        
        if self.feature_embedding is not None:
            pd.DataFrame(self.feature_embedding).to_csv(f"{save_path}/feature_embedding_{name}.csv")
            
        combined = self.get_embedding()
        if combined is not None:
            pd.DataFrame(combined).to_csv(f"{save_path}/combined_embedding_{name}.csv")
            
        self.logger.info(f"Embeddings saved to {save_path}")