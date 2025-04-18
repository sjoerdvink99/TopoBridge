# core/layout.py
import numpy as np
from sklearn.decomposition import PCA

class LayoutManager:
    def __init__(self, graph, embedding_manager):
        """Initialize layout manager.
        
        Args:
            graph: NetworkX graph
            embedding_manager: An initialized EmbeddingManager instance
        """
        self.graph = graph
        self.embedding_manager = embedding_manager
        self.layout_cache = {}
        self.pca_model = PCA(n_components=2, random_state=42)
        
    def compute_layouts(self):
        """Compute 2D layouts for all cached embeddings.
        
        Returns:
            Dictionary of layouts indexed by alpha values
        """
        self.layout_cache.clear()  # Clear previous layouts
        
        # Get all embeddings for PCA
        embeddings = list(self.embedding_manager.embedding_cache.values())
        combined_embeddings = np.vstack(embeddings)
        nodes = list(self.graph.nodes())
        
        # Fit PCA model
        self.pca_model.fit(combined_embeddings)
        
        # Transform embeddings to 2D
        for alpha, embedding in self.embedding_manager.embedding_cache.items():
            layout_2d = self.pca_model.transform(embedding)
            
            # Normalize for better visualization
            layout_2d = layout_2d - layout_2d.mean(axis=0)
            max_val = np.max(np.abs(layout_2d))
            if max_val > 0:
                layout_2d = layout_2d / max_val
                
            # Store position dictionary
            pos = {nodes[i]: (layout_2d[i, 0], layout_2d[i, 1]) for i in range(len(nodes))}
            self.layout_cache[alpha] = pos
        
        return self.layout_cache
    
    def get_layout(self, alpha):
        """Get the layout for a specific alpha value.
        
        Args:
            alpha: The alpha value
            
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        if not self.layout_cache:
            self.compute_layouts()
            
        # Get nearest alpha if exact value not available
        nearest_alpha = self.embedding_manager.get_nearest_alpha(alpha)
        return self.layout_cache[nearest_alpha]