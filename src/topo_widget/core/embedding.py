import numpy as np

class EmbeddingManager:
    def __init__(self, g2dm_model):
        """Initialize the embedding manager with a Graph2DMatrix model.
        
        Args:
            g2dm_model: An initialized Graph2DMatrix model
        """
        self.g2dm = g2dm_model
        self.embedding_cache = {}
        self.sample_alphas = None
        
    def precompute_embeddings(self, alpha_range=(0.0, 1.0), num_samples=41):
        """Precompute embeddings for a range of alpha values.
        
        Args:
            alpha_range: Tuple with (min_alpha, max_alpha)
            num_samples: Number of alpha values to sample
            
        Returns:
            The cache of computed embeddings
        """
        self.sample_alphas = np.linspace(alpha_range[0], alpha_range[1], num_samples)
        
        for alpha in self.sample_alphas:
            embedding = self.g2dm.get_embedding(alpha=alpha)
            self.embedding_cache[alpha] = embedding
            
        return self.embedding_cache
    
    def get_embedding(self, alpha):
        """Get the embedding for a specific alpha value.
        
        If the exact alpha is not in the cache, return the nearest one.
        
        Args:
            alpha: The alpha value to retrieve
            
        Returns:
            The embedding matrix
        """
        if alpha in self.embedding_cache:
            return self.embedding_cache[alpha]
        
        # Get nearest cached alpha
        nearest_alpha = self.get_nearest_alpha(alpha)
        return self.embedding_cache[nearest_alpha]
    
    def get_nearest_alpha(self, alpha):
        """Get the nearest precomputed alpha value.
        
        Args:
            alpha: The target alpha value
            
        Returns:
            The nearest available alpha
        """
        if self.sample_alphas is None:
            raise ValueError("No embeddings have been precomputed yet")
            
        return min(self.sample_alphas, key=lambda x: abs(x - alpha))