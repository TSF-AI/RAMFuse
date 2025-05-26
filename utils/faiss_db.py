import faiss
import numpy as np
import torch
from typing import List, Tuple

class TemporalKnowledgeBase:
    """Dynamic Temporal Knowledge Base (TK) Manager
    
    Implements:
    - Hierarchical indexing for efficient retrieval (HNSW)
    - Incremental updates with aging mechanism
    - Cross-domain embedding storage
    
    Args:
        embedding_dim (int): Dimension of fused embeddings (tk_dim in paper)
        max_entries (int): Maximum stored entries (for memory management)
        aging_threshold (int): Days before entry expiration (default: 90)
        hnsw_ef (int): HNSW search depth parameter (default: 32)
        
    """
    def __init__(self,
                 embedding_dim: int = 512,
                 max_entries: int = 1000000,
                 aging_threshold: int = 90,
                 hnsw_ef: int = 32):
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries
        self.hnsw_ef = hnsw_ef
        self.aging_threshold = aging_threshold * 86400  # Convert days to seconds
        
        # Core storage components
        self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
        self.labels = []
        self.timestamps = []
        
        # Initialize HNSW parameters (paper defaults)
        self.index.hnsw.efSearch = hnsw_ef
        self.index.hnsw.efConstruction = 200

    def add_entries(self,
                  embeddings: np.ndarray,
                  labels: List[int],
                  timestamps: List[float]):
        """Add entries to TK with aging control
        
        Args:
            embeddings: Normalized embeddings shape (N, tk_dim)
            labels: Trend labels (0/1) corresponding to embeddings
            timestamps: Unix timestamps for aging calculation
        """
        # Validate input dimensions
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch: Expected {self.embedding_dim}, Got {embeddings.shape[1]}")
            
        # Apply aging-based filtering
        valid_mask = self._filter_by_aging(timestamps)
        
        # Add valid entries
        self.index.add(embeddings[valid_mask])
        self.labels.extend(np.array(labels)[valid_mask].tolist())
        self.timestamps.extend(np.array(timestamps)[valid_mask].tolist())
        
        # Enforce capacity limits
        self._enforce_capacity()

    def retrieve(self,
               query: np.ndarray,
               top_k: int = 5
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Similarity-based retrieval 
        
        Args:
            query: Normalized query embeddings (N, tk_dim)
            top_k: Number of neighbors to retrieve (N in paper)
        
        Returns:
            (distances, labels, timestamps)
        """
        # FAISS requires float32 inputs
        query = query.astype('float32')
        
        # Search with current HNSW parameters
        distances, indices = self.index.search(query, top_k)
        
        # Map to labels and timestamps
        retrieved_labels = np.array([self.labels[i] for i in indices.flatten()]).reshape(-1, top_k)
        retrieved_times = np.array([self.timestamps[i] for i in indices.flatten()]).reshape(-1, top_k)
        
        return distances, retrieved_labels, retrieved_times

    def incremental_update(self,
                          new_embeddings: np.ndarray,
                          new_labels: List[int],
                          new_timestamps: List[float]):
        """Incremental update with index rebuild """
        # Add new entries
        self.add_entries(new_embeddings, new_labels, new_timestamps)
        
        # Rebuild index if size exceeds threshold
        if len(self.labels) > self.max_entries * 0.9:
            self._rebuild_index()

    def _filter_by_aging(self,
                       timestamps: List[float]
                      ) -> np.ndarray:
        """Aging mechanism"""
        current_time = time.time()
        return np.array([(current_time - ts) < self.aging_threshold for ts in timestamps])

    def _enforce_capacity(self):
        """Remove oldest entries when exceeding capacity"""
        if len(self.labels) > self.max_entries:
            remove_count = len(self.labels) - self.max_entries
            self.labels = self.labels[remove_count:]
            self.timestamps = self.timestamps[remove_count:]
            self._rebuild_index()

    def _rebuild_index(self):
        """Full index reconstruction for consistency"""
        # Rebuild from current embeddings
        all_embeddings = self.index.reconstruct_n(0, len(self.labels))
        new_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        new_index.hnsw.efSearch = self.hnsw_ef
        new_index.add(all_embeddings)
        self.index = new_index

    @staticmethod
    def normalize_embeddings(embeddings: torch.Tensor) -> np.ndarray:
        """L2 Normalization """
        return F.normalize(embeddings, p=2, dim=-1).cpu().numpy().astype('float32')
