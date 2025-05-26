"""
RAMFuse Inference Pipeline 
"""

import torch
import faiss
import numpy as np
from typing import Optional, Tuple
from models.ramfuse import RAMFuse
from utils.faiss_db import TemporalKnowledgeBase

class RAMFuseInferencer:
    """Retrieval-Augmented Forecasting Inference Engine
    
    Args:
        model_path (str): Path to trained model weights
        tk_index_path (str): Path to FAISS TK index
        config (dict): Model configuration dictionary
        device (str): Inference device ('cuda' or 'cpu')
    
    Usage:
        inferencer = RAMFuseInferencer("best_model.pth", "tk.index", config)
        forecast, explanations = inferencer.predict(X_test, positions_test)
    """
    
    def __init__(self,
                 model_path: str,
                 tk_index_path: str,
                 config: dict,
                 device: str = "cuda"):
        self.device = torch.device(device)
        self.config = config
        
        # Load model architecture
        self.model = RAMFuse(
            seq_len=config['model']['seq_len'],
            n_channels=config['model']['n_channels'],
            tk_dim=config['model']['tk_dim'],
            num_retrievals=config['model']['num_retrievals'],
            temp=config['model']['temperature']
        ).to(self.device)
        
        # Load trained weights
        self._load_model_weights(model_path)
        self.model.eval()
        
        # Initialize knowledge base
        self.tk = TemporalKnowledgeBase(
            embedding_dim=config['model']['tk_dim'],
            max_entries=config['data']['max_tk_entries'],
            aging_threshold=config['model']['aging_threshold']
        )
        self._load_tk_index(tk_index_path)

    def _load_model_weights(self, path: str):
        """Load pretrained model weights with device mapping"""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Loaded model weights from {path}")

    def _load_tk_index(self, path: str):
        """Load FAISS index with safety checks"""
        if not faiss.read_index(path):
            raise FileNotFoundError(f"TK index file {path} not found")
        self.tk.index = faiss.read_index(path)
        print(f"Loaded TK index with {self.tk.index.ntotal} entries")

    def _preprocess_input(self,
                        X: np.ndarray,
                        positions: Optional[np.ndarray] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input Preprocessing Pipeline
        
        Args:
            X: Raw time series (N, L, D)
            positions: Spatial coordinates (N, H, W, 2)
        
        Returns:
            X_tensor: Normalized tensor (N, L, D)
            pos_tensor: Position tensor (N, H, W, 2)
        """
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Generate default positions if not provided
        if positions is None:
            H, W = self.config['model']['seq_len'], self.config['model']['seq_len']
            positions = np.stack(np.mgrid[:H, :W], -1).astype(np.float32)
            positions = np.repeat(positions[np.newaxis], len(X), axis=0)
        
        pos_tensor = torch.FloatTensor(positions).to(self.device)
        
        return X_tensor, pos_tensor

    def _postprocess_output(self,
                          predictions: torch.Tensor,
                          attention_weights: torch.Tensor
                         ) -> Tuple[np.ndarray, dict]:
        """Output Postprocessing
        
        Returns:
            forecasts: Denormalized predictions (N, T)
            explanations: Dictionary with attention weights and retrieval info
        """
        # Convert to numpy (assuming no denormalization needed)
        forecasts = predictions.cpu().detach().numpy()
        
        # Prepare explanation artifacts
        explanations = {
            'attention_weights': attention_weights.cpu().numpy(),
            'retrieved_count': self.tk.index.ntotal
        }
        
        return forecasts, explanations

    def predict(self,
              X: np.ndarray,
              positions: Optional[np.ndarray] = None,
              batch_size: int = 32
             ) -> Tuple[np.ndarray, dict]:
        """End-to-End Prediction Pipeline
        
        Args:
            X: Input time series (N, L, D)
            positions: Spatial coordinates (N, H, W, 2)
            batch_size: Batch size for inference
        
        Returns:
            predictions: Forecasted values (N, T)
            explanations: Interpretation artifacts
        """
        # Preprocessing
        X_tensor, pos_tensor = self._preprocess_input(X, positions)
        
        # Batch inference
        predictions = []
        attentions = []
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_pos = pos_tensor[i:i+batch_size]
                
                preds, attn = self.model(batch_X, batch_pos)
                
                predictions.append(preds)
                attentions.append(attn)
        
        # Postprocessing
        predictions = torch.cat(predictions)
        attentions = torch.cat(attentions)
        
        return self._postprocess_output(predictions, attentions)

    def explain_prediction(self,
                         X: np.ndarray,
                         positions: np.ndarray,
                         top_k: int = 3
                        ) -> dict:
        """Enhanced Prediction Explanation 
        
        Returns:
            explanation: {
                'causal_narrative': generated text,
                'retrieved_patterns': top-K similar patterns,
                'spatial_heatmap': attention weights visualization data
            }
        """
        # Single sample processing
        X_tensor, pos_tensor = self._preprocess_input(X[np.newaxis], positions[np.newaxis])
        
        with torch.no_grad():
            # Get full model outputs
            _, attn_weights = self.model(X_tensor, pos_tensor)
            
            # Generate causal text
            gaf_images = self.model.gaf(X_tensor)
            _, narratives = self.model.lvlm(gaf_images)
            
            # Retrieve similar patterns
            query = self.model._encode_multimodal(X_tensor, pos_tensor)
            distances, labels, timestamps = self.tk.retrieve(
                TemporalKnowledgeBase.normalize_embeddings(query.cpu())
            )
        
        return {
            'causal_narrative': narratives[0],
            'retrieved_patterns': {
                'distances': distances[0].tolist(),
                'labels': labels[0].tolist(),
                'timestamps': timestamps[0].tolist()
            },
            'spatial_heatmap': {
                'attention_scores': attn_weights[0].cpu().numpy().tolist()
            }
        }
