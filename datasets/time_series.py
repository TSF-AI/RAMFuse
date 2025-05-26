
"""
Time Series Data Loading Module
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

class TimeSeriesDataset(Dataset):
    """Multivariate Time Series Dataset Handler
    
    Args:
        data_path (str): Path to .npz file containing raw data
        seq_len (int): Historical time steps (L)
        pred_len (int): Prediction horizon (T)
        split (str): Data split type ('train', 'val', 'test')
        scale (bool): Enable z-score normalization
        spatial_info (str): Position encoding type ('coord' or 'learned')
        freq (str): Time series frequency (for timestamp features)
    
    Data Format:
        .npz file should contain:
        - train_X: (num_train, L+T, D)
        - val_X: (num_val, L+T, D)
        - test_X: (num_test, L+T, D)
        - positions: (num_nodes, 2) spatial coordinates
    """
    
    def __init__(self,
                 data_path: str,
                 seq_len: int = 128,
                 pred_len: int = 24,
                 split: str = 'train',
                 scale: bool = True,
                 spatial_info: str = 'coord',
                 freq: str = 'h'):
        super().__init__()
        
        # Validate parameters
        assert split in ['train', 'val', 'test'], "Invalid split type"
        assert spatial_info in ['coord', 'learned', None], "Invalid spatial type"
        
        # Load raw data
        data = np.load(data_path)
        self.raw_data = data[f'{split}_X']  # (num_samples, L+T, D)
        self.positions = data['positions'] if 'positions' in data else None
        
        # Data dimensions
        self.num_nodes, self.num_features = self.raw_data.shape[2], self.raw_data.shape[2]
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Normalization
        self.scaler = None
        if scale:
            self._normalize_data(data['train_X'])  # Fit on training data
            
        # Generate position encoding
        self.spatial_embeddings = self._get_spatial_embeddings(spatial_info)
        
        # Generate time features
        self.time_embeddings = self._get_time_features(freq)
        
        # Preprocess data
        self.data_x, self.data_y = self._preprocess(self.raw_data)
        
    def _normalize_data(self, train_data: np.ndarray):
        """Z-score normalization """
        self.scaler = StandardScaler()
        # Reshape to (num_samples*seq_len, num_features)
        train_data = train_data.reshape(-1, self.num_features)
        self.scaler.fit(train_data)
        
    def _get_spatial_embeddings(self, mode: str) -> Optional[np.ndarray]:
        """Generate spatial position encoding"""
        if mode == 'coord' and self.positions is not None:
            # Normalize coordinates to [0,1]
            pos_min = self.positions.min(axis=0)
            pos_max = self.positions.max(axis=0)
            return (self.positions - pos_min) / (pos_max - pos_min)
        return None
    
    def _get_time_features(self, freq: str) -> np.ndarray:
        """Generate temporal positional features"""
        # Implement timestamp features (hour, day, week, etc)
        # Placeholder implementation
        num_samples = self.raw_data.shape[0]
        return np.zeros((num_samples, self.seq_len + self.pred_len, 4))
    
    def _preprocess(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert raw data into supervised learning format"""
        num_samples, total_len, num_features = data.shape
        
        # Split into history and prediction segments
        x = data[:, :self.seq_len, :]
        y = data[:, -self.pred_len:, 0]  # Predict first feature
        
        # Normalization
        if self.scaler is not None:
            original_shape = x.shape
            x = self.scaler.transform(x.reshape(-1, num_features))
            x = x.reshape(original_shape)
            
        return x, y
    
    def __len__(self) -> int:
        return self.data_x.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """Return a sample with:
        - X: (seq_len, num_features)
        - positions: (H, W, 2) spatial coordinates
        - y: (pred_len,)
        """
        x = torch.FloatTensor(self.data_x[idx])
        y = torch.FloatTensor(self.data_y[idx])
        
        # Generate position encoding
        if self.spatial_embeddings is not None:
            pos = torch.FloatTensor(self.spatial_embeddings)
        else:
            pos = torch.zeros((self.seq_len, 2))  # Dummy positions
            
        # Time features
        time_feat = torch.FloatTensor(self.time_embeddings[idx])
        
        return {
            'X': x,                  # (seq_len, num_features)
            'positions': pos,         # (seq_len, 2) or (H, W, 2)
            'time_features': time_feat, 
            'y': y                   # (pred_len,)
        }

def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Generate data loaders for all splits"""
    train_set = TimeSeriesDataset(
        data_path=config['data_path'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        split='train',
        scale=True,
        spatial_info=config['spatial_info']
    )
    
    val_set = TimeSeriesDataset(
        data_path=config['data_path'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        split='val',
        scale=True,
        spatial_info=config['spatial_info']
    )
    
    test_set = TimeSeriesDataset(
        data_path=config['data_path'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        split='test',
        scale=True,
        spatial_info=config['spatial_info']
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    return train_loader, val_loader, test_loader
