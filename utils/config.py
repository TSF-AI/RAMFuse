"""
Configuration Management Module (Implements Experiment Reproducibility)
Reference: NIPS_2025_RAMFuse_TimeTrend.pdf
"""

import os
import yaml
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model Architecture Configuration """
    seq_len: int          # Historical time steps (L)
    n_channels: int       # Variable dimension (D)
    tk_dim: int           # Temporal Knowledge Base dimension
    num_retrievals: int   # Top-N retrievals (N)
    temperature: float    # Attention fusion temperature (β)
    aging_threshold: int  # Knowledge entry expiration days

@dataclass
class DataConfig:
    """Dataset Configuration """
    train_path: str       # Path to training data
    val_path: str         # Path to validation data
    test_path: str        # Path to test data
    max_tk_entries: int   # Maximum entries in Temporal Knowledge Base
    spatial_info: str     # Position encoding type ('coord'/'learned')

@dataclass
class TrainingConfig:
    """Training Process Configuration """
    device: str           # Compute device ('cuda'/'cpu')
    batch_size: int       # Training batch size
    epochs: int           # Total training epochs
    lr: float            # Learning rate
    weight_decay: float  # AdamW weight decay
    grad_clip: float     # Gradient clipping threshold
    lambda: float        # Alignment loss weight (λ)

class ConfigManager:
    """Central Configuration Controller
    
    Features:
    - YAML configuration loading
    - Path resolution and validation
    - Type checking and default values
    - Nested parameter access
    """
    
    def __init__(self, config_path: str):
        self.raw_config = self._load_yaml(config_path)
        self._validate_structure()
        self._resolve_paths()
        
        # Initialize configuration sections
        self.model = ModelConfig(**self.raw_config['model'])
        self.data = DataConfig(**self.raw_config['data'])
        self.training = TrainingConfig(**self.raw_config['training'])
        
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load and parse YAML configuration file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file {path} not found")
            
        with open(path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML format: {str(e)}")

    def _validate_structure(self):
        """Validate configuration structure against dataclasses"""
        required_sections = ['model', 'data', 'training']
        for section in required_sections:
            if section not in self.raw_config:
                raise KeyError(f"Missing required config section: {section}")
                
        # Validate model parameters
        model_fields = ModelConfig.__annotations__.keys()
        for field in model_fields:
            if field not in self.raw_config['model']:
                raise ValueError(f"Missing model config field: {field}")

    def _resolve_paths(self):
        """Convert relative paths to absolute paths"""
        path_fields = ['train_path', 'val_path', 'test_path']
        for field in path_fields:
            if field in self.raw_config['data']:
                path = self.raw_config['data'][field]
                if not os.path.isabs(path):
                    self.raw_config['data'][field] = os.path.abspath(path)
                    
    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """Access nested configuration parameters using dot notation"""
        keys = key_path.split('.')
        value = self.raw_config
        for k in keys:
            value = value.get(k)
            if value is None:
                return default
        return value

def load_config(config_path: str) -> ConfigManager:
    """Public interface for configuration loading"""
    return ConfigManager(config_path)
