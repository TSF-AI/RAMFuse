import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.ramfuse import RAMFuse
from utils.faiss_db import TemporalKnowledgeBase
from core.trainer import RAMFuseTrainer
from utils.logger import ExperimentLogger


def load_config(config_path: str) -> dict:
    """Load and validate configuration parameters"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Essential parameter validation
    required_keys = {'data', 'model', 'training'}
    assert required_keys.issubset(config.keys()), "Missing essential config sections"
    
    return config

def prepare_data(config: dict) -> Tuple[DataLoader, DataLoader]:
    """Load and preprocess time series data
    Args:
        config: Data configuration parameters
    Returns:
        train_loader, val_loader: Batched data loaders
    """
    # Load raw data (Example: numpy arrays)
    train_data = np.load(config['data']['train_path'])
    val_data = np.load(config['data']['val_path'])
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(train_data['X'])
    pos_train = torch.FloatTensor(train_data['positions'])
    y_train = torch.FloatTensor(train_data['y'])
    
    X_val = torch.FloatTensor(val_data['X'])
    pos_val = torch.FloatTensor(val_data['positions'])
    y_val = torch.FloatTensor(val_data['y'])
    
    # Create datasets
    train_dataset = TensorDataset(X_train, pos_train, y_train)
    val_dataset = TensorDataset(X_val, pos_val, y_val)
    
    # Create loaders with paper-specified parameters
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=os.cpu_count()//2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size']*2,
        shuffle=False
    )
    
    return train_loader, val_loader

def initialize_system(config: dict, device: torch.device) -> Tuple[RAMFuse, TemporalKnowledgeBase]:
    """System initialization"""
    # Model initialization
    model = RAMFuse(
        seq_len=config['model']['seq_len'],
        n_channels=config['model']['n_channels'],
        tk_dim=config['model']['tk_dim'],
        num_retrievals=config['model']['num_retrievals'],
        temp=config['model']['temperature']
    ).to(device)
    
    # Knowledge base initialization
    tk = TemporalKnowledgeBase(
        embedding_dim=config['model']['tk_dim'],
        max_entries=config['data']['max_tk_entries'],
        aging_threshold=config['model']['aging_threshold']
    )
    
    return model, tk

def main():
    # System configuration
    config = load_config('configs/base.yaml')
    device = torch.device(config['training']['device'])
    torch.manual_seed(config['training']['seed'])
    
    # Experiment tracking
    logger = ExperimentLogger(config['logging']['log_dir'])
    logger.save_config(config)
    
    # Data preparation
    train_loader, val_loader = prepare_data(config['data'])
    
    # System initialization
    model, tk = initialize_system(config, device)
    logger.log_model_architecture(model)
    
    # Trainer setup 
    trainer = RAMFuseTrainer(
        model=model,
        tk=tk,
        config=config['training'],
        device=device
    )
    
    # Training loop (Algorithm 1)
    print("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs']
    )
    
    # Final evaluation
    print("\nFinal evaluation:")
    final_loss = trainer.validate(val_loader)
    logger.log_metrics({'final_val_loss': final_loss})
    
    # Save final artifacts
    torch.save(model.state_dict(), os.path.join(logger.log_dir, 'final_model.pth'))
    tk._rebuild_index()  # Optimize final index
    faiss.write_index(tk.index, os.path.join(logger.log_dir, 'final_tk.index'))
    
    print(f"Experiment artifacts saved to {logger.log_dir}")

if __name__ == "__main__":
    main()
