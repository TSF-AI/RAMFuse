"""
RAMFuse Experiment Logger
"""

import os
import json
import time
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

class ExperimentLogger:
    """Unified Experiment Tracking System
    
    Features:
    - TensorBoard integration for real-time monitoring
    - Config version control
    - Model architecture logging
    - Metrics persistence
    - Artifact management
    
    Args:
        log_dir (str): Root directory for all experiment artifacts
        use_tb (bool): Enable TensorBoard logging (default: True)
        max_artifacts (int): Maximum stored checkpoints (default: 5)
    """
    
    def __init__(self,
                 log_dir: str = "./experiments",
                 use_tb: bool = True,
                 max_artifacts: int = 5):
        # Create unique experiment ID
        self.exp_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, self.exp_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize subsystems
        self._init_tensorboard(use_tb)
        self.artifact_queue = []
        self.max_artifacts = max_artifacts
        
        # State tracking
        self.start_time = time.time()
        self.step_counter = 0
        
        print(f"Experiment logging initialized at {self.log_dir}")

    def _init_tensorboard(self, use_tb: bool):
        """Initialize TensorBoard writer with safe fallback"""
        self.tb_writer = None
        if use_tb:
            try:
                self.tb_writer = SummaryWriter(log_dir=self.log_dir)
            except Exception as e:
                print(f"TensorBoard initialization failed: {str(e)}")

    def save_config(self, config: Dict[str, Any]):
        """Version-controlled config preservation"""
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # TensorBoard text logging
        if self.tb_writer:
            self.tb_writer.add_text("Config", json.dumps(config, indent=2))

    def log_model_architecture(self, model: torch.nn.Module):
        """Model structure logging with graph visualization"""
        # Save architecture summary
        arch_path = os.path.join(self.log_dir, "model_arch.txt")
        with open(arch_path, 'w') as f:
            f.write(str(model))
        
        # TensorBoard graph (requires sample input)
        try:
            sample_input = torch.randn(1, model.seq_len, model.n_channels)
            self.tb_writer.add_graph(model, sample_input)
        except Exception as e:
            print(f"Model graph logging failed: {str(e)}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Multi-channel metrics logging"""
        # Automatic step management
        if step is None:
            step = self.step_counter
            self.step_counter += 1
        
        # TensorBoard scalar logging
        if self.tb_writer:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, step)
        
        # Persistent JSON logging
        metrics_path = os.path.join(self.log_dir, "metrics.jsonl")
        with open(metrics_path, 'a') as f:
            record = {"step": step, **metrics, "timestamp": time.time()}
            f.write(json.dumps(record) + "\n")

    def log_artifact(self,
                   artifact: object,
                   artifact_name: str,
                   metadata: Optional[Dict] = None):
        """Artifact versioning with rotation policy"""
        # Create artifact directory
        artifact_dir = os.path.join(self.log_dir, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{artifact_name}_{timestamp}.pt"
        filepath = os.path.join(artifact_dir, filename)
        
        # Save with metadata
        if isinstance(artifact, torch.nn.Module):
            torch.save({
                'state_dict': artifact.state_dict(),
                'metadata': metadata or {}
            }, filepath)
        else:
            torch.save(artifact, filepath)
        
        # Maintain artifact rotation
        self.artifact_queue.append(filepath)
        while len(self.artifact_queue) > self.max_artifacts:
            old_artifact = self.artifact_queue.pop(0)
            os.remove(old_artifact)

    def log_embeddings(self,
                     embeddings: np.ndarray,
                     labels: np.ndarray,
                     tag: str = "TK_Embeddings",
                     step: Optional[int] = None):
        """High-dimensional embedding visualization"""
        if self.tb_writer:
            try:
                self.tb_writer.add_embedding(
                    mat=embeddings,
                    metadata=labels,
                    tag=tag,
                    global_step=step
                )
            except Exception as e:
                print(f"Embedding logging failed: {str(e)}")

    def log_images(self,
                 images: torch.Tensor,
                 tag: str = "GAF_Images",
                 step: Optional[int] = None):
        """Image data logging with normalization"""
        if self.tb_writer and images.dim() == 4:  # (B,C,H,W)
            try:
                img_grid = torchvision.utils.make_grid(images)
                self.tb_writer.add_image(tag, img_grid, step)
            except Exception as e:
                print(f"Image logging failed: {str(e)}")

    def close(self):
        """Safe resource cleanup"""
        if self.tb_writer:
            self.tb_writer.close()
        print(f"Experiment duration: {time.time()-self.start_time:.1f}s")
