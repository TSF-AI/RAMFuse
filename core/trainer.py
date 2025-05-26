import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, Tuple
from ..models.ramfuse import RAMFuse
from ..utils.faiss_db import TemporalKnowledgeBase

class RAMFuseTrainer:
    """Training Pipeline Manager for RAMFuse
    
    Args:
        model (RAMFuse): Initialized RAMFuse model
        tk (TemporalKnowledgeBase): Temporal Knowledge Base instance
        config (Dict): Training configuration dictionary
        device (torch.device): Target compute device
    
    Key Functionalities:
        - End-to-end training loop
        - Loss calculation
        - Knowledge base incremental updates
        - Gradient clipping and optimization
    """
    def __init__(self,
                 model: RAMFuse,
                 tk: TemporalKnowledgeBase,
                 config: Dict,
                 device: torch.device):
        self.model = model.to(device)
        self.tk = tk
        self.device = device
        self.config = config
        
        # Optimization setup 
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=3
        )
        
        # Loss components 
        self.pred_criterion = nn.MSELoss()
        self.align_criterion = nn.CosineEmbeddingLoss()
        
        # Training state
        self.best_loss = float('inf')
        self.step_counter = 0

    def _compute_loss(self,
                    preds: torch.Tensor,
                    targets: torch.Tensor,
                    img_feats: torch.Tensor,
                    text_feats: torch.Tensor) -> torch.Tensor:
        """Composite Loss Calculation 
        
        Args:
            preds: Model predictions (B, T)
            targets: Ground truth values (B, T)
            img_feats: Spatial features from PCNN (B, F1)
            text_feats: Causal features from LVLM (B, F2)
        
        Returns:
            Total loss weighted by λ (config['lambda'])
        """
        # Prediction loss 
        pred_loss = self.pred_criterion(preds, targets)
        
        # Alignment loss 
        align_loss = 1 - self.align_criterion(
            img_feats, text_feats,
            torch.ones(img_feats.size(0)).to(self.device)
        
        # Total loss 
        return pred_loss + self.config['lambda'] * align_loss

    def _update_knowledge_base(self,
                             X: torch.Tensor,
                             positions: torch.Tensor,
                             labels: torch.Tensor):
        """TK Incremental Update """
        with torch.no_grad():
            # Generate normalized embeddings
            fused = self.model._encode_multimodal(X, positions)
            normalized = TemporalKnowledgeBase.normalize_embeddings(fused)
            
            # Add entries with timestamps
            timestamps = [time.time()] * len(X)
            self.tk.add_entries(
                normalized.cpu().numpy(),
                labels.cpu().numpy(),
                timestamps
            )

    def train_step(self,
                 X_batch: torch.Tensor,
                 pos_batch: torch.Tensor,
                 y_batch: torch.Tensor) -> float:
        """Single Training Step Workflow"""
        self.model.train()
        X, pos, y = X_batch.to(self.device), pos_batch.to(self.device), y_batch.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        preds, attn_weights = self.model(X, pos, update_tk=False)
        
        # Extract intermediate features
        with torch.no_grad():
            gaf_images = self.model.gaf(X)
            img_feats = self.model.pcnn(gaf_images, pos)
            text_feats, _ = self.model.lvlm(gaf_images)
        
        # Loss calculation
        loss = self._compute_loss(preds, y, img_feats, text_feats)
        
        # Backpropagation
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
        self.optimizer.step()
        
        # Update knowledge base periodically
        if self.step_counter % self.config['tk_update_interval'] == 0:
            self._update_knowledge_base(X, pos, y)
        
        self.step_counter += 1
        return loss.item()

    def validate(self,
               val_loader: torch.utils.data.DataLoader) -> float:
        """Validation Phase"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X, pos, y in val_loader:
                X, pos, y = X.to(self.device), pos.to(self.device), y.to(self.device)
                preds, _ = self.model(X, pos)
                
                gaf_images = self.model.gaf(X)
                img_feats = self.model.pcnn(gaf_images, pos)
                text_feats, _ = self.model.lvlm(gaf_images)
                
                loss = self._compute_loss(preds, y, img_feats, text_feats)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.scheduler.step(avg_loss)
        return avg_loss

    def train(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            num_epochs: int = 50):
        """Full Training Loop"""
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for batch in progress_bar:
                X, pos, y = batch
                step_loss = self.train_step(X, pos, y)
                epoch_loss += step_loss
                progress_bar.set_postfix({"loss": f"{step_loss:.4f}"})
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch+1} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), f"best_model_epoch{epoch+1}.pt")
                print(f"New best model saved with val loss {val_loss:.4f}")
