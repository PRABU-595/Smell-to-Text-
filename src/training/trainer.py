"""
Training loop for smell-to-molecule models.
Supports NeoBERT, BERT baseline, and any model with compatible forward() API.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for smell-to-molecule models."""
    
    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model: nn.Module with forward(input_ids, attention_mask, labels)
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Dict with training hyperparameters
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 2e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.get('epochs', 10)
        warmup_steps = config.get('warmup_steps', int(total_steps * 0.1))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training settings
        self.gradient_clip = config.get('gradient_clip', 1.0)
        self.epochs = config.get('epochs', 10)
        self.patience = config.get('early_stopping_patience', 5)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'models/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_f1': [], 'learning_rate': [],
        }
    
    def train_epoch(self):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            result = self.model(input_ids, attention_mask, labels=labels)
            loss = result['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            )
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / max(num_batches, 1)
    
    def validate(self):
        """Run validation and compute metrics."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                result = self.model(input_ids, attention_mask, labels=labels)
                total_loss += result['loss'].item()
                num_batches += 1
                
                preds = (result['probs'] > 0.5).int().cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute F1
        all_preds = np.vstack(all_preds) if all_preds else np.array([])
        all_labels = np.vstack(all_labels) if all_labels else np.array([])
        
        try:
            val_f1 = f1_score(
                (all_labels > 0.5).astype(int), all_preds,
                average='micro', zero_division=0
            )
        except Exception:
            val_f1 = 0.0
        
        return avg_loss, val_f1
    
    def train(self):
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_f1 = self.validate()
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val F1:     {val_f1:.4f}")
            print(f"  LR:         {current_lr:.2e}")
            print(f"  Time:       {epoch_time:.1f}s")
            
            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss, val_f1, is_best=True)
                print(f"  ✓ New best model saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  ✗ No improvement ({patience_counter}/{self.patience})")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, val_loss, val_f1, is_best=False)
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"\n⚠ Early stopping after {epoch+1} epochs")
                break
        
        # Save final history
        self._save_history()
        
        print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
        return self.history
    
    def _save_checkpoint(self, epoch, val_loss, val_f1, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_f1': val_f1,
            'config': self.config,
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        
        torch.save(checkpoint, path)
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0)
