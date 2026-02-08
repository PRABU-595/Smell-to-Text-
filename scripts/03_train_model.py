#!/usr/bin/env python3
"""
Main training script
"""
import sys
sys.path.append('.')

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import yaml
import wandb

from src.data.dataset import SmellDataset
from src.models.neobert_model import SmellToMoleculeModel
from src.training.trainer import Trainer

def main():
    # Load configs
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb
    wandb.init(project="smell-to-molecule", config=config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Create datasets
    train_dataset = SmellDataset(
        'data/processed/train.csv',
        tokenizer,
        max_length=config['model']['max_length']
    )
    val_dataset = SmellDataset(
        'data/processed/val.csv',
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size']
    )
    
    # Initialize model
    model = SmellToMoleculeModel(
        model_name=config['model']['name'],
        num_chemicals=config['model']['num_chemicals'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout']
    )
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, config['training'])
    trainer.train()
    
    print("Training complete!")

if __name__ == '__main__':
    main()
