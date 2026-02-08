"""
Unit tests for models
"""
import pytest
import torch
from src.models.neobert_model import SmellToMoleculeModel

def test_model_initialization():
    model = SmellToMoleculeModel(num_chemicals=100)
    assert model is not None

def test_forward_pass():
    model = SmellToMoleculeModel(num_chemicals=100)
    
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128)
    
    outputs, _ = model(input_ids, attention_mask)
    
    assert outputs.shape == (2, 100)
    assert torch.all(outputs >= 0) and torch.all(outputs <= 1)

def test_model_save_load():
    model = SmellToMoleculeModel(num_chemicals=100)
    torch.save(model.state_dict(), 'test_model.pt')
    
    new_model = SmellToMoleculeModel(num_chemicals=100)
    new_model.load_state_dict(torch.load('test_model.pt'))
    
    assert new_model is not None
