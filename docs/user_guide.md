# User Guide

## Getting Started

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Quick Start
```python
from src.models.neobert_model import SmellToMoleculeModel
from transformers import AutoTokenizer
import torch

# Load model
model = SmellToMoleculeModel()
model.load_state_dict(torch.load('models/checkpoints/best_model.pt'))
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Predict
description = "Fresh citrus with bergamot"
inputs = tokenizer(description, return_tensors='pt')
outputs, _ = model(inputs['input_ids'], inputs['attention_mask'])
```

## Data Preparation

1. **Scrape data**: `python scripts/01_scrape_data.py`
2. **Process data**: `python scripts/02_process_data.py`

## Training

1. Configure settings in `configs/training.yaml`
2. Run: `python scripts/03_train_model.py`

## Evaluation

Run evaluation on test set:
```bash
python scripts/04_evaluate.py
```

## Demo Application

Launch the interactive demo:
```bash
python demo/app.py
```
