# NeoBERT for Smell-to-Molecule Translation

## Overview
This project implements a novel NLP system that translates natural language 
smell descriptions into chemical formulas using fine-tuned NeoBERT.

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```bash
# 1. Scrape data
python scripts/01_scrape_data.py

# 2. Process dataset
python scripts/02_process_data.py

# 3. Train model
python scripts/03_train_model.py

# 4. Evaluate
python scripts/04_evaluate.py
```

## Demo
```bash
python demo/app.py
```

## Citation
[Your citation here]
