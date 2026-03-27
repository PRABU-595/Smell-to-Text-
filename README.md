# NeoBERT for Smell-to-Molecule Translation

## Overview
This project implements a novel NLP system that translates natural language smell descriptions into chemical profiles using a fine-tuned **NeoBERT** model. It predicts odor families and maps them to real-world chemicals using the PubChem API.

## Key Features
- **Interactive Prediction Engine**: Live terminal app for real-time scent analysis.
- **PubChem Integration**: Dynamically fetches chemical names for predicted CAS numbers.
- **Micro/Macro F1 Optimization**: Achieved Macro F1 of 0.73 on complex multi-label data.
- **Leak-Free Split**: Strict molecule-level separation between training and test sets.

## Installation
```bash
# Activate your venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Interactive Predictor
Run the interactive terminal app to predict molecules from natural language:
```bash
python scripts/05_generate_predictions.py
```

## Training Lifecycle
1. **Prepare Data**: `python scripts/02_process_data.py`
2. **Train Model**: `python scripts/03_train_model.py`
3. **Generate Visuals**: `python scripts/08_generate_metric_plots.py`

## Model Performance
The following metrics are available in `outputs/visualizations/metrics/`:
- **ROC-AUC**: Confirms high classification accuracy.
- **Precision@K**: Measures reliability in top-K predictions.
- **Confusion Matrix**: Visualizes family-level overlap.

## License
MIT License

