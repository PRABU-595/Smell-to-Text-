#!/bin/bash
# Smell-to-Molecule: Full Pipeline
# Run all steps sequentially
set -e

echo "================================================"
echo "  Smell-to-Molecule Full Pipeline"
echo "================================================"

cd "$(dirname "$0")/.."

echo ""
echo "[1/5] Generating dataset..."
python scripts/01_scrape_data.py --n_samples 5000

echo ""
echo "[2/5] Processing data..."
python scripts/02_process_data.py

echo ""
echo "[3/5] Training models..."
python scripts/03_train_model.py --epochs 5 --batch_size 8 --patience 3

echo ""
echo "[4/5] Evaluating models..."
python scripts/04_evaluate.py

echo ""
echo "[5/5] Generating predictions..."
python scripts/05_generate_predictions.py

echo ""
echo "================================================"
echo "  Pipeline complete!"
echo "  Results: outputs/reports/evaluation_results.json"
echo "  Predictions: outputs/predictions/predictions.json"
echo "================================================"
