#!/bin/bash
# Run full pipeline

echo "=== Smell-to-Molecule Pipeline ==="

echo "Step 1: Scraping data..."
python scripts/01_scrape_data.py

echo "Step 2: Processing data..."
python scripts/02_process_data.py

echo "Step 3: Training model..."
python scripts/03_train_model.py

echo "Step 4: Evaluating..."
python scripts/04_evaluate.py

echo "Step 5: Generating predictions..."
python scripts/05_generate_predictions.py

echo "=== Pipeline complete! ==="
