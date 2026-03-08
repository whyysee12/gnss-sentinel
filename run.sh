#!/bin/bash
echo "=== GNSS-SENTINEL PIPELINE ==="
cd src
echo "Step 1: Feature Engineering..."
python features.py
echo "Step 2: Model Training..."
python model.py
echo "Step 3: Generating Submission..."
python predict.py
echo "=== DONE. Check outputs/submission.csv ==="
