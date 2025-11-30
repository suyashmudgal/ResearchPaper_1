#!/bin/bash

echo "========================================"
echo "   SASKC Research Project Setup"
echo "========================================"

# Install dependencies
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt

# Run experiments
echo "[2/4] Running experiments (100 cycles)..."
python src/experiment_runs.py

# Train final model
echo "[3/4] Training final SASKC model..."
python src/train_best_model.py

echo "========================================"
echo "   Setup Complete!"
echo "========================================"
echo ""
echo "To run the Dashboard:"
echo "    streamlit run dashboard/streamlit_app.py"
echo ""
echo "To run the API:"
echo "    python src/api/flask_api.py"
echo ""
