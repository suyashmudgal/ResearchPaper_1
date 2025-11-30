import os
import sys
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithm_utils import load_wine_data
from saskc import SASKC

# Paths
# This file is in src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model_saskc.pkl")

def train_and_save():
    print("Loading Wine dataset...")
    X, y, _, target_names = load_wine_data(add_noise=False)
    
    print("Training final SASKC model on full dataset...")
    # Using k=5 as default robust choice
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SASKC(n_neighbors=5))
    ])
    
    pipeline.fit(X, y)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Save model and target names
    joblib.dump((pipeline, target_names), MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    
    # Quick verification
    print("Verifying model...")
    loaded_model, loaded_names = joblib.load(MODEL_PATH)
    sample_pred = loaded_model.predict(X[:5])
    print(f"Predictions for first 5 samples: {sample_pred}")
    print(f"Actual labels:                   {y[:5]}")

if __name__ == "__main__":
    train_and_save()
