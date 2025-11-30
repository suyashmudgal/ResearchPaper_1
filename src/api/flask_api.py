import os
import sys
import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.datasets import load_wine

# Ensure src is in path to load SASKC class if needed by pickle
# (Pickle needs the class definition to be importable)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(SRC_DIR)

from saskc import SASKC # Import so joblib can find it

app = Flask(__name__)

# Load model and class names
BASE_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model_saskc.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Get class names from dataset
wine = load_wine()
CLASS_NAMES = list(wine.target_names)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict class for a wine sample.
    Input: JSON {"features": [v1, v2, ..., v13]}
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Missing 'features' in request body"}), 400
            
        features = data['features']
        
        if not isinstance(features, list) or len(features) != 13:
            return jsonify({"error": "Features must be a list of 13 numeric values"}), 400
            
        # Convert to numpy array (1, 13)
        X_input = np.array(features).reshape(1, -1)
        
        # Predict
        prediction_idx = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        
        class_name = CLASS_NAMES[prediction_idx]
        
        # Format probabilities
        probs_dict = {
            name: float(p) for name, p in zip(CLASS_NAMES, proba)
        }
        
        result = {
            "class_index": int(prediction_idx),
            "class_name": class_name,
            "probabilities": probs_dict
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on 0.0.0.0:5000
    app.run(host='0.0.0.0', port=5000, debug=True)
