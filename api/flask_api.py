import os
import sys
from typing import Any, Dict, List

import joblib
import numpy as np
from flask import Flask, jsonify, request

# -------------------------------------------------------------
# Make sure we can import `saskc` from src/
# Project structure:
#   SASKC_Research_Project/
#     api/flask_api.py   <-- this file
#     src/saskc.py
# -------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # .../SASKC_Research_Project
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# IMPORTANT: this import must be BEFORE joblib.load
# so that pickle can resolve saskc.SASKC
from saskc import SASKC  # noqa: F401

# We’ll use the Wine model
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_wine.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. "
        "Train it first with: python src/train_best_model.py"
    )

# We saved (pipeline, class_names, feature_names)
model, class_names, feature_names = joblib.load(MODEL_PATH)

app = Flask(__name__)


def _validate_features(payload: Dict[str, Any]) -> List[float]:
    """
    Expected JSON:
    {
        "features": [13 numbers in same order as feature_names]
    }
    """
    if "features" not in payload:
        raise ValueError("Missing 'features' key in JSON body.")

    features = payload["features"]

    if not isinstance(features, list):
        raise ValueError("'features' must be a list of numbers.")

    expected_len = len(feature_names)
    if len(features) != expected_len:
        raise ValueError(f"Expected {expected_len} features, got {len(features)}.")

    try:
        features = [float(v) for v in features]
    except (TypeError, ValueError) as exc:
        raise ValueError("All features must be numeric.") from exc

    return features


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "model_loaded": bool(model is not None),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "class_labels": class_names.tolist(),
        }
    )


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    try:
        features = _validate_features(request.get_json(silent=True) or {})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    X = np.array(features, dtype=float).reshape(1, -1)
    pred_label = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0].tolist()
    else:
        proba = None

    return jsonify(
        {
            "pred_class_label": int(pred_label),
            "class_labels": class_names.tolist(),
            "probabilities": proba,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
