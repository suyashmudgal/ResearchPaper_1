import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional

# Directory setup
# This file is in src/algorithm_utils.py
# We want data/ in the project root (one level up from src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_wine_data(add_noise: bool = False, noise_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Load the Wine dataset from scikit-learn.
    
    Parameters
    ----------
    add_noise : bool
        If True, adds Gaussian noise to the features.
    noise_level : float
        Standard deviation of the Gaussian noise (mean=0).
        
    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    feature_names : list of str
    target_names : ndarray of shape (n_classes,)
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    
    if add_noise:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, noise_level, X.shape)
        X = X + noise
        
    return X, y, feature_names, target_names

def save_runs_and_summary(df: pd.DataFrame, filename_suffix: str = "") -> Tuple[str, str, pd.DataFrame]:
    """
    Save per-run metrics and aggregated summary to CSV.
    
    Parameters
    ----------
    df : DataFrame
        Must contain columns: ["run", "model", "accuracy", "precision", "recall", "f1"].
    filename_suffix : str
        Optional suffix for filenames (e.g., "_noisy").
        
    Returns
    -------
    runs_path : str
        Path to the saved runs CSV.
    summary_path : str
        Path to the saved summary CSV.
    summary : DataFrame
        Aggregated summary by model.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    runs_filename = f"results_runs{filename_suffix}.csv"
    summary_filename = f"results_summary{filename_suffix}.csv"
    
    runs_path = os.path.join(DATA_DIR, runs_filename)
    summary_path = os.path.join(DATA_DIR, summary_filename)
    
    # Save detailed runs
    # If file exists, we might want to append or overwrite. 
    # For this project, we overwrite to keep it clean or append if we run multiple experiments.
    # Let's overwrite for simplicity as per spec "Save per-run logs".
    df.to_csv(runs_path, index=False)
    
    # Aggregate statistics by model
    # We want mean, std, min, max for each metric
    metrics = ["accuracy", "precision", "recall", "f1"]
    # Filter metrics that exist in df
    metrics = [m for m in metrics if m in df.columns]
    
    summary = (
        df.groupby("model")[metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    
    # Flatten MultiIndex columns: ("accuracy", "mean") -> "accuracy_mean"
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    
    summary.to_csv(summary_path, index=False)
    
    return runs_path, summary_path, summary
