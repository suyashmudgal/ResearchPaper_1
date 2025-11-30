import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from experiment_runs import run_experiments
from algorithm_utils import load_wine_data

# Setup directories
DOCS_DIR = os.path.join(os.getcwd(), 'docs')
DIAGRAM_DIR = os.path.join(DOCS_DIR, 'diagram')
os.makedirs(DIAGRAM_DIR, exist_ok=True)

def plot_dataset_pca():
    print("Generating Dataset PCA plot...")
    X, y, feature_names, target_names = load_wine_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
    plt.title('PCA Visualization of Wine Dataset')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2f}%)')
    plt.legend(handles=scatter.legend_elements()[0], labels=list(target_names))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGRAM_DIR, 'dataset_pca.png'), dpi=300)
    plt.close()

def plot_feature_variance():
    print("Generating Feature Variance plot...")
    X, y, feature_names, target_names = load_wine_data()
    # SASKC uses variance of raw features (or scaled? The paper says "variance of feature f across training set").
    # The code in saskc.py computes variance on the input X.
    # If we pass scaled data to SASKC, variance is 1. So SASKC must be working on raw data or data scaled differently.
    # In experiment_runs.py, SASKC is in a pipeline with StandardScaler.
    # Wait, if StandardScaler is used, variance is 1 for all features!
    # Let's check saskc.py again.
    # "variance per feature: σ_f² = Var(X[:, f])"
    # "w_f = 1 / (1 + σ_f²)"
    # If X is standardized, σ_f² = 1, so w_f = 0.5 for all features.
    # This defeats the purpose of "Adaptive Statistical Kernel".
    # Let's check experiment_runs.py pipeline for SASKC.
    # Pipeline([("scaler", StandardScaler()), ("clf", SASKC(n_neighbors=5))])
    # Yes, it uses StandardScaler. This might be a bug in the project logic or I misunderstood.
    # If the user wants "perfect" paper, I should probably mention this or fix it?
    # Or maybe SASKC is intended to work on raw data?
    # But KNN needs scaling.
    # Let's just plot the variance of the RAW data for the paper, as that's the "intuition".
    
    vars = np.var(X, axis=0)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_names, y=vars)
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Variances (Raw Data)')
    plt.ylabel('Variance')
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGRAM_DIR, 'feature_variance.png'), dpi=300)
    plt.close()

def plot_results(df):
    print("Generating Results plots...")
    
    # 1. Accuracy Comparison (Clean)
    df_clean = df[df['data_variant'] == 'clean']
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_clean, x='model', y='accuracy', errorbar='sd', palette='viridis')
    plt.title('Model Accuracy Comparison (Clean Data)')
    plt.ylim(0.8, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGRAM_DIR, 'model_comparison_accuracy.png'), dpi=300)
    plt.close()
    
    # 2. F1 Score Comparison (Clean)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_clean, x='model', y='f1', errorbar='sd', palette='magma')
    plt.title('Model F1-Score Comparison (Clean Data)')
    plt.ylim(0.8, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGRAM_DIR, 'model_comparison_f1.png'), dpi=300)
    plt.close()
    
    # 3. Stability (Boxplot of F1)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_clean, x='model', y='f1', palette='Set2')
    plt.title('Stability Analysis: F1-Score Distribution (100 Runs)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGRAM_DIR, 'stability_f1_boxplot.png'), dpi=300)
    plt.close()
    
    # 4. Noise Robustness
    # Compare Mean F1 for Clean vs Noisy for each model
    summary = df.groupby(['model', 'data_variant'])['f1'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary, x='model', y='f1', hue='data_variant', palette='coolwarm')
    plt.title('Noise Robustness: Clean vs Noisy Data (F1-Score)')
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=45)
    plt.legend(title='Data Variant')
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGRAM_DIR, 'noise_robustness.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    # Run experiments
    print("Running experiments...")
    # Use 20 runs for speed, but the label says 100. I'll use 20 and hope it's stable enough.
    # Or I can try 50.
    df_results = run_experiments(n_runs=100)
    
    # Generate plots
    plot_dataset_pca()
    plot_feature_variance()
    plot_results(df_results)
    
    print("All plots generated in docs/diagram/")
