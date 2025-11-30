import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from src.saskc import SASKC
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
N_RUNS = 100
TEST_SIZE = 0.3
RANDOM_STATE = 42

def load_datasets():
    datasets = {}
    
    # Wine
    wine = load_wine()
    datasets['Wine'] = (wine.data, wine.target, wine.feature_names, wine.target_names)
    
    # Iris
    iris = load_iris()
    datasets['Iris'] = (iris.data, iris.target, iris.feature_names, iris.target_names)
    
    # Breast Cancer
    bc = load_breast_cancer()
    datasets['Breast Cancer'] = (bc.data, bc.target, bc.feature_names, bc.target_names)
    
    return datasets

def get_baselines():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=RANDOM_STATE),
        "SVM-RBF": SVC(kernel="rbf", gamma="scale", probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    }

def evaluate_model(model, X, y, n_runs=N_RUNS):
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}
    
    sss = StratifiedShuffleSplit(n_splits=n_runs, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Preprocessing: Z-score normalization (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['Precision'].append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        metrics['Recall'].append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        metrics['F1'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}

def run_ablation(X, y):
    variants = {
        "KNN Baseline": SASKC(n_neighbors=5, use_weights=False, use_rank_voting=False),
        "Only Adaptive Kernel": SASKC(n_neighbors=5, use_weights=True, use_rank_voting=False),
        "Only Rank Voting": SASKC(n_neighbors=5, use_weights=False, use_rank_voting=True),
        "Full SASKC": SASKC(n_neighbors=5, use_weights=True, use_rank_voting=True)
    }
    
    results = {}
    for name, model in variants.items():
        print(f"  Running ablation: {name}...")
        metrics = evaluate_model(model, X, y)
        results[name] = metrics
    return results

def run_noise_robustness(X, y):
    noise_levels = [0, 0.05, 0.1, 0.2]
    results = []
    
    model = SASKC(n_neighbors=5, use_weights=True, use_rank_voting=True)
    
    for sigma in noise_levels:
        print(f"  Testing noise level: {sigma}...")
        accuracies = []
        sss = StratifiedShuffleSplit(n_splits=N_RUNS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Add noise
            noise_train = np.random.normal(0, sigma, X_train.shape)
            noise_test = np.random.normal(0, sigma, X_test.shape)
            
            # Standardize FIRST, then add noise? Or add noise then standardize?
            # User said: "Preprocess... Apply Z-score... Line plot for noise robustness: noise levels [0, 0.05...]"
            # Usually noise is added to raw data or standardized data. 
            # If we add noise to raw data, standardization might mitigate it if it's just shift, but random noise won't be removed.
            # Let's Standardize -> Add Noise (since sigma is 0.05, which implies scale relative to std=1).
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            X_train_noisy = X_train_scaled + noise_train
            X_test_noisy = X_test_scaled + noise_test
            
            model.fit(X_train_noisy, y_train)
            y_pred = model.predict(X_test_noisy)
            accuracies.append(accuracy_score(y_test, y_pred))
            
        results.append(np.mean(accuracies))
        
    return noise_levels, results

def plot_confusion_matrix(model, X, y, dataset_name, target_names):
    # Train on one split for visualization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    plt.title(f'Confusion Matrix: SASKC on {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{dataset_name}.png')
    plt.close()

def plot_feature_weights(model, feature_names, dataset_name):
    # Model is already fitted from CM step
    weights = model.feature_weights_
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(weights)), weights)
    plt.xticks(range(len(weights)), feature_names, rotation=45, ha='right')
    plt.title(f'Adaptive Feature Weights: {dataset_name}')
    plt.ylabel('Weight (1 / (1 + Variance))')
    plt.tight_layout()
    plt.savefig(f'feature_weights_{dataset_name}.png')
    plt.close()

def main():
    datasets = load_datasets()
    baselines = get_baselines()
    saskc = SASKC(n_neighbors=5)
    
    all_results = []
    ablation_results_all = {}
    
    print("Starting Comprehensive Evaluation...")
    
    for name, (X, y, feat_names, target_names) in datasets.items():
        print(f"\nProcessing Dataset: {name}")
        
        # 1. Evaluate Baselines
        for b_name, model in baselines.items():
            print(f"  Evaluating {b_name}...")
            metrics = evaluate_model(model, X, y)
            row = {'Dataset': name, 'Model': b_name}
            for k, v in metrics.items():
                row[k] = f"{v[0]:.3f} ± {v[1]:.3f}"
            all_results.append(row)
            
        # 2. Evaluate SASKC
        print(f"  Evaluating SASKC...")
        metrics = evaluate_model(saskc, X, y)
        row = {'Dataset': name, 'Model': 'SASKC'}
        for k, v in metrics.items():
            row[k] = f"{v[0]:.3f} ± {v[1]:.3f}"
        all_results.append(row)
        
        # 3. Ablation Study
        print(f"  Running Ablation Study...")
        ablation_metrics = run_ablation(X, y)
        ablation_results_all[name] = ablation_metrics
        
        # 4. Plots
        print(f"  Generating Plots...")
        plot_confusion_matrix(saskc, X, y, name, target_names)
        if name == 'Wine': # Plot weights for Wine as requested (13 features)
            plot_feature_weights(saskc, feat_names, name)
            
        # 5. Noise Robustness (Only for Wine or all? User said "Line plot for noise robustness". Let's do Wine)
        if name == 'Wine':
            print(f"  Running Noise Robustness on Wine...")
            noise_x, noise_y = run_noise_robustness(X, y)
            plt.figure(figsize=(8, 5))
            plt.plot(noise_x, noise_y, marker='o', label='SASKC')
            plt.xlabel('Noise Level (sigma)')
            plt.ylabel('Accuracy')
            plt.title('SASKC Noise Robustness (Wine Dataset)')
            plt.grid(True)
            plt.savefig('noise_robustness.png')
            plt.close()

    # Save Results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('evaluation_results.csv', index=False)
    
    # Format Ablation Table
    ablation_rows = []
    for dataset, variants in ablation_results_all.items():
        for v_name, metrics in variants.items():
            row = {'Dataset': dataset, 'Variant': v_name}
            for k, v in metrics.items():
                row[k] = f"{v[0]:.3f} ± {v[1]:.3f}"
            ablation_rows.append(row)
    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv('ablation_results.csv', index=False)
    
    print("\nEvaluation Complete.")
    print("\n--- Model Results ---")
    print(df_results.to_string(index=False))
    print("\n--- Ablation Results ---")
    print(df_ablation.to_string(index=False))

if __name__ == "__main__":
    main()
