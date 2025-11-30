import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from tqdm import tqdm
from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from src.saskc import SASKC

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
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Gaussian NB": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=RANDOM_STATE),
        "SVM-RBF": SVC(kernel="rbf", gamma="scale", probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    }

def evaluate_model_runs(model, X, y, dataset_name, model_name, n_runs=N_RUNS, add_noise=False, noise_level=0.05):
    """
    Run Monte Carlo Cross-Validation and return per-run metrics.
    """
    results = []
    sss = StratifiedShuffleSplit(n_splits=n_runs, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    run_id = 0
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if add_noise:
            # Add noise to standardized data
            rng = np.random.default_rng(RANDOM_STATE + run_id)
            X_train_scaled += rng.normal(0, noise_level, X_train_scaled.shape)
            X_test_scaled += rng.normal(0, noise_level, X_test_scaled.shape)
            variant = "noisy"
        else:
            variant = "clean"
            
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        metrics = {
            "run": run_id,
            "dataset": dataset_name,
            "model": model_name,
            "data_variant": variant,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='macro', zero_division=0)
        }
        results.append(metrics)
        run_id += 1
        
    return results

def run_ablation(X, y):
    variants = {
        "KNN Baseline": SASKC(n_neighbors=5, use_weights=False, use_rank_voting=False),
        "Only Adaptive Kernel": SASKC(n_neighbors=5, use_weights=True, use_rank_voting=False),
        "Only Rank Voting": SASKC(n_neighbors=5, use_weights=False, use_rank_voting=True),
        "Full SASKC": SASKC(n_neighbors=5, use_weights=True, use_rank_voting=True)
    }
    
    results = {}
    for name, model in variants.items():
        # Just run one pass of evaluation (summary)
        metrics_list = evaluate_model_runs(model, X, y, "Wine", name, n_runs=N_RUNS)
        # Aggregate
        df = pd.DataFrame(metrics_list)
        results[name] = {
            "Accuracy": (df["accuracy"].mean(), df["accuracy"].std()),
            "Precision": (df["precision"].mean(), df["precision"].std()),
            "Recall": (df["recall"].mean(), df["recall"].std()),
            "F1-Score": (df["f1"].mean(), df["f1"].std())
        }
    return results

def run_noise_robustness(X, y):
    noise_levels = [0, 0.05, 0.1, 0.2]
    results = []
    
    model = SASKC(n_neighbors=5, use_weights=True, use_rank_voting=True)
    
    for sigma in noise_levels:
        metrics_list = evaluate_model_runs(model, X, y, "Wine", "SASKC", n_runs=N_RUNS, add_noise=(sigma > 0), noise_level=sigma)
        df = pd.DataFrame(metrics_list)
        results.append(df["accuracy"].mean())
        
    return noise_levels, results

def plot_confusion_matrix(model, X, y, dataset_name, target_names):
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
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig(f'results/plots/confusion_matrix_{dataset_name}.png')
    plt.close()

def plot_feature_weights(model, feature_names, dataset_name):
    weights = model.feature_weights_
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(weights)), weights)
    plt.xticks(range(len(weights)), feature_names, rotation=45, ha='right')
    plt.title(f'Adaptive Feature Weights: {dataset_name}')
    plt.ylabel('Weight (1 / (1 + Variance))')
    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig(f'results/plots/feature_weights_{dataset_name}.png')
    plt.close()

def main():
    datasets = load_datasets()
    baselines = get_baselines()
    saskc = SASKC(n_neighbors=5)
    
    all_runs_data = []
    
    print("Starting Comprehensive Evaluation...")
    
    # 1. Main Evaluation (Clean Data)
    for name, (X, y, feat_names, target_names) in datasets.items():
        print(f"\nProcessing Dataset: {name}")
        
        # Baselines
        for b_name, model in tqdm(baselines.items(), desc="Baselines"):
            runs = evaluate_model_runs(model, X, y, name, b_name)
            all_runs_data.extend(runs)
            
        # SASKC
        print(f"  Evaluating SASKC...")
        runs = evaluate_model_runs(saskc, X, y, name, "SASKC")
        all_runs_data.extend(runs)
        
        # Plots
        plot_confusion_matrix(saskc, X, y, name, target_names)
        if name == 'Wine':
            plot_feature_weights(saskc, feat_names, name)
            
            # Noise Robustness
            print(f"  Running Noise Robustness on Wine...")
            noise_x, noise_y = run_noise_robustness(X, y)
            plt.figure(figsize=(8, 5))
            plt.plot(noise_x, noise_y, marker='o', label='SASKC')
            plt.xlabel('Noise Level (sigma)')
            plt.ylabel('Accuracy')
            plt.title('SASKC Noise Robustness (Wine Dataset)')
            plt.grid(True)
            plt.savefig('results/plots/noise_robustness.png')
            plt.close()
            
            # Run SASKC on Noisy Wine for Dashboard (Data Variant: Noisy)
            print(f"  Generating Noisy Data Runs for Dashboard...")
            noisy_runs = evaluate_model_runs(saskc, X, y, name, "SASKC", add_noise=True, noise_level=0.05)
            all_runs_data.extend(noisy_runs)
            # Also run baselines on noisy data for comparison?
            # Dashboard expects variants. Let's run KNN and RF on noisy too.
            for b_name in ["KNN", "Random Forest"]:
                runs = evaluate_model_runs(baselines[b_name], X, y, name, b_name, add_noise=True, noise_level=0.05)
                all_runs_data.extend(runs)

    # 2. Ablation Study (Wine only)
    print(f"\nRunning Ablation Study on Wine...")
    wine_X, wine_y, _, _ = datasets['Wine']
    ablation_results = run_ablation(wine_X, wine_y)
    
    # Save Results
    os.makedirs('results/tables', exist_ok=True)
    
    # Raw Runs CSV
    df_runs = pd.DataFrame(all_runs_data)
    df_runs.to_csv('results/tables/results_runs.csv', index=False)
    
    # Summary CSV
    summary = df_runs.groupby(['dataset', 'model', 'data_variant'])[['accuracy', 'precision', 'recall', 'f1']].agg(['mean', 'std'])
    # Flatten columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.reset_index().to_csv('results/tables/evaluation_results.csv', index=False)
    
    # Ablation CSV
    ablation_rows = []
    for v_name, metrics in ablation_results.items():
        row = {'Dataset': 'Wine', 'Variant': v_name}
        for k, v in metrics.items():
            row[k] = f"{v[0]:.3f} ± {v[1]:.3f}"
        ablation_rows.append(row)
    pd.DataFrame(ablation_rows).to_csv('results/tables/ablation_results.csv', index=False)
    
    # Save Best SASKC Model for API (Train on full Wine dataset)
    print("\nSaving SASKC model for API...")
    import joblib
    os.makedirs('models', exist_ok=True)
    wine_X, wine_y, _, _ = datasets['Wine']
    
    # We must use the same preprocessing as expected by the API.
    # The API expects raw features? Or scaled?
    # The API code does: `X_input = np.array(features).reshape(1, -1); model.predict(X_input)`
    # It does NOT scale.
    # So the model must be trained on RAW data, OR the model must include scaling in a pipeline.
    # SASKC handles raw data (computes weights), but distance is Euclidean.
    # If we train on raw data, SASKC works (weights adjust for variance).
    # But if we trained on Scaled data in evaluation, we should probably stick to a standard pipeline.
    # However, SASKC is designed to be robust.
    # Let's train a Pipeline(StandardScaler, SASKC) and save that.
    
    from sklearn.pipeline import Pipeline
    api_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SASKC(n_neighbors=5))
    ])
    api_model.fit(wine_X, wine_y)
    joblib.dump(api_model, 'models/best_model_saskc.pkl')
    print("Model saved to models/best_model_saskc.pkl")
    
    print("\nEvaluation Complete. Results saved to results/tables/")

if __name__ == "__main__":
    main()
