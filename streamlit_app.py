import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from algorithm_utils import load_wine_data, save_runs_and_summary
from saskc import SASKC


def get_models():
    """
    Define all models + their (optional) hyperparameter grids.
    """
    models_config = [
        (
            "LogisticRegression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000))
            ]),
            None
        ),
        (
            "SVC",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True))
            ]),
            None
        ),
        (
            "KNN",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier())
            ]),
            {"clf__n_neighbors": [3, 5, 7]}
        ),
        (
            "GaussianNB",
            GaussianNB(),
            None
        ),
        (
            "DecisionTree",
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [3, 5, 10, None]}
        ),
        (
            "RandomForest",
            RandomForestClassifier(random_state=42),
            {"n_estimators": [50, 100]}
        ),
        (
            "SASKC",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SASKC(n_neighbors=5))
            ]),
            {"clf__n_neighbors": [3, 5, 7]}
        ),
    ]
    return models_config


def run_experiments(n_runs: int = 100) -> pd.DataFrame:
    """
    Run experiments for both clean and noisy Wine data.
    For each model and each run, log accuracy, precision, recall and f1.
    """
    results = []
    data_variants = ["clean", "noisy"]

    for variant in data_variants:
        print(f"\n=== Starting experiments for {variant.upper()} data ===")
        add_noise = (variant == "noisy")
        X, y, _, _ = load_wine_data(add_noise=add_noise)

        for run in tqdm(range(1, n_runs + 1), desc=f"Runs ({variant})"):
            rs = 42 + run
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=rs,
                stratify=y,
            )

            for name, estimator, param_grid in get_models():
                if param_grid:
                    clf = GridSearchCV(
                        estimator,
                        param_grid,
                        cv=3,
                        n_jobs=-1
                    )
                    clf.fit(X_train, y_train)
                    best_model = clf.best_estimator_
                else:
                    best_model = estimator
                    best_model.fit(X_train, y_train)

                y_pred = best_model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
                rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

                results.append(
                    {
                        "run": run,
                        "model": name,
                        "data_variant": variant,
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                    }
                )

    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    print("Running SASKC Experiments on Wine Dataset...")
    df_results = run_experiments(n_runs=100)

    runs_path, summary_path, summary = save_runs_and_summary(df_results)

    print("\nExperiments completed ✅")
    print(f"Detailed per-run results saved to: {runs_path}")
    print(f"Summary statistics saved to       : {summary_path}")
    print("\nSummary (head):")
    print(summary.head())
