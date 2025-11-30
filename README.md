# SASKC Research Project

**Supervised Adaptive Statistical Kernel Classifier (SASKC)**

A complete, production-quality, research-grade machine learning project implementing and evaluating a novel supervised classification algorithm designed for small-sample, high-dimensional datasets.

## 📌 Project Overview

This project introduces **SASKC**, a new non-parametric classification algorithm that improves upon K-Nearest Neighbors (KNN) by mitigating the "curse of dimensionality." It incorporates:
1.  **Adaptive Statistical Kernel**: Dynamically weights features based on their global variance to suppress noise.
2.  **Riemannian Manifold Construction**: Transforms the feature space to emphasize reliable signal directions.
3.  **Rank-Based Voting**: Prioritizes the most similar neighbors to reduce the influence of outliers.

The project is structured as a rigorous research study, comparing SASKC against standard ML models (Logistic Regression, SVM, KNN, Naive Bayes, Decision Tree, Random Forest) on three benchmark datasets: **Wine, Iris, and Breast Cancer**.

## 🧠 Algorithm Logic

SASKC operates on the principle that feature variance is a proxy for reliability in small-sample regimes.

### 1. Feature Weighting (The Statistical Kernel)
For each feature $f$, we calculate the global variance $\sigma_f^2$. The adaptive weight $w_f$ is defined as:
$$w_f = \frac{1}{1 + \sigma_f^2 + \epsilon}$$
*   **Intuition**: High variance features ($\sigma^2 \to \infty$) are treated as noise and down-weighted ($w \to 0$). Low variance features are preserved.

### 2. Weighted Distance Metric
We compute the distance between a query point $x_q$ and a training point $x_i$ on the learned manifold:
$$D(x_q, x_i) = \sqrt{ \sum_{f=1}^F w_f (x_{q,f} - x_{i,f})^2 }$$

### 3. Rank-Based Voting
Instead of simple majority voting, SASKC weights the contribution of each neighbor $j$ by its rank:
$$\text{Vote}_j = \underbrace{\frac{1}{1 + D(x_q, x_j)}}_{\text{Kernel Similarity}} \times \underbrace{\frac{1}{\text{rank}_j}}_{\text{Rank Decay}}$$
This ensures that the closest neighbors have exponentially more influence than distant ones.

## 📂 Project Structure

```
SASKC_Research_Project/
├── data/                   # Dataset storage
├── results/
│   ├── plots/              # Generated plots (Confusion Matrices, Noise Robustness)
│   └── tables/             # CSV results (Evaluation, Ablation, Raw Runs)
├── src/
│   ├── saskc.py            # Core SASKC algorithm implementation (Optimized)
│   ├── comprehensive_evaluation.py # Main evaluation pipeline
│   └── api/                # Flask API for inference
├── dashboard/              # Streamlit interactive dashboard
├── docs/                   # Research paper (LaTeX source & PDF)
└── requirements.txt        # Python dependencies
```

## 🚀 How to Run

### 1. Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Comprehensive Evaluation
Execute the main evaluation pipeline to train models, run experiments, and generate all artifacts:
```bash
python -m src.comprehensive_evaluation
```
*   **Output**: Generates `results/tables/evaluation_results.csv`, `results/tables/results_runs.csv`, and plots in `results/plots/`.

### 3. Interactive Dashboard
Visualize the experiment results, stability plots, and trade-off analysis:
```bash
streamlit run dashboard/streamlit_app.py
```

### 4. REST API
Serve the trained model via Flask for real-time predictions:
```bash
python src/api/flask_api.py
```

## 📥 Input/Output Format

### API Input (JSON)
```json
{
  "features": [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050]
}
```

### API Output (JSON)
```json
{
  "prediction": "class_0",
  "probabilities": {
    "class_0": 0.98,
    "class_1": 0.02,
    "class_2": 0.00
  }
}
```

## 📄 Research Paper
The full 11-page IEEE conference paper is available in the `docs/` directory:
- **PDF**: `docs/IEEE_template.pdf`
- **LaTeX Source**: `docs/IEEE_template.tex`

## 📜 License
MIT License.
