# SASKC Research Project

**Supervised Adaptive Statistical Kernel Classifier (SASKC)**

A complete, production-quality, research-grade machine learning project implementing and evaluating a novel supervised classification algorithm.

## 📌 Project Overview

This project introduces **SASKC**, a new non-parametric classification algorithm that improves upon K-Nearest Neighbors (KNN) by incorporating feature-wise statistical weighting and a rank-based voting mechanism. The project is structured as a rigorous research study, comparing SASKC against standard ML models (Logistic Regression, SVM, KNN, Naive Bayes, Decision Tree, Random Forest) on three benchmark datasets: **Wine, Iris, and Breast Cancer**.

**Key Features:**
- **Novel Algorithm**: Full implementation of SASKC from scratch.
- **Rigorous Evaluation**: 100 Monte Carlo cross-validation runs on multiple datasets.
- **Ablation Study**: Analysis of the impact of adaptive weights vs. rank-based voting.
- **Robustness Testing**: Evaluation on both clean and noisy data (Gaussian noise).
- **Interactive Dashboard**: Streamlit app for visualizing results and stability.
- **Research Paper**: Full 11-page IEEE-format conference paper included.

## 🧠 What is SASKC?

**Supervised Adaptive Statistical Kernel Classifier** is designed to handle features with varying degrees of importance automatically.

### Algorithm Steps:

1.  **Feature Weighting**:
    Calculate variance $\sigma_f^2$ for each feature $f$.
    $$w_f = \frac{1}{1 + \sigma_f^2}$$
    (Low variance features get higher weight).

2.  **Weighted Distance**:
    Compute distance between test sample $x_i$ and train sample $x_j$:
    $$D_{ij} = \sqrt{ \sum_f w_f (x_{i,f} - x_{j,f})^2 }$$

3.  **Nearest Neighbors**:
    Select top $k$ neighbors based on $D_{ij}$.

4.  **Similarity & Rank Weights**:
    $$S_j = \frac{1}{1 + D_{ij}}, \quad R_j = \frac{1}{\text{rank}_j}$$

5.  **Class Support Score**:
    $$C_c = \sum_{j \in \text{neighbors}} S_j \cdot R_j \cdot \mathbb{I}(y_j = c)$$

6.  **Prediction**:
    $$\hat{y} = \arg\max_c C_c$$

## 📂 Project Structure

```
SASKC_Research_Project/
├── data/
│   └── raw/                # Raw datasets (e.g., wine.data)
├── results/
│   ├── plots/              # Generated plots (Confusion Matrices, Noise Robustness)
│   └── tables/             # CSV results (Evaluation, Ablation, Runs)
├── src/
│   ├── saskc.py            # Core algorithm implementation
│   ├── comprehensive_evaluation.py # Main evaluation script
│   ├── experiment_runs.py  # Dashboard data generator
│   └── api/                # Flask API
├── dashboard/              # Streamlit app
├── docs/                   # Research paper (LaTeX source & PDF)
└── requirements.txt        # Dependencies
```

## 🚀 How to Run

### 1. Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Comprehensive Evaluation
Execute the main evaluation script to generate all results, tables, and plots:
```bash
python -m src.comprehensive_evaluation
```
This will:
- Evaluate all models on Wine, Iris, and Breast Cancer datasets.
- Run the ablation study.
- Generate plots in `results/plots/`.
- Save CSV tables in `results/tables/`.

### 3. Interactive Dashboard
Visualize the experiment results, stability plots, and comparisons:
```bash
streamlit run dashboard/streamlit_app.py
```
*Note: The dashboard will automatically generate the necessary data if it's missing.*

### 4. REST API
Serve the trained model via Flask:
```bash
python src/api/flask_api.py
```

## 📄 Research Paper
The full 11-page IEEE conference paper is available in the `docs/` directory:
- **PDF**: `docs/IEEE_template.pdf`
- **LaTeX Source**: `docs/IEEE_template.tex`

## 📜 License
MIT License.
