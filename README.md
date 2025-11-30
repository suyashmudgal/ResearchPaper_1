# SASKC Research Project

**Supervised Adaptive Statistical Kernel Classifier (SASKC)**

A complete, production-quality, research-grade machine learning project implementing and evaluating a novel supervised classification algorithm.

## 📌 Project Overview

This project introduces **SASKC**, a new non-parametric classification algorithm that improves upon K-Nearest Neighbors (KNN) by incorporating feature-wise statistical weighting and a rank-based voting mechanism. The project is structured as a rigorous research study, comparing SASKC against standard ML models (Logistic Regression, SVM, KNN, Naive Bayes, Decision Tree, Random Forest) on the **Wine dataset**.

Key Features:
- **Novel Algorithm**: Full implementation of SASKC from scratch.
- **Rigorous Evaluation**: 100 Monte Carlo cross-validation runs.
- **Robustness Testing**: Evaluation on both clean and noisy data (Gaussian noise).
- **Production Ready**: Includes a REST API (Flask) and an interactive Dashboard (Streamlit).
- **Research Grade**: IEEE-format paper draft and experiment logs.

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

## 📊 Dataset

We use the **Wine dataset** from scikit-learn:
- **Samples**: 178
- **Features**: 13 numeric (Alcohol, Malic acid, Ash, etc.)
- **Classes**: 3 (Cultivars)

## 🧪 Experiments

We perform **100 training cycles** (Monte Carlo Cross-Validation) with:
- **Test Size**: 30%
- **Stratified Split**
- **Models**: LR, SVC, KNN, GNB, DT, RF, SASKC.
- **Metrics**: Accuracy, Precision, Recall, F1.
- **Noise Injection**: Experiments are repeated with 5% Gaussian noise added to features to test robustness.

## 🚀 How to Run

### 1. Setup & Experiments
Run the automated setup script to install dependencies, run experiments, and train the final model:

**Windows (PowerShell):**
```powershell
./run.ps1
```

**Linux/Mac (Bash):**
```bash
./run.sh
```

Or run manually:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run experiments (Generates data/results_runs.csv)
python src/experiment_runs.py

# 3. Train final model (Generates models/best_model_saskc.pkl)
python src/train_best_model.py
```

### 2. Interactive Dashboard
Visualize the experiment results, stability plots, and comparisons:
```bash
streamlit run dashboard/streamlit_app.py
```

### 3. REST API
Serve the trained model via Flask:
```bash
python src/api/flask_api.py
```

**Test with cURL:**
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]}'
```

## 📂 Project Structure

```
SASKC_Research_Project/
├── data/               # Experiment results (CSVs)
├── models/             # Trained .pkl models
├── src/
│   ├── saskc.py        # Core algorithm implementation
│   ├── experiment_runs.py # Experiment runner
│   └── api/            # Flask API
├── dashboard/          # Streamlit app
├── docs/               # Research paper & presentation
└── requirements.txt    # Dependencies
```

## 📜 License
MIT License.
