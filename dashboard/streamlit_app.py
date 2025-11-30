import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import cdist

# Ensure src is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from src.saskc import SASKC

# ---------- CONFIGURATION ----------
st.set_page_config(
    page_title="SASKC Research Dashboard",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Research-Grade" Aesthetics & Dark Theme
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; font-size: 2rem; }
    h2 { font-family: 'Helvetica Neue', sans-serif; font-weight: 600; font-size: 1.5rem; margin-top: 1.5rem; }
    h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 500; font-size: 1.2rem; }
    .stPlotlyChart { border-radius: 8px; background-color: #1e1e1e; } 
    .metric-card { background-color: #262730; padding: 1rem; border-radius: 8px; border-left: 5px solid #4e73df; }
    /* Fix margins */
    .main > div { padding-left: 2rem; padding-right: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ---------- DATA LOADING ----------
@st.cache_data
def load_results():
    runs_path = os.path.join(PROJECT_ROOT, "results", "tables", "results_runs.csv")
    if os.path.exists(runs_path):
        return pd.read_csv(runs_path)
    return None

@st.cache_data
def load_raw_data():
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    return X, y, feature_names, target_names

df_runs = load_results()
X_raw, y_raw, feat_names, target_names = load_raw_data()

# ---------- MODEL FACTORY ----------
def get_model_instance(model_name):
    if model_name == "SASKC":
        return SASKC(n_neighbors=5, use_weights=True, use_rank_voting=True)
    elif model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "SVM-RBF":
        return SVC(kernel="rbf", probability=True, random_state=42)
    elif model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "Gaussian NB":
        return GaussianNB()
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    return None

# ---------- SIDEBAR ----------
st.sidebar.title("🧠 SASKC Control Panel")
st.sidebar.markdown("---")

if df_runs is not None:
    all_models = sorted(df_runs["model"].unique())
    default_models = [m for m in all_models if m in ["SASKC", "Random Forest", "SVM-RBF"]]
    selected_models = st.sidebar.multiselect("Select Models", all_models, default=default_models)
    
    selected_metric = st.sidebar.selectbox("Primary Metric", ["f1", "accuracy", "precision", "recall"], format_func=lambda x: x.capitalize())
else:
    selected_models = []
    selected_metric = "f1"

st.sidebar.markdown("---")
st.sidebar.info(
    "**SASKC Formulation**:\n"
    r"1. **Variance Weighting**:\n   $w_f = 1/(1+\sigma_f^2+\epsilon)$" "\n"
    r"2. **Kernel**:\n   $K(x_q, x_i) = 1/(1+\sqrt{d^2})$" "\n"
    r"3. **Voting**:\n   $Vote_j = K(x_q, x_j) \cdot (1/j)$"
)

# ---------- VISUALIZATION FUNCTIONS ----------

def plot_performance_table(df, models, metric):
    df_filt = df[df["model"].isin(models)]
    summary = df_filt.groupby("model")[metric].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values("mean", ascending=False)
    summary["mean"] = summary["mean"].round(4)
    summary["std"] = summary["std"].round(4)
    summary.columns = ["Model", f"Mean {metric.capitalize()}", "Std Dev"]
    return summary

def plot_metric_distribution(df, models, metric):
    df_filt = df[df["model"].isin(models)]
    fig = px.box(
        df_filt, x="model", y=metric, color="model",
        title=f"<b>Metric Distribution</b>: {metric.capitalize()} (100 Runs)",
        template="plotly_dark",
        points="all"
    )
    fig.update_layout(height=400, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_convergence_trend(df, models, metric):
    df_filt = df[df["model"].isin(models)].copy()
    df_filt = df_filt.sort_values(["model", "run"])
    
    # Calculate cumulative mean
    df_filt["cumulative_metric"] = df_filt.groupby("model")[metric].expanding().mean().reset_index(level=0, drop=True)
    
    fig = px.line(
        df_filt, x="run", y="cumulative_metric", color="model",
        title=f"<b>Convergence Trend</b>: Cumulative Mean {metric.capitalize()}",
        template="plotly_dark",
        labels={"cumulative_metric": f"Mean {metric.capitalize()}", "run": "Run Index"}
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_noise_robustness(df, models, metric):
    if "data_variant" not in df.columns: return None
    df_filt = df[df["model"].isin(models)]
    summary = df_filt.groupby(["model", "data_variant"])[metric].mean().reset_index()
    
    fig = px.bar(
        summary, x="model", y=metric, color="data_variant", barmode="group",
        title=f"<b>Noise Robustness</b>: Clean vs Noisy",
        template="plotly_dark",
        color_discrete_map={"clean": "#4e73df", "noisy": "#e74a3b"}
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), yaxis_range=[0, 1.05])
    return fig

def plot_confusion_heatmap(model_name, X, y):
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = get_model_instance(model_name)
    if clf is None: return go.Figure()
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(
        cm, text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=target_names, y=target_names,
        color_continuous_scale="Blues",
        title=f"<b>Confusion Matrix</b>: {model_name}",
        template="plotly_dark"
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_manifold_comparison(X, y):
    """Visual: Compare Raw PCA vs. SASKC Weighted PCA"""
    # 1. Raw PCA (Standardized)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca_raw = PCA(n_components=2)
    X_pca_raw = pca_raw.fit_transform(X_std)
    
    # 2. SASKC Weighted PCA (Variance-Informed)
    # We use raw variance to demonstrate the suppression effect
    variances = np.var(X, axis=0)
    weights = 1.0 / (1.0 + variances + 1e-6)
    
    # Apply weights to centered data (not scaled to unit variance, to preserve natural spread)
    X_centered = X - np.mean(X, axis=0)
    X_weighted = X_centered * np.sqrt(weights)
    
    pca_weighted = PCA(n_components=2)
    X_pca_weighted = pca_weighted.fit_transform(X_weighted)
    
    # Prepare DataFrames
    df_raw = pd.DataFrame(X_pca_raw, columns=['PC1', 'PC2'])
    df_raw['Class'] = [target_names[i] for i in y]
    df_raw['Type'] = 'Standard PCA (Equal Weights)'
    
    df_weighted = pd.DataFrame(X_pca_weighted, columns=['PC1', 'PC2'])
    df_weighted['Class'] = [target_names[i] for i in y]
    df_weighted['Type'] = 'SASKC Weighted (Noise Suppressed)'
    
    df_combined = pd.concat([df_raw, df_weighted])
    
    fig = px.scatter(
        df_combined, x='PC1', y='PC2', color='Class', facet_col='Type',
        title="<b>Manifold Transformation</b>: Impact of Variance-Based Weighting",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.8,
        hover_data={'PC1': False, 'PC2': False}
    )
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig

def plot_kernel_heatmap(X, y):
    """Visual: Pairwise Kernel Similarity Matrix"""
    # Sort data by class for block-diagonal visualization
    sort_idx = np.argsort(y)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Compute SASKC Weighted Distances
    variances = np.var(X, axis=0)
    weights = 1.0 / (1.0 + variances + 1e-6)
    V = 1.0 / (weights + 1e-10) # cdist expects V = 1/w for 'seuclidean'
    
    # Distance Matrix
    dists = cdist(X_sorted, X_sorted, metric='seuclidean', V=V)
    
    # Kernel Matrix: K = 1 / (1 + d)
    K = 1.0 / (1.0 + dists)
    
    # Create Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=K,
        x=[target_names[i] for i in y_sorted],
        y=[target_names[i] for i in y_sorted],
        colorscale='Viridis',
        colorbar=dict(title="Similarity K(x,y)")
    ))
    
    fig.update_layout(
        title="<b>Kernel Similarity Matrix</b> (Sorted by Class)",
        xaxis_title="Sample Index (Grouped by Class)",
        yaxis_title="Sample Index (Grouped by Class)",
        template="plotly_dark",
        height=550,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showticklabels=False), # Too many labels
        yaxis=dict(showticklabels=False)
    )
    return fig

def plot_similarity_distribution(X, y):
    """Visual: Violin Plot of Intra- vs Inter-class Similarity"""
    # Compute SASKC Weighted Distances
    variances = np.var(X, axis=0)
    weights = 1.0 / (1.0 + variances + 1e-6)
    V = 1.0 / (weights + 1e-10)
    
    # Distance & Kernel
    dists = cdist(X, X, metric='seuclidean', V=V)
    K = 1.0 / (1.0 + dists)
    
    # Flatten and categorize
    n = len(y)
    similarities = []
    types = []
    
    # Subsample for performance if needed (e.g., max 5000 pairs)
    indices = np.triu_indices(n, k=1)
    # Randomly sample 5000 pairs if too many
    if len(indices[0]) > 5000:
        idx_sample = np.random.choice(len(indices[0]), 5000, replace=False)
        rows = indices[0][idx_sample]
        cols = indices[1][idx_sample]
    else:
        rows, cols = indices
        
    for i, j in zip(rows, cols):
        sim = K[i, j]
        is_same = (y[i] == y[j])
        similarities.append(sim)
        types.append("Intra-Class (Same Label)" if is_same else "Inter-Class (Diff Label)")
        
    df_sim = pd.DataFrame({"Similarity": similarities, "Type": types})
    
    fig = px.violin(
        df_sim, x="Type", y="Similarity", color="Type",
        title="<b>Kernel Similarity Distribution</b>: Class Separation",
        template="plotly_dark",
        box=True, points=False,
        color_discrete_map={"Intra-Class (Same Label)": "#2ca02c", "Inter-Class (Diff Label)": "#d62728"}
    )
    fig.update_layout(height=450, showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    return fig



def plot_metric_comparison_bar(df, models):
    """Visual: Grouped Bar Chart for Multi-Metric Comparison"""
    df_filt = df[df["model"].isin(models)]
    if df_filt.empty: return go.Figure()
    
    metrics = ["accuracy", "precision", "recall", "f1"]
    summary = df_filt.groupby("model")[metrics].mean().reset_index()
    
    # Melt for grouped bar chart
    df_melt = summary.melt(id_vars="model", var_name="Metric", value_name="Score")
    
    fig = px.bar(
        df_melt, x="model", y="Score", color="Metric", barmode="group",
        title="<b>Holistic Model Comparison</b> (Grouped Bar)",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_layout(
        height=450, 
        margin=dict(l=40, r=40, t=50, b=20),
        yaxis_range=[0, 1.05],
        xaxis_title="",
        yaxis_title="Score"
    )
    return fig

def plot_class_performance_heatmap(models, X, y):
    """Visual: Heatmap of Per-Class F1 Scores"""
    from sklearn.metrics import f1_score
    
    # Train/Test Split once
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    results = []
    
    for m_name in models:
        clf = get_model_instance(m_name)
        if clf is not None:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            # Per-class F1
            scores = f1_score(y_test, y_pred, average=None)
            for cls_idx, score in enumerate(scores):
                results.append({
                    "Model": m_name,
                    "Class": target_names[cls_idx],
                    "F1 Score": score
                })
                
    if not results: return go.Figure()
    
    df_res = pd.DataFrame(results)
    
    # Pivot for heatmap
    df_pivot = df_res.pivot(index="Class", columns="Model", values="F1 Score")
    
    fig = px.imshow(
        df_pivot,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        title="<b>Per-Class Performance</b> (F1 Score)",
        template="plotly_dark",
        aspect="auto"
    )
    fig.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_feature_importance_lollipop(X, feature_names):
    """Visual: Lollipop Chart of Top 10 Weighted Features"""
    # Calculate Weights
    variances = np.var(X, axis=0)
    weights = 1.0 / (1.0 + variances + 1e-6)
    
    # Identify Top 10 Features
    top_indices = np.argsort(weights)[-10:] # Ascending order for plot
    top_feats = [feature_names[i] for i in top_indices]
    top_weights = weights[top_indices]
    top_vars = variances[top_indices]
    
    fig = go.Figure()
    
    # Draw lines (sticks)
    for feat, w, v in zip(top_feats, top_weights, top_vars):
        fig.add_trace(go.Scatter(
            x=[0, w], y=[feat, feat],
            mode='lines',
            line=dict(color='#555', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
    # Draw markers (heads)
    fig.add_trace(go.Scatter(
        x=top_weights, y=top_feats,
        mode='markers',
        marker=dict(
            size=12,
            color=top_vars,
            colorscale='RdYlGn_r', # Red=High Var (Bad), Green=Low Var (Good)
            showscale=True,
            colorbar=dict(title="Variance")
        ),
        text=[f"Var: {v:.2f}" for v in top_vars],
        hovertemplate="<b>%{y}</b><br>Weight: %{x:.4f}<br>%{text}<extra></extra>",
        name="Features"
    ))
    
    fig.update_layout(
        title="<b>Top 10 Reliable Features</b> (Lollipop Chart)",
        xaxis_title="SASKC Weight (Importance)",
        yaxis_title="",
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
    )
    return fig

def plot_feature_weights(X, feature_names):
    variances = np.var(X, axis=0)
    weights = 1.0 / (1.0 + variances + 1e-6)
    
    df_feat = pd.DataFrame({"Feature": feature_names, "Weight": weights, "Variance": variances})
    df_feat = df_feat.sort_values("Weight", ascending=True)
    
    fig = px.bar(
        df_feat, y="Feature", x="Weight", color="Variance",
        title="<b>SASKC Feature Weights</b> (Variance-Based)",
        template="plotly_dark",
        orientation='h',
        color_continuous_scale="RdYlGn_r"
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_kernel_curve():
    d = np.linspace(0, 5, 100)
    k_saskc = 1.0 / (1.0 + d)
    k_rbf = np.exp(-d**2)
    
    df_k = pd.DataFrame({"Distance": d, "SASKC": k_saskc, "RBF": k_rbf})
    df_melt = df_k.melt("Distance", var_name="Kernel", value_name="Similarity")
    
    fig = px.line(
        df_melt, x="Distance", y="Similarity", color="Kernel",
        title="<b>Kernel Response</b>: SASKC vs RBF",
        template="plotly_dark",
        color_discrete_map={"SASKC": "#2ca02c", "RBF": "#d62728"}
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# ---------- MAIN LAYOUT ----------

st.title("📊 SASKC Research Dashboard")

if not selected_models:
    st.warning("Please select at least one model from the sidebar.")
    st.stop()

tab_overview, tab1, tab2 = st.tabs(["🧠 Algorithm Overview", "📈 Performance Evaluation", "🔬 Manifold Visualization"])

with tab_overview:
    st.markdown("### SASKC Algorithm Pipeline")
    
    # Visual Pipeline using HTML/CSS for cleaner control than columns
    st.markdown("""
    <div style="background-color: #262730; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; border: 1px solid #444;">
        <h4 style="text-align: center; color: #4e73df; margin-bottom: 1.5rem; font-family: 'Helvetica Neue', sans-serif;">End-to-End Classification Flow</h4>
        <div style="display: flex; justify-content: space-between; align-items: center; text-align: center; flex-wrap: wrap; gap: 10px;">
            <div style="flex: 1; min-width: 80px;">
                <div style="font-size: 1.2rem;">📂</div>
                <b>Raw Data</b><br><span style="font-size: 0.8em; color: #aaa;">Input X</span>
            </div>
            <div style="font-size: 1.2em; color: #666;">➔</div>
            <div style="flex: 1; min-width: 80px;">
                <div style="font-size: 1.2rem;">📊</div>
                <b>Variance</b><br><span style="font-size: 0.8em; color: #aaa;">Compute σ²</span>
            </div>
            <div style="font-size: 1.2em; color: #666;">➔</div>
            <div style="flex: 1; min-width: 100px;">
                <div style="font-size: 1.2rem;">⚖️</div>
                <b>Weighting</b><br><span style="font-size: 0.8em; color: #aaa;">w = 1/(1+σ²)</span>
            </div>
            <div style="font-size: 1.2em; color: #666;">➔</div>
            <div style="flex: 1; min-width: 100px;">
                <div style="font-size: 1.2rem;">📏</div>
                <b>Distance</b><br><span style="font-size: 0.8em; color: #aaa;">Weighted d_W</span>
            </div>
            <div style="font-size: 1.2em; color: #666;">➔</div>
            <div style="flex: 1; min-width: 100px;">
                <div style="font-size: 1.2rem;">🔔</div>
                <b>Kernel</b><br><span style="font-size: 0.8em; color: #aaa;">K = 1/(1+d)</span>
            </div>
            <div style="font-size: 1.2em; color: #666;">➔</div>
            <div style="flex: 1; min-width: 100px;">
                <div style="font-size: 1.2rem;">🗳️</div>
                <b>Voting</b><br><span style="font-size: 0.8em; color: #aaa;">Rank Decay</span>
            </div>
            <div style="font-size: 1.2em; color: #666;">➔</div>
            <div style="flex: 1; min-width: 80px;">
                <div style="font-size: 1.2rem;">🎯</div>
                <b>Prediction</b><br><span style="font-size: 0.8em; color: #aaa;">Class Label</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1. Adaptive Feature Suppression")
        st.markdown("SASKC automatically down-weights features with high global variance (noise), focusing the manifold on stable structures.")
        st.plotly_chart(plot_feature_weights(X_raw, feat_names), use_container_width=True, key="overview_weights")
        
    with col2:
        st.markdown("#### 2. Heavy-Tail Kernel Response")
        st.markdown("The statistical kernel decays slower than Gaussian (RBF), maintaining gradients for distant neighbors in high-dimensional space.")
        st.plotly_chart(plot_kernel_curve(), use_container_width=True, key="overview_kernel")

with tab1:
    st.markdown("### Model Performance Analysis")
    
    # Row 1: Table & Boxplot
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Summary Statistics")
        st.dataframe(plot_performance_table(df_runs, selected_models, selected_metric), use_container_width=True, hide_index=True)
    with col2:
        st.plotly_chart(plot_metric_distribution(df_runs, selected_models, selected_metric), use_container_width=True)
        
    # Row 2: Convergence & Noise
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_convergence_trend(df_runs, selected_models, selected_metric), use_container_width=True)
    with col4:
        st.plotly_chart(plot_noise_robustness(df_runs, selected_models, selected_metric), use_container_width=True)
        
    # Row 3: Advanced Comparison (New)
    st.markdown("#### Advanced Model Comparison")
    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(plot_metric_comparison_bar(df_runs, selected_models), use_container_width=True)
    with col6:
        st.plotly_chart(plot_class_performance_heatmap(selected_models, X_raw, y_raw), use_container_width=True)

    # Row 4: Confusion Matrix (Dynamic)
    st.markdown("#### Detailed Error Analysis")
    cm_model = st.selectbox("Select Model for Confusion Matrix", selected_models, index=0)
    st.plotly_chart(plot_confusion_heatmap(cm_model, X_raw, y_raw), use_container_width=True)

with tab2:
    st.markdown("### Manifold & Algorithm Behavior")
    
    st.markdown("#### 1. Manifold Transformation")
    st.markdown("Comparing standard PCA (left) with SASKC's weighted feature space (right). Notice how high-variance features are suppressed, potentially altering class separability.")
    st.plotly_chart(plot_manifold_comparison(X_raw, y_raw), use_container_width=True, key="manifold_comparison")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("#### 2. Learned Similarity Structure")
        st.markdown("Violin plot showing the distribution of kernel similarity scores. **SASKC aims for high Intra-Class similarity (Green) and low Inter-Class similarity (Red).**")
        st.plotly_chart(plot_similarity_distribution(X_raw, y_raw), use_container_width=True, key="manifold_similarity")
        
    with col2:
        st.markdown("#### 3. Feature Reliability")
        st.markdown("Lollipop chart of the **Top 10 Reliable Features**. Features with low variance (Green) receive higher weights, while high-variance features are suppressed.")
        st.plotly_chart(plot_feature_importance_lollipop(X_raw, feat_names), use_container_width=True, key="manifold_lollipop")

st.markdown("---")
st.caption("SASKC Research Project | Optimized for IEEE Paper Alignment")
