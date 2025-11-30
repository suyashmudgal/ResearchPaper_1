import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- PATHS ----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
RUNS_PATH = os.path.join(RESULTS_DIR, "results_runs.csv")

st.set_page_config(page_title="SASKC Research Dashboard", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    h1 { font-size: 2.2rem; margin-bottom: 0.5rem; }
    h2 { font-size: 1.6rem; margin-top: 2rem; margin-bottom: 1rem; }
    .stPlotlyChart { height: 450px; }
    </style>
""", unsafe_allow_html=True)

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np # Added for plot_kernel_response

# ---------- PATHS ----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
RUNS_PATH = os.path.join(RESULTS_DIR, "results_runs.csv")

st.set_page_config(page_title="SASKC Research Dashboard", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    h1 { font-size: 2.2rem; margin-bottom: 0.5rem; }
    h2 { font-size: 1.6rem; margin-top: 2rem; margin-bottom: 1rem; }
    .stPlotlyChart { height: 550px; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 SASKC Research: Model Performance Analysis")
st.markdown("Interactive analysis of model stability, accuracy, and trade-offs across 100 Monte Carlo runs.")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    if not os.path.exists(RUNS_PATH):
        return None
    df = pd.read_csv(RUNS_PATH)
    return df

df_runs = load_data()

if df_runs is None:
    st.error(f"Data file not found at `{RUNS_PATH}`.")
    st.info("Please run the comprehensive evaluation script first to generate results:")
    st.code("python -m src.comprehensive_evaluation", language="bash")
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.header("Configuration")

# 1. Data Variant
if "data_variant" in df_runs.columns:
    variants = df_runs["data_variant"].unique()
    selected_variant = st.sidebar.radio("Data Variant", variants, index=0, key="variant_select")
    df_runs = df_runs[df_runs["data_variant"] == selected_variant]
else:
    selected_variant = "clean"

st.sidebar.markdown("---")

# 2. Model Selection
all_models = sorted(df_runs["model"].unique())
selected_models = st.sidebar.multiselect(
    "Models to Compare", 
    all_models, 
    default=all_models,
    key="model_select"
)

if not selected_models:
    st.warning("Select at least one model.")
    st.stop()

df_viz = df_runs[df_runs["model"].isin(selected_models)]

# 3. Metric Selection
metrics = ["accuracy", "precision", "recall", "f1"]
primary_metric = st.sidebar.selectbox(
    "Primary Metric (for Distribution/Trends)", 
    metrics, 
    index=0, 
    format_func=lambda x: x.capitalize(),
    key="metric_select"
)

# ---------- PLOT FUNCTIONS ----------

def plot_multi_metric_bar(df):
    """Grouped Bar Chart: Compare models across ALL metrics"""
    summary = df.groupby("model")[metrics].mean().reset_index()
    melted = summary.melt(id_vars="model", var_name="Metric", value_name="Score")
    
    fig = px.bar(
        melted, 
        x="Metric", 
        y="Score", 
        color="model", 
        barmode="group",
        title="Mean Performance Across Metrics",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Mean Score",
        xaxis_title=""
    )
    fig.update_yaxes(range=[0, 1.05])
    return fig

def plot_stability_box(df, metric):
    """Box Plot: Show distribution stability"""
    fig = px.box(
        df, 
        x="model", 
        y=metric, 
        color="model", 
        points="all", # Show all points for research transparency
        title=f"{metric.capitalize()} Stability Distribution (100 Runs)",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(
        height=550,
        showlegend=False,
        yaxis_title=metric.capitalize(),
        xaxis_title=""
    )
    return fig

def plot_kernel_response():
    """Line Plot: Compare SASKC Kernel vs Gaussian Kernel decay"""
    d = np.linspace(0, 5, 100)
    
    # SASKC Kernel: 1 / (1 + d)
    k_saskc = 1.0 / (1.0 + d)
    
    # Gaussian Kernel: exp(-d^2)
    k_rbf = np.exp(-d**2)
    
    df_kernel = pd.DataFrame({
        "Distance": d,
        "SASKC (Inverse Soft-Plus)": k_saskc,
        "Gaussian (RBF)": k_rbf
    })
    
    df_melted = df_kernel.melt(id_vars="Distance", var_name="Kernel", value_name="Similarity")
    
    fig = px.line(
        df_melted,
        x="Distance",
        y="Similarity",
        color="Kernel",
        title="Kernel Response Analysis: Heavy-Tail Behavior",
        template="plotly_white",
        color_discrete_map={
            "SASKC (Inverse Soft-Plus)": "#00CC96", # Greenish
            "Gaussian (RBF)": "#EF553B" # Reddish
        }
    )
    
    fig.update_layout(
        height=550,
        xaxis_title="Distance (d)",
        yaxis_title="Similarity K(d)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add annotation explaining the heavy tail
    fig.add_annotation(
        x=3.5, y=0.25,
        text="SASKC preserves gradient<br>at larger distances",
        showarrow=True,
        arrowhead=1,
        ax=0, ay=-40
    )
    
    return fig

def plot_convergence(df, metric):
    """Cumulative Mean Line Plot: Check if 100 runs were sufficient"""
    # Calculate cumulative mean for each model
    df_sorted = df.sort_values("run")
    df_sorted["cumulative_mean"] = df_sorted.groupby("model")[metric].transform(lambda x: x.expanding().mean())
    
    fig = px.line(
        df_sorted, 
        x="run", 
        y="cumulative_mean", 
        color="model",
        title=f"Convergence Check: Cumulative Mean {metric.capitalize()}",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(
        height=550,
        yaxis_title=f"Cumulative Mean {metric.capitalize()}",
        xaxis_title="Number of Runs",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ---------- LAYOUT ----------

# 1. Multi-Metric Comparison (The "Executive Summary" View)
st.markdown("### 🏆 Overall Performance Comparison")
st.markdown("Comparison of mean scores across all evaluated metrics. Grouped bars allow for easy side-by-side comparison.")
st.plotly_chart(plot_multi_metric_bar(df_viz), use_container_width=True)

col1, col2 = st.columns(2)

# 2. Stability Analysis (Deep Dive into Primary Metric)
with col1:
    st.markdown(f"### 🎻 Stability Analysis ({primary_metric.capitalize()})")
    st.markdown("Distribution of scores across 100 runs. Tighter boxes indicate more stable models.")
    st.plotly_chart(plot_stability_box(df_viz, primary_metric), use_container_width=True)

# 3. Kernel Response Analysis (Theoretical Insight)
with col2:
    st.markdown("### 📉 Kernel Behavior Analysis")
    st.markdown("Comparison of similarity decay. SASKC's **heavy tail** allows it to capture relationships even in sparse, high-dimensional spaces where Gaussian kernels vanish too quickly.")
    st.plotly_chart(plot_kernel_response(), use_container_width=True)

# 4. Convergence Check (Research Quality)
st.markdown("---")
st.markdown(f"### 🔁 Experiment Convergence ({primary_metric.capitalize()})")
st.markdown("Verifying that the mean score stabilizes as more runs are added (Monte Carlo convergence).")
st.plotly_chart(plot_convergence(df_viz, primary_metric), use_container_width=True)

# 5. Raw Data Table
with st.expander("📄 View Raw Summary Statistics"):
    summary_stats = df_viz.groupby("model")[primary_metric].agg(["mean", "std", "min", "max"]).sort_values("mean", ascending=False)
    st.dataframe(summary_stats.style.format("{:.4f}").background_gradient(cmap="Blues", subset=["mean"]), use_container_width=True)

st.markdown("---")
st.caption(f"SASKC Research Dashboard | Data Variant: {selected_variant.upper()} | Runs: {len(df_runs) // len(df_runs['model'].unique())}")
