import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- PATHS ----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RUNS_PATH = os.path.join(DATA_DIR, "results_runs.csv")

st.set_page_config(page_title="SASKC Research Dashboard", layout="wide", page_icon="📊")

# Custom CSS for better spacing and readability
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    h1 { font-size: 2.2rem; margin-bottom: 0.5rem; }
    h2 { font-size: 1.6rem; margin-top: 2rem; margin-bottom: 1rem; }
    .stPlotlyChart { height: 450px; }
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
    st.error("Data not found. Run `python src/experiment_runs.py` first.")
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

# 3. Metric Selection (for specific deep-dive plots)
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
    # Aggregate mean for all metrics
    summary = df.groupby("model")[metrics].mean().reset_index()
    # Melt for plotting
    melted = summary.melt(id_vars="model", var_name="Metric", value_name="Score")
    
    fig = px.bar(
        melted, 
        x="Metric", 
        y="Score", 
        color="model", 
        barmode="group",
        title="Model Performance by Metric (Mean)",
        text_auto='.3f',
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(
        height=450,
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
        height=450,
        showlegend=False,
        yaxis_title=metric.capitalize(),
        xaxis_title=""
    )
    return fig

def plot_radar_chart(df):
    """Radar Chart: Holistic view of model shape"""
    summary = df.groupby("model")[metrics].mean().reset_index()
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Bold
    
    for i, model in enumerate(summary["model"]):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatterpolar(
            r=summary.loc[summary["model"] == model, metrics].values.flatten(),
            theta=[m.capitalize() for m in metrics],
            fill='toself',
            name=model,
            line_color=color,
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Holistic Performance Comparison (Radar)",
        height=450,
        template="plotly_white",
        margin=dict(l=80, r=80, t=40, b=40)
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
        height=450,
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

# 3. Radar Chart (Trade-off Analysis)
with col2:
    st.markdown("### 🕸️ Trade-off Analysis")
    st.markdown("Shape comparison to identify balanced models vs. specialists.")
    st.plotly_chart(plot_radar_chart(df_viz), use_container_width=True)

# 4. Convergence Check (Research Quality)
st.markdown("---")
st.markdown(f"### � Experiment Convergence ({primary_metric.capitalize()})")
st.markdown("Verifying that the mean score stabilizes as more runs are added (Monte Carlo convergence).")
st.plotly_chart(plot_convergence(df_viz, primary_metric), use_container_width=True)

# 5. Raw Data Table
with st.expander("� View Raw Summary Statistics"):
    summary_stats = df_viz.groupby("model")[primary_metric].agg(["mean", "std", "min", "max"]).sort_values("mean", ascending=False)
    st.dataframe(summary_stats.style.format("{:.4f}").background_gradient(cmap="Blues", subset=["mean"]), use_container_width=True)

st.markdown("---")
st.caption(f"SASKC Research Dashboard | Data Variant: {selected_variant.upper()} | Runs: {len(df_runs) // len(df_runs['model'].unique())}")
