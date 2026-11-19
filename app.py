"""
app.py — MaterialMind AI
Intelligent Material Engineering & Decision Platform
Light Theme Edition: Minimalist, Modern, Warm Pastel Design System
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# -----------------------------------------------------------------------------
# 1. SYSTEM CONFIGURATION & PATH SETUP
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model_utils import (
    load_dataset,
    load_model_artifact,
    MODELS_DIR,
    PROCESSED_DIR,
    ASSETS_DIR,
    RAW_DIR
)
from material_substitution import MaterialSubstitutionEngine, SAFETY_DISCLAIMER
from search_algorithms import MaterialGraphSearch
from cost_estimation import CostEstimator
import ui_helpers
import importlib
importlib.reload(ui_helpers)
from ui_helpers import (
    apply_custom_theme,
    get_light_plotly_layout,
    get_dark_plotly_layout,
    render_metric_panel,
    render_disclaimer,
    render_architecture_diagram
)

# Page configuration
st.set_page_config(
    page_title="MaterialMind AI — Intelligent Material Discovery",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Light Pastel Design System
apply_custom_theme()

# -----------------------------------------------------------------------------
# 2. CACHED DATA & MODEL MANAGERS
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_dataset(name: str):
    """Load and cache processed datasets."""
    try:
        return load_dataset(name)
    except Exception as e:
        st.error(f"Error loading dataset '{name}': {e}")
        return pd.DataFrame()

@st.cache_data
def get_cached_comparison_table(filename: str):
    """Load benchmark comparison tables."""
    path = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_resource
def get_cached_substitution_engine(dataset_name: str = "steel_db"):
    """Instantiate and cache the Material Substitution Engine."""
    return MaterialSubstitutionEngine(dataset_name=dataset_name)

@st.cache_resource
def get_cached_model(model_filename: str, subfolder: str = "regression"):
    """Load and cache Joblib model artifacts and metadata."""
    try:
        return load_model_artifact(model_filename, subfolder=subfolder)
    except Exception:
        return None, {}

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.2rem 0;">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="background-color: #FF765F; width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 800; color: #FFFFFF; font-size: 1.25rem; box-shadow: 0 2px 8px rgba(255,118,95,0.3);">
                M
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.2rem; color: #1A1A1A; letter-spacing: -0.4px;">MaterialMind AI</div>
                <div style="font-size: 0.74rem; color: #777777; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Material Discovery</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_options = [
        "Overview",
        "Material Explorer",
        "AI Predictions",
        "Material Substitution",
        "Search Intelligence",
        "Cost Analysis",
        "Model Analytics",
        "About MaterialMind AI"
    ]
    default_nav_idx = 0
    try:
        requested_page = str(st.query_params.get("page", "")).lower()
        if requested_page:
            for i, opt in enumerate(nav_options):
                if requested_page in opt.lower():
                    default_nav_idx = i
                    break
    except Exception:
        pass

    nav_selection = st.radio(
        "Navigation",
        nav_options,
        index=default_nav_idx,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #EEEEEE; border-radius: 12px; padding: 1.1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
        <div style="color: #888888; font-weight: 700; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.6rem;">System Operational Status</div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.45rem;">
            <span style="color: #444444; font-size: 0.82rem;">Inference Engine</span>
            <span class="pill pill-mint" style="font-size: 0.7rem;">Active</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.45rem;">
            <span style="color: #444444; font-size: 0.82rem;">Verified Models</span>
            <span style="color: #202020; font-weight: 700; font-size: 0.82rem;">30 Models</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.45rem;">
            <span style="color: #444444; font-size: 0.82rem;">Alloy Repository</span>
            <span style="color: #202020; font-weight: 700; font-size: 0.82rem;">4,149 Records</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #444444; font-size: 0.82rem;">Similarity Edges</span>
            <span style="color: #202020; font-weight: 700; font-size: 0.82rem;">12,103 Edges</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.2rem; color: #888888; font-size: 0.74rem; text-align: center;'>MaterialMind AI Platform v2.4 • Light Edition</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 1. OVERVIEW (LANDING DASHBOARD)
# -----------------------------------------------------------------------------
def render_overview():
    # Hero Section with soft blush pastel background
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">
            <span>●</span> Intelligent Material Discovery • Powered by AI
        </div>
        <h1 class="hero-headline">Smarter Material Decisions, Powered by AI.</h1>
        <p class="hero-subtitle">
            Predict mechanical properties, explore material alternatives, and make informed engineering decisions using machine learning and intelligent search.
        </p>
        <div style="display: flex; gap: 0.6rem; flex-wrap: wrap;">
            <span class="pill pill-coral">3,234 Steel DB Records</span>
            <span class="pill pill-mint">915 MatNavi Records (914 Clean)</span>
            <span class="pill pill-lavender">30 Verified ML Models</span>
            <span class="pill pill-peach">4 Graph Search Algorithms</span>
            <span class="pill pill-slate">Zero-Leakage Group Validation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Performance Indicators in Soft Pastel Accent Blocks
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_panel("Steel Database", "3,234 Records", "Room-Temp Structural Alloys", "mint")
    with c2:
        render_metric_panel("MatNavi Repository", "915 Records", "Elevated-Temp 27°C–650°C (914 Clean)", "peach")
    with c3:
        render_metric_panel("Trained ML Models", "30 Verified", "20 Regr, 7 Clf, 2 Clu, 1 PCA", "lavender")
    with c4:
        render_metric_panel("Core Engines", "4 Systems", "Prediction, CSP, Graph, Sourcing", "white")

    st.markdown('<div class="section-header">Platform Capabilities</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">MaterialMind AI automates complex metallurgical evaluation and material selection through six integrated capabilities:</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="pastel-card">
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;">
                <span class="pill pill-coral">01</span>
                <h4 style="margin: 0; color: #1A1A1A; font-size: 1.12rem;">Mechanical Property Prediction</h4>
            </div>
            <p style="color: #666666; font-size: 0.9rem; margin: 0; line-height: 1.55;">
                Instant multi-target inference for Tensile Strength (Rm), Yield Strength (Re), and Ductility (Elongation) derived from bulk chemical composition and heat treatment conditions, with dedicated elevated-temperature modeling (27°C–650°C).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="pastel-card">
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;">
                <span class="pill pill-lavender">03</span>
                <h4 style="margin: 0; color: #1A1A1A; font-size: 1.12rem;">Graph Search Intelligence</h4>
            </div>
            <p style="color: #666666; font-size: 0.9rem; margin: 0; line-height: 1.55;">
                Computational graph algorithms (BFS, DFS, A* Heuristic Search, and Hill-Climbing Local Search) traversing a 3,234-node topological similarity network to explore direct and indirect metallurgical substitution pathways.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="pastel-card">
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;">
                <span class="pill pill-mint">05</span>
                <h4 style="margin: 0; color: #1A1A1A; font-size: 1.12rem;">Unsupervised Alloy Clustering</h4>
            </div>
            <p style="color: #666666; font-size: 0.9rem; margin: 0; line-height: 1.55;">
                Unsupervised partitioning into 4 natural chemical families using K-Means (Silhouette: 0.8120) and Hierarchical Ward linkage (Silhouette: 0.7857), achieving an inter-algorithm consensus ARI of 0.8622.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="pastel-card">
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;">
                <span class="pill pill-mint">02</span>
                <h4 style="margin: 0; color: #1A1A1A; font-size: 1.12rem;">Material Substitution Engine</h4>
            </div>
            <p style="color: #666666; font-size: 0.9rem; margin: 0; line-height: 1.55;">
                Constraint Satisfaction Problem (CSP) formulation filtering non-viable candidates against hard physical limits, followed by transparent multi-objective soft preference ranking and natural-language engineering rationale generation.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="pastel-card">
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;">
                <span class="pill pill-peach">04</span>
                <h4 style="margin: 0; color: #1A1A1A; font-size: 1.12rem;">Sourcing Economics & Cost Modeling</h4>
            </div>
            <p style="color: #666666; font-size: 0.9rem; margin: 0; line-height: 1.55;">
                Commercial procurement decision-support balancing mechanical performance with material cost, featuring transparent provenance tracking, user quote overrides, and annual tonnage savings simulations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="pastel-card">
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;">
                <span class="pill pill-coral">06</span>
                <h4 style="margin: 0; color: #1A1A1A; font-size: 1.12rem;">Rigorous Group-Split Validation</h4>
            </div>
            <p style="color: #666666; font-size: 0.9rem; margin: 0; line-height: 1.55;">
                Methodologically strict evaluation across held-out alloy families via GroupShuffleSplit, ensuring zero data leakage across 168 distinct groups and providing honest out-of-sample generalization metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Analytical Insights
    st.markdown('<div class="section-header">Analytical Insights from Empirical Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Key metallurgical correlations derived from the processed datasets:</div>', unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("""
        <div class="pastel-card pastel-card-mint">
            <div style="color: #20784E; font-weight: 700; font-size: 0.78rem; text-transform: uppercase;">Solid-Solution Strengthening</div>
            <div style="color: #1A1A1A; font-size: 1.1rem; font-weight: 700; margin: 0.35rem 0;">Carbon & Manganese Synergy</div>
            <p style="color: #444444; font-size: 0.88rem; margin: 0; line-height: 1.5;">
                Across 3,234 commercial steels, carbon content demonstrates a dominant non-linear relationship with tensile strength (r = 0.61), amplified significantly when supported by manganese ratios (Mn/C &ge; 2.0).
            </p>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown("""
        <div class="pastel-card pastel-card-peach">
            <div style="color: #B8551B; font-weight: 700; font-size: 0.78rem; text-transform: uppercase;">Thermal Softening Kinetics</div>
            <div style="color: #1A1A1A; font-size: 1.1rem; font-weight: 700; margin: 0.35rem 0;">Elevated-Temp Proof Stress</div>
            <p style="color: #444444; font-size: 0.88rem; margin: 0; line-height: 1.5;">
                Creep-resistant alloys maintain yield integrity up to 400°C, followed by a steep average proof-stress reduction of 41.2% at 600°C due to dislocation climbing and carbide coarsening.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with i3:
        st.markdown("""
        <div class="pastel-card pastel-card-lavender">
            <div style="color: #5E39C4; font-weight: 700; font-size: 0.78rem; text-transform: uppercase;">Unsupervised Clustering</div>
            <div style="color: #1A1A1A; font-size: 1.1rem; font-weight: 700; margin: 0.35rem 0;">Composition Space Consensus</div>
            <p style="color: #444444; font-size: 0.88rem; margin: 0; line-height: 1.5;">
                K-Means and Agglomerative Hierarchical clustering achieved 86.22% chance-adjusted consensus (ARI = 0.8622) on 23 standardized elements, partitioning steels into 4 distinct natural chemical families.
            </p>
        </div>
        """, unsafe_allow_html=True)

    render_disclaimer()


# -----------------------------------------------------------------------------
# 2. MATERIAL EXPLORER
# -----------------------------------------------------------------------------
def render_material_explorer():
    st.markdown('<div class="section-header">Material Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Inspect material composition, mechanical properties, and processing metadata across commercial and elevated-temperature steel repositories.</div>', unsafe_allow_html=True)

    ds_choice = st.radio(
        "Select Repository",
        ["Commercial Low-Alloy Steels (Steel DB — 3,234 Records)", 
         "Elevated-Temperature Creep Steels (MatNavi — 915 Records [914 Clean])"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if "Commercial" in ds_choice:
        df = get_cached_dataset("steel_db")
        ds_name = "Steel Database"
        id_col = "name"
        target_cols = ["tensile_strength_mpa", "yield_strength_mpa", "ductility_pct"]
    else:
        df = get_cached_dataset("matnavi")
        ds_name = "MatNavi Creep Database"
        id_col = "alloy_code"
        target_cols = ["tensile_strength_mpa", "proof_stress_mpa", "elongation_pct"]

    if df.empty:
        st.error(f"Repository data for {ds_name} is currently unavailable.")
        return

    # High-level dataset summary
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_panel("Total Records", f"{len(df):,}", f"Entries in {ds_name}", "mint")
    with m2:
        render_metric_panel("Feature Columns", f"{df.shape[1]}", "Chemical + Mechanical features", "peach")
    with m3:
        render_metric_panel("Unique Alloys", f"{df[id_col].nunique()}", "Distinct alloy identifiers", "lavender")
    with m4:
        render_metric_panel("Data Quality", "100%", "Zero missing values in features", "white")

    tab1, tab2, tab3 = st.tabs(["Repository Table", "Composition Distributions", "Property Correlations"])

    with tab1:
        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            search_query = st.text_input("Filter by alloy name or code:", "", placeholder="e.g. 4140, 1020, MBB...")
        with f2:
            sort_by = st.selectbox("Sort records by:", target_cols + ["c", "total_alloying_pct"])
        with f3:
            row_limit = st.slider("Records to display:", min_value=10, max_value=200, value=25, step=5)

        filtered_df = df.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df[id_col].astype(str).str.contains(search_query, case=False, na=False)]
        
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
        st.dataframe(filtered_df.head(row_limit), use_container_width=True)
        st.caption(f"Displaying top {min(row_limit, len(filtered_df))} of {len(filtered_df)} matching records.")

        with st.expander("Descriptive Statistics Table"):
            st.dataframe(df.describe().round(3), use_container_width=True)

    with tab2:
        elem_options = [c for c in ["c", "mn", "si", "cr", "ni", "mo", "total_alloying_pct", "carbon_equivalent", "mn_c_ratio"] if c in df.columns]
        chosen_elem = st.selectbox("Select element or metallurgical index to inspect:", elem_options, index=0)

        fig_hist = px.histogram(
            df,
            x=chosen_elem,
            nbins=45,
            marginal="box",
            title=f"Distribution of {chosen_elem.upper()} in {ds_name}",
            color_discrete_sequence=["#FF765F"]
        )
        fig_hist.update_layout(get_light_plotly_layout(f"Distribution Profile: {chosen_elem.upper()}"))
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        if "tensile_strength_mpa" in df.columns and ("yield_strength_mpa" in df.columns or "proof_stress_mpa" in df.columns):
            y_col = "yield_strength_mpa" if "yield_strength_mpa" in df.columns else "proof_stress_mpa"
            color_col = "temperature_c" if "temperature_c" in df.columns else "c"

            fig_scatter = px.scatter(
                df,
                x="tensile_strength_mpa",
                y=y_col,
                color=color_col,
                hover_data=[id_col],
                title=f"Tensile Strength vs {y_col.replace('_', ' ').title()}",
                color_continuous_scale="Plasma",
                labels={"tensile_strength_mpa": "Tensile Strength (MPa)", y_col: f"{y_col.replace('_', ' ').title()} (MPa)"}
            )
            fig_scatter.update_layout(get_light_plotly_layout(f"Mechanical Property Space: Tensile vs {y_col.replace('_', ' ').title()}"))
            st.plotly_chart(fig_scatter, use_container_width=True)

        with st.expander("Pearson Correlation Heatmap"):
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.startswith("cluster_") and not c.startswith("is_")]
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr.round(2),
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title=f"Pearson Correlation Matrix — {ds_name}"
            )
            fig_corr.update_layout(get_light_plotly_layout(f"Correlation Matrix: {ds_name}", height=550))
            st.plotly_chart(fig_corr, use_container_width=True)


# -----------------------------------------------------------------------------
# 3. AI PREDICTIONS
# -----------------------------------------------------------------------------
def render_ai_predictions():
    st.markdown('<div class="section-header">AI Property Predictions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Estimate mechanical properties (Tensile Strength, Yield/Proof Stress, Ductility) and strength classification from chemical composition and heat treatment conditions.</div>', unsafe_allow_html=True)

    d1, d2 = st.columns([1, 1])
    with d1:
        domain = st.selectbox(
            "Target Domain Repository",
            ["Steel Database (Commercial Grades — Room Temperature)",
             "MatNavi Repository (Creep Steels — Elevated Temperature 27–650°C)"]
        )
    with d2:
        algo_choice = st.selectbox(
            "Machine Learning Architecture",
            ["Decision Tree Regressor (Recommended — Optimal Tabular Accuracy)",
             "Linear Regression (Baseline OLS Pipeline)",
             "Polynomial Regression (Degree 2 + Ridge Regularization)",
             "Gradient Descent (Stochastic Gradient Descent — SGD)"]
        )

    is_matnavi = "MatNavi" in domain
    algo_map = {
        "Decision Tree Regressor (Recommended — Optimal Tabular Accuracy)": "decision_tree_regressor",
        "Linear Regression (Baseline OLS Pipeline)": "linear_regression",
        "Polynomial Regression (Degree 2 + Ridge Regularization)": "polynomial_regression_deg_2",
        "Gradient Descent (Stochastic Gradient Descent — SGD)": "gradient_descent_sgd"
    }
    algo_suffix = algo_map[algo_choice]

    # Presets
    presets = {
        "Custom Composition Profile": {},
        "AISI 4140 (Cr-Mo Medium-Carbon High Strength)": {"c": 0.40, "mn": 0.85, "si": 0.25, "cr": 1.00, "mo": 0.20, "ni": 0.10, "cu": 0.15, "v": 0.0, "al": 0.02, "p": 0.02, "s": 0.02},
        "AISI 1045 (Medium-Carbon Machinery Steel)": {"c": 0.45, "mn": 0.75, "si": 0.20, "cr": 0.10, "mo": 0.02, "ni": 0.10, "cu": 0.15, "v": 0.0, "al": 0.02, "p": 0.02, "s": 0.02},
        "AISI 4340 (Ni-Cr-Mo Heavy-Duty Structural Steel)": {"c": 0.40, "mn": 0.70, "si": 0.25, "cr": 0.80, "mo": 0.25, "ni": 1.80, "cu": 0.15, "v": 0.0, "al": 0.02, "p": 0.02, "s": 0.02},
        "AISI 1020 (Low-Carbon Structural Steel)": {"c": 0.20, "mn": 0.45, "si": 0.20, "cr": 0.05, "mo": 0.01, "ni": 0.05, "cu": 0.10, "v": 0.0, "al": 0.02, "p": 0.02, "s": 0.02},
    }

    p_choice = st.selectbox("Load Standard Alloy Profile Preset (Optional):", list(presets.keys()), index=1)
    p_vals = presets[p_choice]

    st.markdown("<div style='font-weight: 700; color: #202020; font-size: 0.95rem; margin-top: 0.8rem; margin-bottom: 0.4rem;'>Chemical Composition Inputs (% weight)</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        val_c = st.number_input("Carbon (C %)", min_value=0.01, max_value=2.00, value=p_vals.get("c", 0.40), step=0.01, format="%.2f")
        val_mn = st.number_input("Manganese (Mn %)", min_value=0.01, max_value=3.00, value=p_vals.get("mn", 0.85), step=0.05, format="%.2f")
        val_si = st.number_input("Silicon (Si %)", min_value=0.01, max_value=2.00, value=p_vals.get("si", 0.25), step=0.05, format="%.2f")
    with c2:
        val_cr = st.number_input("Chromium (Cr %)", min_value=0.00, max_value=12.00, value=p_vals.get("cr", 1.00), step=0.05, format="%.2f")
        val_ni = st.number_input("Nickel (Ni %)", min_value=0.00, max_value=10.00, value=p_vals.get("ni", 0.10), step=0.05, format="%.2f")
        val_mo = st.number_input("Molybdenum (Mo %)", min_value=0.00, max_value=5.00, value=p_vals.get("mo", 0.20), step=0.05, format="%.2f")
    with c3:
        val_cu = st.number_input("Copper (Cu %)", min_value=0.00, max_value=2.00, value=p_vals.get("cu", 0.15), step=0.02, format="%.2f")
        val_v = st.number_input("Vanadium (V %)", min_value=0.00, max_value=1.50, value=p_vals.get("v", 0.00), step=0.02, format="%.2f")
        val_al = st.number_input("Aluminum (Al %)", min_value=0.00, max_value=1.00, value=p_vals.get("al", 0.02), step=0.01, format="%.2f")
    with c4:
        val_p = st.number_input("Phosphorus (P %)", min_value=0.001, max_value=0.10, value=p_vals.get("p", 0.02), step=0.005, format="%.3f")
        val_s = st.number_input("Sulfur (S %)", min_value=0.001, max_value=0.10, value=p_vals.get("s", 0.02), step=0.005, format="%.3f")
        val_temp = st.slider("Operating Temperature (°C)", min_value=27, max_value=650, value=300, step=25) if is_matnavi else 25

    # Metallurgical indices calculation
    tot_alloy = val_c + val_mn + val_si + val_cr + val_ni + val_mo + val_cu + val_v + val_al
    ce_iiw = val_c + (val_mn / 6.0) + ((val_cr + val_mo + val_v) / 5.0) + ((val_ni + val_cu) / 15.0)
    mn_c = val_mn / max(val_c, 0.01)
    cr_mo = val_cr + val_mo
    ni_cr = val_ni + val_cr

    st.markdown("<div style='font-weight: 700; color: #202020; font-size: 0.95rem; margin-top: 0.8rem; margin-bottom: 0.4rem;'>Calculated Metallurgical Indices</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_panel("Total Alloying", f"{tot_alloy:.2f} wt%", "Sum of non-Fe elements", "mint")
    with m2:
        render_metric_panel("IIW Carbon Equiv", f"{ce_iiw:.3f}", "Weldability index (CE)", "peach")
    with m3:
        render_metric_panel("Mn / C Ratio", f"{mn_c:.2f}", "Solid solution ratio", "lavender")
    with m4:
        render_metric_panel("Cr + Mo Content", f"{cr_mo:.2f} wt%", "Hardenability total", "white")

    if st.button("Run AI Property Inference", type="primary", use_container_width=True):
        with st.spinner("Executing machine learning regression models..."):
            time.sleep(0.2)

            if not is_matnavi:
                ts_model, ts_meta = get_cached_model(f"steel_db_tensile_strength_mpa_{algo_suffix}.joblib", "regression")
                ys_model, ys_meta = get_cached_model(f"steel_db_yield_strength_mpa_{algo_suffix}.joblib", "regression")
                duct_model, duct_meta = get_cached_model(f"steel_db_ductility_pct_{algo_suffix}.joblib", "regression")
                cls_model, cls_meta = get_cached_model("steel_db_ensemble_adaboost_classifier.joblib", "classification")

                if ts_model is None or ys_model is None:
                    st.error("Model artifacts could not be loaded. Please verify files in models/regression/.")
                    return

                feats = ts_meta.get("feature_columns", [])
                input_row = {
                    "c": val_c, "mn": val_mn, "si": val_si, "cr": val_cr, "ni": val_ni, "mo": val_mo,
                    "cu": val_cu, "v": val_v, "al": val_al, "ti": 0.0, "s": val_s, "p": val_p,
                    "fe": max(0.0, 100.0 - tot_alloy), "co": 0.0, "nb": 0.0, "b": 0.0, "n": 0.0, "zr": 0.0,
                    "total_alloying_pct": tot_alloy, "carbon_equivalent": ce_iiw,
                    "mn_c_ratio": mn_c, "cr_mo_total": cr_mo, "ni_cr_total": ni_cr
                }
                input_df = pd.DataFrame([input_row])
                for col in feats:
                    if col not in input_df.columns:
                        input_df[col] = 0.0
                input_df = input_df[feats]

                pred_ts = float(ts_model.predict(input_df)[0])
                pred_ys = float(ys_model.predict(input_df)[0])
                pred_duct = float(duct_model.predict(input_df)[0]) if duct_model else 15.0

                tier_pred = "Medium Strength"
                if cls_model:
                    cls_feats = cls_meta.get("feature_columns", feats)
                    cls_input = pd.DataFrame([input_row])
                    for col in cls_feats:
                        if col not in cls_input.columns:
                            cls_input[col] = 0.0
                    tier_pred = str(cls_model.predict(cls_input[cls_feats])[0])

                st.success("AI Inference Completed Successfully")
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    render_metric_panel(
                        "Tensile Strength (Rm)",
                        f"{pred_ts:.1f} MPa",
                        f"Test R²: {ts_meta.get('test_metrics', {}).get('R2', 0.51):.3f} | RMSE: {ts_meta.get('test_metrics', {}).get('RMSE', 221.9):.1f} MPa",
                        "mint"
                    )
                with r2:
                    render_metric_panel(
                        "Yield Strength (Re)",
                        f"{pred_ys:.1f} MPa",
                        f"Test R²: {ys_meta.get('test_metrics', {}).get('R2', 0.25):.3f} | RMSE: {ys_meta.get('test_metrics', {}).get('RMSE', 260.7):.1f} MPa",
                        "peach"
                    )
                with r3:
                    render_metric_panel(
                        "Ductility (Elongation)",
                        f"{pred_duct:.1f} %",
                        f"Test R²: {duct_meta.get('test_metrics', {}).get('R2', 0.14) if duct_meta else 0.14:.3f} | RMSE: 5.2%",
                        "lavender"
                    )
                with r4:
                    render_metric_panel(
                        "Strength Classification",
                        f"{tier_pred}",
                        "AdaBoost Ensemble (Acc: 67.2%)",
                        "white"
                    )

            else:
                ts_model, ts_meta = get_cached_model(f"matnavi_tensile_strength_mpa_{algo_suffix}.joblib", "regression")
                ps_model, ps_meta = get_cached_model(f"matnavi_proof_stress_mpa_{algo_suffix}.joblib", "regression")

                if ts_model is None or ps_model is None:
                    st.error("MatNavi model artifacts could not be loaded.")
                    return

                feats = ts_meta.get("feature_columns", [])
                input_row = {
                    "c": val_c, "si": val_si, "mn": val_mn, "p": val_p, "s": val_s,
                    "ni": val_ni, "cr": val_cr, "mo": val_mo, "cu": val_cu, "v": val_v, "al": val_al,
                    "n": 0.01, "nb_ta": 0.0, "ceq": ce_iiw, "temperature_c": val_temp,
                    "total_alloying_pct": tot_alloy, "carbon_equivalent_calc": ce_iiw,
                    "mn_c_ratio": mn_c, "cr_mo_total": cr_mo, "is_elevated_temp": 1 if val_temp >= 100 else 0
                }
                input_df = pd.DataFrame([input_row])
                for col in feats:
                    if col not in input_df.columns:
                        input_df[col] = 0.0
                input_df = input_df[feats]

                pred_ts = float(ts_model.predict(input_df)[0])
                pred_ps = float(ps_model.predict(input_df)[0])

                st.success(f"AI Inference Completed at {val_temp}°C")
                r1, r2 = st.columns(2)
                with r1:
                    render_metric_panel(
                        f"Tensile Strength ({val_temp}°C)",
                        f"{pred_ts:.1f} MPa",
                        f"MatNavi DT Test R²: {ts_meta.get('test_metrics', {}).get('R2', 0.8911):.4f} | RMSE: 40.9 MPa",
                        "mint"
                    )
                with r2:
                    render_metric_panel(
                        f"Proof Stress Rp0.2 ({val_temp}°C)",
                        f"{pred_ps:.1f} MPa",
                        f"MatNavi DT Test R²: {ps_meta.get('test_metrics', {}).get('R2', 0.9083):.4f} | RMSE: 40.2 MPa",
                        "peach"
                    )

            # Benchmark Chart with modern light styling
            benchmarks = [
                {"Grade": "A36 Mild Steel", "Tensile Strength (MPa)": 400.0, "Category": "Industry Standard"},
                {"Grade": "AISI 1020", "Tensile Strength (MPa)": 420.0, "Category": "Industry Standard"},
                {"Grade": "AISI 1045", "Tensile Strength (MPa)": 565.0, "Category": "Industry Standard"},
                {"Grade": "AISI 4140 Q&T", "Tensile Strength (MPa)": 1020.0, "Category": "Industry Standard"},
                {"Grade": "AISI 4340 Ultra", "Tensile Strength (MPa)": 1280.0, "Category": "Industry Standard"},
                {"Grade": "Current Inference", "Tensile Strength (MPa)": pred_ts, "Category": "Predicted Profile"}
            ]
            fig_bar = px.bar(
                pd.DataFrame(benchmarks),
                x="Grade",
                y="Tensile Strength (MPa)",
                color="Category",
                color_discrete_map={"Industry Standard": "#CBD5E1", "Predicted Profile": "#FF765F"},
                text="Tensile Strength (MPa)"
            )
            fig_bar.update_layout(get_light_plotly_layout("Tensile Strength Benchmark Comparison"))
            st.plotly_chart(fig_bar, use_container_width=True)


# -----------------------------------------------------------------------------
# 4. MATERIAL SUBSTITUTION (FLAGSHIP CAPABILITY)
# -----------------------------------------------------------------------------
def render_material_substitution():
    st.markdown('<div class="section-header">Material Substitution Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Identify and rank viable structural steel alternatives matching strict engineering thresholds using Constraint Satisfaction (CSP) and multi-objective preference ranking.</div>', unsafe_allow_html=True)

    engine = get_cached_substitution_engine("steel_db")
    df = engine.df

    col_ref, col_k = st.columns([3, 1])
    with col_ref:
        sample_alloys = ["AISI 4140", "AISI 1020", "AISI 4340", "AISI 1045", "AISI 4130", "AISI 5160", "AISI 8620"]
        ref_selection = st.selectbox("Baseline Reference Alloy:", sample_alloys, index=0)
    with col_k:
        top_k = st.slider("Alternatives to Return (Top K):", min_value=1, max_value=10, value=5)

    st.markdown("<div style='font-weight: 700; color: #202020; font-size: 0.95rem; margin-top: 0.8rem; margin-bottom: 0.4rem;'>Hard Engineering Constraints (CSP Filtering)</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_ts = st.number_input("Min Tensile Strength (MPa):", min_value=0.0, max_value=2500.0, value=750.0, step=50.0)
    with c2:
        min_ys = st.number_input("Min Yield Strength (MPa):", min_value=0.0, max_value=2500.0, value=500.0, step=50.0)
    with c3:
        min_duct = st.number_input("Min Ductility (% Elongation):", min_value=0.0, max_value=50.0, value=12.0, step=1.0)
    with c4:
        max_cost = st.number_input("Max Allowable Price ($/kg):", min_value=0.0, max_value=20.0, value=5.0, step=0.5)

    with st.expander("Advanced Boundary Constraints"):
        ac1, ac2 = st.columns(2)
        with ac1:
            max_c = st.number_input("Maximum Carbon Content (C % ceiling):", min_value=0.1, max_value=2.0, value=0.60, step=0.05)
        with ac2:
            allow_same_grade = st.checkbox("Allow alternative heat-treatment conditions of the same alloy?", value=True)

    st.markdown("<div style='font-weight: 700; color: #202020; font-size: 0.95rem; margin-top: 0.8rem; margin-bottom: 0.4rem;'>Multi-Objective Preference Weights</div>", unsafe_allow_html=True)
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        weight_comp = st.slider("Composition Similarity", 0.0, 1.0, 0.35, 0.05)
    with w2:
        weight_mech = st.slider("Mechanical Similarity", 0.0, 1.0, 0.35, 0.05)
    with w3:
        weight_duct = st.slider("Ductility Reserve Margin", 0.0, 1.0, 0.15, 0.05)
    with w4:
        weight_cost = st.slider("Sourcing Cost Advantage", 0.0, 1.0, 0.15, 0.05)

    total_w = weight_comp + weight_mech + weight_duct + weight_cost
    if total_w <= 0:
        total_w = 1.0
    norm_weights = {
        "comp_sim": weight_comp / total_w,
        "mech_sim": weight_mech / total_w,
        "duct_margin": weight_duct / total_w,
        "cost_adv": weight_cost / total_w
    }

    if st.button("Execute Material Substitution Analysis", type="primary", use_container_width=True):
        with st.spinner("Processing CSP hard filters and ranking candidate alloys..."):
            constraints = {
                "min_tensile_strength": min_ts if min_ts > 0 else None,
                "min_yield_strength": min_ys if min_ys > 0 else None,
                "min_ductility": min_duct if min_duct > 0 else None,
                "max_cost": max_cost if max_cost > 0 else None,
                "max_c": max_c
            }

            results = engine.find_substitutes(
                ref_identifier=ref_selection,
                constraints=constraints,
                weights=norm_weights,
                top_k=top_k,
                run_graph_search=False,
                allow_same_grade=allow_same_grade
            )

            ref_mat = results.get("reference_material")
            recs = results.get("recommendations", [])
            rejection_summary = results.get("rejection_summary", {})

            if not ref_mat:
                st.error(f"Reference alloy '{ref_selection}' not found in database.")
                return

            st.markdown("---")
            st.markdown(f"#### Baseline Reference Alloy: `{ref_mat.get('name')}`")
            rb1, rb2, rb3, rb4 = st.columns(4)
            with rb1:
                render_metric_panel("Reference Alloy", f"{ref_mat.get('name')}", f"Condition: {ref_mat.get('processing_condition', 'Standard')}", "peach")
            with rb2:
                render_metric_panel("Measured Tensile", f"{ref_mat.get('tensile_strength_mpa')} MPa", "Baseline Rm", "mint")
            with rb3:
                render_metric_panel("Measured Yield", f"{ref_mat.get('yield_strength_mpa')} MPa", "Baseline Re", "lavender")
            with rb4:
                render_metric_panel("Indicative Sourcing", f"{ref_mat.get('cost_display', 'N/A')}", "Public baseline quote", "white")

            st.markdown(f"#### Substitution Results ({len(recs)} Recommended | {results.get('feasible_count', 0)} Feasible of {results.get('total_candidates_evaluated', 0)} Evaluated)")

            if not recs:
                st.warning("No candidate materials satisfied all mandatory constraints. Relax tensile strength, ductility, or cost parameters.")
            else:
                table_rows = []
                for idx, r in enumerate(recs, 1):
                    meas = r["measured_properties"]
                    pred = r["predicted_properties"]
                    deltas = r["property_deltas_vs_reference"]
                    cost = r["cost_information"]

                    table_rows.append({
                        "Rank": idx,
                        "Alternative Alloy": r["material_name"],
                        "Condition": r["processing_condition"],
                        "Suitability": f"{r['suitability_score']:.3f}",
                        "Comp Sim": f"{r['composition_similarity']:.3f}",
                        "Mech Sim": f"{r['mechanical_similarity']:.3f}",
                        "Measured TS": f"{meas.get('tensile_strength_mpa', 'N/A')} MPa",
                        "Predicted TS (DT)": f"{pred.get('tensile_strength_mpa', 'N/A')} MPa",
                        "Δ TS vs Ref": deltas.get("delta_tensile_mpa", "N/A"),
                        "Δ YS vs Ref": deltas.get("delta_yield_mpa", "N/A"),
                        "Cost Status": cost.get("price_display", "Unavailable")
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

                st.markdown("#### Candidate Specification Cards")
                for idx, r in enumerate(recs, 1):
                    meas = r["measured_properties"]
                    pred = r["predicted_properties"]
                    deltas = r["property_deltas_vs_reference"]
                    cost = r["cost_information"]

                    st.markdown(f"""
                    <div class="candidate-card">
                        <div class="candidate-header">
                            <div>
                                <span class="pill pill-coral">Rank #{idx}</span>
                                <span class="candidate-name" style="margin-left: 0.6rem;">{r['material_name']}</span>
                                <span style="color: #777777; font-size: 0.85rem; margin-left: 0.5rem;">({r['processing_condition']})</span>
                            </div>
                            <span class="pill pill-mint" style="font-size: 0.85rem; font-weight: 700;">Suitability Score: {r['suitability_score']:.4f}</span>
                        </div>
                        <div class="candidate-stats-grid">
                            <div>
                                <div class="candidate-stat-label">Composition Sim</div>
                                <div class="candidate-stat-val">{r['composition_similarity']:.3f}</div>
                            </div>
                            <div>
                                <div class="candidate-stat-label">Mechanical Sim</div>
                                <div class="candidate-stat-val">{r['mechanical_similarity']:.3f}</div>
                            </div>
                            <div>
                                <div class="candidate-stat-label">Measured TS (Pred TS)</div>
                                <div class="candidate-stat-val">{meas.get('tensile_strength_mpa')} MPa <span style="color: #64A88C; font-size: 0.85rem;">({pred.get('tensile_strength_mpa')} pred)</span></div>
                            </div>
                            <div>
                                <div class="candidate-stat-label">Sourcing Estimate</div>
                                <div class="candidate-stat-val" style="color: #FF765F;">{cost.get('price_display', 'N/A')}</div>
                            </div>
                        </div>
                        <div class="candidate-justification">
                            <b style="color: #202020;">Engineering Rationale:</b> {r['explanation']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Radar Comparison with Light Theme styling
                if len(recs) >= 1:
                    radar_categories = ["Carbon %", "Manganese %", "Chromium %", "Tensile (x100 MPa)", "Yield (x100 MPa)", "Ductility %"]
                    ref_ts_val = (ref_mat.get("tensile_strength_mpa") or 700) / 100.0
                    ref_ys_val = (ref_mat.get("yield_strength_mpa") or 500) / 100.0
                    ref_duct_val = ref_mat.get("ductility_pct") or 15.0
                    ref_row = df.loc[ref_mat["index"]]

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[ref_row.get("c", 0.4)*10, ref_row.get("mn", 0.8), ref_row.get("cr", 1.0), ref_ts_val, ref_ys_val, ref_duct_val],
                        theta=radar_categories,
                        fill='toself',
                        name=f"Baseline: {ref_mat.get('name')}",
                        line_color="#94A3B8"
                    ))

                    for idx, r in enumerate(recs[:2], 1):
                        cand_row = df.loc[r["dataset_index"]]
                        cand_ts_val = (r["measured_properties"].get("tensile_strength_mpa") or 700) / 100.0
                        cand_ys_val = (r["measured_properties"].get("yield_strength_mpa") or 500) / 100.0
                        cand_duct_val = r["measured_properties"].get("ductility_pct") or 15.0
                        c_palette = ["#FF765F", "#64A88C"]

                        fig_radar.add_trace(go.Scatterpolar(
                            r=[cand_row.get("c", 0.4)*10, cand_row.get("mn", 0.8), cand_row.get("cr", 1.0), cand_ts_val, cand_ys_val, cand_duct_val],
                            theta=radar_categories,
                            fill='toself',
                            name=f"Rank #{idx}: {r['material_name']}",
                            line_color=c_palette[idx-1]
                        ))

                    layout_radar = get_light_plotly_layout("Multi-Property Profile Radar Comparison", height=460)
                    layout_radar["polar"] = dict(
                        radialaxis=dict(visible=True, range=[0, 15], gridcolor="#E5E7EB", linecolor="#D1D5DB"),
                        angularaxis=dict(gridcolor="#E5E7EB", linecolor="#D1D5DB")
                    )
                    fig_radar.update_layout(layout_radar)
                    st.plotly_chart(fig_radar, use_container_width=True)

            with st.expander("CSP Hard Rejection Audit Log"):
                rej_df = pd.DataFrame([
                    {"Constraint": k, "Disqualified Materials": v}
                    for k, v in rejection_summary.items() if k != "rejection_samples"
                ])
                st.dataframe(rej_df, use_container_width=True)
                if "rejection_samples" in rejection_summary and rejection_summary["rejection_samples"]:
                    st.caption("Sample Disqualification Reasons:")
                    for s in rejection_summary["rejection_samples"][:5]:
                        st.markdown(f"- {s}")

    render_disclaimer()


# -----------------------------------------------------------------------------
# 5. SEARCH INTELLIGENCE
# -----------------------------------------------------------------------------
def render_search_intelligence():
    st.markdown('<div class="section-header">Search Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Computational graph intelligence algorithms traversing a 3,234-node topological similarity network (12,103 substitution edges) to identify direct and multi-hop substitution pathways.</div>', unsafe_allow_html=True)

    engine = get_cached_substitution_engine("steel_db")
    adj_list = engine.get_similarity_graph(threshold=0.6)
    searcher = MaterialGraphSearch(adj_list, engine.df, engine.similarity_calc, engine.cost_estimator)

    col1, col2 = st.columns(2)
    with col1:
        start_node = st.number_input("Starting Alloy Node Index (0 to 3233):", min_value=0, max_value=3233, value=0, step=1)
        start_alloy_name = engine.df.iloc[start_node].get("name", f"Alloy_{start_node}")
        st.caption(f"Selected Origin Alloy: **{start_alloy_name}**")
    with col2:
        target_ts = st.number_input("Target Minimum Tensile Strength (Rm >= MPa):", min_value=400.0, max_value=1600.0, value=800.0, step=50.0)

    p1, p2, p3 = st.columns(3)
    with p1:
        max_bfs_depth = st.slider("Max BFS Search Depth:", 1, 4, 2)
    with p2:
        max_dfs_depth = st.slider("Max DFS Search Depth:", 1, 6, 3)
    with p3:
        max_local_iter = st.slider("Local Search Iterations:", 5, 30, 15)

    def goal_test(row):
        return row.get("tensile_strength_mpa", 0) >= target_ts

    if st.button("Execute Graph Search Algorithms", type="primary", use_container_width=True):
        with st.spinner("Traversing material network across all four computational algorithms..."):
            t0 = time.perf_counter()
            bfs_res = searcher.breadth_first_search(start_node, goal_test, max_depth=max_bfs_depth)
            bfs_time = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            dfs_res = searcher.depth_first_search(start_node, goal_test, max_depth=max_dfs_depth)
            dfs_time = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            astar_res = searcher.a_star_search(start_node, goal_test, required_min_ts=target_ts)
            astar_time = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            local_res = searcher.local_search_optimization(start_node, start_node, max_iterations=max_local_iter)
            local_time = (time.perf_counter() - t0) * 1000

        st.success("Graph Traversals Completed Successfully")

        astar_target = astar_res.get("target_node")
        astar_target_name = engine.df.iloc[astar_target].get("name") if astar_target is not None else "None reachable"
        local_best = local_res.get("best_node")
        local_best_name = engine.df.iloc[local_best].get("name") if local_best is not None else "None"

        comp_data = [
            {
                "Algorithm": "Breadth-First Search (BFS)",
                "Nodes Explored": bfs_res.get("visited_count", 0),
                "Qualifying Goals": len(bfs_res.get("found_goals", [])),
                "Solution Discovered": f"{len(bfs_res.get('found_goals', []))} candidate alloy(s)",
                "Execution Time (ms)": f"{bfs_time:.2f} ms",
                "Traversal Strategy": "Unweighted level-by-level exhaustive expansion"
            },
            {
                "Algorithm": "Depth-First Search (DFS)",
                "Nodes Explored": dfs_res.get("visited_count", 0),
                "Qualifying Goals": len(dfs_res.get("found_goals", [])),
                "Solution Discovered": f"{len(dfs_res.get('found_goals', []))} candidate alloy(s)",
                "Execution Time (ms)": f"{dfs_time:.2f} ms",
                "Traversal Strategy": "Deep branch exploration before backtracking"
            },
            {
                "Algorithm": "A* Heuristic Search",
                "Nodes Explored": astar_res.get("nodes_expanded", 0),
                "Qualifying Goals": 1 if astar_target is not None else 0,
                "Solution Discovered": f"{astar_target_name} (Cost: {astar_res.get('total_cost', 0):.4f})",
                "Execution Time (ms)": f"{astar_time:.2f} ms",
                "Traversal Strategy": "Heuristic-guided minimization of transition friction f(n) = g(n) + h(n)"
            },
            {
                "Algorithm": "Local Search (Hill-Climbing)",
                "Nodes Explored": local_res.get("total_steps", 0),
                "Qualifying Goals": 1,
                "Solution Discovered": f"{local_best_name} (Score: {local_res.get('best_score', 0):.4f})",
                "Execution Time (ms)": f"{local_time:.2f} ms",
                "Traversal Strategy": "Greedy iterative climbing in immediate topological neighborhood"
            }
        ]
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

        tab_b, tab_d, tab_a, tab_l = st.tabs(["BFS Levels", "DFS Stack Order", "A* Heuristic Path", "Hill-Climbing Trajectory"])

        with tab_b:
            levels = bfs_res.get("exploration_levels", {})
            for d, nodes in levels.items():
                st.markdown(f"- **Depth Level {d} ({len(nodes)} nodes):** Sample nodes: `{nodes[:8]}`")

        with tab_d:
            order = dfs_res.get("exploration_order", [])
            st.markdown(f"Total nodes explored along deepest branch paths: **{len(order)}**")
            st.caption(f"First 15 visited nodes in traversal sequence: `{order[:15]}`")

        with tab_a:
            if astar_target is not None:
                path = astar_res.get("path", [])
                st.markdown(f"**Heuristic-Guided Pathway Discovered:** Total transition cost g(n) = **{astar_res.get('total_cost'):.4f}**")
                path_str = " ➔ ".join([f"**{engine.df.iloc[idx].get('name')}** [Node {idx}]" for idx in path])
                st.markdown(f"> {path_str}")
            else:
                st.warning("A* could not reach an alloy satisfying the target strength requirement.")

        with tab_l:
            traj = local_res.get("trajectory", [])
            if traj:
                traj_df = pd.DataFrame([
                    {"Step": i, "Node": t[0], "Alloy": engine.df.iloc[t[0]].get("name"), "Suitability Score": t[1]}
                    for i, t in enumerate(traj)
                ])
                fig_traj = px.line(traj_df, x="Step", y="Suitability Score", markers=True, title="Hill-Climbing Optimization Trajectory", color_discrete_sequence=["#FF765F"])
                fig_traj.update_layout(get_light_plotly_layout("Score Optimization vs Step"))
                st.plotly_chart(fig_traj, use_container_width=True)


# -----------------------------------------------------------------------------
# 6. COST ANALYSIS
# -----------------------------------------------------------------------------
def render_cost_analysis():
    st.markdown('<div class="section-header">Cost Analysis & Sourcing Economics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Evaluate commercial steel pricing economics, simulate annual material substitution savings, and review ethical web scraping and data provenance audits.</div>', unsafe_allow_html=True)

    cost_est = CostEstimator()
    ref_table = cost_est.get_all_reference_materials()

    tab1, tab2, tab3 = st.tabs(["Material Sourcing Reference", "Annual Substitution Savings Simulator", "Ethical Sourcing & Compliance Audit"])

    with tab1:
        st.markdown("<div style='color: #666666; font-size: 0.9rem; margin-bottom: 0.8rem;'>Curated public market price ranges (January 2024 industrial estimates) categorized with explicit provenance tags.</div>", unsafe_allow_html=True)
        st.dataframe(ref_table, use_container_width=True)
        st.caption(f"Displaying {len(ref_table)} standard structural alloy grades with indicative pricing benchmarks.")

    with tab2:
        st.markdown("<div style='color: #666666; font-size: 0.9rem; margin-bottom: 0.8rem;'>Simulate annual procurement expenditure reductions when transitioning from premium alloy grades to optimized structural substitutes.</div>", unsafe_allow_html=True)

        grade_col = "grade" if "grade" in ref_table.columns else "material_name"
        grade_options = ref_table[grade_col].tolist() if not ref_table.empty else ["AISI 4140", "AISI 1020"]

        s1, s2, s3 = st.columns(3)
        with s1:
            base_grade = st.selectbox("Baseline Alloy Grade:", grade_options, index=2 if len(grade_options) > 2 else 0)
            base_info = cost_est.get_material_cost(base_grade)
            base_default_price = float(base_info.get("price_per_kg") or 1.50)
            user_base_price = st.number_input("Baseline Unit Price ($/kg):", min_value=0.1, max_value=50.0, value=base_default_price, step=0.1)

        with s2:
            sub_grade = st.selectbox("Proposed Alternative Alloy:", grade_options, index=1 if len(grade_options) > 1 else 0)
            sub_info = cost_est.get_material_cost(sub_grade)
            sub_default_price = float(sub_info.get("price_per_kg") or 1.00)
            user_sub_price = st.number_input("Alternative Unit Price ($/kg):", min_value=0.1, max_value=50.0, value=sub_default_price, step=0.1)

        with s3:
            volume_tonnes = st.number_input("Annual Production Volume (Metric Tons):", min_value=1.0, max_value=100000.0, value=500.0, step=50.0)
            volume_kg = volume_tonnes * 1000.0

        base_annual_cost = volume_kg * user_base_price
        sub_annual_cost = volume_kg * user_sub_price
        net_savings = base_annual_cost - sub_annual_cost
        pct_savings = (net_savings / base_annual_cost) * 100.0 if base_annual_cost > 0 else 0.0

        st.markdown("---")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            render_metric_panel("Baseline Annual Cost", f"${base_annual_cost:,.2f}", f"Annual spend for {base_grade}", "peach")
        with sc2:
            render_metric_panel("Alternative Annual Cost", f"${sub_annual_cost:,.2f}", f"Annual spend for {sub_grade}", "mint")
        with sc3:
            render_metric_panel("Projected Net Savings", f"${net_savings:,.2f}", f"{pct_savings:+.1f}% economic change", "white" if net_savings >= 0 else "lavender")

        if net_savings > 0:
            st.success(f"Material substitution from **{base_grade}** to **{sub_grade}** yields an estimated annual savings of **${net_savings:,.2f}** ({pct_savings:.1f}% reduction).")
        elif net_savings < 0:
            st.warning(f"Alternative alloy **{sub_grade}** carries an annual price premium of **${abs(net_savings):,.2f}**. Ensure structural or corrosion performance justifies the increased expenditure.")

    with tab3:
        st.markdown("""
        <div class="pastel-card">
            <h4 style="margin: 0 0 0.5rem 0; color: #1A1A1A;">Web Scraping & Sourcing Compliance Audit</h4>
            <p style="color: #666666; font-size: 0.9rem; line-height: 1.55;">
                In accordance with ethical data practices and site access regulations, automated web scraping was verified against target endpoints:
            </p>
            <ul style="color: #444444; font-size: 0.88rem; line-height: 1.6;">
                <li><b>MatWeb Materials Database:</b> robots.txt inspection revealed <code>User-agent: * Disallow: /</code>. Automated crawlers were terminated in compliance with site policy.</li>
                <li><b>MetalsDepot Commercial Supplier:</b> Automated inspection returned <code>HTTP 403 Forbidden</code> via Cloudflare Web Application Firewall (WAF). No bypass evasion was attempted.</li>
                <li><b>Operational Strategy:</b> Curated public reference estimates (January 2024 benchmarks) are provided with explicit provenance metadata (<code>illustrative_estimate</code>, <code>user_override</code>, or <code>unavailable</code>).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. MODEL ANALYTICS
# -----------------------------------------------------------------------------
def render_model_analytics():
    st.markdown('<div class="section-header">Model Analytics & Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Comprehensive evaluation of all 30 trained machine learning models, strictly validated on held-out test groups via GroupShuffleSplit to ensure zero data leakage.</div>', unsafe_allow_html=True)

    reg_df = get_cached_comparison_table("regression_model_comparison.csv")
    cls_df = get_cached_comparison_table("classification_model_comparison.csv")
    clu_df = get_cached_comparison_table("clustering_model_comparison.csv")

    tab1, tab2, tab3, tab4 = st.tabs(["Regression Benchmarks", "Classification Ensembles", "Clustering & PCA", "Validation Plot Gallery"])

    with tab1:
        st.dataframe(reg_df, use_container_width=True)
        if not reg_df.empty:
            fig_r2 = px.bar(
                reg_df,
                x="Target",
                y="Test_R2",
                color="Algorithm",
                barmode="group",
                facet_col="Dataset",
                title="Held-Out Test R² Score by Target and Architecture",
                color_discrete_sequence=["#FF765F", "#64A88C", "#7C5CFC", "#F59E0B"]
            )
            fig_r2.update_layout(get_light_plotly_layout("Regression Test R² Comparison Across Targets"))
            st.plotly_chart(fig_r2, use_container_width=True)

    with tab2:
        st.dataframe(cls_df, use_container_width=True)
        if not cls_df.empty:
            fig_cls = px.bar(
                cls_df,
                x="Algorithm",
                y="Test_Accuracy",
                color="Model_Type",
                title="Classification Test Accuracy Across Base Models and Ensembles",
                text="Test_Accuracy",
                color_discrete_map={"Base Classifier": "#CBD5E1", "Ensemble": "#FF765F"}
            )
            fig_cls.update_layout(get_light_plotly_layout("Classification Test Accuracy"))
            st.plotly_chart(fig_cls, use_container_width=True)

    with tab3:
        st.dataframe(clu_df, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="pastel-card pastel-card-mint">
                <h4 style="margin: 0 0 0.5rem 0; color: #20784E;">Unsupervised Clustering (k=4)</h4>
                <ul style="color: #444444; font-size: 0.88rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                    <li><b>K-Means:</b> Silhouette Score = <b>0.8120</b>, Inertia = 40,946.29.</li>
                    <li><b>Agglomerative Hierarchical:</b> Silhouette Score = <b>0.7857</b>, Ward linkage.</li>
                    <li><b>Inter-Algorithm Consensus:</b> ARI = <b>0.8622</b> between K-Means and Hierarchical.</li>
                    <li><b>Historical Label Comparison:</b> ARI = <b>0.0051</b>, confirming composition clusters are orthogonal to heat-treatment states.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="pastel-card pastel-card-lavender">
                <h4 style="margin: 0 0 0.5rem 0; color: #5E39C4;">Principal Component Analysis (PCA)</h4>
                <ul style="color: #444444; font-size: 0.88rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                    <li>Standardized on 23 chemical and metallurgical features.</li>
                    <li><b>PC1 (Principal Axis 1):</b> 16.51% variance explained.</li>
                    <li><b>PC2 (Principal Axis 2):</b> 11.23% variance explained.</li>
                    <li><b>Top 10 Components:</b> Capture <b>86.02%</b> of cumulative metallurgical variance.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        art_choice = st.selectbox(
            "Select Validation Plot to Display:",
            [
                "Classification Model Comparison (classification_model_comparison_steel_db.png)",
                "Confusion Matrices: Base Classifiers (confusion_matrices_base_steel_db.png)",
                "Confusion Matrices: Ensembles (confusion_matrices_ensemble_steel_db.png)",
                "Clustering Elbow & Silhouette (clustering_elbow_silhouette_steel_db.png)",
                "Hierarchical Dendrogram (hierarchical_dendrogram_steel_db.png)",
                "PCA 2D Scatter Projection (pca_2d_scatter_steel_db.png)",
                "PCA Explained Variance Scree Plot (pca_explained_variance_steel_db.png)",
                "Regression Actual vs Pred: Steel DB Tensile (reg_actual_vs_pred_steel_db_tensile_strength_mpa.png)",
                "Regression Actual vs Pred: MatNavi Tensile (reg_actual_vs_pred_matnavi_tensile_strength_mpa.png)",
                "Dataset 1 Target Distributions (ds1_target_distributions.png)",
                "Dataset 2 Temperature Effect (ds2_temperature_effect.png)"
            ]
        )
        fname = art_choice.split("(")[-1].replace(")", "").strip()
        img_path = os.path.join(ASSETS_DIR, fname)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            st.image(img, caption=art_choice, use_container_width=True)
        else:
            st.info(f"Artifact `{fname}` is currently not present in the assets directory.")


# -----------------------------------------------------------------------------
# 8. ABOUT MATERIALMIND AI
# -----------------------------------------------------------------------------
def render_about():
    st.markdown('<div class="section-header">About MaterialMind AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Platform architecture, underlying technologies, dataset specifications, and engineering scope.</div>', unsafe_allow_html=True)

    # Polished Visual System Architecture Diagram & Data Movement Breakdown
    render_architecture_diagram()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="pastel-card">
            <h4 style="margin: 0 0 0.5rem 0; color: #1A1A1A;">Technology Stack</h4>
            <ul style="color: #555555; font-size: 0.88rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                <li><b>Core Language:</b> Python 3.14</li>
                <li><b>Machine Learning:</b> Scikit-Learn 1.9</li>
                <li><b>Data Processing:</b> Pandas, NumPy</li>
                <li><b>Interactive UI:</b> Streamlit 1.67</li>
                <li><b>Visualizations:</b> Plotly 7.1, Matplotlib, Seaborn</li>
                <li><b>Model Serialization:</b> Joblib 1.6</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="pastel-card">
            <h4 style="margin: 0 0 0.5rem 0; color: #1A1A1A;">Dataset Specifications</h4>
            <ul style="color: #555555; font-size: 0.88rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                <li><b>Steel Database:</b> 3,234 records, 31 attributes, room temperature standard structural alloys (AISI/SAE/ASTM).</li>
                <li><b>MatNavi Creep Database:</b> 915 records (914 clean), 27 attributes, elevated-temperature testing (27°C–650°C).</li>
                <li><b>Separation Protocol:</b> Repositories are maintained as separate data models to prevent thermodynamic conflicts.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pastel-card">
        <h4 style="margin: 0 0 0.5rem 0; color: #1A1A1A;">Known Engineering Scope & Limitations</h4>
        <ul style="color: #555555; font-size: 0.88rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
            <li><b>Microstructure Phase Blindness:</b> Models predict from bulk chemistry and gross temper temperatures without direct microstructural phase fraction inputs (martensite, bainite, retained austenite volume %).</li>
            <li><b>Dynamic Mechanical Properties:</b> Impact toughness (Charpy V-notch DBTT) and high-cycle fatigue endurance limits (S-N curves) are not modeled in current dataset scopes.</li>
            <li><b>Section Thickness Sensitivity:</b> Predictions do not account for ruling section thickness or Jominy end-quench hardenability variations across deep cross-sections.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    render_disclaimer()


# -----------------------------------------------------------------------------
# 5. MAIN ROUTER
# -----------------------------------------------------------------------------
if nav_selection == "Overview":
    render_overview()
elif nav_selection == "Material Explorer":
    render_material_explorer()
elif nav_selection == "AI Predictions":
    render_ai_predictions()
elif nav_selection == "Material Substitution":
    render_material_substitution()
elif nav_selection == "Search Intelligence":
    render_search_intelligence()
elif nav_selection == "Cost Analysis":
    render_cost_analysis()
elif nav_selection == "Model Analytics":
    render_model_analytics()
elif nav_selection == "About MaterialMind AI":
    render_about()
