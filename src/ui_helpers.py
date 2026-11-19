"""
ui_helpers.py — UI/UX Design System and Theming Helpers for MaterialMind AI
Light Theme Design System: Minimalist, Modern, Warm Pastel Accents
"""

import os
import streamlit as st
import streamlit.components.v1 as components
import textwrap
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. LIGHT PASTEL THEME CSS
# -----------------------------------------------------------------------------
LIGHT_THEME_CSS = """
<style>
    /* Global Font and Clean Baseline */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Main Canvas Background */
    .stApp {
        background-color: #FFFFFF !important;
        color: #202020 !important;
    }

    /* Streamlit Main Content Container Spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* Hero Section Container */
    .hero-container {
        background-color: #FFF1F0;
        border: 1px solid #FFE4E1;
        border-radius: 18px;
        padding: 2.2rem 2.6rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background-color: #FFFFFF;
        color: #FF765F;
        border: 1px solid #FFD9D4;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        padding: 0.3rem 0.85rem;
        margin-bottom: 0.9rem;
    }
    .hero-headline {
        color: #1A1A1A !important;
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.6px;
        margin: 0 0 0.65rem 0;
        line-height: 1.25;
    }
    .hero-subtitle {
        color: #555555 !important;
        font-size: 1.05rem;
        margin: 0 0 1.25rem 0;
        font-weight: 400;
        max-width: 860px;
        line-height: 1.6;
    }

    /* Modern Pastel Feature Cards */
    .pastel-card {
        background-color: #FFFFFF;
        border: 1px solid #EEEEEE;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .pastel-card:hover {
        transform: translateY(-2px);
        border-color: #FF765F;
        box-shadow: 0 6px 20px rgba(255, 118, 95, 0.10);
    }
    .pastel-card-mint {
        background-color: #E8F6EC;
        border: 1px solid #CDECD6;
    }
    .pastel-card-peach {
        background-color: #FFF0E2;
        border: 1px solid #FFDFC2;
    }
    .pastel-card-lavender {
        background-color: #F1EBFF;
        border: 1px solid #DFD2FC;
    }

    /* Key Performance Metric Panels */
    .metric-panel {
        background: #FFFFFF;
        border: 1px solid #EEEEEE;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
        position: relative;
        overflow: hidden;
    }
    .metric-panel.mint, .metric-panel.cyan {
        background-color: #E8F6EC;
        border: 1px solid #CDECD6;
    }
    .metric-panel.peach, .metric-panel.teal {
        background-color: #FFF0E2;
        border: 1px solid #FFDFC2;
    }
    .metric-panel.lavender, .metric-panel.purple {
        background-color: #F1EBFF;
        border: 1px solid #DFD2FC;
    }
    .metric-panel.white, .metric-panel.amber {
        background-color: #FFFFFF;
        border: 1px solid #EEEEEE;
    }

    .metric-label {
        color: #666666;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        color: #202020;
        font-size: 1.85rem;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
        line-height: 1.2;
    }
    .metric-subtext {
        color: #777777;
        font-size: 0.8rem;
        margin-top: 0.35rem;
        font-weight: 500;
    }

    /* Status Badges & Pills */
    .pill {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    .pill-coral, .pill-cyan {
        background: #FFF0ED;
        color: #E0533C;
        border: 1px solid #FFD2C9;
    }
    .pill-mint, .pill-teal {
        background: #E8F6EC;
        color: #20784E;
        border: 1px solid #C6EAD0;
    }
    .pill-peach, .pill-amber {
        background: #FFF0E2;
        color: #B8551B;
        border: 1px solid #FFD9B5;
    }
    .pill-lavender, .pill-purple {
        background: #F1EBFF;
        color: #5E39C4;
        border: 1px solid #DCD0FC;
    }
    .pill-slate {
        background: #F4F5F6;
        color: #555555;
        border: 1px solid #E5E7EB;
    }

    /* Section Subheadings */
    .section-header {
        color: #1A1A1A;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.4px;
        margin-top: 0.8rem;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-desc {
        color: #666666;
        font-size: 0.94rem;
        margin-bottom: 1.35rem;
        line-height: 1.55;
    }

    /* Recommendation Candidate Card */
    .candidate-card {
        background: #FFFFFF;
        border: 1px solid #EEEEEE;
        border-left: 4px solid #FF765F;
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
    }
    .candidate-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.7rem;
    }
    .candidate-name {
        color: #202020;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0;
    }
    .candidate-stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 14px 0;
        background: #F9FAFB;
        border: 1px solid #F0F2F5;
        padding: 12px 16px;
        border-radius: 10px;
    }
    .candidate-stat-label {
        color: #777777;
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 600;
    }
    .candidate-stat-val {
        color: #202020;
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: 2px;
    }
    .candidate-justification {
        color: #333333;
        font-size: 0.88rem;
        line-height: 1.55;
        background: #F6FAF7;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 3px solid #64A88C;
    }

    /* Engineering Disclaimer Box */
    .disclaimer-container {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #F59E0B;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 2rem 0 1rem 0;
        color: #78350F;
        font-size: 0.86rem;
        line-height: 1.6;
    }
    .disclaimer-title {
        color: #B45309;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.35rem;
    }

    /* Streamlit Sidebar Clean Styling */
    div[data-testid="stSidebar"] {
        background-color: #FAFAFB !important;
        border-right: 1px solid #EBECEF;
    }
    div[data-testid="stSidebar"] hr {
        border-color: #EBECEF;
    }

    /* Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background-color: #F4F5F7;
        padding: 0.35rem;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #666666;
        padding: 0.45rem 1rem;
        font-weight: 500;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #202020 !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
    }

    /* Primary CTA Buttons */
    button[kind="primary"] {
        background-color: #FF765F !important;
        border: 1px solid #FF6247 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.4rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 6px rgba(255, 118, 95, 0.25) !important;
    }
    button[kind="primary"]:hover {
        background-color: #F05C43 !important;
        box-shadow: 0 4px 14px rgba(255, 118, 95, 0.35) !important;
    }

    /* Secondary Buttons */
    button[kind="secondary"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        color: #374151 !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
    }
    button[kind="secondary"]:hover {
        border-color: #FF765F !important;
        color: #FF765F !important;
    }

    /* Form Input Labels */
    label[data-testid="stWidgetLabel"] p {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
    }

    /* Dataframe Container */
    div[data-testid="stDataFrame"] {
        border: 1px solid #EEEEEE;
        border-radius: 10px;
        overflow: hidden;
    }
</style>
"""


def apply_custom_theme():
    """Inject the clean light pastel CSS stylesheet."""
    st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)


def get_light_plotly_layout(title="", height=420):
    """
    Return a standardized clean light Plotly layout dictionary.
    """
    return dict(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color="#202020", family="Inter, sans-serif"),
            x=0.01,
            y=0.96
        ),
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        height=height,
        margin=dict(l=45, r=30, t=50, b=40),
        font=dict(color="#444444", family="Inter, sans-serif", size=12),
        xaxis=dict(
            gridcolor="#F0F0F0",
            linecolor="#E0E0E0",
            tickcolor="#E0E0E0",
            tickfont=dict(color="#666666", size=11),
            title_font=dict(color="#333333", size=12)
        ),
        yaxis=dict(
            gridcolor="#F0F0F0",
            linecolor="#E0E0E0",
            tickcolor="#E0E0E0",
            tickfont=dict(color="#666666", size=11),
            title_font=dict(color="#333333", size=12)
        ),
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#E5E7EB",
            borderwidth=1,
            font=dict(color="#333333", size=11)
        )
    )


def get_dark_plotly_layout(title="", height=420):
    """
    Backward-compatible alias: redirects to light plotly layout.
    """
    return get_light_plotly_layout(title=title, height=height)


def render_metric_panel(label, value, subtext="", panel_type="white"):
    """
    Render a styled pastel KPI metric panel.
    Available panel_type options: 'mint', 'peach', 'lavender', 'white' (or legacy aliases).
    """
    type_map = {
        "cyan": "mint",
        "teal": "peach",
        "purple": "lavender",
        "amber": "white"
    }
    actual_type = type_map.get(panel_type, panel_type)
    st.markdown(f"""
    <div class="metric-panel {actual_type}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtext">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)


def render_disclaimer():
    """Render the standardized engineering safety disclaimer."""
    st.markdown("""
    <div class="disclaimer-container">
        <div class="disclaimer-title">Engineering Safety & Regulatory Notice</div>
        MaterialMind AI is developed strictly for educational screening and decision-support modeling. 
        Model outputs and substitution recommendations do not constitute structural certification or design approval. 
        Industrial adoption requires physical coupon testing, metallurgical phase quantification, fatigue verification, and compliance with applicable engineering standards (ASTM, ASME, ISO, SAE).
    </div>
    """, unsafe_allow_html=True)


def render_architecture_diagram():
    """
    Renders the official MaterialMind AI architecture pipeline diagram
    using the high-resolution vector flow image with light-theme styling.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    img_path = os.path.join(project_root, "assets", "materialmind_architecture_pipeline.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(project_root, "assets", "materialmind_architecture_pipeline.jpg")

    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 4px 14px rgba(0,0,0,0.03);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.9rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 0.80rem; font-weight: 700; color: #FF765F; text-transform: uppercase; letter-spacing: 0.05em; background: #FFF1F0; padding: 0.25rem 0.65rem; border-radius: 20px;">
                    End-to-End System Pipeline
                </span>
                <span style="font-size: 0.88rem; font-weight: 600; color: #2D3748;">MaterialMind AI Architectural Workflow</span>
            </div>
            <span style="font-size: 0.78rem; color: #718096; background: #F7FAFC; padding: 0.2rem 0.55rem; border-radius: 12px; border: 1px solid #EDF2F7;">5-Stage Dual Intelligence Engine</span>
        </div>
    """, unsafe_allow_html=True)

    if os.path.exists(img_path):
        st.image(img_path, caption="MaterialMind AI: End-to-End Materials Intelligence & Machine Learning Architecture Flowchart", use_container_width=True)
    else:
        st.warning("Architecture diagram image asset not found.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Technical workflow details
    st.markdown("""
    <div class="pastel-card" style="margin-bottom: 1.5rem;">
        <h4 style="margin: 0 0 0.6rem 0; color: #1A1A1A; font-size: 1.12rem; font-weight: 700;">Data Movement & Pipeline Workflow</h4>
        <p style="color: #4A5568; font-size: 0.88rem; line-height: 1.6; margin-bottom: 0.7rem;">
            <b>1. Ingestion & Partitioning:</b> Raw empirical materials data from the commercial Steel Database (3,234 heats) and MatNavi Creep repository (915 records, 914 clean) are ingested and maintained under strict data model separation to prevent thermodynamic conflicts.
        </p>
        <p style="color: #4A5568; font-size: 0.88rem; line-height: 1.6; margin-bottom: 0.7rem;">
            <b>2. Data Transformation & Feature Engineering:</b> 23 chemical elements are standardized and combined into metallurgical indices (IIW Carbon Equivalent, Mn/C ratio, Cr+Mo hardenability sum) without data leakage.
        </p>
        <p style="color: #4A5568; font-size: 0.88rem; line-height: 1.6; margin-bottom: 0.7rem;">
            <b>3. Dual Intelligence Engines:</b> The supervised ML pipeline infers continuous mechanical properties (Rm, Re, A) and classifies strength tiers. In parallel, the graph and CSP engine traverses 12,103 topological similarity edges and filters candidate alloys against mandatory physical constraints.
        </p>
        <p style="color: #4A5568; font-size: 0.88rem; line-height: 1.6; margin-bottom: 0.7rem;">
            <b>4. Multi-Objective Decision Synthesis:</b> Qualifying candidate alternatives are ranked through a weighted multi-criteria preference function balancing chemical similarity, mechanical parity, ductility reserve margin, and sourcing cost advantage.
        </p>
        <p style="color: #4A5568; font-size: 0.88rem; line-height: 1.6; margin: 0;">
            <b>5. User Application Delivery:</b> The Streamlit light-theme interface surfaces real-time inference, radar profile comparisons, graph search traces, and economic simulations for materials engineers and researchers.
        </p>
    </div>
    """, unsafe_allow_html=True)


__all__ = [
    "LIGHT_THEME_CSS",
    "apply_custom_theme",
    "get_light_plotly_layout",
    "get_dark_plotly_layout",
    "render_metric_panel",
    "render_disclaimer",
    "render_architecture_diagram"
]


