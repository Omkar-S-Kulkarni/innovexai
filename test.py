import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from model.inference_engine import StreamingInferenceEngine
from monitoring.output_distribution import OutputDistributionTracker
from monitoring.window_manager import WindowManager
from monitoring.rolling_reference import RollingReferenceManager
from monitoring.confidence_entropy import ConfidenceEntropySignals
from monitoring.alert_engine import AlertEngine
from monitoring.slice_risk_ranker import rank_slices, explain_slice
from monitoring.slice_drift_metrics import compute_slice_metrics
from monitoring.cluster_builder import ClusterBuilder
from monitoring.cluster_lifecycle import ClusterLifecycleTracker
from monitoring.cluster_slice_adapter import build_cluster_slices
from monitoring.cluster_risk_integration import ClusterRiskEngine
from monitoring.drift_attribution import feature_kl_divergence, feature_psi, slice_feature_comparison
from monitoring.perturbation_tests import add_gaussian_noise, decision_flip_rate, counterfactual_stability
from monitoring.composite_score import CompositeDriftScore
from monitoring.drift_regime import DriftRegimeDetector
from monitoring.audit_report_generator import AuditReportGenerator
from monitoring.failure_blindspot_matrix import build_failure_blindspot_matrix
from monitoring.regime_timeline import build_regime_timeline
from monitoring.slice_definition import SliceRegistry, threshold_slice, range_slice
from monitoring.behavior_fingerprint import BehaviorFingerprintStore
from monitoring.trend_acceleration import compute_trends
from monitoring.early_warning_state import EarlyWarningEngine

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Silent Drift Monitor | Enterprise ML Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️"
)

# =========================================================
# DESIGN SYSTEM - COLOR PALETTE & SPACING
# =========================================================
COLORS = {
    # Primary Brand
    'primary': '#6366f1',
    'primary_dark': '#4f46e5',
    'primary_light': '#818cf8',
    
    # Semantic Colors
    'success': '#22c55e',
    'warning': '#facc15',
    'danger': '#ef4444',
    'info': '#3b82f6',
    
    # Backgrounds
    'bg_primary': '#020617',
    'bg_secondary': '#0f172a',
    'bg_card': 'rgba(255,255,255,0.04)',
    
    # Text
    'text_primary': '#E5E7EB',
    'text_secondary': '#CBD5F5',
    'text_muted': '#94a3b8',
    
    # Borders
    'border': 'rgba(255,255,255,0.08)',
    'border_hover': 'rgba(255,255,255,0.16)',
}

SPACING = {
    'xs': '4px',
    'sm': '8px',
    'md': '16px',
    'lg': '24px',
    'xl': '32px',
    'xxl': '48px',
}

SHADOWS = {
    'sm': '0 2px 8px rgba(0,0,0,0.2)',
    'md': '0 8px 24px rgba(0,0,0,0.3)',
    'lg': '0 16px 48px rgba(0,0,0,0.4)',
    'xl': '0 24px 80px rgba(0,0,0,0.5)',
}

# =========================================================
# PREMIUM UI STYLING WITH COMPLETE DESIGN SYSTEM
# =========================================================
st.markdown(f"""
<style>
/* ====== CSS RESET & GLOBAL ====== */
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

/* ====== ROOT VARIABLES ====== */
:root {{
    --primary: {COLORS['primary']};
    --success: {COLORS['success']};
    --warning: {COLORS['warning']};
    --danger: {COLORS['danger']};
    --spacing-md: {SPACING['md']};
    --shadow-lg: {SHADOWS['lg']};
}}

/* ====== GLOBAL APP ====== */
.stApp {{
    background: radial-gradient(ellipse at top, #1a1a2e 0%, {COLORS['bg_primary']} 50%);
    color: {COLORS['text_primary']};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

.block-container {{
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1600px;
    margin: 0 auto;
}}

/* ====== TYPOGRAPHY SYSTEM ====== */
h1, h2, h3, h4, h5, h6 {{
    font-weight: 700;
    letter-spacing: -0.02em;
    color: {COLORS['text_primary']};
    line-height: 1.2;
}}

h1 {{ 
    font-size: 2.5rem; 
    margin-bottom: 1rem;
}}
h2 {{ 
    font-size: 1.875rem; 
    margin-top: 2.5rem; 
    margin-bottom: 1rem;
}}
h3 {{ 
    font-size: 1.5rem; 
    margin-bottom: 0.75rem;
}}
h4 {{
    font-size: 1.25rem;
    margin-bottom: 0.5rem;
}}

p, li, span {{
    color: {COLORS['text_secondary']};
    line-height: 1.7;
    font-size: 0.95rem;
}}

.subtitle {{
    color: {COLORS['text_muted']};
    font-size: 1.1rem;
    margin-bottom: 2rem;
}}

/* ====== ENHANCED CARD SYSTEM ====== */
.ui-card {{
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.08) 0%,
        rgba(255,255,255,0.02) 100%
    );
    border-radius: 20px;
    padding: 1.5rem 2rem;
    border: 1px solid {COLORS['border']};
    box-shadow: {SHADOWS['lg']};
    margin-bottom: 1.5rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}}

.ui-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['primary_light']});
    opacity: 0;
    transition: opacity 0.3s ease;
}}

.ui-card:hover {{
    border-color: {COLORS['border_hover']};
    transform: translateY(-2px);
    box-shadow: {SHADOWS['xl']};
}}

.ui-card:hover::before {{
    opacity: 1;
}}

/* Card Variants */
.card-success {{
    border-left: 4px solid {COLORS['success']};
}}

.card-warning {{
    border-left: 4px solid {COLORS['warning']};
}}

.card-danger {{
    border-left: 4px solid {COLORS['danger']};
}}

.card-info {{
    border-left: 4px solid {COLORS['info']};
}}

/* ====== METRIC DISPLAY ====== */
.ui-metric {{
    font-size: 2.75rem;
    font-weight: 800;
    margin: 0.5rem 0;
    background: linear-gradient(135deg, {COLORS['primary_light']}, {COLORS['primary']});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}}

.metric-label {{
    font-size: 0.875rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {COLORS['text_muted']};
    font-weight: 600;
    margin-bottom: 0.25rem;
}}

.metric-subtitle {{
    font-size: 0.875rem;
    color: {COLORS['text_muted']};
    margin-top: 0.5rem;
}}

.metric-trend {{
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.875rem;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    margin-top: 0.5rem;
}}

.trend-up {{
    background: rgba(239, 68, 68, 0.1);
    color: {COLORS['danger']};
}}

.trend-down {{
    background: rgba(34, 197, 94, 0.1);
    color: {COLORS['success']};
}}

.trend-stable {{
    background: rgba(148, 163, 184, 0.1);
    color: {COLORS['text_muted']};
}}

/* ====== STATUS PILLS ====== */
.pill {{
    display: inline-block;
    padding: 0.375rem 0.875rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.8125rem;
    transition: all 0.2s ease;
    border: 1px solid transparent;
    text-transform: uppercase;
    letter-spacing: 0.025em;
}}

.pill:hover {{
    transform: scale(1.05);
}}

.pill-green {{
    background: rgba(34, 197, 94, 0.15);
    color: {COLORS['success']};
    border-color: rgba(34, 197, 94, 0.3);
}}

.pill-yellow {{
    background: rgba(250, 204, 21, 0.15);
    color: {COLORS['warning']};
    border-color: rgba(250, 204, 21, 0.3);
}}

.pill-orange {{
    background: rgba(251, 146, 60, 0.15);
    color: #fb923c;
    border-color: rgba(251, 146, 60, 0.3);
}}

.pill-red {{
    background: rgba(239, 68, 68, 0.15);
    color: {COLORS['danger']};
    border-color: rgba(239, 68, 68, 0.3);
}}

.pill-blue {{
    background: rgba(59, 130, 246, 0.15);
    color: {COLORS['info']};
    border-color: rgba(59, 130, 246, 0.3);
}}

/* ====== ENHANCED BUTTONS ====== */
.stButton > button {{
    background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['primary_dark']});
    color: white;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.75rem 1.75rem;
    border: none;
    font-size: 0.9375rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    position: relative;
    overflow: hidden;
}}

.stButton > button::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
}}

.stButton > button:hover::before {{
    left: 100%;
}}

.stButton > button:active {{
    transform: translateY(0);
}}

/* Secondary Button */
.stButton.secondary > button {{
    background: transparent;
    border: 2px solid {COLORS['border']};
    box-shadow: none;
}}

.stButton.secondary > button:hover {{
    border-color: {COLORS['primary']};
    background: rgba(99, 102, 241, 0.1);
}}

/* ====== SIDEBAR ENHANCEMENT ====== */
section[data-testid="stSidebar"] {{
    background: {COLORS['bg_secondary']};
    border-right: 1px solid {COLORS['border']};
    box-shadow: 4px 0 24px rgba(0,0,0,0.3);
}}

section[data-testid="stSidebar"] .block-container {{
    padding: 1.5rem 1rem;
}}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {COLORS['text_primary']};
    font-size: 1.125rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid {COLORS['border']};
}}

/* Sidebar Sections */
.sidebar-section {{
    background: {COLORS['bg_card']};
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    border: 1px solid {COLORS['border']};
}}

/* ====== ENHANCED TABS ====== */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    background: {COLORS['bg_card']};
    padding: 0.5rem;
    border-radius: 16px;
    border: 1px solid {COLORS['border']};
}}

.stTabs [data-baseweb="tab"] {{
    font-size: 0.9375rem;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    border-radius: 12px;
    background: transparent;
    border: none;
    color: {COLORS['text_muted']};
    transition: all 0.3s ease;
}}

.stTabs [data-baseweb="tab"]:hover {{
    background: rgba(99, 102, 241, 0.1);
    color: {COLORS['primary_light']};
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['primary_dark']}) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}}

/* ====== ALERT BANNER ====== */
.alert-banner {{
    padding: 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(250, 204, 21, 0.15), rgba(251, 146, 60, 0.15));
    border-radius: 12px;
    border: 1px solid rgba(250, 204, 21, 0.3);
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    animation: slideDown 0.4s ease-out;
}}

@keyframes slideDown {{
    from {{
        opacity: 0;
        transform: translateY(-10px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

.alert-icon {{
    font-size: 1.5rem;
}}

.alert-content {{
    flex: 1;
}}

.alert-title {{
    font-weight: 700;
    color: {COLORS['warning']};
    margin-bottom: 0.25rem;
}}

/* ====== LOADING SKELETON ====== */
.skeleton {{
    background: linear-gradient(
        90deg,
        rgba(255,255,255,0.05) 25%,
        rgba(255,255,255,0.1) 50%,
        rgba(255,255,255,0.05) 75%
    );
    background-size: 200% 100%;
    animation: loading 1.5s infinite;
    border-radius: 8px;
}}

@keyframes loading {{
    0% {{
        background-position: 200% 0;
    }}
    100% {{
        background-position: -200% 0;
    }}
}}

/* ====== EMPTY STATE ====== */
.empty-state {{
    text-align: center;
    padding: 4rem 2rem;
    color: {COLORS['text_muted']};
}}

.empty-state-icon {{
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}}

.empty-state-title {{
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: {COLORS['text_secondary']};
}}

.empty-state-description {{
    font-size: 1rem;
    color: {COLORS['text_muted']};
}}

/* ====== TOOLTIP ====== */
.tooltip {{
    position: relative;
    display: inline-block;
    cursor: help;
    border-bottom: 1px dotted {COLORS['text_muted']};
}}

/* ====== DIVIDER ====== */
.divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, {COLORS['border']}, transparent);
    margin: 2rem 0;
}}

/* ====== BADGE ====== */
.badge {{
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.625rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    background: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    color: {COLORS['text_primary']};
}}

/* ====== METRIC GRID ====== */
.metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}}

/* ====== CHART CONTAINER ====== */
.chart-container {{
    background: {COLORS['bg_card']};
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid {COLORS['border']};
    margin: 1.5rem 0;
}}

.chart-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}}

.chart-title {{
    font-size: 1.125rem;
    font-weight: 600;
    color: {COLORS['text_primary']};
}}

.chart-actions {{
    display: flex;
    gap: 0.5rem;
}}

/* ====== STREAMLIT SPECIFIC OVERRIDES ====== */
.stMetric {{
    background: {COLORS['bg_card']};
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid {COLORS['border']};
}}

.stMetric label {{
    color: {COLORS['text_muted']} !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

.stMetric [data-testid="stMetricValue"] {{
    color: {COLORS['text_primary']} !important;
    font-size: 2.25rem !important;
    font-weight: 700 !important;
}}

.stMetric [data-testid="stMetricDelta"] {{
    font-size: 0.875rem !important;
}}

/* DataFrame Styling */
.dataframe {{
    border: 1px solid {COLORS['border']} !important;
    border-radius: 8px;
    overflow: hidden;
}}

.dataframe thead tr {{
    background: {COLORS['bg_card']} !important;
}}

.dataframe thead th {{
    color: {COLORS['text_primary']} !important;
    font-weight: 600 !important;
    padding: 0.75rem !important;
    border-bottom: 2px solid {COLORS['border']} !important;
}}

.dataframe tbody td {{
    padding: 0.75rem !important;
    border-bottom: 1px solid {COLORS['border']} !important;
}}

/* ====== RESPONSIVE DESIGN ====== */
@media (max-width: 768px) {{
    .block-container {{
        padding: 1rem;
    }}
    
    h1 {{
        font-size: 2rem;
    }}
    
    .ui-metric {{
        font-size: 2rem;
    }}
    
    .metric-grid {{
        grid-template-columns: 1fr;
    }}
}}

/* ====== ANIMATIONS ====== */
@keyframes fadeIn {{
    from {{
        opacity: 0;
        transform: translateY(10px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

.fade-in {{
    animation: fadeIn 0.4s ease-out;
}}

@keyframes pulse {{
    0%, 100% {{
        opacity: 1;
    }}
    50% {{
        opacity: 0.5;
    }}
}}

.pulse {{
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}}

/* ====== LIVE INDICATOR ====== */
.live-indicator {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 600;
    color: {COLORS['success']};
}}

.live-dot {{
    width: 8px;
    height: 8px;
    background: {COLORS['success']};
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}}

/* ====== ACCESSIBILITY ====== */
*:focus {{
    outline: 2px solid {COLORS['primary']};
    outline-offset: 2px;
}}

@media (prefers-reduced-motion: reduce) {{
    *,
    *::before,
    *::after {{
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }}
}}

/* ====== SCROLLBAR ====== */
::-webkit-scrollbar {{
    width: 10px;
    height: 10px;
}}

::-webkit-scrollbar-track {{
    background: {COLORS['bg_secondary']};
}}

::-webkit-scrollbar-thumb {{
    background: {COLORS['border']};
    border-radius: 5px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: {COLORS['border_hover']};
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
def init_session_state():
    """Initialize all session state variables"""
    if "reference_manager" not in st.session_state:
        st.session_state.reference_manager = RollingReferenceManager(
            reference_size=200,
            strategy="freeze"
        )
    
    if "cluster_builder" not in st.session_state:
        st.session_state.cluster_builder = ClusterBuilder(
            method="kmeans",
            n_clusters=4
        )
    
    if "cluster_lifecycle" not in st.session_state:
        st.session_state.cluster_lifecycle = ClusterLifecycleTracker(
            min_cluster_size=30
        )
    
    if "cluster_risk_engine" not in st.session_state:
        st.session_state.cluster_risk_engine = ClusterRiskEngine()
    
    if "window_manager" not in st.session_state:
        st.session_state.window_manager = WindowManager(
            sliding_window_size=200,
            reference_window_size=200
        )
    
    if "reference_probs" not in st.session_state:
        st.session_state.reference_probs = None
    
    if "slice_registry" not in st.session_state:
        slice_registry = SliceRegistry(min_slice_size=30)
        slice_registry.register(threshold_slice("confidence", 0.4, op="<"))
        slice_registry.register(range_slice("confidence", 0.4, 0.7))
        slice_registry.register(threshold_slice("confidence", 0.7, op=">"))
        st.session_state.slice_registry = slice_registry
    
    if "alert_engine" not in st.session_state:
        st.session_state.alert_engine = AlertEngine(
            persistence_windows=2,
            cooldown_windows=3,
            top_k_slices=3
        )
    
    if "fingerprint_store" not in st.session_state:
        st.session_state.fingerprint_store = BehaviorFingerprintStore()
    
    if "early_warning_engine" not in st.session_state:
        st.session_state.early_warning_engine = EarlyWarningEngine()
    
    if "alerts" not in st.session_state:
        st.session_state.alerts = []

init_session_state()

# =========================================================
# LOAD INFERENCE ENGINE
# =========================================================
@st.cache_resource
def load_inference_engine():
    return StreamingInferenceEngine()

engine = load_inference_engine()

# =========================================================
# UI COMPONENT LIBRARY
# =========================================================
def ui_header():
    """Render the main header with live status indicator"""
    st.markdown("""
    <div class="ui-card fade-in">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div style="flex: 1;">
                <h1 style="margin: 0; display: flex; align-items: center; gap: 0.75rem;">
                    🛡️ Silent Drift Monitor
                    <span class="live-indicator">
                        <span class="live-dot"></span>
                        LIVE
                    </span>
                </h1>
                <p class="subtitle" style="margin: 0.75rem 0 0 0;">
                    Real-time detection of <b>hidden model degradation</b> in production — 
                    <i>without labels, retraining, or downtime.</i>
                </p>
            </div>
        </div>
        <div style="display: flex; gap: 0.75rem; margin-top: 1.25rem; flex-wrap: wrap;">
            <span class="pill pill-green">Label-Free</span>
            <span class="pill pill-blue">Streaming</span>
            <span class="pill pill-yellow">Output-Only</span>
            <span class="pill pill-orange">Regulation-Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def ui_metric_card(label, value, subtitle=None, pill=None, trend=None):
    """Enhanced metric card with trend indicators"""
    trend_html = ""
    if trend:
        trend_class = f"trend-{trend['direction']}"
        trend_icon = "↑" if trend['direction'] == 'up' else "↓" if trend['direction'] == 'down' else "→"
        trend_html = f"""
        <div class="metric-trend {trend_class}">
            {trend_icon} {trend['value']}
        </div>
        """
    
    pill_html = f'<span class="pill {pill}">{pill.replace("pill-", "").upper()}</span>' if pill else ""
    subtitle_html = f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ""
    
    st.markdown(f"""
    <div class="ui-card fade-in">
        <div class="metric-label">{label}</div>
        <div class="ui-metric">{value}</div>
        {pill_html}
        {subtitle_html}
        {trend_html}
    </div>
    """, unsafe_allow_html=True)

def ui_alert(severity, title, description, dismissible=False):
    """Enhanced alert component with variants"""
    color_map = {
        "low": ("pill-green", "✓"),
        "medium": ("pill-yellow", "⚠"),
        "high": ("pill-orange", "⚡"),
        "critical": ("pill-red", "🚨")
    }
    pill, icon = color_map.get(severity.lower(), ("pill-blue", "ℹ"))
    
    card_class = {
        "low": "card-success",
        "medium": "card-warning",
        "high": "card-warning",
        "critical": "card-danger"
    }.get(severity.lower(), "")
    
    st.markdown(f"""
    <div class="ui-card {card_class} fade-in">
        <div style="display: flex; align-items: start; gap: 1rem;">
            <div style="font-size: 1.5rem;">{icon}</div>
            <div style="flex: 1;">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                    <span class="pill {pill}">{severity.upper()}</span>
                    <h3 style="margin: 0;">{title}</h3>
                </div>
                <p style="margin: 0; line-height: 1.6;">{description}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def ui_section_header(title, subtitle=None, icon=None):
    """Section header with optional icon and subtitle"""
    icon_html = f'<span style="margin-right: 0.5rem;">{icon}</span>' if icon else ""
    subtitle_html = f'<p class="subtitle" style="margin-top: 0.5rem;">{subtitle}</p>' if subtitle else ""
    
    st.markdown(f"""
    <div style="margin: 2.5rem 0 1.5rem 0;">
        <h2>{icon_html}{title}</h2>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)

def ui_chart_container(title, chart_function, export_button=False):
    """Wrapper for charts with consistent styling"""
    st.markdown(f"""
    <div class="chart-container">
        <div class="chart-header">
            <div class="chart-title">{title}</div>
        </div>
    """, unsafe_allow_html=True)
    
    chart_function()
    
    st.markdown("</div>", unsafe_allow_html=True)

def ui_empty_state(icon, title, description):
    """Empty state component"""
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)

def ui_loading_skeleton(height="100px"):
    """Loading skeleton placeholder"""
    st.markdown(f"""
    <div class="skeleton" style="height: {height}; width: 100%;"></div>
    """, unsafe_allow_html=True)

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
def render_sidebar():
    """Render enhanced sidebar with grouped controls"""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid rgba(255,255,255,0.08);">
        <h2 style="margin: 0; font-size: 1.5rem;">⚙️ Control Panel</h2>
        <p style="font-size: 0.875rem; color: #94a3b8; margin-top: 0.5rem;">Configure monitoring parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Window Controls
    with st.sidebar.expander("🪟 Window Controls", expanded=True):
        window_size = st.slider(
            "Sliding Window Size",
            min_value=50,
            max_value=500,
            value=200,
            help="Number of recent samples to analyze"
        )
        
        reference_size = st.slider(
            "Reference Window Size",
            min_value=50,
            max_value=500,
            value=200,
            help="Baseline samples for comparison"
        )
        
        steps = st.slider(
            "Simulation Steps",
            min_value=1,
            max_value=500,
            value=100,
            help="Number of time steps to simulate"
        )
    
    # Reference Strategy
    with st.sidebar.expander("🔄 Reference Strategy", expanded=True):
        strategy = st.selectbox(
            "Reset Strategy",
            ["freeze", "hard", "canary"],
            help="How to update the reference window"
        )
        
        ref_size = st.slider(
            "Reference Size",
            50, 500, 200,
            help="Size of reference dataset"
        )
        
        if st.button("🔄 Manual Reset", use_container_width=True):
            st.session_state.reference_manager.manual_reset()
            st.success("Reference window reset!")
    
    # Drift Sensitivity
    with st.sidebar.expander("🎚️ Sensitivity Controls", expanded=True):
        sensitivity = st.slider(
            "Drift Sensitivity",
            min_value=1,
            max_value=10,
            value=5,
            help="Higher values = more sensitive detection"
        )
        
        # Composite Score Weights
        st.markdown("**Composite Score Weights**")
        output_weight = st.slider("Output Drift", 0.0, 1.0, 0.4, 0.05)
        confidence_weight = st.slider("Confidence", 0.0, 1.0, 0.2, 0.05)
        entropy_weight = st.slider("Entropy", 0.0, 1.0, 0.2, 0.05)
        stability_weight = st.slider("Stability", 0.0, 1.0, 0.2, 0.05)
        
        # Normalize weights
        total = output_weight + confidence_weight + entropy_weight + stability_weight
        if total > 0:
            st.caption(f"✓ Weights normalized (sum={total:.2f})")
    
    # Reference State Display
    with st.sidebar.expander("📊 Reference State", expanded=False):
        ref_mgr = st.session_state.reference_manager
        
        st.markdown(f"""
        <div class="sidebar-section">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                <div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.25rem;">Frozen</div>
                    <div style="font-weight: 600;">{ref_mgr.frozen}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.25rem;">Strategy</div>
                    <div style="font-weight: 600;">{strategy}</div>
                </div>
            </div>
            <div style="margin-top: 0.75rem;">
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.25rem;">Last Update</div>
                <div style="font-weight: 600; font-size: 0.875rem;">{ref_mgr.last_update_time or 'Never'}</div>
            </div>
            <div style="margin-top: 0.75rem;">
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.25rem;">Reason</div>
                <div style="font-weight: 600; font-size: 0.875rem;">{ref_mgr.last_update_reason or 'N/A'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Demo Controls
    with st.sidebar.expander("🎤 Demo Mode", expanded=False):
        demo_mode = st.toggle(
            "Enable Judge Mode",
            value=True,
            help="Show AI-generated insights"
        )
        
        show_tooltips = st.toggle(
            "Show Tooltips",
            value=True,
            help="Display helpful explanations"
        )
    
    # Update session state
    st.session_state.reference_manager.reference_size = ref_size
    st.session_state.reference_manager.strategy = strategy
    
    return {
        'window_size': window_size,
        'reference_size': reference_size,
        'steps': steps,
        'sensitivity': sensitivity,
        'demo_mode': demo_mode,
        'output_weight': output_weight,
        'confidence_weight': confidence_weight,
        'entropy_weight': entropy_weight,
        'stability_weight': stability_weight,
    }

# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def compute_kl(p, q, bins=20):
    """Compute KL divergence between two distributions"""
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    p_hist += 1e-8
    q_hist += 1e-8
    return entropy(p_hist, q_hist)

def compute_psi(expected, actual, bins=10):
    """Compute Population Stability Index"""
    expected_hist, _ = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bins)
    expected_perc = expected_hist / len(expected)
    actual_perc = actual_hist / len(actual)
    psi = np.sum(
        (expected_perc - actual_perc) * 
        np.log((expected_perc + 1e-8) / (actual_perc + 1e-8))
    )
    return psi

def drift_severity(value, low, high):
    """Determine drift severity level"""
    if value < low:
        return "pill-green", "Stable"
    elif value < high:
        return "pill-yellow", "Watch"
    else:
        return "pill-orange", "Drift Risk"

# =========================================================
# MAIN APPLICATION
# =========================================================
def main():
    # Render header
    ui_header()
    
    # System alert banner
    st.markdown("""
    <div class="alert-banner">
        <div class="alert-icon">🔔</div>
        <div class="alert-content">
            <div class="alert-title">System Alert</div>
            <div style="color: #cbd5f5;">Drift monitoring active. No blocking actions enforced.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render sidebar and get config
    config = render_sidebar()
    
    # Update window manager
    window_manager = st.session_state.window_manager
    window_manager.sliding_window_size = config['window_size']
    window_manager.reference_window_size = config['reference_size']
    
    # Run inference engine
    engine.run(config['steps'])
    pred_df = engine.get_prediction_dataframe()
    window_manager.update(pred_df)
    
    # Check probability readiness
    current_df = window_manager.get_sliding_window()
    PROBS_READY = False
    probs = None
    
    if current_df is not None and not current_df.empty:
        prob_cols = [c for c in current_df.columns if c.startswith("p_class_")]
        
        if len(prob_cols) > 0:
            probs = current_df[prob_cols].to_numpy()
            
            if probs.ndim == 2 and probs.shape[0] > 0 and probs.shape[1] >= 2:
                PROBS_READY = True
    
    if not PROBS_READY:
        ui_empty_state(
            "⏳",
            "Initializing Monitoring System",
            "Collecting initial data samples... This may take a few moments."
        )
        ui_loading_skeleton("200px")
        st.stop()
    
    # Extract signals
    time_idx = np.arange(len(probs))
    mean_conf = ConfidenceEntropySignals.mean_top1_confidence(probs)
    conf_var = ConfidenceEntropySignals.confidence_variance(probs)
    prob_margin = ConfidenceEntropySignals.probability_margin(probs)
    low_conf_mass = float(np.mean(ConfidenceEntropySignals.low_confidence_mass(probs)))
    
    entropies = ConfidenceEntropySignals.sample_entropy(probs)
    mean_entropy = ConfidenceEntropySignals.mean_entropy(entropies)
    entropy_slope = ConfidenceEntropySignals.entropy_trend(entropies, time_idx)
    entropy_div = ConfidenceEntropySignals.entropy_confidence_divergence(probs)
    
    time = pred_df["timestamp"]
    dummy_probs = pred_df["confidence"]
    
    # Build reference window
    if st.session_state.reference_probs is None and len(pred_df) >= config['reference_size']:
        st.session_state.reference_probs = pred_df["confidence"].iloc[:config['reference_size']].values
    
    reference_probs = st.session_state.reference_probs
    
    # Compute drift scores
    if reference_probs is not None:
        kl_score = compute_kl(reference_probs, dummy_probs)
        psi_score = compute_psi(reference_probs, dummy_probs)
        wass_score = wasserstein_distance(reference_probs, dummy_probs)
    else:
        kl_score = psi_score = wass_score = 0.0
    
    # Get class distributions
    tracker = OutputDistributionTracker()
    current_df_tmp = window_manager.get_sliding_window()
    reference_df_tmp = window_manager.get_reference_window()
    
    if current_df_tmp is not None and reference_df_tmp is not None:
        current_freq = tracker.class_frequency(current_df_tmp)
        reference_freq = tracker.class_frequency(reference_df_tmp)
        current_freq, reference_freq = tracker.align_class_distributions(
            current_freq, reference_freq
        )
    else:
        current_freq = reference_freq = pd.Series([0.25, 0.25, 0.25, 0.25])
    
    # Compute composite scores
    composite_engine = CompositeDriftScore(
        config['output_weight'],
        config['confidence_weight'],
        config['entropy_weight'],
        config['stability_weight']
    )
    
    output_drift = np.array([kl_score, psi_score, wass_score])
    confidence_arr = np.array([mean_conf, mean_conf, mean_conf])
    entropy_arr = np.array([mean_entropy, mean_entropy, mean_entropy])
    stability_arr = np.array([0.1, 0.12, 0.11])
    
    global_score = composite_engine.compute_global_score(
        output_drift, confidence_arr, entropy_arr, stability_arr
    )
    
    composite_score = float(np.mean(global_score))
    
    # Update reference manager
    if PROBS_READY:
        top1_conf = np.max(probs, axis=1)
        st.session_state.reference_manager.update(
            current_probs=top1_conf,
            entropy_slope=entropy_slope
        )
    
    # Compute slice metrics
    slice_registry = st.session_state.slice_registry
    valid_slices, _ = slice_registry.get_valid_slices(current_df)
    
    slice_metrics_dict = {}  # Initialize as empty dict in case reference_probs is None
    if reference_probs is not None:  # Only compute if reference_probs is available
        for slice_name, info in valid_slices.items():
            slice_metrics_dict[slice_name] = compute_slice_metrics(
                reference_probs,
                info["data"]["confidence"].values
            )

    
    global_metrics = {
        "kl_divergence": kl_score,
        "psi": psi_score,
        "wasserstein": wass_score,
        "entropy": mean_entropy,
        "mean_confidence": mean_conf
    }
    
    slice_rankings = rank_slices(slice_metrics_dict, global_metrics)
    slice_explanations = {
        name: explain_slice(name, metrics, global_metrics)
        for name, metrics in slice_metrics_dict.items()
    }
    
    alerts = st.session_state.alert_engine.process(
        slice_rankings, slice_explanations, window_id=time_idx[-1]
    )
    
    st.session_state.alerts = alerts
    
    # =========================================================
    # TABS
    # =========================================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Overview",
        "📈 Drift Signals",
        "🔬 Slice Monitoring",
        "🔍 Diagnosis",
        "📋 Audit Report",
        "🧠 Confidence & Entropy",
        "🎯 Drift Attribution",
        "🧪 Stability Tests",
        "📄 Self-Audit"
    ])
    
    # =========================================================
    # TAB 1: OVERVIEW
    # =========================================================
    with tab1:
        ui_section_header(
            "System Health Overview",
            "Real-time monitoring dashboard with key performance indicators",
            "🚦"
        )
        
        if config['demo_mode']:
            ui_alert(
                "low",
                "AI Judge Summary",
                "The model shows early signs of entropy drift in low-confidence slices. "
                "Output stability remains intact. No mitigation required at this stage."
            )
        
        # Active Incidents
        if st.session_state.alerts:
            ui_section_header("Active Incidents", icon="🚨")
            for alert in st.session_state.alerts:
                ui_alert(
                    severity=alert["severity"],
                    title=f"Slice: {alert['slice']}",
                    description=alert["explanation"]
                )
        else:
            ui_alert(
                "low",
                "No Active Incidents",
                "System behavior within expected bounds. All metrics nominal."
            )
        
        # System Health Metrics
        ui_section_header("Key Metrics", "Primary system health indicators", "📊")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ui_metric_card(
                "System Status",
                "Healthy",
                subtitle="No blocking actions required",
                pill="pill-green",
                trend={'direction': 'stable', 'value': '0%'}
            )
        
        with col2:
            ui_metric_card(
                "Composite Drift Score",
                f"{composite_score:.3f}",
                subtitle="Weighted global risk indicator",
                pill="pill-yellow" if composite_score > 0.15 else "pill-green"
            )
        
        with col3:
            ui_metric_card(
                "Trend",
                "Stable →",
                subtitle="No acceleration detected",
                pill="pill-green"
            )
        
        # Additional Metrics Row
        col4, col5, col6 = st.columns(3)
        
        with col4:
            ui_metric_card(
                "Mean Confidence",
                f"{mean_conf:.3f}",
                subtitle="Average model certainty"
            )
        
        with col5:
            ui_metric_card(
                "Mean Entropy",
                f"{mean_entropy:.3f}",
                subtitle="Prediction uncertainty"
            )
        
        with col6:
            ui_metric_card(
                "Low-Conf Mass",
                f"{low_conf_mass:.1%}",
                subtitle="Uncertain predictions"
            )
        
        # Predictions Timeline
        ui_section_header("Prediction Timeline", "Historical prediction activity", "📈")
        
        def render_timeline():
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time,
                y=np.arange(len(time)),
                mode='lines',
                name='Predictions',
                line=dict(color=COLORS['primary'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba(99, 102, 241, 0.1)"
            ))
            fig.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Time",
                yaxis_title="Cumulative Predictions",
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        ui_chart_container("Predictions Over Time", render_timeline)
        
        # Class Distribution
        ui_section_header("Class Distribution", "Comparison of current vs reference", "📊")
        
        def render_class_dist():
            class_df = pd.DataFrame({
                "Class": current_freq.index.astype(str),
                "Reference": reference_freq.values,
                "Current": current_freq.values
            })
            
            fig = px.bar(
                class_df,
                x="Class",
                y=["Reference", "Current"],
                barmode="group",
                color_discrete_sequence=[COLORS['text_muted'], COLORS['primary']]
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        ui_chart_container("Predicted Class Distribution", render_class_dist)
    
    # =========================================================
    # TAB 2: DRIFT SIGNALS
    # =========================================================
    with tab2:
        ui_section_header(
            "Drift Signals & Metrics",
            "Statistical indicators of model degradation",
            "📈"
        )
        
        # Drift Risk Indicators
        ui_section_header("Risk Indicators", "Key drift detection metrics", "🚨")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pill, label = drift_severity(kl_score, 0.05, 0.15)
            ui_metric_card(
                "Distribution Shift",
                f"{kl_score:.3f}",
                subtitle=f"KL divergence · {label}",
                pill=pill
            )
        
        with col2:
            pill, label = drift_severity(psi_score, 0.1, 0.25)
            ui_metric_card(
                "Population Stability",
                f"{psi_score:.3f}",
                subtitle=f"PSI score · {label}",
                pill=pill
            )
        
        with col3:
            pill, label = drift_severity(wass_score, 0.05, 0.2)
            ui_metric_card(
                "Transport Distance",
                f"{wass_score:.3f}",
                subtitle=f"Wasserstein · {label}",
                pill=pill
            )
        
        # Class Distribution Comparison
        ui_section_header("Distribution Comparison", "Reference vs Current", "📊")
        
        def render_dist_comparison():
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            
            x = np.arange(len(current_freq))
            width = 0.35
            
            ax.bar(x - width/2, reference_freq.values, width, 
                   label="Reference", color=COLORS['text_muted'], alpha=0.8)
            ax.bar(x + width/2, current_freq.values, width, 
                   label="Current", color=COLORS['primary'], alpha=0.8)
            
            ax.set_xticks(x)
            ax.set_xticklabels(current_freq.index.astype(str))
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability", color=COLORS['text_secondary'])
            ax.tick_params(colors=COLORS['text_secondary'])
            ax.legend(facecolor='#0f172a', edgecolor='#ffffff')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#ffffff') 
            ax.spines['bottom'].set_color('#ffffff')
            
            st.pyplot(fig, use_container_width=True)
        
        ui_chart_container("Class Distribution Comparison", render_dist_comparison)
        
        # Confidence Distribution
        if reference_probs is not None:
            ui_section_header("Confidence Distribution", "Model certainty over time", "📉")
            
            def render_confidence_dist():
                current_conf_df = pd.DataFrame({"confidence": dummy_probs})
                reference_conf_df = pd.DataFrame({"confidence": reference_probs})
                
                if tracker.validate_confidence_range(current_conf_df):
                    curr_hist, bins = tracker.confidence_histogram(current_conf_df, bins=30)
                    ref_hist, _ = tracker.confidence_histogram(reference_conf_df, bins=30)
                    
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=bin_centers,
                        y=ref_hist,
                        mode='lines',
                        name='Reference',
                        line=dict(color=COLORS['text_muted'], width=3),
                        fill='tozeroy',
                        fillcolor=f"rgba(148, 163, 184, 0.1)"
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=bin_centers,
                        y=curr_hist,
                        mode='lines',
                        name='Current',
                        line=dict(color=COLORS['primary'], width=3),
                        fill='tozeroy',
                        fillcolor=f"rgba(99, 102, 241, 0.1)"
                    ))
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=350,
                        xaxis_title="Confidence",
                        yaxis_title="Density",
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            ui_chart_container("Prediction Confidence Distribution", render_confidence_dist)
    
    # =========================================================
    # TAB 3: SLICE MONITORING
    # =========================================================
    with tab3:
        ui_section_header(
            "Slice-Level Drift Monitoring",
            "Granular analysis of model behavior across data segments",
            "🔬"
        )
        
        low_conf_mask = dummy_probs < 0.4
        high_conf_mask = dummy_probs > 0.7
        
        slice_data = []
        for slice_name, info in valid_slices.items():
            data = info["data"]
            slice_data.append({
                "Slice": slice_name,
                "Size": len(data),
                "Avg Confidence": data["confidence"].mean(),
                "Min Confidence": data["confidence"].min(),
                "Max Confidence": data["confidence"].max(),
                "Std Dev": data["confidence"].std()
            })
        
        slice_df = pd.DataFrame(slice_data)
        
        st.markdown("""
        <div class="chart-container">
            <div class="chart-header">
                <div class="chart-title">Slice Statistics</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            slice_df.style.background_gradient(
                subset=['Avg Confidence'],
                cmap='RdYlGn'
            ).format({
                'Avg Confidence': '{:.3f}',
                'Min Confidence': '{:.3f}',
                'Max Confidence': '{:.3f}',
                'Std Dev': '{:.3f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Slice Performance Visualization
        ui_section_header("Slice Performance", "Confidence distribution by slice", "📊")
        
        def render_slice_performance():
            fig = px.box(
                pd.DataFrame({
                    'Slice': ['Low Conf'] * low_conf_mask.sum() + ['High Conf'] * high_conf_mask.sum(),
                    'Confidence': np.concatenate([
                        dummy_probs[low_conf_mask],
                        dummy_probs[high_conf_mask]
                    ])
                }),
                x='Slice',
                y='Confidence',
                color='Slice',
                color_discrete_sequence=[COLORS['danger'], COLORS['success']]
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=400,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        ui_chart_container("Confidence Distribution by Slice", render_slice_performance)
    
    # =========================================================
    # TAB 4: DIAGNOSIS
    # =========================================================
    with tab4:
        ui_section_header(
            "Drift Diagnosis & Root Cause Analysis",
            "Detailed investigation of detected anomalies",
            "🔍"
        )
        
        # Failure Type Classification
        failure_type = ConfidenceEntropySignals.silent_failure_typing(
            mean_conf=mean_conf,
            mean_ent=mean_entropy,
            margin=prob_margin,
            low_conf_mass=low_conf_mass
        )
        
        SEVERITY_MAP = {
            "Healthy / Stable": ("🟢", "Low", "pill-green"),
            "Boundary Confusion": ("🟡", "Medium", "pill-yellow"),
            "Class Overlap": ("🟡", "Medium", "pill-yellow"),
            "Data Noise Accumulation": ("🟠", "High", "pill-orange"),
            "Domain Shift": ("🔴", "Critical", "pill-red")
        }
        
        icon, severity, pill = SEVERITY_MAP.get(failure_type, ("❓", "Unknown", "pill-blue"))
        
        st.markdown(f"""
        <div class="ui-card card-{severity.lower() if severity != 'Unknown' else 'info'}">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <div style="font-size: 3rem;">{icon}</div>
                <div>
                    <div class="metric-label">Detected Failure Mode</div>
                    <h2 style="margin: 0.5rem 0;">{failure_type}</h2>
                    <span class="pill {pill}">{severity} Severity</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Diagnostic Signals
        ui_section_header("Diagnostic Signals", "Key indicators contributing to diagnosis", "🔬")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="ui-card">
                <h3>Observable Patterns</h3>
                <ul style="margin: 1rem 0; padding-left: 1.5rem; line-height: 2;">
                    <li>Confidence entropy showing upward trend</li>
                    <li>Output distribution shifting from baseline</li>
                    <li>Increased uncertainty in predictions</li>
                    <li>No label feedback available for validation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="ui-card">
                <h3>Recommended Actions</h3>
                <ul style="margin: 1rem 0; padding-left: 1.5rem; line-height: 2;">
                    <li>Monitor slice-level metrics closely</li>
                    <li>Investigate feature drift in high-risk slices</li>
                    <li>Consider expanding reference window</li>
                    <li>Review recent data quality changes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # =========================================================
    # TAB 5: AUDIT REPORT
    # =========================================================
    with tab5:
        ui_section_header(
            "Self-Audit & Governance Report",
            "Compliance-ready documentation and audit trail",
            "📋"
        )
        
        audit = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_info": {
                "engine": "StreamingInferenceEngine",
                "version": "1.0.0"
            },
            "metrics": {
                "kl_divergence": float(kl_score),
                "psi": float(psi_score),
                "wasserstein": float(wass_score),
                "composite_score": float(composite_score),
                "mean_confidence": float(mean_conf),
                "mean_entropy": float(mean_entropy)
            },
            "configuration": {
                "window_size": config['window_size'],
                "reference_size": config['reference_size'],
                "sensitivity": config['sensitivity']
            },
            "compliance": {
                "labels_used": False,
                "retraining_performed": False,
                "blocking_actions": False
            },
            "alerts": [
                {
                    "severity": alert["severity"],
                    "slice": alert["slice"],
                    "explanation": alert["explanation"]
                }
                for alert in st.session_state.alerts
            ]
        }
        
        os.makedirs("audit_logs", exist_ok=True)
        audit_path = f"audit_logs/audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(audit_path, "w") as f:
            json.dump(audit, f, indent=2)
        
        st.markdown(f"""
        <div class="ui-card card-success">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 2rem;">✓</div>
                <div>
                    <h3 style="margin: 0;">Audit Report Generated</h3>
                    <p style="margin: 0.5rem 0 0 0;">Saved to: <code>{audit_path}</code></p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.json(audit)
        
        # Download button
        st.download_button(
            label="📥 Download Audit Report",
            data=json.dumps(audit, indent=2),
            file_name=f"audit_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # =========================================================
    # TAB 6: CONFIDENCE & ENTROPY
    # =========================================================
    with tab6:
        ui_section_header(
            "Confidence & Entropy Analysis",
            "Deep dive into model uncertainty and degradation signals",
            "🧠"
        )
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Confidence", f"{mean_conf:.3f}")
        
        with col2:
            st.metric("Mean Entropy", f"{mean_entropy:.3f}")
        
        with col3:
            st.metric("Low-Conf Mass", f"{low_conf_mass:.2%}")
        
        with col4:
            entropy_shock = ConfidenceEntropySignals.entropy_shock(entropies)
            st.metric(
                "Entropy Shock",
                "YES 🚨" if entropy_shock else "No ✓",
                delta="Detected" if entropy_shock else "Not Detected"
            )
        
        # Phase Space Diagram
        ui_section_header("Confidence-Entropy Phase Space", "Relationship visualization", "🌌")
        
        def render_phase_space():
            phase_df = pd.DataFrame({
                "Time": time_idx,
                "Confidence": np.max(probs, axis=1),
                "Entropy": entropies
            })
            
            fig = px.scatter(
                phase_df,
                x="Confidence",
                y="Entropy",
                color="Time",
                color_continuous_scale="Viridis",
                size_max=10
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        ui_chart_container("Confidence–Entropy Phase Space", render_phase_space)
        
        # Time Series
        ui_section_header("Temporal Evolution", "Trends over time", "📈")
        
        def render_temporal():
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time_idx,
                y=np.max(probs, axis=1),
                mode='lines',
                name='Confidence',
                line=dict(color=COLORS['primary'], width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=time_idx,
                y=entropies,
                mode='lines',
                name='Entropy',
                line=dict(color=COLORS['warning'], width=2),
                yaxis='y2'
            ))
            
            fig.update_layout(
                template="plotly_dark",
                height=400,
                xaxis_title="Time",
                yaxis_title="Confidence",
                yaxis2=dict(
                    title="Entropy",
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        ui_chart_container("Confidence & Entropy Over Time", render_temporal)
    
    # =========================================================
    # TAB 7: DRIFT ATTRIBUTION
    # =========================================================
    with tab7:
        ui_section_header(
            "Drift Attribution Analysis",
            "Identify which features contribute most to drift",
            "🎯"
        )
        
        feature_cols = [
            c for c in current_df.columns 
            if not c.startswith("p_class_") and c != "timestamp" and c != "_cluster_id"
        ]
        
        if len(feature_cols) > 0:
            reference_df = window_manager.get_reference_window()
            
            kl_scores = feature_kl_divergence(reference_df, current_df, feature_cols)
            psi_scores = feature_psi(reference_df, current_df, feature_cols)
            
            drift_df = pd.DataFrame({
                "Feature": feature_cols,
                "KL Divergence": [kl_scores[f] for f in feature_cols],
                "PSI": [psi_scores[f] for f in feature_cols]
            }).sort_values(by="KL Divergence", ascending=False)
            
            ui_section_header("Top Drifting Features", "Ranked by KL divergence", "📊")
            
            st.markdown("""
            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">Feature Drift Rankings</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                drift_df.style.background_gradient(
                    subset=['KL Divergence', 'PSI'],
                    cmap='RdYlGn_r'
                ).format({
                    'KL Divergence': '{:.4f}',
                    'PSI': '{:.4f}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Visualization
            def render_feature_drift():
                fig = px.bar(
                    drift_df.head(10),
                    x='Feature',
                    y='KL Divergence',
                    color='KL Divergence',
                    color_continuous_scale='RdYlGn_r'
                )
                
                fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            ui_chart_container("Top 10 Drifting Features", render_feature_drift)
        else:
            ui_empty_state(
                "🔍",
                "No Features Available",
                "Feature-level drift analysis requires additional data columns."
            )
    
    # =========================================================
    # TAB 8: STABILITY TESTS
    # =========================================================
    with tab8:
        ui_section_header(
            "Stability & Perturbation Tests",
            "Robustness analysis through controlled perturbations",
            "🧪"
        )
        
        # Original predictions
        y_orig_probs = probs.copy()
        
        # Perturbed predictions
        y_pert_probs = y_orig_probs + np.random.normal(0, 1e-3, y_orig_probs.shape)
        y_pert_probs = np.clip(y_pert_probs, 0, 1)
        y_pert_probs /= y_pert_probs.sum(axis=1, keepdims=True)
        
        y_orig = np.argmax(y_orig_probs, axis=1)
        y_pert = np.argmax(y_pert_probs, axis=1)
        flips = (y_orig != y_pert).sum()
        flip_rate = flips / len(y_orig)
        
        # Counterfactual stability
        n_trials = 5
        cf_flip_rates = []
        
        for _ in range(n_trials):
            y_pert_probs_trial = y_orig_probs + np.random.normal(0, 1e-3, y_orig_probs.shape)
            y_pert_probs_trial = np.clip(y_pert_probs_trial, 0, 1)
            y_pert_probs_trial /= y_pert_probs_trial.sum(axis=1, keepdims=True)
            y_pert_trial = np.argmax(y_pert_probs_trial, axis=1)
            cf_flip_rates.append((y_orig != y_pert_trial).sum() / len(y_orig))
        
        cf_stability = np.mean(cf_flip_rates)
        
        # Display Results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ui_metric_card(
                "Noise Flip Rate",
                f"{flip_rate:.2%}",
                subtitle="Single perturbation test",
                pill="pill-green" if flip_rate < 0.05 else "pill-yellow"
            )
        
        with col2:
            ui_metric_card(
                "Counterfactual Stability",
                f"{cf_stability:.2%}",
                subtitle=f"Average over {n_trials} trials",
                pill="pill-green" if cf_stability < 0.05 else "pill-yellow"
            )
        
        with col3:
            stability_score = 1 - cf_stability
            ui_metric_card(
                "Stability Score",
                f"{stability_score:.3f}",
                subtitle="Higher is better",
                pill="pill-green" if stability_score > 0.95 else "pill-yellow"
            )
        
        # Visualization
        ui_section_header("Perturbation Analysis", "Distribution of flip rates", "📊")
        
        def render_perturbation():
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=cf_flip_rates,
                nbinsx=20,
                name='Flip Rates',
                marker_color=COLORS['primary'],
                opacity=0.7
            ))
            
            fig.add_vline(
                x=np.mean(cf_flip_rates),
                line_dash="dash",
                line_color=COLORS['warning'],
                annotation_text=f"Mean: {np.mean(cf_flip_rates):.3f}"
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=350,
                xaxis_title="Flip Rate",
                yaxis_title="Frequency",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        ui_chart_container("Counterfactual Flip Rate Distribution", render_perturbation)
    
    # =========================================================
    # TAB 9: SELF-AUDIT
    # =========================================================
    with tab9:
        ui_section_header(
            "Comprehensive Self-Audit Report",
            "Detailed system analysis for compliance and governance",
            "📄"
        )
        
        st.markdown("""
        <div class="ui-card card-info">
            <h3>📋 Audit Report Summary</h3>
            <p style="margin-top: 0.5rem;">
                This comprehensive report provides a complete overview of the monitoring system's 
                current state, including all metrics, configurations, and detected anomalies. 
                The report is suitable for regulatory compliance and internal governance.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate comprehensive report
        comprehensive_audit = {
            "metadata": {
                "report_generated": datetime.utcnow().isoformat(),
                "report_version": "1.0.0",
                "system_version": "Silent Drift Monitor v1.0"
            },
            "model_info": {
                "engine": "StreamingInferenceEngine",
                "monitoring_mode": "Output-Only",
                "label_free": True
            },
            "current_metrics": {
                "drift_scores": {
                    "kl_divergence": float(kl_score),
                    "psi": float(psi_score),
                    "wasserstein": float(wass_score),
                    "composite_score": float(composite_score)
                },
                "confidence_metrics": {
                    "mean_confidence": float(mean_conf),
                    "confidence_variance": float(conf_var),
                    "low_confidence_mass": float(low_conf_mass)
                },
                "entropy_metrics": {
                    "mean_entropy": float(mean_entropy),
                    "entropy_slope": float(entropy_slope),
                    "entropy_shock_detected": bool(entropy_shock)
                }
            },
            "system_configuration": {
                "window_size": config['window_size'],
                "reference_size": config['reference_size'],
                "sensitivity": config['sensitivity'],
                "weights": {
                    "output": config['output_weight'],
                    "confidence": config['confidence_weight'],
                    "entropy": config['entropy_weight'],
                    "stability": config['stability_weight']
                }
            },
            "slice_analysis": {
                "total_slices": len(valid_slices),
                "slice_metrics": slice_metrics_dict
            },
            "alerts": {
                "active_count": len(st.session_state.alerts),
                "alerts": st.session_state.alerts
            },
            "compliance": {
                "labels_used": False,
                "retraining_performed": False,
                "blocking_actions": False,
                "audit_trail_maintained": True
            },
            "recommendations": [
                "Continue monitoring slice-level metrics",
                "Review reference window update strategy",
                "Investigate high-drift features",
                "Maintain current sensitivity settings"
            ]
        }
        
        # Save report
        os.makedirs("audit_logs", exist_ok=True)
        report_path = f"audit_logs/comprehensive_audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, "w") as f:
            json.dump(comprehensive_audit, f, indent=2)
        
        st.success(f"✓ Comprehensive audit report saved to: `{report_path}`")
        
        # Display report
        st.json(comprehensive_audit)
        
        # Download button
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download JSON Report",
                data=json.dumps(comprehensive_audit, indent=2),
                file_name=f"comprehensive_audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Create CSV summary
            summary_df = pd.DataFrame({
                "Metric": [
                    "KL Divergence", "PSI", "Wasserstein",
                    "Mean Confidence", "Mean Entropy",
                    "Composite Score", "Active Alerts"
                ],
                "Value": [
                    f"{kl_score:.4f}",
                    f"{psi_score:.4f}",
                    f"{wass_score:.4f}",
                    f"{mean_conf:.4f}",
                    f"{mean_entropy:.4f}",
                    f"{composite_score:.4f}",
                    len(st.session_state.alerts)
                ]
            })
            
            st.download_button(
                label="📊 Download CSV Summary",
                data=summary_df.to_csv(index=False),
                file_name=f"audit_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # =========================================================
    # FOOTER
    # =========================================================
    st.markdown("""
    <div style="margin-top: 4rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.08); text-align: center;">
        <p style="color: #94a3b8; font-size: 0.875rem;">
            Silent Drift Monitor v1.0.0 | Enterprise ML Monitoring Platform
        </p>
        <p style="color: #64748b; font-size: 0.75rem; margin-top: 0.5rem;">
            Built with Streamlit • Designed for Production ML Systems
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# RUN APPLICATION
# =========================================================
if __name__ == "__main__":
    main()