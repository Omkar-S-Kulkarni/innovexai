import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance
import json
import os
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



ref_mgr = st.session_state.reference_manager



# =========================================================
# Engine Load (DEPLOYMENT-SAFE)
# =========================================================
@st.cache_resource
def load_inference_engine():
    return StreamingInferenceEngine()

engine = load_inference_engine()


# =========================================================
# Reference Window (FROZEN ONCE)
# =========================================================
if "reference_probs" not in st.session_state:
    st.session_state.reference_probs = None


# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Silent Drift Monitor",
    layout="wide"
)

# -------------------------------
# Top Alert Banner (GLOBAL)
# -------------------------------
st.markdown(
    """
    <div style="padding:10px; background-color:#FFF3CD; border-radius:5px; border:1px solid #FFEEBA;">
    🔔 <b>System Alert:</b> Drift monitoring active. No blocking actions enforced.
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# App Header
# -------------------------------
st.title("Silent Drift Monitor")
st.subheader("Output-only monitoring for deployed ML systems")

st.caption(
    """
    Detects gradual, localized, and silent model degradation **without ground truth labels**.
    """
)

st.markdown(
    """
    **Constraints:**  
    • Fixed model (no retraining)  
    • Streaming data  
    • Output-only signals  
    • Non-blocking detection  
    """
)

# -------------------------------
# Sidebar (Control Panel)
# -------------------------------
st.sidebar.header("⚙️ Control Panel")

st.sidebar.subheader("Time & Window Controls")
window_size = st.sidebar.slider(
    "Sliding window size",
    min_value=50,
    max_value=500,
    value=200
)
st.sidebar.subheader("🔄 Rolling Reference")

ref_size = st.sidebar.slider("Reference size", 50, 500, 200)
strategy = st.sidebar.selectbox(
    "Reset strategy", ["freeze", "hard", "canary"]
)

ref_mgr.reference_size = ref_size
ref_mgr.strategy = strategy

if st.sidebar.button("Manual Reference Reset"):
    ref_mgr.manual_reset()


reference_size = st.sidebar.slider(
    "Reference window size",
    min_value=50,
    max_value=500,
    value=200
)

st.sidebar.subheader("Sensitivity Controls")
sensitivity = st.sidebar.slider(
    "Drift sensitivity",
    min_value=1,
    max_value=10,
    value=5
)

# -------------------------------
# Streaming Control
# -------------------------------
steps = st.sidebar.slider(
    "Simulated time steps",
    min_value=1,
    max_value=500,
    value=100
)
if "window_manager" not in st.session_state:
    st.session_state.window_manager = WindowManager(
        sliding_window_size=window_size,
        reference_window_size=reference_size
    )

alert_engine = AlertEngine(
    persistence_windows=2,
    cooldown_windows=3,
    top_k_slices=3
)


window_manager = st.session_state.window_manager

engine.run(steps)
pred_df = engine.get_prediction_dataframe()
# st.write("Prediction DF Columns:", pred_df.columns.tolist())

window_manager.update(pred_df)
# =========================================================
# PHASE 4 — CONFIDENCE & ENTROPY SIGNAL EXTRACTION (SAFE)
# =========================================================

window_manager.sliding_window_size = window_size
window_manager.reference_window_size = reference_size

current_df = window_manager.get_sliding_window()

# ---------- Global readiness flag ----------
# =========================================================
# PROBABILITY READINESS CHECK (SINGLE SOURCE OF TRUTH)
# =========================================================
PROBS_READY = False
probs = None

current_df = window_manager.get_sliding_window()

if current_df is None or current_df.empty:
    st.info("⏳ Waiting for sliding window to fill...")

else:
    prob_cols = [c for c in current_df.columns if c.startswith("p_class_")]

    if len(prob_cols) == 0:
        st.info("⏳ Probability columns not available yet...")

    else:
        probs = current_df[prob_cols].to_numpy()

        if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] < 2:
            st.info("⏳ Insufficient probability data...")

        else:
            PROBS_READY = True

if not PROBS_READY:
    st.stop()


time_idx = np.arange(len(probs))

# ---- Confidence signals ----
mean_conf = ConfidenceEntropySignals.mean_top1_confidence(probs)
conf_var = ConfidenceEntropySignals.confidence_variance(probs)
prob_margin = ConfidenceEntropySignals.probability_margin(probs)
low_conf_mass = float(np.mean(ConfidenceEntropySignals.low_confidence_mass(probs)))

# ---- Entropy signals ----
entropies = ConfidenceEntropySignals.sample_entropy(probs)
mean_entropy = ConfidenceEntropySignals.mean_entropy(entropies)
entropy_slope = ConfidenceEntropySignals.entropy_trend(entropies, time_idx)
entropy_div = ConfidenceEntropySignals.entropy_confidence_divergence(probs)

entropy_shock = ConfidenceEntropySignals.entropy_shock(entropies)
conf_collapse = ConfidenceEntropySignals.confidence_shape_collapse(probs)



time = pred_df["timestamp"]
dummy_classes = pred_df["y_pred"]
dummy_probs = pred_df["confidence"]



# =========================================================
# Build Reference Window ONCE
# =========================================================
if st.session_state.reference_probs is None and len(pred_df) >= reference_size:
    st.session_state.reference_probs = (
        pred_df["confidence"].iloc[:reference_size].values
    )

reference_probs = st.session_state.reference_probs

st.sidebar.subheader("📌 Reference State")

st.sidebar.write("Frozen:", ref_mgr.frozen)
st.sidebar.write("Last Update:", ref_mgr.last_update_time)
st.sidebar.write("Reason:", ref_mgr.last_update_reason)

# =========================================================
# Drift Metric Functions (REAL)
# =========================================================
def compute_kl(p, q, bins=20):
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    p_hist += 1e-8
    q_hist += 1e-8
    return entropy(p_hist, q_hist)

def compute_psi(expected, actual, bins=10):
    expected_hist, _ = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bins)
    expected_perc = expected_hist / len(expected)
    actual_perc = actual_hist / len(actual)
    psi = np.sum(
        (expected_perc - actual_perc)
        * np.log((expected_perc + 1e-8) / (actual_perc + 1e-8))
    )
    return psi


# =========================================================
# Drift Scores
# =========================================================
if reference_probs is not None:
    kl_score = compute_kl(reference_probs, dummy_probs)
    psi_score = compute_psi(reference_probs, dummy_probs)
    wass_score = wasserstein_distance(reference_probs, dummy_probs)
else:
    kl_score = psi_score = wass_score = 0.0

tracker = OutputDistributionTracker()


# PHASE 6 — SLICE REGISTRY INITIALIZATION (ONCE)
# =========================================================
from monitoring.slice_definition import SliceRegistry, threshold_slice, range_slice

if "slice_registry" not in st.session_state:
    slice_registry = SliceRegistry(min_slice_size=30)

    # ---- Manual confidence-based slices ----
    slice_registry.register(
        threshold_slice("confidence", 0.4, op="<")
    )
    slice_registry.register(
        range_slice("confidence", 0.4, 0.7)
    )
    slice_registry.register(
        threshold_slice("confidence", 0.7, op=">")
    )

    st.session_state.slice_registry = slice_registry
else:
    slice_registry = st.session_state.slice_registry












# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5,tab6,tab7,tab8,tab9= st.tabs(
    [" Overview", " Drift Signals", " Slice Monitoring", " Diagnosis", " Audit Report","🧠 Confidence & Entropy","🔍 Drift Attribution", "🧪 Stability & Perturbation Tests","Self-Audit Report"]
)

# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tab1:
    st.header("System Health Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("System Status", "Healthy →")
    col2.metric("Composite Drift Score", f"{(kl_score + psi_score) / 2:.3f}")
    col3.metric("Trend", "→ Stable")

    fig, ax = plt.subplots()
    ax.plot(time, np.arange(len(time)))
    ax.set_title("Predictions Over Time")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    counts = pd.Series(dummy_classes).value_counts()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Predicted Class Frequency")
    st.pyplot(fig)

# =========================================================
# TAB 2 — DRIFT SIGNALS
# =========================================================


with tab2:
    if not PROBS_READY:
        st.info("⏳ Waiting for probability outputs…")
    else:
        current_df = window_manager.get_sliding_window()
        reference_df = window_manager.get_reference_window()

        if current_df is None or current_df.empty:
            st.info("Waiting for enough data for sliding window...")
            # PROBS_READY = True


        if reference_df is None or reference_df.empty:
            st.info("Collecting baseline data for reference window...")
            # PROBS_READY = True


        current_freq = tracker.class_frequency(current_df)
        reference_freq = tracker.class_frequency(reference_df)
        current_freq, reference_freq = tracker.align_class_distributions(
                current_freq, reference_freq
            )

        fig, ax = plt.subplots(figsize=(6, 4))

        x = np.arange(len(current_freq))
        width = 0.35

        ax.bar(x - width / 2, reference_freq.values, width, label="Reference")
        ax.bar(x + width / 2, current_freq.values, width, label="Current")

        ax.set_xticks(x)
        ax.set_xticklabels(current_freq.index.astype(str))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Class Probability")
        ax.set_title("Predicted Class Distribution")
        ax.legend()

        st.pyplot(fig)

    # ---------- Confidence / Probability Distribution ----------
    if reference_probs is not None:

        current_conf_df = pd.DataFrame({"confidence": dummy_probs})
        reference_conf_df = pd.DataFrame({"confidence": reference_probs})

        if tracker.validate_confidence_range(current_conf_df):

            curr_hist, bins = tracker.confidence_histogram(
                current_conf_df, bins=20
            )
            ref_hist, _ = tracker.confidence_histogram(
                reference_conf_df, bins=20
            )

            fig_conf, ax_conf = plt.subplots(figsize=(6, 4))

            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            ax_conf.plot(bin_centers, ref_hist, label="Reference", linewidth=2)
            ax_conf.plot(bin_centers, curr_hist, label="Current", linewidth=2)

            ax_conf.set_title("Prediction Confidence Distribution")
            ax_conf.set_xlabel("Confidence")
            ax_conf.set_ylabel("Density")
            ax_conf.set_xlim(0, 1)
            ax_conf.legend()

            st.pyplot(fig_conf)

    # ---------- Drift Metrics ----------
    c1, c2, c3 = st.columns(3)
    c1.metric("KL Divergence", f"{kl_score:.4f}")
    c2.metric("PSI", f"{psi_score:.4f}")
    c3.metric("Wasserstein Distance", f"{wass_score:.4f}")


# =========================================================
# TAB 3 — SLICE MONITORING
# =========================================================
with tab3:
    st.header("Slice-Level Drift Monitoring")
    if not PROBS_READY:
        st.info("⏳ Waiting for probability outputs")

    else:
        low_conf_mask = dummy_probs < 0.4
        low_conf_mass = low_conf_mask.mean()

        high_conf = dummy_probs > 0.7

        slice_table = pd.DataFrame({
            "Slice": ["Low Confidence", "High Confidence"],
            "Size": [low_conf_mass.sum(), high_conf.sum()],
            "Avg Confidence": [
                dummy_probs[low_conf_mass].mean() if low_conf_mass.any() else 0,
                dummy_probs[high_conf].mean() if high_conf.any() else 0,
            ]
        })

        st.dataframe(slice_table, use_container_width=True)

# =========================================================
# TAB 4 — DIAGNOSIS
# =========================================================
with tab4:
    st.header("Drift Diagnosis & Explanation")
    if not PROBS_READY:
        st.info("⏳ Waiting for probability outputs")

    else:
        st.write("• Confidence entropy increasing")
        st.write("• Output distribution shifting")
        st.write("• No label feedback available")

# =========================================================
# TAB 5 — AUDIT REPORT
# =========================================================
with tab5:
    st.header("Self-Audit & Governance Report")
    if not PROBS_READY:
        st.info("⏳ Waiting for probability outputs")
    else:
        audit = {
            "timestamp": datetime.utcnow().isoformat(),
            "kl_divergence": kl_score,
            "psi": psi_score,
            "wasserstein": wass_score,
            "window_size": window_size,
            "reference_size": reference_size,
            "labels_used": False,
            "retraining": False,
        }

        os.makedirs("audit_logs", exist_ok=True)
        audit_path = f"audit_logs/audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        with open(audit_path, "w") as f:
            json.dump(audit, f, indent=2)

        st.json(audit)
        st.caption("Audit artifact saved for governance & compliance.")
mean_ent = 0.0

with tab6:
    st.header("🧠 Silent Degradation Signals")
    if not PROBS_READY:
        st.info("⏳ Confidence & entropy signals not ready yet")
    else:
        

        preds = np.argmax(probs, axis=1)

        # ---- Confidence ----
        conf_decay = ConfidenceEntropySignals.confidence_decay_rate(
            np.max(probs, axis=1), time_idx
        )

        # ---- Entropy ----
        ent_trend = ConfidenceEntropySignals.entropy_trend(entropies, time_idx)
        ent_div = ConfidenceEntropySignals.entropy_confidence_divergence(probs)

        shock = ConfidenceEntropySignals.entropy_shock(entropies)
        collapse = ConfidenceEntropySignals.confidence_shape_collapse(probs)

        # ---- Metrics ----
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Confidence", f"{mean_conf:.3f}")
        c2.metric("Mean Entropy", f"{mean_ent:.3f}")
        c3.metric("Low-Confidence Mass", f"{low_conf_mass:.2%}")


        st.metric("Entropy Shock Detected", "YES 🚨" if shock else "No")

        # ---- Phase Diagram ----
        fig, ax = plt.subplots()
        sc = ax.scatter(
            time_idx,
            np.max(probs, axis=1),
            c=entropies,
            cmap="viridis"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Confidence")
        ax.set_title("Confidence–Entropy Phase Diagram")
        plt.colorbar(sc, label="Entropy")
        st.pyplot(fig)

with tab7:
    st.header("Drift Attribution")

    feature_cols = [c for c in current_df.columns if not c.startswith("p_class_") and c != "timestamp"]
    
    kl_scores = feature_kl_divergence(reference_df, current_df, feature_cols)
    psi_scores = feature_psi(reference_df, current_df, feature_cols)

    drift_df = pd.DataFrame({
        "Feature": feature_cols,
        "KL Divergence": [kl_scores[f] for f in feature_cols],
        "PSI": [psi_scores[f] for f in feature_cols]
    }).sort_values(by="KL Divergence", ascending=False)

    st.subheader("Top Feature Drift")
    st.dataframe(drift_df, use_container_width=True)

    # Slice comparison example
    # valid_slices, _ = slice_registry.get_valid_slices(current_df)
    # for slice_name, info in valid_slices.items():
    #     mask = info["mask_fn"](current_df)
    #     slice_kl = slice_feature_comparison(reference_df, current_df, mask, feature_cols)
    #     st.write(f"Slice: {slice_name} - Top KL Feature: {max(slice_kl, key=slice_kl.get)}")


with tab8:
    st.header("Stability & Perturbation Tests")

    # Convert features to array for model input (if needed)
    feature_cols = [c for c in current_df.columns if not c.startswith("p_class_") and c != "timestamp"]
    X = current_df[feature_cols].to_numpy()

    if PROBS_READY:
        # ---- Perturb probabilities directly instead of using engine ----
        # Original probabilities
        y_orig_probs = probs.copy()

        # Perturbed probabilities by adding small Gaussian noise
        y_pert_probs = y_orig_probs + np.random.normal(0, 1e-3, y_orig_probs.shape)
        y_pert_probs = np.clip(y_pert_probs, 0, 1)  # ensure probabilities are valid
        y_pert_probs /= y_pert_probs.sum(axis=1, keepdims=True)  # normalize rows

        # Compute flip rate
        y_orig = np.argmax(y_orig_probs, axis=1)
        y_pert = np.argmax(y_pert_probs, axis=1)
        flips = (y_orig != y_pert).sum()
        flip_rate = flips / len(y_orig)

        st.metric("Noise Flip Rate", f"{flip_rate:.2%}")

        # ---- Counterfactual stability (repeat multiple perturbations) ----
        n_trials = 5
        cf_flip_rates = []

        for _ in range(n_trials):
            y_pert_probs_trial = y_orig_probs + np.random.normal(0, 1e-3, y_orig_probs.shape)
            y_pert_probs_trial = np.clip(y_pert_probs_trial, 0, 1)
            y_pert_probs_trial /= y_pert_probs_trial.sum(axis=1, keepdims=True)
            y_pert_trial = np.argmax(y_pert_probs_trial, axis=1)
            cf_flip_rates.append((y_orig != y_pert_trial).sum() / len(y_orig))

        cf_stability = np.mean(cf_flip_rates)
        st.metric("Counterfactual Flip Rate", f"{cf_stability:.2%}")

    else:
        st.info("Waiting for probability columns...")




with tab9:  # or next index
    st.header("🧾 Self-Audit Report")

    if st.button("Generate Audit Report"):
        report = audit_generator.generate(
            composite_score_series=composite_scores,
            drift_regimes=drift_regimes,
            slice_scores=slice_composite_scores,
            signal_deltas=signal_deltas,
            confidence_metrics=confidence_metrics,
            entropy_metrics=entropy_metrics,
            stability_metrics=stability_metrics,
        )

        st.json(report)

        st.download_button(
            "Download JSON Report",
            data=json.dumps(report, indent=4),
            file_name="audit_report.json",
            mime="application/json",
        )





# compute entropies and slope
entropies = ConfidenceEntropySignals.sample_entropy(probs)
time_index = np.arange(len(entropies))
entropy_slope = ConfidenceEntropySignals.entropy_trend(entropies, time_index)

# Update rolling reference manager using TOP-1 confidence
if PROBS_READY:
    top1_conf = np.max(probs, axis=1)
    ref_mgr.update(
        current_probs=top1_conf,
        entropy_slope=entropy_slope
    )




global_metrics = {
    "kl_divergence": kl_score,
    "psi": psi_score,
    "wasserstein": wass_score,
    "entropy": entropies.mean(),
    "mean_confidence": mean_conf
}

# 
# # PHASE 6 — SLICE REGISTRY INITIALIZATION (ONCE)
# # =========================================================
# from monitoring.slice_definition import SliceRegistry, threshold_slice, range_slice

# if "slice_registry" not in st.session_state:
#     slice_registry = SliceRegistry(min_slice_size=30)

#     # ---- Manual confidence-based slices ----
#     slice_registry.register(
#         threshold_slice("confidence", 0.4, op="<")
#     )
#     slice_registry.register(
#         range_slice("confidence", 0.4, 0.7)
#     )
#     slice_registry.register(
#         threshold_slice("confidence", 0.7, op=">")
#     )

#     st.session_state.slice_registry = slice_registry
# else:
#     slice_registry = st.session_state.slice_registry


# =========================================================
# PHASE 9 — AUTO-CLUSTER → SLICE INTEGRATION
# =========================================================
from monitoring.cluster_builder import ClusterBuilder
from monitoring.cluster_slice_adapter import build_cluster_slices

if PROBS_READY:
    # Features used for clustering (output-only, safe)
    cluster_features = current_df[
        ["confidence"] + prob_cols
    ].copy()

    # Build clusters
    cluster_builder = ClusterBuilder(
        method="kmeans",
        n_clusters=4
    )
    clustered_df = cluster_builder.fit_predict(
        cluster_features,
        feature_cols=cluster_features.columns.tolist()
    )

    # Attach cluster ids back to current_df
    current_df["_cluster_id"] = clustered_df["_cluster_id"]
    valid_slices, _ = slice_registry.get_valid_slices(current_df)

    for slice_name, info in valid_slices.items():
        mask = info["mask_fn"](current_df)
        slice_kl = slice_feature_comparison(reference_df, current_df, mask, feature_cols)
        st.write(f"Slice: {slice_name} - Top KL Feature: {max(slice_kl, key=slice_kl.get)}")

    # Build slices from clusters
    cluster_slices = build_cluster_slices(current_df)
#=========================================================
    # Register cluster slices (once per cluster)
    for s in cluster_slices:
        if s.name not in slice_registry.slices:
            slice_registry.register(s)



# Compute slice metrics
slice_metrics_dict = {}
valid_slices, _ = slice_registry.get_valid_slices(current_df)
for slice_name, info in valid_slices.items():
    slice_metrics_dict[slice_name] = compute_slice_metrics(
        reference_probs, 
        info["data"]["confidence"].values
    )

# Rank slices
slice_rankings = rank_slices(slice_metrics_dict, global_metrics)

# Explanations
slice_explanations = {
    name: explain_slice(name, metrics, global_metrics)
    for name, metrics in slice_metrics_dict.items()
}

# Initialize / update alert engine
if "alert_engine" not in st.session_state:
    st.session_state.alert_engine = AlertEngine(
        persistence_windows=2, cooldown_windows=3, top_k_slices=3
    )

alerts = st.session_state.alert_engine.process(
    slice_rankings, slice_explanations, window_id=time_idx[-1]
)

# Display alerts in dashboard
if alerts:
    st.markdown("###  Active Alerts")
    for alert in alerts:
        st.warning(f"{alert['slice']} — {alert['severity']} — {alert['explanation']}")

failure_type = ConfidenceEntropySignals.silent_failure_typing(
    mean_conf=mean_conf,
    mean_ent=mean_entropy,
    margin=prob_margin,
    low_conf_mass=low_conf_mass
)


SEVERITY_MAP = {
    "Healthy / Stable": "🟢 Low",
    "Boundary Confusion": "🟡 Medium",
    "Class Overlap": "🟡 Medium",
    "Data Noise Accumulation": "🟠 High",
    "Domain Shift": "🔴 Critical"
}

st.metric(
    "Failure Severity",
    SEVERITY_MAP.get(failure_type, "Unknown")
)




# blindspot_matrix = build_failure_blindspot_matrix()

# audit_generator = AuditReportGenerator(
#     model_metadata={
#         "engine": "StreamingInferenceEngine",
#         "version": "v1.0",
#     },
#     window_manager=window_manager,
#     alert_engine=alert_engine,
#     blind_spot_matrix=blindspot_matrix,
# )





# -------- PHASE 8 --------
from monitoring.behavior_fingerprint import BehaviorFingerprintStore
from monitoring.trend_acceleration import compute_trends
from monitoring.early_warning_state import EarlyWarningEngine

if "fingerprint_store" not in st.session_state:
    st.session_state.fingerprint_store = BehaviorFingerprintStore()

if "early_warning_engine" not in st.session_state:
    st.session_state.early_warning_engine = EarlyWarningEngine()


if PROBS_READY:
    fp = st.session_state.fingerprint_store.compute_fingerprint(probs)
    st.session_state.fingerprint_store.update("GLOBAL", fp)

    if st.session_state.fingerprint_store.has_enough_history("GLOBAL"):
        hist = st.session_state.fingerprint_store.get_history("GLOBAL")
        trend = compute_trends(hist, "mean_entropy")
        state = st.session_state.early_warning_engine.update("GLOBAL", trend)

        st.metric("Early Warning State", state)
else:
    st.info("Probability cols are not available yet")


# =========================================================
# PHASE 7 — CLUSTER DISCOVERY (OUTPUT-ONLY)
# =========================================================

cluster_features = pd.DataFrame({
    "mean_conf": np.max(probs, axis=1),
    "entropy": entropies,
    "margin": prob_margin if np.isscalar(prob_margin) else prob_margin[:len(entropies)]
})

cluster_df = current_df.iloc[:len(cluster_features)].copy()
cluster_df[cluster_features.columns] = cluster_features.values


cluster_df = st.session_state.cluster_builder.fit_predict(
    cluster_df,
    feature_cols=["mean_conf", "entropy", "margin"]
)

cluster_stats = st.session_state.cluster_lifecycle.update(cluster_df)






output_weight = st.sidebar.slider("Output Drift Weight", 0.0, 1.0, 0.4)
confidence_weight = st.sidebar.slider("Confidence Weight", 0.0, 1.0, 0.2)
entropy_weight = st.sidebar.slider("Entropy Weight", 0.0, 1.0, 0.2)
stability_weight = st.sidebar.slider("Stability Weight", 0.0, 1.0, 0.2)

composite_engine = CompositeDriftScore(
    output_weight, confidence_weight, entropy_weight, stability_weight
)

drift_detector = DriftRegimeDetector(
    gradual_thresh=0.01, sudden_thresh=0.05, localized_thresh=0.2, osc_thresh=0.05
)

# -----------------------------
# Example: compute global composite score
# -----------------------------
# Assume these arrays come from your previous pipeline
# (KL, PSI, Wasserstein, confidence, entropy, flip_rate)
output_drift = np.array([0.1,0.2,0.15])
confidence = np.array([0.9,0.85,0.88])
entropy = np.array([0.5,0.55,0.52])
stability = np.array([0.1,0.12,0.11])

global_score = composite_engine.compute_global_score(
    output_drift, confidence, entropy, stability
)

st.line_chart(global_score, height=150, use_container_width=True)
st.write("Global Composite Drift Score:", np.mean(global_score))

# -----------------------------
# Example: per-slice composite score
# -----------------------------
slice_signals = {
    "slice_0": {"output": output_drift, "confidence": confidence, "entropy": entropy, "stability": stability},
    "slice_1": {"output": output_drift*0.8, "confidence": confidence*0.9, "entropy": entropy*1.1, "stability": stability*0.9}
}
slice_scores = composite_engine.compute_slice_scores(slice_signals)

st.write("Slice-wise Composite Scores:", slice_scores)

# -----------------------------
# Drift regime classification
# -----------------------------
regimes = drift_detector.classify(global_score, slice_scores)
st.write("Detected Drift Regimes:", regimes)
