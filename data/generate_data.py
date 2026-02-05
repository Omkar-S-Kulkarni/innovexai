import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION (JUDGE-IMPORTANT: REPRODUCIBLE & CONTROLLABLE)
# ============================================================
SEED = 42
N_SAMPLES = 1500

GLOBAL_DRIFT_START = 300
LOCAL_DRIFT_START = 600

GLOBAL_DRIFT_STRENGTH = 0.0008
LOCAL_DRIFT_STRENGTH = 0.0025

NOISE_INCREASE_RATE = 0.0006  # for confidence collapse

np.random.seed(SEED)

# ============================================================
# FEATURE DEFINITIONS
# ============================================================
FEATURE_NAMES = [
    "feature_1_activity_score",
    "feature_2_reliability",
    "feature_3_complexity",
    "feature_4_temporal_behavior",
    "feature_5_interaction_depth",
    "feature_6_variability"
]

N_FEATURES = len(FEATURE_NAMES)

# ============================================================
# CLUSTER (HIDDEN SUBGROUP) DEFINITION
# ============================================================
# Cluster distribution (not uniform on purpose)
CLUSTER_PROBS = [0.6, 0.25, 0.15]  # cluster 2 will drift locally
N_CLUSTERS = len(CLUSTER_PROBS)

# ============================================================
# STORAGE
# ============================================================
data_rows = []

# ============================================================
# TRUE LABEL FUNCTION (HIDDEN — EVALUATION ONLY)
# ============================================================
def true_label_function(x):
    """
    Ground truth decision boundary.
    NEVER used for monitoring.
    """
    weights = np.array([0.6, 0.8, -0.7, 0.5, 0.4, -0.6])
    score = np.dot(x, weights)
    return 1 if score > 0 else 0

# ============================================================
# DATA GENERATION LOOP (STRICTLY TIME-ORDERED)
# ============================================================
for t in range(N_SAMPLES):

    # ----------------------------
    # Time index
    # ----------------------------
    time_index = t

    # ----------------------------
    # Assign hidden cluster
    # ----------------------------
    cluster_id = np.random.choice(
        np.arange(N_CLUSTERS),
        p=CLUSTER_PROBS
    )

    # ----------------------------
    # Base feature distribution
    # ----------------------------
    features = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=N_FEATURES
    )

    # ========================================================
    # GLOBAL GRADUAL DRIFT (ALL SAMPLES)
    # ========================================================
    if t > GLOBAL_DRIFT_START:
        drift_amount = (t - GLOBAL_DRIFT_START) * GLOBAL_DRIFT_STRENGTH
        features += drift_amount

    # ========================================================
    # LOCALIZED SUBGROUP DRIFT (ONLY CLUSTER 2)
    # ========================================================
    if t > LOCAL_DRIFT_START and cluster_id == 2:
        local_drift = (t - LOCAL_DRIFT_START) * LOCAL_DRIFT_STRENGTH
        features[2] += local_drift      # feature_3_complexity
        features[5] -= local_drift      # feature_6_variability

    # ========================================================
    # CONFIDENCE COLLAPSE SETUP
    # (Noise increases → boundary proximity → instability)
    # ========================================================
    noise_scale = 1.0 + (t * NOISE_INCREASE_RATE)
    features += np.random.normal(0, noise_scale, size=N_FEATURES)

    # ========================================================
    # TRUE LABEL (HIDDEN)
    # ========================================================
    true_label = true_label_function(features)

    # ========================================================
    # STORE ROW
    # ========================================================
    row = {
        "time": time_index,
        "cluster_id": cluster_id,      # HIDDEN in demo
        "true_label": true_label       # HIDDEN in demo
    }

    for i, fname in enumerate(FEATURE_NAMES):
        row[fname] = features[i]

    data_rows.append(row)

# ============================================================
# CREATE DATAFRAME (NO SHUFFLING — EVER)
# ============================================================
df = pd.DataFrame(data_rows)

# ============================================================
# SAVE DATASET
# ============================================================
OUTPUT_PATH = "D:\innovexai\data\dataset.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(" Phase 1 dataset generated successfully")
print(f" Saved to: {OUTPUT_PATH}")
print(f" Samples: {len(df)}")
print(f" Features: {N_FEATURES}")
print(" Labels & clusters are for evaluation only (not monitoring)")
