# ============================================================
# PHASE 2.1 — ONE-TIME MODEL TRAINING
# MODEL IS FROZEN — DO NOT RETRAIN
# ============================================================

import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_PATH = "D:\innovexai\data\dataset.csv"
MODEL_PATH = "D:\innovexai\model\model.pkl"
METADATA_PATH = "D:\innovexai\model\model_metadata.json"

TRAIN_FRACTION = 0.25      # Use ONLY first 25% of timeline
RANDOM_STATE = 42

# -----------------------------
# LOAD DATA (TIME-ORDERED)
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Ensure strict time ordering
df = df.sort_values("time").reset_index(drop=True)

# -----------------------------
# TRAINING WINDOW (NO FUTURE DATA)
# -----------------------------
train_size = int(len(df) * TRAIN_FRACTION)
train_df = df.iloc[:train_size].copy()

# -----------------------------
# FEATURE SELECTION (NO LEAKAGE)
# -----------------------------
LEAKAGE_COLUMNS = {
    "timestamp",
    "cluster_id",
    "true_label"
}

feature_columns = [c for c in train_df.columns if c not in LEAKAGE_COLUMNS]

X_train = train_df[feature_columns]
y_train = train_df["true_label"]

# -----------------------------
# MODEL DEFINITION
# -----------------------------
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs"
        ))
    ]
)

# -----------------------------
# TRAIN ONCE
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# SAVE MODEL (FROZEN)
# -----------------------------
joblib.dump(model, MODEL_PATH)

# -----------------------------
# SAVE METADATA FOR AUDITABILITY
# -----------------------------
metadata = {
    "model_type": "LogisticRegression",
    "training_fraction": TRAIN_FRACTION,
    "training_sample_count": int(len(train_df)),
    "training_time_range": {
        "start_timestamp": int(train_df["time"].iloc[0]),
        "end_timestamp": int(train_df["time"].iloc[-1])
    },
    "feature_list": feature_columns,
    "random_state": RANDOM_STATE,
    "retraining_allowed": False
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print(" Model trained ONCE and frozen.")
print(f"Samples used: {float(len(train_df))}")
print(f"Features used: {float(len(feature_columns))}")
