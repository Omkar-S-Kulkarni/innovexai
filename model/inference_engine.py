# ============================================================
# PHASE 2.2 — STREAMING INFERENCE ENGINE
# MODEL IS FROZEN — OUTPUT ONLY — NO LABEL ACCESS
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_PATH = "D:\innovexai\data\dataset.csv"
MODEL_PATH = "D:\innovexai\model\model.pkl"
METADATA_PATH = "D:\innovexai\model\model_metadata.json"

# ------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------
def entropy(probs: np.ndarray) -> float:
    """Shannon entropy of probability vector"""
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs))


def probability_margin(probs: np.ndarray) -> float:
    """Difference between top-1 and top-2 probabilities"""
    sorted_probs = np.sort(probs)[::-1]
    if len(sorted_probs) < 2:
        return 0.0
    return sorted_probs[0] - sorted_probs[1]


# ------------------------------------------------------------
# INFERENCE ENGINE
# ------------------------------------------------------------
class StreamingInferenceEngine:
    """
    Simulates a deployed ML model performing
    sequential, output-only inference on streaming data.
    """

    def __init__(self):
        # Load frozen model ONCE
        self.model = joblib.load(MODEL_PATH)
        self.model = joblib.load("D:\innovexai\model\model.pkl")


        # Load metadata for feature order validation
        with open(METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

        self.feature_list = self.metadata["feature_list"]

        # Load full dataset (time-ordered)
        self.data = pd.read_csv(DATA_PATH)
        self.data = self.data.sort_values("time").reset_index(drop=True)

        # Remove forbidden columns
        self.data = self.data.drop(
            columns=[c for c in ["true_label", "cluster_id"] if c in self.data.columns]
        )

        # Internal pointer for streaming
        self.current_index = 0

        # Output-only buffer (append-only)
        self.prediction_log = []

    # --------------------------------------------------------
    # STREAM ONE RECORD
    # --------------------------------------------------------
    def step(self):
        """
        Process exactly ONE new data point.
        """
        if self.current_index >= len(self.data):
            return None  # End of stream

        row = self.data.iloc[self.current_index]
        timestamp = row["time"]

        # Extract features in correct order
        X = row[self.feature_list].values.reshape(1, -1)

        # Predict probabilities
        probs = self.model.predict_proba(X)[0]

        # Derived signals
        y_pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        ent = float(entropy(probs))
        margin = float(probability_margin(probs))

        # Build output record
        record = {
            "timestamp": timestamp,
            "y_pred": y_pred,
            "confidence": conf,
            "entropy": ent,
            "margin": margin
        }

        # Add per-class probabilities explicitly
        for i, p in enumerate(probs):
            record[f"p_class_{i}"] = float(p)

        # Append to output stream
        self.prediction_log.append(record)

        # Advance stream
        self.current_index += 1

        return record
    def get_prediction_probabilities(self) -> np.ndarray:
        """
        Reconstruct probability matrix from prediction_log.

        Returns:
        --------
        np.ndarray of shape (N, num_classes)
        """

        if not self.prediction_log:
            return np.empty((0, 0))

        # Infer number of classes from keys
        prob_keys = sorted(
            [k for k in self.prediction_log[0].keys() if k.startswith("p_class_")]
        )

        probs = []
        for record in self.prediction_log:
            probs.append([record[k] for k in prob_keys])

        return np.array(probs)

    # --------------------------------------------------------
    # STREAM MULTIPLE RECORDS
    # --------------------------------------------------------
    def run(self, steps: int):
        """
        Stream multiple sequential predictions.
        """
        outputs = []
        for _ in range(steps):
            result = self.step()
            if result is None:
                break
            outputs.append(result)
        return outputs
    

    # --------------------------------------------------------
    # GET OUTPUT STREAM AS DATAFRAME
    # --------------------------------------------------------
    def get_prediction_dataframe(self) -> pd.DataFrame:
        """
        Return the append-only prediction log
        with reconstructed probability vectors.
        """
        if not self.prediction_log:
            return pd.DataFrame()

        df = pd.DataFrame(self.prediction_log)

        # Identify probability columns
        prob_cols = sorted([c for c in df.columns if c.startswith("p_class_")])

        # Reconstruct probability vectors per row
        df["probs"] = df[prob_cols].values.tolist()

        return df


# # ============================================================
# # PHASE 2.2 — STREAMING INFERENCE ENGINE
# # MODEL IS FROZEN — OUTPUT ONLY — NO LABEL ACCESS
# # ============================================================

# import json
# import joblib
# import numpy as np
# import pandas as pd
# import os

# # ------------------------------------------------------------
# # CONFIG - CROSS-PLATFORM PATHS
# # ------------------------------------------------------------
# # Use relative paths or environment variables for better portability
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
# MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
# METADATA_PATH = os.path.join(BASE_DIR, "model", "model_metadata.json")

# # ------------------------------------------------------------
# # UTILITY FUNCTIONS
# # ------------------------------------------------------------
# def entropy(probs: np.ndarray) -> float:
#     """Shannon entropy of probability vector"""
#     probs = np.clip(probs, 1e-12, 1.0)
#     return -np.sum(probs * np.log(probs))


# def probability_margin(probs: np.ndarray) -> float:
#     """Difference between top-1 and top-2 probabilities"""
#     sorted_probs = np.sort(probs)[::-1]
#     if len(sorted_probs) < 2:
#         return 0.0
#     return sorted_probs[0] - sorted_probs[1]


# # ------------------------------------------------------------
# # INFERENCE ENGINE
# # ------------------------------------------------------------
# class StreamingInferenceEngine:
#     """
#     Simulates a deployed ML model performing
#     sequential, output-only inference on streaming data.
#     """

#     def __init__(self):
#         # Verify files exist
#         if not os.path.exists(MODEL_PATH):
#             raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
#         if not os.path.exists(METADATA_PATH):
#             raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
#         if not os.path.exists(DATA_PATH):
#             raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        
#         # Load frozen model ONCE
#         self.model = joblib.load(MODEL_PATH)

#         # Load metadata for feature order validation
#         with open(METADATA_PATH, "r") as f:
#             self.metadata = json.load(f)

#         self.feature_list = self.metadata["feature_list"]

#         # Load full dataset (time-ordered)
#         self.data = pd.read_csv(DATA_PATH)
#         self.data = self.data.sort_values("time").reset_index(drop=True)

#         # Remove forbidden columns
#         self.data = self.data.drop(
#             columns=[c for c in ["true_label", "cluster_id"] if c in self.data.columns],
#             errors='ignore'
#         )

#         # Internal pointer for streaming
#         self.current_index = 0

#         # Output-only buffer (append-only)
#         self.prediction_log = []

#     # --------------------------------------------------------
#     # STREAM ONE RECORD
#     # --------------------------------------------------------
#     def step(self):
#         """
#         Process exactly ONE new data point.
#         """
#         if self.current_index >= len(self.data):
#             return None  # End of stream

#         row = self.data.iloc[self.current_index]
#         timestamp = row["time"]

#         # Extract features in correct order
#         X = row[self.feature_list].values.reshape(1, -1)

#         # Predict probabilities
#         probs = self.model.predict_proba(X)[0]

#         # Derived signals
#         y_pred = int(np.argmax(probs))
#         conf = float(np.max(probs))
#         ent = float(entropy(probs))
#         margin = float(probability_margin(probs))

#         # Build output record
#         record = {
#             "timestamp": timestamp,
#             "y_pred": y_pred,
#             "confidence": conf,
#             "entropy": ent,
#             "margin": margin
#         }

#         # Add per-class probabilities explicitly
#         for i, p in enumerate(probs):
#             record[f"p_class_{i}"] = float(p)

#         # Append to output stream
#         self.prediction_log.append(record)

#         # Advance stream
#         self.current_index += 1

#         return record
    
#     def get_prediction_probabilities(self) -> np.ndarray:
#         """
#         Reconstruct probability matrix from prediction_log.

#         Returns:
#         --------
#         np.ndarray of shape (N, num_classes)
#         """

#         if not self.prediction_log:
#             return np.empty((0, 0))

#         # Infer number of classes from keys
#         prob_keys = sorted(
#             [k for k in self.prediction_log[0].keys() if k.startswith("p_class_")]
#         )

#         probs = []
#         for record in self.prediction_log:
#             probs.append([record[k] for k in prob_keys])

#         return np.array(probs)

#     # --------------------------------------------------------
#     # STREAM MULTIPLE RECORDS
#     # --------------------------------------------------------
#     def run(self, steps: int):
#         """
#         Stream multiple sequential predictions.
#         """
#         outputs = []
#         for _ in range(steps):
#             result = self.step()
#             if result is None:
#                 break
#             outputs.append(result)
#         return outputs
    
#     # --------------------------------------------------------
#     # GET OUTPUT STREAM AS DATAFRAME
#     # --------------------------------------------------------
#     def get_prediction_dataframe(self) -> pd.DataFrame:
#         """
#         Return the append-only prediction log
#         with reconstructed probability vectors.
#         """
#         if not self.prediction_log:
#             return pd.DataFrame()

#         df = pd.DataFrame(self.prediction_log)

#         # Identify probability columns
#         prob_cols = sorted([c for c in df.columns if c.startswith("p_class_")])

#         # Reconstruct probability vectors per row
#         if len(prob_cols) > 0:
#             df["probs"] = df[prob_cols].values.tolist()

#         return df