# import pandas as pd
# from collections import deque
# from typing import Optional


# class WindowManager:
#     """
#     Manages:
#     - Sliding (current) window
#     - Frozen reference window

#     This class is STATEFUL by design.
#     It must be cached in Streamlit using st.cache_resource.
#     """

#     def __init__(
#         self,
#         sliding_window_size: int,
#         reference_window_size: int,
#     ):
#         # -------------------------------
#         # Configuration
#         # -------------------------------
#         self.sliding_window_size = sliding_window_size
#         self.reference_window_size = reference_window_size

#         # -------------------------------
#         # Sliding Window (Deque = bounded)
#         # -------------------------------
#         self._sliding_window = deque(maxlen=sliding_window_size)

#         # -------------------------------
#         # Reference Window (Frozen)
#         # -------------------------------
#         self._reference_window: Optional[pd.DataFrame] = None
#         self._reference_locked: bool = False

#     # =========================================================
#     # INGESTION
#     # =========================================================
#     def ingest(self, prediction_row: dict):
#         """
#         Ingest ONE prediction at a time.
#         prediction_row must contain ONLY output-level data.
#         Example:
#         {
#             "timestamp": 123,
#             "y_pred": 1,
#             "confidence": 0.82
#         }
#         """
#         self._sliding_window.append(prediction_row)

#         # Auto-capture reference window ONCE
#         if not self._reference_locked:
#             self._try_capture_reference()

#     # =========================================================
#     # SLIDING WINDOW
#     # =========================================================
#     def get_sliding_window(self) -> pd.DataFrame:
#         """
#         Returns the most recent N predictions.
#         Handles warm-up safely.
#         """
#         if len(self._sliding_window) == 0:
#             return pd.DataFrame()

#         return pd.DataFrame(list(self._sliding_window))

#     def sliding_window_ready(self) -> bool:
#         """
#         Indicates whether sliding window is fully populated.
#         """
#         return len(self._sliding_window) >= self.sliding_window_size

#     # =========================================================
#     # REFERENCE WINDOW
#     # =========================================================
#     def _try_capture_reference(self):
#         """
#         Capture reference window ONCE when enough early data exists.
#         """
#         if len(self._sliding_window) >= self.reference_window_size:
#             self._reference_window = pd.DataFrame(
#                 list(self._sliding_window)[: self.reference_window_size]
#             )
#             self._reference_locked = True

#     def get_reference_window(self) -> Optional[pd.DataFrame]:
#         """
#         Returns frozen reference window.
#         """
#         return self._reference_window

#     def reference_ready(self) -> bool:
#         return self._reference_window is not None

#     # =========================================================
#     # SAFETY / GOVERNANCE
#     # =========================================================
#     def is_frozen(self) -> bool:
#         """
#         Reference window immutability check.
#         """
#         return self._reference_locked

#     def reset(self):
#         """
#         HARD RESET — SHOULD NEVER BE CALLED IN PRODUCTION.
#         Exists only for testing.
#         """
#         self._sliding_window.clear()
#         self._reference_window = None
#         self._reference_locked = False
        
#     # =========================================================
#     # BATCH UPDATE (STREAMLIT ADAPTER)
#     # =========================================================
#     def update(self, df: pd.DataFrame):
#         """
#         Ingest a batch of predictions safely.
#         Designed for Streamlit reruns.

#         df must contain:
#         - timestamp
#         - y_pred
#         - confidence
#         """

#         if df is None or df.empty:
#             return

#         # Prevent duplicate ingestion on Streamlit rerun
#         existing_timestamps = {
#             row["timestamp"] for row in self._sliding_window
#         }

#         for _, row in df.iterrows():
#             ts = row["timestamp"]

#             if ts in existing_timestamps:
#                 continue  # already ingested

#             self.ingest(
#                 {
#                     "timestamp": ts,
#                     "y_pred": row["y_pred"],
#                     "confidence": row["confidence"],
#                 }
#             )




import pandas as pd
from collections import deque
from typing import Optional


class WindowManager:
    """
    Manages:
    - Sliding (current) window
    - Frozen reference window

    This class is STATEFUL by design.
    It must be cached in Streamlit using st.cache_resource.
    """

    def __init__(
        self,
        sliding_window_size: int,
        reference_window_size: int,
    ):
        # -------------------------------
        # Configuration
        # -------------------------------
        self.sliding_window_size = sliding_window_size
        self.reference_window_size = reference_window_size

        # -------------------------------
        # Sliding Window (Deque = bounded)
        # -------------------------------
        self._sliding_window = deque(maxlen=sliding_window_size)

        # -------------------------------
        # Reference Window (Frozen)
        # -------------------------------
        self._reference_window: Optional[pd.DataFrame] = None
        self._reference_locked: bool = False

    # =========================================================
    # INGESTION
    # =========================================================
    def ingest(self, prediction_row: dict):
        """
        Ingest ONE prediction at a time.
        prediction_row must contain ONLY output-level data.
        Example:
        {
            "timestamp": 123,
            "y_pred": 1,
            "confidence": 0.82,
            "p_class_0": 0.1,
            "p_class_1": 0.82,
            "p_class_2": 0.08
        }
        """
        self._sliding_window.append(prediction_row)

        # Auto-capture reference window ONCE
        if not self._reference_locked:
            self._try_capture_reference()

    # =========================================================
    # SLIDING WINDOW
    # =========================================================
    def get_sliding_window(self) -> pd.DataFrame:
        """
        Returns the most recent N predictions.
        Handles warm-up safely.
        """
        if len(self._sliding_window) == 0:
            return pd.DataFrame()

        return pd.DataFrame(list(self._sliding_window))

    def sliding_window_ready(self) -> bool:
        """
        Indicates whether sliding window is fully populated.
        """
        return len(self._sliding_window) >= self.sliding_window_size

    # =========================================================
    # REFERENCE WINDOW
    # =========================================================
    def _try_capture_reference(self):
        """
        Capture reference window ONCE when enough early data exists.
        """
        if len(self._sliding_window) >= self.reference_window_size:
            self._reference_window = pd.DataFrame(
                list(self._sliding_window)[: self.reference_window_size]
            )
            self._reference_locked = True

    def get_reference_window(self) -> Optional[pd.DataFrame]:
        """
        Returns frozen reference window.
        """
        return self._reference_window

    def reference_ready(self) -> bool:
        return self._reference_window is not None

    # =========================================================
    # SAFETY / GOVERNANCE
    # =========================================================
    def is_frozen(self) -> bool:
        """
        Reference window immutability check.
        """
        return self._reference_locked

    def reset(self):
        """
        HARD RESET — SHOULD NEVER BE CALLED IN PRODUCTION.
        Exists only for testing.
        """
        self._sliding_window.clear()
        self._reference_window = None
        self._reference_locked = False
        
    # =========================================================
    # BATCH UPDATE (STREAMLIT ADAPTER)
    # =========================================================
    def update(self, df: pd.DataFrame):
        """
        Ingest a batch of predictions safely.
        Designed for Streamlit reruns.

        df must contain:
        - timestamp
        - y_pred
        - confidence
        - p_class_0, p_class_1, ... (all probability columns)
        - any other output columns
        """

        if df is None or df.empty:
            return

        # Prevent duplicate ingestion on Streamlit rerun
        existing_timestamps = {
            row["timestamp"] for row in self._sliding_window
        }

        for _, row in df.iterrows():
            ts = row["timestamp"]

            if ts in existing_timestamps:
                continue  # already ingested

            # CRITICAL FIX: Include ALL columns, not just the 3 basic ones
            # This preserves p_class_0, p_class_1, etc.
            row_dict = row.to_dict()
            
            self.ingest(row_dict)