from collections import defaultdict


class EarlyWarningEngine:
    def __init__(
        self,
        accel_threshold=0.02,
        persistence_windows=2
    ):
        self.accel_threshold = accel_threshold
        self.persistence_windows = persistence_windows
        self.counters = defaultdict(int)
        self.states = defaultdict(lambda: "STABLE")

    def update(self, slice_name, trend_info):
        accel = abs(float(trend_info.get("acceleration", 0.0)))
        direction = trend_info.get("direction", "STABLE")

        if accel >= self.accel_threshold and direction == "WORSENING":
            self.counters[slice_name] += 1
        else:
            self.counters[slice_name] = 0
            self.states[slice_name] = "STABLE"
            return self.states[slice_name]

        if self.counters[slice_name] >= self.persistence_windows:
            self.states[slice_name] = "EARLY_WARNING"
        else:
            self.states[slice_name] = "WATCH"

        return self.states[slice_name]

    def get_state(self, slice_name):
        return self.states[slice_name]
