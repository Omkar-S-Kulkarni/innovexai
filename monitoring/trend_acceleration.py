import numpy as np


def _slope(values):
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    return float(np.polyfit(x, values, 1)[0])


def compute_trends(history, key):
    values = [float(h.get(key, 0.0)) for h in history]

    if len(values) < 3:
        return {
            "slope": 0.0,
            "acceleration": 0.0,
            "direction": "STABLE"
        }

    slope_now = _slope(values[-3:])
    slope_prev = _slope(values[-4:-1]) if len(values) >= 4 else 0.0

    acceleration = slope_now - slope_prev

    if acceleration > 0.01:
        direction = "WORSENING"
    elif acceleration < -0.01:
        direction = "RECOVERING"
    else:
        direction = "STABLE"

    return {
        "slope": slope_now,
        "acceleration": acceleration,
        "direction": direction
    }
