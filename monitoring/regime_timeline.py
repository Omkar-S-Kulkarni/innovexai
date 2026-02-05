# monitoring/regime_timeline.py

def build_regime_timeline(regimes):
    timeline = []
    last = None

    for i, r in enumerate(regimes):
        if r != last:
            timeline.append({
                "window": i,
                "regime": r
            })
            last = r

    return timeline
