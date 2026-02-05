# cooldown_manager.py
"""
Cooldown Manager prevents alert spam by enforcing a window
during which repeated alerts for the same slice are suppressed.
"""

from collections import defaultdict


class CooldownManager:
    def __init__(self, cooldown_windows: int = 3):
        self.cooldown_windows = cooldown_windows
        self.cooldowns = defaultdict(int)

    def is_in_cooldown(self, alert_key: str) -> bool:
        """Check if a slice alert is in cooldown."""
        return self.cooldowns[alert_key] > 0

    def trigger_cooldown(self, alert_key: str):
        """Start cooldown for a slice alert."""
        self.cooldowns[alert_key] = self.cooldown_windows

    def tick(self):
        """Decrement cooldown counters for all slices."""
        for key in list(self.cooldowns.keys()):
            if self.cooldowns[key] > 0:
                self.cooldowns[key] -= 1

    def reset(self, alert_key: str):
        """Reset cooldown for a slice alert."""
        self.cooldowns[alert_key] = 0
