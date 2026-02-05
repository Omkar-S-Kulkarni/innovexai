import json
import os
from datetime import datetime
from uuid import uuid4


class ReferenceAuditLogger:
    def __init__(self, path="audit_logs"):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def log(self, payload: dict):
        payload["timestamp"] = datetime.utcnow().isoformat()
        payload["id"] = str(uuid4())

        fname = f"reference_{payload['timestamp'].replace(':', '')}.json"
        with open(os.path.join(self.path, fname), "w") as f:
            json.dump(payload, f, indent=2)
