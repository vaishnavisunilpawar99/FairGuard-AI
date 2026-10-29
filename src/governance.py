import json
import os
from datetime import datetime, timezone
from typing import Optional

PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "decisions.json")


def can_approve(equalized_odds_gap, reviewer=None):
    if float(equalized_odds_gap) > 0.1:
        return False
    return reviewer is None or bool(reviewer.strip())


def save_decision(model, status, reviewer, reason, path=PATH, assessment_id="default_assessment"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entries = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            entries = []
    entry = {
        "model": model,
        "status": status,
        "reviewer": reviewer,
        "reason": reason,
        "time": datetime.now(timezone.utc).isoformat()
    }
    entries.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    # Persist to database if available
    try:
        from .database import crud
        crud.save_review(assessment_id, reviewer, status, reason)
    except Exception:
        pass

    return entry


def latest_status(path=PATH):
    if not os.path.exists(path):
        return "Pending"
    try:
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
        return entries[-1]["status"] if entries else "Pending"
    except Exception:
        return "Pending"
