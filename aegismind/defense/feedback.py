"""Analyst feedback loop (Sprint 5).

Every recommendation an analyst approves or rejects is appended to a JSONL
log. The optimizer turns the log into a small, transparent preference term per
action type:

    reject_rate = (rejects + 1) / (approves + rejects + 2)      # Beta(1,1) prior
    penalty     = rho * (reject_rate - 0.5)

So an action type analysts keep rejecting is ranked lower next time, and one
they keep approving is ranked higher. With no feedback the penalty is 0. This
is what makes the system "self-learning" in a reviewable way: the rule is
simple and every decision that shaped it is on record.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DECISIONS = ("approve", "reject")


class FeedbackStore:
    def __init__(self, path: str | Path = "experiments/feedback.jsonl", rho: float = 0.2):
        self.path = Path(path)
        self.rho = rho

    def record(self, scenario: str, action: dict, decision: str, note: str = "") -> dict:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scenario": scenario,
            "action_kind": action["kind"],
            "action": action.get("description", action["kind"]),
            "decision": decision,
            "note": note,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def counts(self) -> dict[str, Counter]:
        out: dict[str, Counter] = {}
        for e in self.entries():
            out.setdefault(e["action_kind"], Counter())[e["decision"]] += 1
        return out

    def preference_penalty(self) -> dict[str, float]:
        pen = {}
        for kind, c in self.counts().items():
            a, r = c["approve"], c["reject"]
            reject_rate = (r + 1) / (a + r + 2)
            pen[kind] = round(self.rho * (reject_rate - 0.5), 4)
        return pen
