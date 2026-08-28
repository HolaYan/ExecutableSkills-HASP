"""The ledger — what has been tried, what it measured, and what was retired.

This is the part that makes the loop a loop rather than a repeated first
round. Without it the proposer re-derives the same dead PFs every time: the
six "more skills" PFs, the drift anchor, the relation probe were each
plausible on their face and each cost a run to falsify. The ledger carries
that verdict forward and hands it to the next proposer as "do not propose
this again, here is what it measured".

One record per (skill_id, round). Status transitions:

    proposed ─screen─> screened_out         (offline precision gate)
             └────────> probed ─probe─> probed_out    (no marginal rescue / broke a correct one)
                              └────────> refine       (fired, but the verdict was not decisive)
                              └────────> measured ─acc─> admitted | probed_out

`admitted` is the only status that may enter a training rollout, and it
requires a measured accuracy change, not a screening number.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

_HASP = Path(__file__).resolve().parents[2]

STATUSES = ("proposed", "screened_out", "probed", "probed_out", "refine",
            "measured", "admitted", "retired")
# statuses that mean "this idea is dead — tell the next proposer"
DEAD = ("screened_out", "probed_out", "retired")


class Ledger:
    def __init__(self, path: Optional[Path] = None, domain: str = "math"):
        self.path = Path(path) if path else _HASP / "data" / "forge" / f"ledger_{domain}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict] = []
        if self.path.exists():
            self.records = [json.loads(l) for l in self.path.open() if l.strip()]

    # ── writing ──
    def upsert(self, skill_id: str, **fields) -> Dict:
        rec = self.get(skill_id)
        if rec is None:
            rec = dict(skill_id=skill_id, first_round=fields.get("round", 0),
                       status="proposed", history=[])
            self.records.append(rec)
        old = rec.get("status")
        rec.update({k: v for k, v in fields.items() if k != "history"})
        rec["updated"] = time.strftime("%Y-%m-%d %H:%M")
        if fields.get("status") and fields["status"] != old:
            rec.setdefault("history", []).append(
                dict(round=fields.get("round", rec.get("round", 0)),
                     status=fields["status"], why=(fields.get("reasons") or [""])[0]))
        return rec

    def save(self) -> None:
        self.path.write_text("".join(json.dumps(r) + "\n" for r in self.records))

    # ── reading ──
    def get(self, skill_id: str) -> Optional[Dict]:
        return next((r for r in self.records if r["skill_id"] == skill_id), None)

    def by_status(self, *statuses: str) -> List[Dict]:
        return [r for r in self.records if r.get("status") in statuses]

    def admitted_ids(self) -> List[str]:
        return [r["skill_id"] for r in self.by_status("admitted")]

    def next_round(self) -> int:
        return max((r.get("round", 0) for r in self.records), default=0) + 1

    def dead_families(self, min_dead: int = 3) -> List[str]:
        """Families where enough candidates died that proposing there again is
        a poor use of a round. Reported, never silently enforced."""
        from collections import Counter
        c = Counter(r.get("family", "") for r in self.by_status(*DEAD))
        live = {r.get("family", "") for r in self.by_status("admitted", "measured", "refine")}
        return [f for f, k in c.items() if k >= min_dead and f and f not in live]

    def falsified_note(self, limit: int = 24) -> str:
        """The block handed to the proposer: what already failed and why."""
        rows = []
        for r in self.by_status(*DEAD)[-limit:]:
            why = (r.get("reasons") or ["falsified"])[0]
            sc = r.get("screen") or {}
            nums = (f" (fired on {sc.get('fire_wrong', 0):.0%} of wrong / "
                    f"{sc.get('fire_correct', 0):.0%} of correct)" if sc else "")
            rows.append(f"  - {r['skill_id']} [{r.get('family','?')}]: {why}{nums}")
        if not rows:
            return "  (nothing falsified yet)"
        return "\n".join(rows)

    def summary(self) -> str:
        from collections import Counter
        c = Counter(r.get("status") for r in self.records)
        parts = [f"{s}={c[s]}" for s in STATUSES if c[s]]
        return f"ledger[{self.path.name}] {len(self.records)} candidates: " + ", ".join(parts)
