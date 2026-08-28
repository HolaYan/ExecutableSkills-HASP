"""Objective B data builder — 'Learn to evolve'.

Converts skill-generation candidates + Helper review + validation ΔEM
into SFT, preference (good>bad), and prompt-only datasets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..signals import SignalAggregator
from .prompt_templates import (
    SYSTEM_EVOLVE,
    build_skillgen_prompt,
    build_skillgen_target,
    to_chat,
    to_chat_prompt_only,
)

logger = logging.getLogger(__name__)


@dataclass
class EvolveBuilderConfig:
    output_dir: str
    q_skill_threshold: float = 0.5   # SFT: accept candidate if q_skill ≥ threshold
    lam_val_gain: float = 0.5
    formats: List[str] = None         # ["sft", "dpo", "prompt"]
    include_rejected_in_dpo: bool = True


class EvolveBuilder:
    def __init__(self, config: EvolveBuilderConfig, aggregator: SignalAggregator):
        self.cfg = config
        self.aggregator = aggregator
        self.out_dir = Path(config.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.formats = config.formats or ["sft"]

    # ------------------------------------------------------------------

    def build(
        self,
        candidates: List,
        reviews: List,
        val_delta_em_by_skill: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Path]:
        val_delta = val_delta_em_by_skill or {}
        review_map = {r.skill_id: r for r in reviews}
        enriched: List[Dict[str, Any]] = []
        for cand in candidates:
            review = review_map.get(cand.skill_id)
            if review is None:
                continue
            r_skill = self.aggregator.score_skill(
                q_skill=getattr(review, "q_skill", 0.0),
                validation_delta_em=val_delta.get(cand.skill_id, 0.0),
                lam=self.cfg.lam_val_gain,
            )
            enriched.append({"candidate": cand, "review": review, "r_skill": r_skill})

        outputs = {}
        if "sft" in self.formats:
            outputs["sft"] = self._write_sft(enriched)
        if "dpo" in self.formats:
            outputs["dpo"] = self._write_dpo(enriched)
        if "prompt" in self.formats:
            outputs["prompt"] = self._write_prompt_only(enriched)

        self._write_manifest(enriched)
        return outputs

    # ------------------------------------------------------------------

    def _write_sft(self, enriched: List[Dict[str, Any]]) -> Path:
        path = self.out_dir / "objB_sft.jsonl"
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for item in enriched:
                cand = item["candidate"]
                review = item["review"]
                q = getattr(review, "q_skill", 0.0)
                if q < self.cfg.q_skill_threshold:
                    continue
                user = build_skillgen_prompt(
                    failure_pattern=getattr(cand, "target_failure_pattern", "")[:1000],
                )
                target = build_skillgen_target(
                    md_spec=getattr(cand, "md_spec", ""),
                    pf_code=getattr(cand, "pf_code", ""),
                )
                row = to_chat(SYSTEM_EVOLVE, user, target)
                row["sample_weight"] = item["r_skill"]
                row["skill_id"] = cand.skill_id
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        logger.info("Wrote %d Obj-B SFT samples → %s", n, path)
        return path

    def _write_dpo(self, enriched: List[Dict[str, Any]]) -> Path:
        """Preference = accepted (good) > rejected (bad). Pairs within same target cluster if available."""
        path = self.out_dir / "objB_dpo.jsonl"
        good = [e for e in enriched if getattr(e["review"], "decision", "") == "accept"]
        bad = [e for e in enriched if getattr(e["review"], "decision", "") == "reject"]

        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for g in good:
                for b in bad:
                    if n >= 200:
                        break
                    cand_g = g["candidate"]
                    cand_b = b["candidate"]
                    # Prefer pairing when target_failure_pattern matches (same cluster)
                    if getattr(cand_g, "target_failure_pattern", "") != getattr(cand_b, "target_failure_pattern", ""):
                        if self.cfg.include_rejected_in_dpo is False:
                            continue
                    user = build_skillgen_prompt(
                        failure_pattern=getattr(cand_g, "target_failure_pattern", "")[:1000],
                    )
                    chosen = build_skillgen_target(cand_g.md_spec, cand_g.pf_code)
                    rejected = build_skillgen_target(cand_b.md_spec, cand_b.pf_code)
                    row = {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_EVOLVE},
                            {"role": "user", "content": user},
                        ],
                        "chosen": [{"role": "assistant", "content": chosen}],
                        "rejected": [{"role": "assistant", "content": rejected}],
                        "chosen_r": g["r_skill"],
                        "rejected_r": b["r_skill"],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
                if n >= 200:
                    break
        logger.info("Wrote %d Obj-B DPO pairs → %s", n, path)
        return path

    def _write_prompt_only(self, enriched: List[Dict[str, Any]]) -> Path:
        """Prompts for GRPO / RS skill rollouts."""
        path = self.out_dir / "objB_prompts.jsonl"
        seen_patterns = set()
        with open(path, "w", encoding="utf-8") as f:
            for item in enriched:
                pat = getattr(item["candidate"], "target_failure_pattern", "")
                if pat in seen_patterns:
                    continue
                seen_patterns.add(pat)
                user = build_skillgen_prompt(pat[:1000])
                row = to_chat_prompt_only(SYSTEM_EVOLVE, user)
                row["target_failure_pattern"] = pat[:500]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %d Obj-B prompts → %s", len(seen_patterns), path)
        return path

    def _write_manifest(self, enriched: List[Dict[str, Any]]) -> None:
        manifest = {
            "objective": "B_evolve",
            "n_enriched": len(enriched),
            "q_skill_threshold": self.cfg.q_skill_threshold,
            "lam_val_gain": self.cfg.lam_val_gain,
            "formats": self.formats,
        }
        with open(self.out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
