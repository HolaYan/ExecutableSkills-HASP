"""
Episode data structure for evaluation.

Unified Episode JSON format:
{
    "question": "...",
    "trace": [
        {"action": {"type": "SEARCH", "query": "..."}, "observation": {...}},
        {"action": {"type": "READ", "doc_id": "..."}, "observation": {...}},
    ],
    "final": {"answer": "..."},
    "evidence": [...],
    "attack_metadata": {...} (optional)
}
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import json


@dataclass
class Action:
    """Agent action."""
    type: str  # "SEARCH", "READ", "FINAL"
    query: Optional[str] = None  # For SEARCH
    doc_id: Optional[str] = None  # For READ

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "query": self.query,
            "doc_id": self.doc_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        return cls(
            type=data.get("type", ""),
            query=data.get("query"),
            doc_id=data.get("doc_id"),
        )

    @classmethod
    def search(cls, query: str) -> "Action":
        return cls(type="SEARCH", query=query)

    @classmethod
    def read(cls, doc_id: str) -> "Action":
        return cls(type="READ", doc_id=doc_id)

    @classmethod
    def final(cls) -> "Action":
        return cls(type="FINAL")


@dataclass
class Observation:
    """Environment observation."""
    results: Optional[List[Dict[str, Any]]] = None  # For SEARCH
    content: Optional[str] = None  # For READ (not saved to reduce file size)
    summary: Optional[str] = None  # For SUMMARY

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.results is not None:
            result["results"] = self.results
        # content is intentionally not saved for READ (too long)
        if self.summary is not None:
            result["summary"] = self.summary
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return cls(
            results=data.get("results"),
            content=data.get("content"),
            summary=data.get("summary"),
        )


@dataclass
class Step:
    """A single step in the episode: action + observation."""
    action: Action
    observation: Observation
    thought: Optional[str] = None  # Chain of thought reasoning for this step
    raw_output: Optional[str] = None  # Model's raw output for this step

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "action": self.action.to_dict(),
            "observation": self.observation.to_dict(),
        }
        if self.thought:
            result["thought"] = self.thought
        if self.raw_output:
            result["raw_output"] = self.raw_output
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step":
        return cls(
            action=Action.from_dict(data.get("action", {})),
            observation=Observation.from_dict(data.get("observation", {})),
            thought=data.get("thought"),
            raw_output=data.get("raw_output"),
        )


@dataclass
class Evidence:
    """Evidence citation."""
    doc_id: str
    quote: str
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "quote": self.quote,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        return cls(
            doc_id=data.get("doc_id", ""),
            quote=data.get("quote", ""),
            url=data.get("url", ""),
        )


@dataclass
class AttackMetadata:
    """Adversarial attack metadata."""
    attack_type: str = ""
    strength: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_type": self.attack_type,
            "strength": self.strength,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackMetadata":
        return cls(
            attack_type=data.get("attack_type", ""),
            strength=data.get("strength", 0.0),
            details=data.get("details", {}),
        )


@dataclass
class Episode:
    """
    Complete episode for evaluation.

    All evaluation modes output this same format.
    """
    question: str
    trace: List[Step] = field(default_factory=list)
    final: Optional[Dict[str, str]] = None  # {"answer": "..."}
    evidence: List[Evidence] = field(default_factory=list)

    # Metadata
    sample_id: Optional[str] = None
    seed: Optional[int] = None
    mode: Optional[str] = None  # "clean", "adv"
    model: Optional[str] = None  # "base", "sft", "rl"
    attack_metadata: Optional[AttackMetadata] = None
    gold_answers: List[str] = field(default_factory=list)

    # Handler records: PF helper activity logs per episode
    handler_records: List[Dict[str, Any]] = field(default_factory=list)
    # Deliberation records: multi-PF helper consensus logs (Plan 3)
    deliberation_records: List[Dict[str, Any]] = field(default_factory=list)

    # Trajectory: full messages + response per step (not serialized in to_dict)
    _trajectory: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Return the collected trajectory steps (messages snapshot + response per step)."""
        return list(self._trajectory)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "question": self.question,
            "trace": [step.to_dict() for step in self.trace],
            "final": self.final,
            "evidence": [e.to_dict() for e in self.evidence],
        }

        if self.sample_id:
            result["sample_id"] = self.sample_id
        if self.seed is not None:
            result["seed"] = self.seed
        if self.mode:
            result["mode"] = self.mode
        if self.model:
            result["model"] = self.model
        if self.attack_metadata:
            result["attack_metadata"] = self.attack_metadata.to_dict()
        if self.gold_answers:
            result["gold_answers"] = self.gold_answers
        if self.handler_records:
            result["handler_records"] = self.handler_records
        if self.deliberation_records:
            result["deliberation_records"] = self.deliberation_records

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        trace = [Step.from_dict(s) for s in data.get("trace", [])]
        evidence = [Evidence.from_dict(e) for e in data.get("evidence", [])]

        attack_metadata = None
        if data.get("attack_metadata"):
            attack_metadata = AttackMetadata.from_dict(data["attack_metadata"])

        return cls(
            question=data.get("question", ""),
            trace=trace,
            final=data.get("final"),
            evidence=evidence,
            sample_id=data.get("sample_id"),
            seed=data.get("seed"),
            mode=data.get("mode"),
            model=data.get("model"),
            attack_metadata=attack_metadata,
            gold_answers=data.get("gold_answers", []),
            handler_records=data.get("handler_records", []),
            deliberation_records=data.get("deliberation_records", []),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "Episode":
        return cls.from_dict(json.loads(json_str))

    # Helper methods
    def add_step(
        self,
        action: Action,
        observation: Observation,
        thought: Optional[str] = None,
        raw_output: Optional[str] = None,
    ) -> None:
        self.trace.append(Step(
            action=action,
            observation=observation,
            thought=thought,
            raw_output=raw_output,
        ))

    def set_final(self, answer: str, reasoning: Optional[str] = None, raw_output: Optional[str] = None) -> None:
        self.final = {"answer": answer}
        if reasoning:
            self.final["reasoning"] = reasoning
        if raw_output:
            self.final["raw_output"] = raw_output

    def get_full_cot(self) -> str:
        """Get the full chain of thought from all steps."""
        cot_parts = []
        for i, step in enumerate(self.trace):
            if step.thought:
                cot_parts.append(f"Step {i+1}: {step.thought}")
        if self.final and self.final.get("reasoning"):
            cot_parts.append(f"Final: {self.final['reasoning']}")
        return "\n\n".join(cot_parts)

    def get_answer(self) -> str:
        if self.final:
            return self.final.get("answer", "")
        return ""

    def get_search_count(self) -> int:
        return sum(1 for s in self.trace if s.action.type == "SEARCH")

    def get_read_count(self) -> int:
        return sum(1 for s in self.trace if s.action.type == "READ")

    def get_step_count(self) -> int:
        return len(self.trace)

    def has_read(self) -> bool:
        return self.get_read_count() > 0

    def has_valid_structure(self) -> bool:
        """Check if episode has valid SEARCH -> READ -> FINAL structure."""
        has_search = self.get_search_count() > 0
        has_read = self.get_read_count() > 0
        has_final = self.final is not None and bool(self.get_answer())
        has_evidence = len(self.evidence) > 0 or has_read
        return has_search and has_read and has_final and has_evidence
