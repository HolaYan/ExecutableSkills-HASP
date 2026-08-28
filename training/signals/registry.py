"""Signal registry — lets domain plugins register new sub-signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional


@dataclass
class SignalOutput:
    """Output of a single signal evaluation on one step (or one skill)."""
    sub_id: str                # e.g. "s1.tp", "s3.domain.math_verify"
    value: float               # scalar score, typically in [0, 1]
    sample_id: str = ""
    step_index: int = -1       # -1 for skill-level signals
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalSpec:
    """Static description of a signal."""
    sub_id: str
    parent: str                # "S1", "S2", "S3", "S4", "S5+" for domain
    description: str
    default_weight: float = 0.25
    level: str = "step"        # "step" or "skill"


class SignalRegistry:
    """Central registry for signal computation functions.

    A signal function has signature:
        fn(trajectory, step, context) -> Optional[SignalOutput]

    For skill-level signals:
        fn(candidate, review, validation_delta, context) -> Optional[SignalOutput]
    """

    _registry: Dict[str, Callable] = {}
    _specs: Dict[str, SignalSpec] = {}

    @classmethod
    def register(
        cls,
        sub_id: str,
        fn: Callable,
        spec: SignalSpec,
    ) -> None:
        cls._registry[sub_id] = fn
        cls._specs[sub_id] = spec

    @classmethod
    def get(cls, sub_id: str) -> Callable:
        if sub_id not in cls._registry:
            raise KeyError(f"Signal '{sub_id}' not registered. Available: {list(cls._registry)}")
        return cls._registry[sub_id]

    @classmethod
    def get_spec(cls, sub_id: str) -> SignalSpec:
        return cls._specs[sub_id]

    @classmethod
    def list_signals(cls, parent: Optional[str] = None) -> List[str]:
        if parent is None:
            return list(cls._registry)
        return [k for k, s in cls._specs.items() if s.parent == parent]

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
        cls._specs.clear()
