"""The evaluation protocol, loaded from configs/protocol.yaml.

Values that used to be argparse defaults and literals inside the harnesses live
in one file now, so the protocol a number was produced under is readable in one
place and a sweep can change it without editing source.

Precedence is ``CLI flag > protocol.yaml > the dataclass default here``. The
dataclass defaults are the same values as the shipped yaml, so the module works
with the file absent or truncated — a missing key falls back rather than
raising, which keeps an old checkout runnable against a newer file.

Usage in a harness::

    from hasp_config import protocol
    P = protocol()
    ap.add_argument("--tp", type=int, default=P.serving.tensor_parallel)

Point elsewhere with ``HASP_PROTOCOL=/path/to/other.yaml``. The parsed object
is cached; call ``protocol(reload=True)`` after changing the file in-process.

NOTE on SKILL.md: a skill document's ``phases.per_action.action_params`` block
is documentation. It is not read at runtime and this module does not merge it;
the checker thresholds here are the ones that fire.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

HASP_ROOT = Path(__file__).resolve().parent
DEFAULT_PATH = HASP_ROOT / "configs" / "protocol.yaml"


# ── sections ────────────────────────────────────────────────────────────────

@dataclass
class Models:
    policy_math: str = "Qwen/Qwen3-4B-Instruct-2507"
    policy_code: str = "Qwen/Qwen2.5-7B-Instruct"
    judge: str = "Qwen/Qwen3-8B"
    arbiter: str = "openai/gpt-oss-20b"


@dataclass
class Serving:
    tensor_parallel: int = 1
    gpu_memory_utilization: float = 0.88


@dataclass
class Accuracy:
    n: int = 64
    temperature: float = 0.7
    max_tokens: int = 8192
    max_model_len: int = 16384
    max_samples: int = 100
    datasets: str = "aime24,amc23,olympiadbench"
    thinking: bool = False


@dataclass
class MathHarness:
    max_model_len: int = 20480
    max_tokens: int = 6144
    n: int = 2
    temperature: float = 0.7


@dataclass
class CodeHarness:
    max_model_len: int = 12288
    max_tokens: int = 2048
    n: int = 2
    n_best_of: int = 4
    temperature: float = 0.7
    per_arm_cap: int = 400


@dataclass
class WebHarness:
    max_model_len: int = 20480
    n: int = 2
    cap_per_file: int = 120
    locate_max_tokens: int = 2048
    locate_temperature: float = 0.0
    answer_max_tokens: int = 64
    temperature: float = 0.7


@dataclass
class Harness:
    math: MathHarness = field(default_factory=MathHarness)
    code: CodeHarness = field(default_factory=CodeHarness)
    web: WebHarness = field(default_factory=WebHarness)


@dataclass
class Locator:
    max_model_len: int = 20480
    max_tokens: int = 6144
    votes: int = 1
    locate_temperature: float = 0.0
    vote_temperature: float = 0.6
    arbiter_temperature: float = 0.0
    regen_temperature: float = 0.0
    locate_max_tokens: int = 4096
    arbiter_max_tokens: int = 3072
    control_per_ds: int = 60
    wrong_per_ds: int = -1


@dataclass
class StepGate:
    max_model_len: int = 20480
    max_tokens: int = 6144
    max_hits: int = 12
    max_audits: int = 3
    n_samples: int = 2
    consent_max_tokens: int = 128
    consent_temperature: float = 0.0
    evidence_max_tokens: int = 2048
    evidence_temperature: float = 0.0
    regen_temperature: float = 0.7


@dataclass
class Helper:
    temperature: float = 0.0
    max_tokens: int = 4096
    max_model_len: int = 20480
    thinking: bool = True


@dataclass
class Segmentation:
    base_min_len: int = 40
    base_max_steps: int = 200
    union_min_len: int = 60
    production_min_len: int = 350


@dataclass
class Checkers:
    compute_tolerance_rel: float = 0.01
    compute_tolerance_abs: float = 1e-6


@dataclass
class Screen:
    min_fire_wrong: float = 0.03
    max_fire_correct: float = 0.06
    min_lift: float = 2.0
    max_error_rate: float = 0.02


def _base_pfs() -> List[str]:
    return ["arithmetic_slip", "algebraic_sign_error", "boundary_violation",
            "case_incompleteness", "verification_missing",
            "unsupported_final_answer", "interval_sign_check",
            "unsupported_known_result"]


@dataclass
class Library:
    base_pfs: List[str] = field(default_factory=_base_pfs)


@dataclass
class Protocol:
    models: Models = field(default_factory=Models)
    serving: Serving = field(default_factory=Serving)
    accuracy: Accuracy = field(default_factory=Accuracy)
    harness: Harness = field(default_factory=Harness)
    locator: Locator = field(default_factory=Locator)
    step_gate: StepGate = field(default_factory=StepGate)
    helper: Helper = field(default_factory=Helper)
    segmentation: Segmentation = field(default_factory=Segmentation)
    checkers: Checkers = field(default_factory=Checkers)
    screen: Screen = field(default_factory=Screen)
    library: Library = field(default_factory=Library)
    source: str = "<defaults>"


# ── loading ─────────────────────────────────────────────────────────────────

def _apply(obj: Any, blk: Optional[Dict[str, Any]], path: str) -> None:
    """Overlay a yaml block onto a dataclass, recursing into nested ones.

    Unknown keys are ignored rather than fatal: a config written for a newer
    checkout must not stop an older one from running.
    """
    if not isinstance(blk, dict):
        return
    known = {f.name: f for f in fields(obj)}
    for k, v in blk.items():
        f = known.get(k)
        if f is None:
            continue
        cur = getattr(obj, k)
        if is_dataclass(cur) and not isinstance(cur, type):
            _apply(cur, v, f"{path}.{k}")
        else:
            setattr(obj, k, v)


def load(path: Optional[Path] = None) -> Protocol:
    """Parse a protocol file. Never raises on a missing file or key."""
    p = Path(path or os.environ.get("HASP_PROTOCOL") or DEFAULT_PATH)
    proto = Protocol()
    if not p.exists():
        return proto
    try:
        import yaml
        blk = yaml.safe_load(p.read_text()) or {}
    except Exception as e:                      # pragma: no cover
        print(f"[hasp_config] could not read {p}: {e}; using defaults")
        return proto
    _apply(proto, blk, "protocol")
    proto.source = str(p)
    return proto


_CACHE: Optional[Protocol] = None


def protocol(reload: bool = False) -> Protocol:
    global _CACHE
    if _CACHE is None or reload:
        _CACHE = load()
    return _CACHE


if __name__ == "__main__":       # `python hasp_config.py` prints the live protocol
    import json
    from dataclasses import asdict
    print(json.dumps(asdict(protocol()), indent=2))
