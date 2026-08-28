"""PF-derived training signals.

Four canonical signals (S1 activation timing, S2 intervention stage,
S3 content quality, S4 benefit), each with sub-signals registered via
`SignalRegistry`. New domains can register additional signals without
touching core training code.
"""

from .registry import SignalRegistry, SignalSpec, SignalOutput
from .s1_activation_timing import compute_s1
from .s2_intervention_stage import compute_s2
from .s3_content_quality import compute_s3
from .s4_benefit import compute_s4
from .aggregator import SignalAggregator

__all__ = [
    "SignalRegistry",
    "SignalSpec",
    "SignalOutput",
    "compute_s1",
    "compute_s2",
    "compute_s3",
    "compute_s4",
    "SignalAggregator",
]
