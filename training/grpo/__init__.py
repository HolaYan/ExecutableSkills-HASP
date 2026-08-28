"""GRPO via TRL with PF-based reward combining S1..S4."""

from .reward import build_reward_fn
from .train import main as train_main

__all__ = ["build_reward_fn", "train_main"]
