"""Rejection sampling / on-policy distillation."""

from .rollout import Rollouter
from .train import run_rs_iteration

__all__ = ["Rollouter", "run_rs_iteration"]
