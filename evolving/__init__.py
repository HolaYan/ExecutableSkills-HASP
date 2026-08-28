"""evolving — in-training PF evolution.

Every N training steps: pause, evaluate the live weights on a held-out set,
distil PFs from the failures, and grow a run-scoped library. See README.md.
"""
