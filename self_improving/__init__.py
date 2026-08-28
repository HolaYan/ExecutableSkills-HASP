"""
Self-Improving Skills Agent — PF-aware skill expansion with model-centric pseudo-gradient updates.

This module implements a self-improving loop where:
  1. Seed skills (MD + PF) drive initial ReAct execution
  2. PF-aware trajectories log step-level interventions
  3. Recurring failures are clustered and analyzed
  4. Student proposes new skills (MD + PF code)
  5. PF helper evaluates skill quality (5-dim scoring)
  6. Accepted skills enter the library
  7. Pseudo-gradients update Student/PF helper models via training data
"""
