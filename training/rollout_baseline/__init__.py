"""Baseline rollout helpers, used by the bootstrap SFT builders.

`run_vllm_skills` drives a vLLM rollout with a skill catalogue and the teacher
review/finalise turns; `prompt` builds the direct-answer chat prompt;
`data_utils` and `sandbox_eval` are its dataset and execution helpers.

They are kept as their own package rather than at the repository root, where
names this generic would collide.
"""
