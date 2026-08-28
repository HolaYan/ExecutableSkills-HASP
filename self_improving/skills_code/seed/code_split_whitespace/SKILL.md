---
skill_id: code_split_whitespace
name: Whitespace-Aware String Split
version: 1
priority: 0.7
error_category: string_split_delimiter
applicable_modes: [all]
applicable_phases: [pre_final]
system_summary: >
  When the prompt asks you to split a string into words / tokens / by
  whitespace, use `s.split()` (no argument), NOT `s.split(' ')`. The latter
  treats every single space literally, leaving empty strings on consecutive
  spaces or tabs; the former collapses all whitespace.
phases:
  pre_final:
    conditions: []
    priority_boost: 0.15
---

# Whitespace-Aware String Split

A surgical PF rewrites `s.split(' ')` → `s.split()` whenever the prompt
mentions "whitespace" / "words" / "spaces" / "split by space".

## When the auto-fix fires
1. The candidate calls `<expr>.split(' ')` (or other single-whitespace literal).
2. The question text contains a whitespace-related hint.

## What gets patched
The single-arg `.split('  ')` form is replaced with the no-arg `.split()`.

## What this protects against
- Tests with multi-whitespace inputs ("hello  world\tfoo").
- HE+ string-tokenizing problems where words are space-separated but the
  test data has irregular spacing.
