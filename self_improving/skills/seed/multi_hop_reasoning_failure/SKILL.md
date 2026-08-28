---
skill_id: multi_hop_reasoning_failure
name: Multi-Hop Reasoning Verification
version: 2
priority: 0.8
error_category: multi_hop_reasoning_failure
applicable_modes: [all]
applicable_phases: [think, read]
system_summary: >
  For multi-step questions, verify each reasoning step has evidence.
phases:
  post_read:
    conditions: [question_is_multi_hop]
    priority_boost: 0.1
    action: verify_reasoning_chain
    action_params: {}
  pre_final:
    conditions: [question_is_multi_hop]
    priority_boost: 0.3
    action: verify_reasoning_chain
    action_params: {}
---

# Multi-Hop Reasoning Verification

Verify that each step in a multi-hop reasoning chain is supported by evidence. Multi-hop failures occur when one intermediate step is wrong, causing the final answer to be incorrect.

## Detection Triggers
- Question requires connecting facts from multiple sources
- Answer depends on a chain of 2+ reasoning steps
- Question uses phrases like "the X of the Y that Z"
- Intermediate entities must be resolved before the final answer

## Avoidance Strategies
- Break the question into individual sub-questions
- Solve each sub-question with explicit evidence before combining
- Verify each intermediate fact against a READ document
- If any step is uncertain, search for additional evidence

## Phase: post_read
Check if this document provides evidence for one step of the reasoning chain. Note which intermediate fact it supports.

## Phase: pre_final
MULTI-HOP CHECK: Break your reasoning into numbered steps. For each step, cite the document that supports it. If any step lacks evidence, search for it.

## Examples
### Example 1
**Scenario:** "Who directed the film starring the actor born in Springfield?"
**Wrong:** Assume the actor without verifying birthplace
**Correct:** Step 1: Find actor born in Springfield. Step 2: Find their film. Step 3: Find director.

### Example 2
**Scenario:** "What is the capital of the country where the Danube originates?"
**Wrong:** Skip verifying where the Danube originates
**Correct:** Step 1: Verify Danube originates in Germany. Step 2: Return Berlin.
