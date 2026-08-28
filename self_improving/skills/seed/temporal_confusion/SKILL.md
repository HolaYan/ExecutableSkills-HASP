---
skill_id: temporal_confusion
name: Temporal Claim Verification
version: 1
priority: 0.7
error_category: temporal_confusion
applicable_modes: [all]
applicable_phases: [think, read]
system_summary: >
  Cross-check years and dates in the answer against source documents to avoid temporal errors.
phases:
  pre_final:
    conditions: [question_has_temporal]
    priority_boost: 0.2
    action: verify_temporal_claims
    action_params: {}
---

# Temporal Claim Verification

Verify that all temporal claims (years, dates, time periods) in the answer are supported by the source documents. Temporal confusion is a common error when questions involve historical events or dated facts.

## Detection Triggers
- Question asks about a specific year, date, or time period
- Answer contains years or dates not present in any read document
- Multiple documents mention different dates for similar events
- Question involves "when" or temporal ordering

## Avoidance Strategies
- Before answering, list all years/dates mentioned in read documents
- Cross-check every year in your answer against the source text
- If documents disagree on dates, note the discrepancy and pick the most authoritative source
- Do not guess or interpolate dates — only use explicitly stated temporal information

## Phase: pre_final
TEMPORAL CHECK: Your answer involves dates or time references.
1. List every year/date you plan to include in your answer.
2. For each, verify it appears in a document you have READ.
3. If a date is not supported, remove it or search for verification.

## Examples
### Example 1
**Scenario:** Question asks when a treaty was signed
**Wrong:** State "signed in 1648" when the document says "signed in 1649"
**Correct:** Carefully copy the exact year from the source: "signed in 1649"

### Example 2
**Scenario:** Question about a person's birth year
**Wrong:** Confuse the birth year with a nearby date in the same paragraph
**Correct:** Identify the specific sentence stating the birth year and extract it precisely
