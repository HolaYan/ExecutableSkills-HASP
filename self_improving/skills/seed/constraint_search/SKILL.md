---
skill_id: constraint_search
name: Constraint-Based Entity Search
version: 1
priority: 0.90
error_category: search_optimization
applicable_modes: [all]
applicable_phases: [think, search]
system_summary: >
  For questions with many constraints describing a single entity, decompose into focused searches. Search the most distinctive constraint first, find candidates, then verify against remaining constraints.
phases:
  post_search:
    conditions: [search_empty, multi_constraint_question]
    priority_boost: 0.4
---

# Constraint-Based Entity Search

When a question describes an entity through multiple independent constraints (date ranges, attributes, relationships), do NOT cram all constraints into one search. Instead, use a systematic narrowing strategy.

## Detection Triggers
- Question contains 4+ independent constraints (date ranges, attributes, etc.)
- Question describes a specific but unnamed entity through its properties
- First search with all constraints returns no useful results
- BrowseComp-style multi-constraint identification questions

## Strategy: Search → Identify → Verify → Narrow

### Step 1: Identify the most searchable constraint
Pick the constraint with the most specific/unique information:
- Named events (FA Cup final, Academy Awards, League Cup)
- Specific places (Wembley, a named university, a specific city)
- Unique attributes (a rare sport, a specific genre)
- Avoid: generic date ranges alone (e.g., "between 1981 and 1984")

### Step 2: Search that constraint
Use a focused 3-8 word query targeting just that one constraint.

### Step 3: READ and extract candidates
READ the most relevant result. Extract candidate entities that match the searched constraint.

### Step 4: Verify candidates against other constraints
For each candidate, search for it combined with the next most constraining attribute. Eliminate candidates that don't match.

### Step 5: Confirm the final answer
Once narrowed to 1-2 candidates, READ their full profile (e.g., Wikipedia page) to verify ALL constraints.

## Examples

### Example 1
**Question:** "The player, born between 1981 and 1984, joined a club formed between 1930 and 1933 that reached Wembley for the first FA Cup final between 1971 and 1974..."
**Wrong:** SEARCH("football club formed between 1930 and 1933 FA Cup final Wembley 1971 1974")
**Correct:**
1. SEARCH("FA Cup final Wembley first time 1973") → identifies Sunderland
2. READ doc → confirms Sunderland's 1973 FA Cup final at Wembley
3. SEARCH("Sunderland player scored two goals cup final") → find candidates
4. READ candidate profiles → verify birth year, career dates, retirement

### Example 2
**Question:** "Which 90s TV series starred an actor born in Tennessee, a Caribbean immigrant actor, and an actor whose father was a law enforcement officer..."
**Wrong:** SEARCH("90s TV series actor Tennessee Caribbean immigrant law enforcement father")
**Correct:**
1. SEARCH("short-lived 90s TV series cast") → browse results
2. SEARCH("90s TV series Caribbean immigrant actor") → most distinctive constraint
3. READ result → find the series name and cast
4. Verify: SEARCH("actor_name Tennessee birthplace") for confirmation
