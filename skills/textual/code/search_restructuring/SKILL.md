---
skill_id: search_restructuring
name: Relevant Library / API Hints
version: 1
priority: 0.86
error_category: wrong_library_api
applicable_modes: [all]
applicable_phases: [pre_final]
system_summary: >
  Identify the library / API family the task needs (matplotlib axes,
  sklearn, pandas, numpy, regex, file IO, graph, sort, datetime) and pull
  in the canonical functions / patterns BEFORE writing code — picking the
  wrong primitive (e.g. `plt.scatter()` instead of `plt.gca()`) wrecks the
  return type even when the algorithm is right.
anchor:
  level: final
  trigger: "the spec matches one of the known task-category patterns"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: []
    priority_boost: 0.3
    action: recall_library_apis
    action_params: {}
---

# Relevant Library / API Hints

## Detection Triggers
- matplotlib / scatter / Axes → plot returns
- sklearn / KMeans / RandomForest / fit_predict → ML pipeline
- pandas / groupby / merge / pivot → tabular ops
- numpy / np.array → vector ops
- regex / re / pattern → text matching
- file / open / csv / json → IO
- graph / BFS / DFS / shortest path → graph algorithms
- sort / key function → sorting
- date / datetime / timezone → time handling

## Avoidance Strategies
- **matplotlib:** `plt.scatter(...)` returns `PathCollection`, NOT
  `plt.Axes`. For Axes use `fig, ax = plt.subplots()` and `ax.scatter(...)`,
  or `plt.gca()`. Do NOT call `plt.show()` in a function the tests inspect.
- **sklearn:** set `random_state` so output is deterministic; validate input
  shape before `.fit_predict()`.
- **pandas:** prefer `df.groupby(...).agg(...)` and `df.merge(...)` to
  Python loops; pick `df.loc` (labels) vs `df.iloc` (positions).
- **numpy:** call `.tolist()` if the spec / tests expect Python lists.
- **regex:** raw strings, compile once, anchor with `^`/`$` if needed.
- **graph:** BFS uses `collections.deque`; Dijkstra uses `heapq`.

## Phase: pre_final
Which library / API does this task actually use? Did you pick the version
that returns the type the spec asks for?

## Examples
### Example 1
**Spec:** "Plot K-means clusters and return the Axes."
**Hint:** `plt.scatter(...)` returns `PathCollection`. Use
`fig, ax = plt.subplots(); ax.scatter(...); return ax`.
