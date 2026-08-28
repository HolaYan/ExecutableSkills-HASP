"""One template for every PF skill: Detect, then Repair.

A program skill has exactly two parts, and this module is what makes that
literal rather than a convention. Both take the same three arguments — the
rollout state, and the action the policy is *proposing* to take:

    Detect   `should_activate(ctx, action, arg)` — do this state and this
             proposed action match the failure pattern?
    Repair   `intervene(ctx, action, arg)` — what should happen instead, as a
             typed intervention: redirect the action, inject corrective
             context, or abstain.

The proposed action is a parameter rather than a field of the state because
that is the distinction the contract turns on. A PF does not audit a rollout in
the abstract; it audits *one action about to be taken*, and says what to do
instead. A signature that hides the action inside the state reads as the
former and behaves as the latter.

    @pf_skill("insufficient_exploration", domain="web",
              anchor=Anchor(level="final", evidence="deterministic",
                            trigger="an answer committed with nothing read"),
              summary="This answer is being given without enough evidence.")
    class InsufficientExploration:
        def should_activate(self, ctx, action, arg) -> bool:
            return action == "FINAL" and ctx.read_count == 0

        def intervene(self, ctx, action, arg) -> Intervention:
            return inject(f"[{ctx.skill_id}] This answer is being committed with "
                          f"nothing read. Take a READ action on one of the "
                          f"search results first, then answer from what it says.")

`redirect("READ", url)` would state the same repair as a rewritten action, and
is the sharper form when the skill can name the URL. Most cannot — see
`redirect`, which is deliberately the rarer of the two.

## Why `ctx.read_count` and not `ctx.get("read_count", 0)`

Because `ctx.raed_count` should be an error. Under a dict a typo is a silent
default, and a Detect that silently reads 0 is a skill that silently never
fires — this library has already lost three skills that way, none of which
raised. `Ctx` allows only the keys in `Ctx.FIELDS`; anything else is an
AttributeError, `tests/test_ctx_fields.py` finds it by AST without running
anything, and a Detect that does raise now logs instead of vanishing.

## Why the split is enforced here

The seed library that produced the measured pf_select gains
(`Agentic_RL/rl/skills/seed`, +16 to +46pp pass@1) had it right: a shared
`_MathVerifyPF` whose `should_activate` delegated to a per-family `trigger()`,
and a Repair that injected a family-specific hint. The HASP rewrite collapsed
that into one class whose `should_activate` returned True for **every** FINAL
and did all its work inside `intervene`. Two things broke:

  * the nine per-family Detects disappeared, so nine distinct skills became one
    generic one — which is why the library reads as a single skill; and
  * every selected PF was recorded as *activated* on every rollout, which is
    why `_build_feedback` (it prints `[sid] reason` for activated PFs) could
    speak for a PF that had found nothing. The empty-NOOP-reason rule existed
    to paper over that. With a real Detect, a skill that finds nothing simply
    does not activate, and the workaround stops being load-bearing.

## Detect returns the finding

`detect` may return a bool or a `Finding`. Returning a `Finding` activates the
skill *and* hands Repair what was found, so a checker runs once per step rather
than once in each half. `exec_pf` passes the same context dict to both calls,
so the hand-off cannot go stale across steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

try:
    from skills_agent.skills.program_functions import (
        Intervention, InterventionType, ProgramFunction, register_pf,
    )
except ImportError:  # pragma: no cover — whichever path the runtime resolved
    from src.skills_agent.skills.program_functions import (  # type: ignore
        Intervention, InterventionType, ProgramFunction, register_pf,
    )

EVIDENCE_KINDS = ("deterministic", "executed", "helper", "reminder")
LEVELS = ("step", "final")


# ── the anchor, in one place ─────────────────────────────────────────────

@dataclass(frozen=True)
class Anchor:
    """WHERE a skill attaches and HOW it decides — the normalised form.

    `level`    "step" (one reasoning step, reported as @step k/N) or "final"
               (the committed answer).
    `trigger`  what must be present for Detect to look at all, in prose. It is
               documentation *and* a contract: if a skill's Detect is broader
               than its trigger says, the trigger is wrong.
    `evidence` how the verdict is produced:
                 deterministic — recomputed (sympy, a test point, a range)
                 executed      — run (sandbox, the spec's own examples)
                 helper        — a PF helper model, family-scoped
                 reminder      — no verdict, a family-specific prompt only
    """
    level: str = "final"
    trigger: str = ""
    evidence: str = "deterministic"

    def __post_init__(self):
        if self.level not in LEVELS:
            raise ValueError(f"anchor.level {self.level!r} not in {LEVELS}")
        if self.evidence not in EVIDENCE_KINDS:
            raise ValueError(f"anchor.evidence {self.evidence!r} not in {EVIDENCE_KINDS}")

    def as_dict(self) -> Dict[str, str]:
        return {"level": self.level, "trigger": self.trigger, "evidence": self.evidence}

    def tag(self, step_idx: Optional[int] = None, n_steps: Optional[int] = None) -> str:
        """The location tag that goes on injected text."""
        if self.level == "step" and step_idx is not None and n_steps:
            return f"@step {step_idx + 1}/{n_steps}"
        return "@final" if self.level == "final" else "@step"


# ── what Detect hands to Repair ──────────────────────────────────────────

@dataclass
class Finding:
    """A matched failure pattern. Truthy, so `detect` can return it directly."""
    verdict: str = ""                      # what is wrong, concretely
    fix: Optional[str] = None              # the corrected value, when known
    step_idx: Optional[int] = None
    n_steps: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return True


# ── the state a skill sees ───────────────────────────────────────────────

class Ctx:
    """The rollout state, with a checked field set.

    Wraps the runtime's `step_context` dict. Every name in `FIELDS` is readable
    as an attribute and defaults to the value given here when the runtime did
    not supply it; **every other name raises AttributeError**. That is the whole
    point: under a plain dict, `ctx.get("raed_count", 0)` returns 0 and the
    skill silently never fires, which is how three skills were lost during the
    template migration without a single traceback.

    `raw` is the underlying dict, for the implementations in
    `implementations.py` that take `step_context` directly, and for the handful
    of web skills that keep cross-step state under private keys.
    """

    #: name -> default. Adding a field here is how a skill gains access to a new
    #: piece of rollout state; `tests/test_ctx_fields.py` checks that every
    #: `ctx.<name>` in the library appears in this dict.
    FIELDS: Dict[str, Any] = {
        # the task
        "question": "",
        "domain": "",
        "uid": None,
        # what the model wrote this step
        "raw_reasoning": "",
        "thought": "",
        # web: the ReAct budget and what has been gathered
        "step_count": 0,
        "max_steps": 0,
        "search_count": 0,
        "read_count": 0,
        "has_read": False,
        "all_read_contents": "",
        "last_search_results_text": "",
        "action_history": (),
        # code: the harness around the candidate program
        "entry_point": "",
        "public_test_code": "",
        "enable_edge_probe": False,
    }

    __slots__ = ("raw", "skill_id", "anchor")

    def __init__(self, skill_id: str, raw: Dict[str, Any], anchor: "Anchor"):
        self.raw = raw
        self.skill_id = skill_id
        self.anchor = anchor        # so a Repair body can tag its own evidence

    def __getattr__(self, name: str) -> Any:
        try:
            default = Ctx.FIELDS[name]
        except KeyError:
            raise AttributeError(
                f"ctx has no field {name!r}. Readable fields are "
                f"{', '.join(sorted(Ctx.FIELDS))}; add one to Ctx.FIELDS, or "
                f"use ctx.raw for a key the library does not otherwise read."
            ) from None
        return self.raw.get(name, default)

    @property
    def reasoning(self) -> str:
        """What the model wrote, under whichever key the caller used."""
        return str(self.raw.get("raw_reasoning") or self.raw.get("thought")
                   or self.raw.get("reasoning") or "")

    @property
    def pf_helper(self):
        """The optional model a helper-backed skill may consult. Usually None."""
        return self.raw.get("_pf_helper")

    def get(self, key: str, default: Any = None) -> Any:
        """Escape hatch for a key deliberately outside `FIELDS`."""
        return self.raw.get(key, default)


def ctx_fields_used(source: str) -> set:
    """Every `ctx.<name>` read in a block of source. Used by the field test."""
    import ast
    used = set()
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "ctx"):
            used.add(node.attr)
    return used


# ── typed interventions, as plain constructors ───────────────────────────

def inject(text: str, *, reason: str = "evidence") -> Intervention:
    """Case C — state the finding and let the policy redo the work."""
    return Intervention(type=InterventionType.INJECT_CONTEXT, context_text=text,
                        reason=reason)


def redirect(to: str, arg: str, *, because: str = "rewrite") -> Intervention:
    """Case B — replace the proposed action with a different one.

    Named for what it does to the action, because that is the thing to think
    twice about: the replacement is taken *instead of* what the policy proposed,
    so a wrong redirect has no fallback. `inject` states the same finding and
    leaves the policy holding the pen; prefer it unless the replacement is
    provably right. Most of this library is `inject` for exactly that reason.
    """
    return Intervention(type=InterventionType.MODIFY_ACTION, new_action_type=to,
                        new_action_arg=arg, reason=because)


def abstain(reason: str = "") -> Intervention:
    """Case A — do nothing. The reason is dropped: `_build_feedback` prints the
    reason of any *activated* PF that produced no injection, so a chatty
    abstention becomes junk feedback."""
    return Intervention(type=InterventionType.NOOP, reason="")


# ── the template ─────────────────────────────────────────────────────────

DetectFn = Callable[["Ctx", str, str], Union[bool, Finding, None]]
RepairFn = Callable[["Ctx", str, str], Optional[Intervention]]


def anchored(ctx: "Ctx", anchor: Anchor, finding: Finding, text: str) -> str:
    """`[skill_id @step k/N] text` — the standard tagged form of injected evidence."""
    return f"[{ctx.skill_id} {anchor.tag(finding.step_idx, finding.n_steps)}] {text}"


# ── the verify-style skill, which is most of the library ─────────────────

import os
import re

#: Inject a family reminder when no concrete verdict was produced. This is the
#: seed library's behaviour, and the seed library is what measured +16 to +46pp
#: pass@1 under pf_select — those PFs "rescued by firing, not by what they
#: said", because on a stalled rollout *any* Case-C feedback produces a Turn-2
#: that finishes the answer. The HASP rewrite dropped it and went silent
#: instead, which is the single largest behavioural regression in the library.
#: Off by one env var, so the two can be compared on the same harness.
REMINDERS_ENABLED = os.environ.get("HASP_PF_REMINDERS", "1") not in ("0", "false", "False")

_COMMIT = re.compile(r"finish\s*\[|\\boxed\s*\{|(?:^|\n)\s*(?:Final answer|Answer)\s*:",
                     re.I | re.M)

CONTINUE_TEXT = (
    "[{sid}] The solution above stops before committing a final answer. Continue the "
    "reasoning from where it stops, write your own Observation after each Action, and "
    "end with `Action: finish[<answer>]`."
)


def committed(text: str) -> bool:
    """Did this rollout actually commit an answer?"""
    return bool(_COMMIT.search(text or ""))


def steps(text: str, min_len: int = 350):
    """Production step segmentation — ReAct markers plus paragraph boundaries,
    merged to `min_len`. The base model writes one giant `Thought:`, so markers
    alone give a median of 2 steps per rollout and no usable anchor."""
    from anchor.anchor import Step, segment_steps
    # An Observation is evidence FOR the Action above it; a checker that sees
    # one without the other can neither verify nor refute (this is how
    # compute_observation_verify missed a planted wrong Observation). So a
    # step may start at Thought/Action, never at an Observation.
    _STEP_START = re.compile(r"(?m)^(?=(?:Thought|Action)\s*:)")
    _OBS_LINE = re.compile(r"(?m)^Observation\s*:")
    obs_starts = {m.start() for m in _OBS_LINE.finditer(text)}
    cuts = {0, len(text)}
    for m in _STEP_START.finditer(text):
        cuts.add(m.start())
    for st in segment_steps(text, min_len=60):
        c = st.char_start
        # a cut inside a token splits values in half ("95/256" became a step
        # ending in "95", which a range checker then read as a probability of
        # 95) -- slide any mid-token cut back to the preceding whitespace
        while c > 0 and c < len(text) and not text[c - 1].isspace() \
                and not text[c].isspace():
            c -= 1
        if c not in obs_starts:
            cuts.add(c)
    pts = sorted(cuts)
    merged = []
    for a, b in zip(pts, pts[1:]):
        if b <= a:
            continue
        if merged and (merged[-1][1] - merged[-1][0]) < min_len:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [Step(i, a, b, text[a:b]) for i, (a, b) in enumerate(merged)]


_CHAIN_LOADED: set = set()


def chain_load(path, name: str) -> None:
    """Exec a sibling skill module once per process.

    The library is chain-loaded (`dynamic_program_functions.py` →
    `evidence_pfs.py` → `skills.py`), and a skill module that reaches back for
    a checker defined in an earlier link re-execs that link, which re-enters the
    chain. Without this guard that recursion is infinite, and `_load_pf_system`
    swallows the resulting error — so the skills simply never register and the
    old classes silently keep the ids.
    """
    import importlib.util as _iu
    from pathlib import Path as _P
    if name in _CHAIN_LOADED:
        return
    p = _P(path)
    if not p.exists():
        return
    _CHAIN_LOADED.add(name)
    spec = _iu.spec_from_file_location(name, str(p))
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)


# ── helpers a Repair body calls explicitly ───────────────────────────────
# These are deliberately small and named after what they produce, so an
# `intervene` body reads as the ordered search for the strongest verdict it can
# give: recompute → helper → continuation → reminder → abstain.
#
# Each takes the prefix of `(ctx, action, arg)` it actually uses, in that order,
# so a call site mirrors the method it sits in.

def first_finding(ctx: Ctx, step_checker: Callable) -> Optional[Finding]:
    """Run a step checker over the segmented reasoning; first hit wins."""
    text = ctx.reasoning
    all_steps = steps(text)
    for st in all_steps:
        try:
            r = step_checker(st.text, text, st.char_start)
        except TypeError:
            try:
                r = step_checker(st)
            except Exception:
                r = None
            if r is not None and not isinstance(r, dict):
                r = {"verdict": getattr(r, "verdict", ""), "fix": None}
        except Exception:
            r = None
        if r and r.get("verdict"):
            return Finding(verdict=r["verdict"], fix=r.get("fix"),
                           step_idx=st.idx, n_steps=len(all_steps),
                           data=r if isinstance(r, dict) else {})
    return None


def answer_finding(ctx: Ctx, arg: str, answer_checker: Callable) -> Optional[Finding]:
    """Run a whole-answer checker over the committed answer.

    A checker returns either a verdict string, or a dict `{"verdict": ...,
    "search": ...}` when it can also name the query that would retrieve what it
    found missing. The second form is what lets a Repair rewrite the action
    instead of describing the gap: a skill that knows *which* entity has no
    evidence knows what to search for, and searching is strictly better than
    telling the policy to search.

    The query lands in `Finding.fix`, which is the field for "the corrected
    value, when known" — here the corrected *action argument*.
    """
    try:
        v = answer_checker(ctx.reasoning, arg, ctx.raw)
    except Exception:
        return None
    if isinstance(v, dict):
        return (Finding(verdict=v.get("verdict", ""), fix=v.get("search"), data=v)
                if v.get("verdict") else None)
    return Finding(verdict=v) if v else None


def verdict(ctx: Ctx, f: Finding, *, redo: bool = False) -> Intervention:
    """Inject the finding, tagged with where it attached."""
    msg = anchored(ctx, ctx.anchor, f, f.verdict)
    if redo and f.fix:
        msg += " Use the correct value and redo the work from that step onward."
    return inject(msg, reason="deterministic evidence")


def helper_verdict(ctx: Ctx, scope: str) -> Optional[Intervention]:
    """Ask the PF helper for a family-scoped audit. Silent unless it says ISSUE."""
    h = ctx.pf_helper
    if h is None or not hasattr(h, "locate"):
        return None
    try:
        resp = h.locate(question=ctx.question, reasoning=ctx.reasoning, family_hint=scope,
                        skill_id=ctx.skill_id, uid=ctx.uid)
    except Exception:
        return None
    if resp and resp.upper().startswith("ISSUE"):
        return inject(f"[{ctx.skill_id}] {resp[6:].strip(': ')}", reason="helper evidence")
    return None


def stalled(ctx: Ctx) -> bool:
    """The rollout never committed an answer — it stopped at `Action: …`."""
    return not committed(ctx.reasoning)


def continuation(ctx: Ctx) -> Intervention:
    """Finish a stalled rollout. This channel carries the stall rescues."""
    return inject(CONTINUE_TEXT.format(sid=ctx.skill_id), reason="stalled rollout")


def reminder(ctx: Ctx, text: str) -> Optional[Intervention]:
    """The seed library's family hint, used when no concrete verdict exists.

    Returns None when reminders are switched off, so a Repair body can fall
    through to `abstain()` unchanged.
    """
    if not REMINDERS_ENABLED or not text:
        return None
    return inject(f"[{ctx.skill_id} {ctx.anchor.tag()}] Before finalizing, double-check: {text}",
                  reason="family reminder")


REDO_TEXT = "Apply this correction and give the corrected answer."
_MAX_SHOW = 700


def format_change(original: str, proposed: str) -> str:
    """State a proposed correction concretely, and briefly.

    Short values are quoted in full — that is the whole verdict. Long ones
    (code) are reduced to the lines that differ, because injecting a whole
    rewritten program invites the model to copy it back without reading it, and
    a verdict it does not read is a verdict that does not work.
    """
    o, p = (original or "").strip(), (proposed or "").strip()
    if not p:
        return ""
    if len(o) <= 120 and len(p) <= 120:
        return f"it should be `{p}` rather than `{o}`."
    same = set(o.splitlines())
    changed = [l for l in p.splitlines() if l not in same][:6]
    if changed and len("\n".join(changed)) <= _MAX_SHOW:
        return "the corrected form differs here:\n" + "\n".join("    " + l.strip() for l in changed)
    return f"the corrected form is:\n{p[:_MAX_SHOW]}"


def correction(ctx: Ctx, arg: str, note: str, new_value: str) -> Intervention:
    """A Case-B finding delivered as Case C: say what is wrong and what it
    should be, and let the model redo it. Byte-identical to what
    `inject_adapter` produces, so the two paths stay comparable."""
    change = format_change(arg, new_value)
    if not change and not note:
        return abstain()
    parts = [f"[{ctx.skill_id} {ctx.anchor.tag()}]"]
    if note:
        parts.append(note)
    if change:
        parts.append(change)
    parts.append(REDO_TEXT)
    return inject(" ".join(parts), reason="rewrite stated as evidence")


def as_action(ctx: Ctx, action: str, arg: str, iv, note: str = ""):
    """Let a Case-B intervention execute, instead of restating it as evidence.

    The library this workspace measured rewrote the action directly, and the
    rewrite is the point: replacing `36` with `\\boxed{36}`, prepending a missing
    import, or turning a premature FINAL into a SEARCH leaves the policy no
    room to ignore the repair. Restating those as injected context costs a turn
    and adds a chance of non-compliance, and buys safety only where the
    replacement could be wrong.

    So the choice belongs to the skill, not to this function. A skill routes
    through here when its replacement is computed from the state and it wants
    the repair executed; through `as_evidence` when the replacement is a
    judgement it would rather the policy check.

    NOOP still normalises to a silent abstention: `_build_feedback` prints the
    reason of any activated PF that injected nothing.
    """
    if iv is None or iv.type is InterventionType.NOOP:
        return abstain()
    if iv.type is InterventionType.MODIFY_ACTION and not (iv.new_action_arg or "").strip():
        # A rewrite with nothing to rewrite to would blank the action.
        return abstain()
    if note and iv.type is InterventionType.MODIFY_ACTION and not (iv.reason or "").strip():
        iv.reason = note
    return iv


def as_evidence(ctx: Ctx, action: str, arg: str, iv, note: str = ""):
    """Restate an intervention as evidence instead of executing it.

    For a repair the skill cannot prove right — a model-reformulated query, a
    model-rewritten program. A MODIFY_ACTION replaces the proposal outright, so
    a wrong one has no fallback, while a wrong injection still leaves the policy
    holding the pen. A rewrite becomes a stated correction, a forced action
    becomes a request to take it, and a silent NOOP stays silent with an empty
    reason.

    Use `as_action` when the replacement is computed rather than guessed.
    """
    if iv is None:
        return abstain()
    if iv.type is InterventionType.MODIFY_ACTION:
        new_type = (iv.new_action_type or action or "").upper()
        if new_type != (action or "").upper():
            body = note or (iv.reason or "").replace("_", " ")
            return inject(f"[{ctx.skill_id} {ctx.anchor.tag()}] {body} Do not answer yet — "
                          f"take a {new_type} action first, then answer from what it "
                          f"returns.", reason="rewrite stated as evidence")
        return correction(ctx, arg, note or (iv.reason or "").replace("_", " ") + ".",
                          iv.new_action_arg or "")
    if iv.type is InterventionType.NOOP:
        return abstain()
    return iv


# ── the decorator: a skill is a class with exactly these two methods ─────

_WARNED: set = set()


def _warn_once(skill_id: str, half: str, exc: BaseException) -> None:
    """A raising Detect or Repair declines, but says so.

    It must not take the dispatch down — one broken skill should not cost the
    rollout every other skill's verdict. But swallowing it silently is how this
    library lost three skills that looked registered and did nothing, so the
    first time each one raises, it goes to the log with the exception type.
    """
    import logging
    key = (skill_id, half)
    if key in _WARNED:
        return
    _WARNED.add(key)
    logging.getLogger(__name__).warning(
        "PF %s: %s raised %s: %s — declining, and staying quiet about it from "
        "here on. This skill is not doing what its card says.",
        skill_id, half, type(exc).__name__, exc)


def _require_signature(skill_id: str, name: str, fn) -> None:
    """Both halves must be `(self, ctx, action, arg)`.

    Checked at import so a half left on an older signature fails the module
    load, rather than becoming a per-rollout TypeError that `_warn_once` turns
    into a permanent, quiet abstention.
    """
    import inspect
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):       # pragma: no cover — builtins, C funcs
        return
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return
    names = [p.name for p in params]
    if len(names) != 4:
        raise TypeError(
            f"{skill_id}.{name}{tuple(names)} — a PF skill's {name} takes "
            f"(self, ctx, action, arg): the rollout state, the action the "
            f"policy proposes, and its argument.")


def pf_skill(skill_id: str, *, domain: str, anchor: Anchor, summary: str,
             needs_helper: bool = False, once_per_episode: bool = True,
             max_fires: int = 1):
    """Register one PF skill. The decorated class MUST define both modules:

        should_activate(self, ctx, action, arg) -> bool          # Detect
        intervene(self, ctx, action, arg) -> Intervention        # Repair

    They are written out in every skill rather than passed as callbacks,
    because the two-module contract is the definition of a program skill and a
    reader should see it in the skill, not infer it from a builder's arguments.

    `summary` is the line the policy reads in the PF menu (`- <id>: <summary>`),
    so it must say what this skill checks.
    """
    if not summary.strip():
        raise ValueError(f"{skill_id}: a skill with no summary can never be selected")

    # Bound to a differently-named local because a class body cannot read a
    # closure variable it also assigns: `needs_helper = needs_helper` inside
    # `class _PF` is a NameError, not a rebind.
    _needs_helper = needs_helper

    def deco(cls):
        for name in ("should_activate", "intervene"):
            fn = getattr(cls, name, None)
            if not callable(fn):
                raise TypeError(
                    f"{skill_id}: a PF skill must define {name}(self, ctx, action, arg); "
                    "Detect and Repair are both required")
            _require_signature(skill_id, name, fn)
        impl = cls()

        class _PF(ProgramFunction):
            __doc__ = f"{skill_id} — {summary}"
            needs_helper = _needs_helper
            pf_domain = domain
            pf_anchor = anchor
            pf_summary = summary
            pf_impl = impl
            pf_max_fires = max_fires

            # ── Detect ──
            def should_activate(self, step_context, action_type, arg) -> bool:
                # A skill may be asked at several step boundaries, not just at
                # FINAL. `max_fires` caps how often it can actually intervene in
                # one episode; `once_per_episode` keeps the historical default of
                # one, so single-dispatch behaviour is unchanged.
                cap = self.pf_max_fires if not once_per_episode else max(1, max_fires)
                if step_context.get("_pf_fire_counts", {}).get(skill_id, 0) >= cap:
                    return False
                if anchor.level == "final" and (action_type or "").upper() != "FINAL":
                    return False
                # A skill only speaks for its own domain. Most Detects never
                # check, which was harmless while every repair was injected --
                # a web skill firing on a math answer added a confusing line.
                # Once repairs are executed it is not: a web skill's rewrite
                # replaces a math answer with a search query. The runtime does
                # not always set `domain`, so an unset one is allowed through.
                dom = str(step_context.get("domain") or "").lower()
                if dom and dom.split("_")[0] not in (domain.lower(), ""):
                    return False
                ctx = Ctx(skill_id, step_context, anchor)
                try:
                    return bool(impl.should_activate(ctx, action_type, str(arg or "")))
                except Exception as e:
                    _warn_once(skill_id, "Detect", e)
                    return False          # a broken Detect must not fire

            # ── Repair ──
            def intervene(self, step_context, action_type, arg, helper=None) -> Intervention:
                counts = step_context.setdefault("_pf_fire_counts", {})
                counts[skill_id] = counts.get(skill_id, 0) + 1
                if helper is not None:
                    step_context["_pf_helper"] = helper
                ctx = Ctx(skill_id, step_context, anchor)
                try:
                    iv = impl.intervene(ctx, action_type, str(arg or ""))
                except Exception as e:
                    _warn_once(skill_id, "Repair", e)
                    return abstain()
                if iv is None:
                    return abstain()
                iv.skill_id = skill_id
                if iv.type is InterventionType.NOOP:
                    iv.reason = ""
                else:
                    # Both an injection and an executed rewrite are interventions
                    # this skill is answerable for, so both are recorded.
                    step_context.setdefault("pf_anchors", []).append(
                        dict(pf=skill_id, level=anchor.level,
                             kind=iv.type.value))
                return iv

        _PF.__name__ = "PF_" + skill_id
        _PF.__qualname__ = _PF.__name__
        register_pf(skill_id)(_PF)
        return cls

    return deco


#: The names a wrapped PF may use for its optional helper parameter. `teacher`
#: is the pre-rename spelling, kept so a PF written against the older interface
#: — or one carried in from upstream — still receives the helper instead of
#: silently running its deterministic fallback forever.
HELPER_PARAM_NAMES = ("helper", "teacher")


def call_base_intervene(base, self_, step_context, action_type, arg, helper):
    """Call a wrapped PF's `intervene`, passing `helper` only if it takes one.

    Not every PF declares the parameter — several web skills are
    `intervene(self, ctx, action_type, arg)`. Passing it positionally raises
    TypeError, which an adapter that catches exceptions turns into a silent
    NOOP: the skill looks registered and active and does nothing. Inspect once,
    per wrapped class, and call correctly.
    """
    import inspect
    cache = getattr(base, "_pf_takes_helper", None)
    if cache is None:
        try:
            params = inspect.signature(base.intervene).parameters
            cache = any(n in params for n in HELPER_PARAM_NAMES) or any(
                q.kind is inspect.Parameter.VAR_KEYWORD for q in params.values())
        except (TypeError, ValueError):
            cache = True
        try:
            base._pf_takes_helper = cache
        except Exception:
            pass
    if cache:
        return base.intervene(self_, step_context, action_type, arg, helper)
    return base.intervene(self_, step_context, action_type, arg)


def adapt_skill(skill_id: str, *, domain: str, anchor: Anchor, summary: str = "",
                note: str = ""):
    """Give an already-registered skill the template's anchor and guarantees.

    For skills whose Detect and Repair are already real and already working —
    the web library's 28 prose skills carry multi-condition gates, per-episode
    fire caps and cross-step state (`_candidate_answers`,
    `_retrieval_failure_fires`, `_wrong_entity_fired`). Retyping those into
    declaration blocks would be transcription with the risk on the transcriber's
    side and no behavioural gain, and web has no end-to-end evaluation to catch
    a slip. So the proven bodies are kept and only the wrapper is added:

      * the normalised `Anchor`, so the skill's level / trigger / evidence are
        data rather than prose in a docstring;
      * injected text tagged with where it attached, and an entry in
        `step_context["pf_anchors"]`;
      * a NOOP reason forced empty, since `_build_feedback` prints the reason
        of any activated PF that produced no injection;
      * exception isolation — a raising Detect declines instead of taking the
        dispatch down with it.

    Composes with `inject_adapter`: whatever is registered at call time is what
    gets wrapped, so a Case-B skill already converted to inject stays converted.
    """
    return _adapt(skill_id, domain=domain, anchor=anchor, summary=summary, note=note)


def _adapt(skill_id: str, *, domain: str, anchor: Anchor, summary: str = "", note: str = ""):
    try:
        from skills_agent.skills.program_functions import _PF_REGISTRY
    except ImportError:
        from src.skills_agent.skills.program_functions import _PF_REGISTRY  # type: ignore
    import logging
    inst = _PF_REGISTRY.get(skill_id)
    if inst is None:
        logging.getLogger(__name__).warning(
            "adapt_skill: %s is not registered; nothing to adapt", skill_id)
        return None
    base = type(inst)

    class _Adapted(base):                       # type: ignore[misc, valid-type]
        __doc__ = (summary or base.__doc__ or skill_id)
        pf_domain = domain
        pf_anchor = anchor
        pf_summary = summary
        pf_note = note
        pf_adapted_from = base

        def should_activate(self, step_context, action_type, arg) -> bool:
            try:
                return bool(base.should_activate(self, step_context, action_type, arg))
            except Exception:
                return False

        def intervene(self, step_context, action_type, arg, helper=None) -> Intervention:
            try:
                iv = call_base_intervene(base, self, step_context, action_type, arg, helper)
            except Exception:
                return abstain()
            if iv is None:
                return abstain()
            iv.skill_id = skill_id
            if iv.type is InterventionType.NOOP:
                iv.reason = ""
            elif iv.type is InterventionType.INJECT_CONTEXT:
                txt = (iv.context_text or "").strip()
                if txt and not txt.startswith(f"[{skill_id}"):
                    iv.context_text = f"[{skill_id} {anchor.tag()}] {txt}"
                step_context.setdefault("pf_anchors", []).append(
                    dict(pf=skill_id, level=anchor.level, via="adapt"))
            return iv

    _Adapted.__name__ = "PF_" + skill_id
    _Adapted.__qualname__ = _Adapted.__name__
    register_pf(skill_id)(_Adapted)
    return _Adapted
