"""Model-driven PF selection eval.

Difference from pf_agent_eval (which mirrors training):
  pf_agent_eval — feeds ALL 51 PFs to exec_pf; PFs gate themselves.
  pf_select_eval — model sees the PF menu and picks which PFs apply, then
                   only those are sent to exec_pf.

Three-phase batched flow:
  1. Turn 1: vllm.generate(question prompt) → t1 candidate answer
  2. Turn-PFselect: vllm.generate(question + t1 + PF menu + "select") →
     model emits <pf>id</pf> tags
  3. exec_pf with active_skill_ids = parsed selection → A/B/C dispatch
     (Case A: keep t1; Case B: append PF-corrected answer; Case C: feedback
     + Turn-2 rewrite)

The PF menu is a compact list `<id> — <one-line summary>`, ordered by skill_id.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# Make `src.skills_agent.skills...` importable.
# HASP layout: <HASP>/src/skills_agent, <HASP>/skills (the seed PF library).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pf_select.react_prompts import build_react_user_prompt  # noqa: E402
from pf_select import step_dispatch as _STEP  # noqa: E402

_STEP_CONSENT_TOKENS = 160


def _merge_feedback(step_fb: str, final_fb: str) -> str:
    """Step evidence first: it names a specific step, and a reader (or a
    policy) acts on the most located claim it is given."""
    parts = [x for x in (step_fb, final_fb) if x and x.strip()]
    return "\n\n".join(parts)


def _load_pf_system(skill_dir: str):
    """Returns (exec_pf_fn, library_dict[skill_id → Skill])."""
    import importlib.util as _iu
    from src.skills_agent.skills.program_functions import execute_program_functions
    from src.skills_agent.skills.skill import SkillLibrary
    from skills_layout import resolve

    lib_spec = resolve(skill_dir)
    for mod_path in lib_spec.pf_modules:
        mod_name = f"_dynamic_pfs_{hash(str(mod_path)) & 0xFFFFFF:06x}"
        spec = _iu.spec_from_file_location(mod_name, str(mod_path))
        if spec and spec.loader:
            try:
                mod = _iu.module_from_spec(spec)
                spec.loader.exec_module(mod)
            except Exception:
                pass

    library = {}
    # textual first, executable second: where both define a skill_id, the
    # executable document is the one that describes what actually runs.
    for d in lib_spec.skill_dirs:
        try:
            for s in SkillLibrary.load_from_directory(str(d)).get_all():
                library[s.skill_id] = s
        except Exception:
            pass
    return execute_program_functions, library


def _build_pf_menu(library: dict) -> str:
    """Compact one-line-per-PF menu used in the selection prompt."""
    lines = []
    for sid in sorted(library.keys()):
        s = library[sid]
        summary = (s.system_summary or s.description or s.name or "").strip()
        # Collapse to one line and clip to keep prompt short.
        summary = " ".join(summary.split())[:140]
        lines.append(f"- {sid}: {summary}")
    return "\n".join(lines)


_PF_TAG_RE = re.compile(r"<pf>\s*([a-zA-Z0-9_\-]+)\s*</pf>", re.IGNORECASE)


# ── Terminal-answer extractors (same behaviour as pf_select_loop.py) ────
_FIN_ANS_RE = re.compile(r"finish\s*\[\s*(.+?)\s*\]", re.DOTALL | re.IGNORECASE)
_BOX_ANS_RE = re.compile(r"\\boxed\s*\{\s*(.+?)\s*\}", re.DOTALL)
_TXT_ANS_RE = re.compile(
    r"(?:^|\n)\s*(?:Answer|Final answer recorded|Final answer|answer)\s*:\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)


def _extract_answer_from_text(text: str) -> str:
    """Last committed finish[]/\\boxed{}/Answer: value in `text`, or ''."""
    for pat in (_FIN_ANS_RE, _BOX_ANS_RE, _TXT_ANS_RE):
        m = pat.findall(text or "")
        if m:
            return m[-1].strip()
    return ""


def _sanitize_final_arg(arg: str, fallback: str = "") -> str:
    """If `arg` looks like a Thought/reasoning blob, extract the answer
    out of it. Otherwise return as-is. See pf_select_loop.py for the
    matching fix; this eval-time copy keeps the two paths consistent."""
    if not arg or not isinstance(arg, str):
        return fallback
    if len(arg) < 100 and "\n" not in arg.strip():
        return arg.strip()
    extracted = _extract_answer_from_text(arg)
    if extracted:
        return extracted
    return fallback


def _parse_pf_selection(text: str, allowed: set) -> List[str]:
    """Pull <pf>id</pf> tags out of model output; restrict to known PF IDs."""
    out = []
    seen = set()
    for m in _PF_TAG_RE.finditer(text):
        sid = m.group(1).strip()
        if sid in allowed and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _build_step_context(question: str, t1_text: str) -> dict:
    return {
        "question": question,
        "step_count": 0,
        "max_steps": 1,
        "search_count": 0,
        "read_count": 0,
        "has_read": False,
        "all_read_contents": "",
        "last_search_results_text": "",
        "action_history": [],
        "thought": t1_text,
        "empty_results": False,
        "contradictory_sources": False,
    }


def _build_feedback(records, injections, final_action, final_arg):
    parts = []
    for text in injections or []:
        if text and text.strip():
            parts.append(text.strip())
    if final_action != "FINAL":
        parts.append(
            "STOP. Your current answer may be incorrect. "
            "The system detected issues that require you to reconsider. "
            "Please revise your answer addressing the concerns above."
        )
    if not parts:
        activated = [r for r in records if getattr(r, "activated", False)]
        for r in activated[:3]:
            reason = getattr(r, "reason", "")
            sid = getattr(r, "skill_id", "")
            if reason:
                parts.append(f"[{sid}] {reason}")
    return "\n\n".join(parts) if parts else ""


_PF_SELECT_INSTRUCTION_TMPL = (
    "You just produced a candidate answer for the problem above. Before we "
    "finalise, you may invoke any number of program-function (PF) checks on "
    "your answer. Each PF inspects a specific failure mode and either accepts "
    "your answer, rewrites it, or asks you to revise.\n\n"
    "Available PFs (id: summary):\n{menu}\n\n"
    "Select the PFs that are MOST RELEVANT to the problem you just solved. "
    "Output ONLY the selections, one tag per PF, in the form:\n"
    "<pf>skill_id_1</pf>\n<pf>skill_id_2</pf>\n"
    "If no PF is needed, output nothing.\n\n"
    "Your candidate answer for review:\n{candidate}\n\n"
    "Selection:"
)


def _try_load_skills_off_t1(
    skills_off_results_path: Optional[str],
    questions: List[str],
    n_eff: int,
) -> Optional[List[str]]:
    """If skills_off_results_path points at a per-dataset skills_off results file
    (e.g. .../step_N/skills_off/aime24_results.json), try to extract its
    `all_responses` lists and align with `questions`. Returns a flat list of
    length `len(questions) * n_eff`, or None if alignment fails."""
    import json
    if not skills_off_results_path:
        return None
    p = Path(skills_off_results_path)
    if not p.is_file():
        return None
    try:
        data = json.load(open(p))
    except Exception as e:
        print(f"[pf_select_eval] skills_off reuse: failed to read {p}: {e}")
        return None
    results = data.get("results") or []
    if len(results) != len(questions):
        print(f"[pf_select_eval] skills_off reuse: result count {len(results)} != "
              f"questions {len(questions)}; falling back to fresh generation")
        return None
    flat: List[str] = []
    for qi, r in enumerate(results):
        responses = r.get("all_responses")
        if not responses or len(responses) < n_eff:
            print(f"[pf_select_eval] skills_off reuse: q{qi} has "
                  f"{len(responses) if responses else 0} responses < n_eff={n_eff}; "
                  f"falling back to fresh generation")
            return None
        flat.extend(responses[:n_eff])
    print(f"[pf_select_eval] skills_off reuse: loaded {len(flat)} t1 trajectories from {p}")
    return flat


def _chat_tmpl(tokenizer, msgs, enable_thinking: bool) -> str:
    """apply_chat_template with the Qwen3 thinking switch; ignored by other templates."""
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def run_inference_pf_select(
    model_path: str,
    questions: List[str],
    domains: List[str],
    *,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    n: int = 1,
    max_model_len: int = 16384,
    tp: int = 1,
    pf_skill_library_dir: Optional[str] = None,
    select_max_tokens: int = 256,
    skills_off_results_path: Optional[str] = None,
    enable_thinking: bool = False,
    return_t1: bool = False,
    force_skill_ids: Optional[Sequence[str]] = None,
):
    """Model-driven PF selection eval.

    Returns list-of-list-of-str when n>1, list-of-str when n==1.

    If `skills_off_results_path` is given (or auto-detected from the env var
    PF_SELECT_REUSE_SKILLS_OFF=1), try to skip Turn 1 by loading
    `all_responses` from the corresponding skills_off results file. Falls
    back to a fresh vllm Turn-1 generation if the file is missing,
    misaligned, or has too few samples.
    """
    from vllm import LLM, SamplingParams

    skill_dir = pf_skill_library_dir or str(_ROOT / "skills")
    exec_pf, library = _load_pf_system(skill_dir)
    menu_text = _build_pf_menu(library)
    allowed_ids = set(library.keys())
    print(f"[pf_select_eval] loaded {len(library)} PFs, menu length {len(menu_text)} chars")

    n_eff = max(1, n)

    # Try to short-circuit Turn 1 from skills_off results.
    reused_t1 = _try_load_skills_off_t1(skills_off_results_path, questions, n_eff)

    # vLLM is needed for the selection turn (and possibly Turn 1 / Turn 2);
    # we always create it because Case C still goes through llm.generate.
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.92,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    grouping: List[List[int]] = [[] for _ in questions]
    ep_question: List[str] = []
    ep_t1_prompt: List[str] = []
    for qi, q in enumerate(questions):
        msgs = [{"role": "user", "content": build_react_user_prompt(q)}]
        prompt = _chat_tmpl(tokenizer, msgs, enable_thinking)
        for _ in range(n_eff):
            grouping[qi].append(len(ep_question))
            ep_question.append(q)
            ep_t1_prompt.append(prompt)

    # ── Turn 1 (re-use or generate) ───────────────────────────────────────
    if reused_t1 is not None and len(reused_t1) == len(ep_question):
        print(f"[pf_select_eval] skipping Turn 1 (reused {len(reused_t1)} t1 trajectories)")
        t1_texts = reused_t1
    else:
        t1_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=1)
        print(f"[pf_select_eval] turn-1 batch size = {len(ep_t1_prompt)}")
        t1_outputs = llm.generate(ep_t1_prompt, t1_params, use_tqdm=True)
        t1_texts: List[str] = [o.outputs[0].text if o.outputs else "" for o in t1_outputs]

    # ── PF selection ──────────────────────────────────────────────────────
    # `force_skill_ids` names the skills directly and skips the selection turn.
    # Model-driven selection is a black box when what you want to see is what
    # ONE skill does, so the showcase and the per-skill measurements name it.
    if force_skill_ids:
        forced = [s for s in force_skill_ids if s in allowed_ids]
        unknown = [s for s in force_skill_ids if s not in allowed_ids]
        if unknown:
            raise SystemExit(f"unknown skill id(s): {unknown}\n"
                             f"run `python -m skills.show --list` to see them all")
        print(f"[pf_select_eval] forced selection ({len(forced)}): {', '.join(forced)}"
              f" — selection turn skipped")
        selected_ids_per_ep = [list(forced) for _ in ep_question]
    else:
        selected_ids_per_ep = None      # filled by the selection turn below

    if selected_ids_per_ep is None:
        select_prompts: List[str] = []
        for q, t1 in zip(ep_question, t1_texts):
            # Multi-turn chat-template: assistant says t1, user asks for selection.
            msgs = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": t1},
                {"role": "user", "content": _PF_SELECT_INSTRUCTION_TMPL.format(
                    menu=menu_text, candidate=t1[-1200:],   # tail-clip if t1 huge
                )},
            ]
            select_prompts.append(_chat_tmpl(tokenizer, msgs, enable_thinking=False))
        select_params = SamplingParams(
            temperature=temperature, max_tokens=select_max_tokens, n=1,
        )
        print(f"[pf_select_eval] selection-turn batch size = {len(select_prompts)}")
        sel_outputs = llm.generate(select_prompts, select_params, use_tqdm=True)
        selected_ids_per_ep = []
        for o in sel_outputs:
            text = o.outputs[0].text if o.outputs else ""
            selected_ids_per_ep.append(_parse_pf_selection(text, allowed_ids))

    n_with_selection = sum(1 for s in selected_ids_per_ep if s)
    avg_sel = (sum(len(s) for s in selected_ids_per_ep) /
               max(1, len(selected_ids_per_ep)))
    print(f"[pf_select_eval] selection: {n_with_selection}/{len(selected_ids_per_ep)} "
          f"episodes selected ≥1 PF; avg {avg_sel:.2f} per episode")

    # ── step channel (optional, additive) ────────────────────────────────
    # Multi-point dispatch: the anchor and the model must BOTH say a step needs
    # work. Off unless HASP_STEP_DISPATCH is set, and it can only add evidence —
    # a rollout where no step reaches dual consent is byte-identical to today's.
    step_feedback: List[str] = [""] * len(ep_question)
    if _STEP.ENABLED:
        _items = [(q, t1, sel) for q, t1, sel in
                  zip(ep_question, t1_texts, selected_ids_per_ep)]

        def _consent_batch(prompts):
            if not prompts:
                return []
            print(f"[pf_select_eval] step-consent batch size = {len(prompts)}")
            outs = llm.generate(prompts, SamplingParams(
                temperature=0.0, max_tokens=_STEP_CONSENT_TOKENS, n=1), use_tqdm=True)
            return [(o.outputs[0].text if o.outputs else "") for o in outs]

        step_feedback, _traces = _STEP.run_batch(
            _items, _consent_batch, domain=(domains[0] if domains else "math"))
        print("[pf_select_eval] " + _STEP.summarise(_traces))

    # ── PF dispatch (CPU) ────────────────────────────────────────────────
    final_texts: List[str] = [""] * len(ep_question)
    case_c_indices: List[int] = []
    case_c_feedback: List[str] = []
    pf_cases: List[str] = ["A"] * len(ep_question)
    n_no_select = 0
    for i, (q, t1, selected) in enumerate(zip(ep_question, t1_texts, selected_ids_per_ep)):
        if not selected:
            final_texts[i] = t1
            pf_cases[i] = "A_no_select"
            n_no_select += 1
            continue
        # Extract clean answer from t1 before handing to skills — mirrors
        # the training-time fix in pf_select_loop.py (pre-E5 v4).
        clean_t1 = _extract_answer_from_text(t1)
        sc = _build_step_context(q, t1)
        sc["raw_reasoning"] = t1
        sc["candidate_answer"] = clean_t1
        try:
            final_action, final_arg, records, injections = exec_pf(
                active_skill_ids=selected,
                step_context=sc,
                action_type="FINAL",
                arg=clean_t1 or t1,
                reasoning=t1,
                teacher_model=None,
            )
        except Exception:
            final_texts[i] = t1
            pf_cases[i] = "A_exec_error"
            continue

        any_fired = any(getattr(r, "activated", False) for r in (records or []))
        if not any_fired:
            # The step channel can carry a rollout on its own: no FINAL-level
            # skill fired, but some step reached dual consent.
            if step_feedback[i]:
                case_c_indices.append(i)
                case_c_feedback.append(step_feedback[i])
                pf_cases[i] = "C_step_only"
                continue
            final_texts[i] = t1
            pf_cases[i] = "A_no_fire"
            continue
        if final_action == "FINAL" and final_arg and final_arg != t1 and str(final_arg).strip():
            cleaned_final_arg = _sanitize_final_arg(str(final_arg),
                                                    fallback=clean_t1)
            if not cleaned_final_arg or cleaned_final_arg == clean_t1:
                # Override collapsed to same-as-turn1 or unparseable —
                # treat as no-op.
                final_texts[i] = t1
                pf_cases[i] = "A_fallback"
                continue
            final_texts[i] = (
                f"{t1.rstrip()}\n\n[PF Final Answer]\nAnswer: {cleaned_final_arg}"
            )
            pf_cases[i] = "B"
            continue
        feedback_text = _build_feedback(records, injections, final_action, final_arg)
        feedback_text = _merge_feedback(step_feedback[i], feedback_text)
        if not feedback_text:
            final_texts[i] = t1
            pf_cases[i] = "A_fallback"
            continue
        case_c_indices.append(i)
        case_c_feedback.append(feedback_text)
        pf_cases[i] = "C_step" if step_feedback[i] else "C"

    n_a = sum(1 for c in pf_cases if c.startswith("A"))
    n_b = pf_cases.count("B")
    n_c = pf_cases.count("C")
    print(f"[pf_select_eval] dispatch: A={n_a} (no-select={n_no_select}) B={n_b} C={n_c}")

    # ── Turn 2 (Case C) ──────────────────────────────────────────────────
    if case_c_indices:
        t2_prompts = []
        for i, fb in zip(case_c_indices, case_c_feedback):
            feedback_block = (
                f"\n\n[System Feedback]\n{fb}\n\n"
                f"Please provide a revised answer:\n"
            )
            t2_prompts.append(ep_t1_prompt[i] + t1_texts[i] + feedback_block)
        t2_params = SamplingParams(
            temperature=temperature,
            max_tokens=max(256, max_tokens // 2),
            n=1,
        )
        print(f"[pf_select_eval] turn-2 batch size = {len(t2_prompts)}")
        t2_outputs = llm.generate(t2_prompts, t2_params, use_tqdm=True)
        for j, o in enumerate(t2_outputs):
            i = case_c_indices[j]
            t2_text = o.outputs[0].text if o.outputs else ""
            final_texts[i] = (
                f"{t1_texts[i].rstrip()}\n\n[System Feedback]\n{case_c_feedback[j]}\n\n"
                f"Revised answer:\n{t2_text}"
            )

    def _group(tx):
        if n == 1:
            return [tx[grouping[qi][0]] for qi in range(len(questions))]
        return [[tx[i] for i in grouping[qi]] for qi in range(len(questions))]
    if return_t1:
        return _group(final_texts), _group(t1_texts), [[pf_cases[i] for i in grouping[qi]] for qi in range(len(questions))]
    return _group(final_texts)
