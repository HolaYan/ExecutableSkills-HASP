"""web — the proven PF implementations, moved here from the textual half.

These classes were written before the two-module template and have been in
service since; several carry cross-step state, per-episode fire caps and
multi-condition gates. They are **moved, not retyped** — transcription is where
this refactor already introduced two silent failures (a load cycle swallowed by
`except: pass`, and an `intervene` signature mismatch turned into a NOOP), and
neither had a test that would have caught it.

`skills.py` loads this module and then declares every skill: the ones with a
self-contained Detect/Repair are rewritten there in template form, and these
are given a normalised anchor by `adapt_skill`. Either way registration happens
once, in `skills.py`.

`skills/textual/web/` now holds only SKILL.md cards, which is what "textual"
should mean.
"""
from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional

# Base types — try the no-`src.` path first so re-registration lands in the
# SAME `_PF_REGISTRY` the runtime uses (else the overrides silently no-op).
try:
    from skills_agent.skills.program_functions import (
        Intervention, InterventionType, ProgramFunction, register_pf,
    )
    from skills_agent.skills.quota import note_api_error
except ImportError:  # pragma: no cover
    from src.skills_agent.skills.program_functions import (  # type: ignore
        Intervention, InterventionType, ProgramFunction, register_pf,
    )
    from src.skills_agent.skills.quota import note_api_error  # type: ignore

logger = logging.getLogger(__name__)


# ============================================================================
# Base web PFs (24 classes ported verbatim from program_functions.py)
# `iterative_refinement` is intentionally omitted here — the override below
# replaces it.
# ============================================================================
# ============================================================================
# Implementations (16 skills)
# ============================================================================

@register_pf("insufficient_exploration")
class InsufficientExplorationPF(ProgramFunction):
    """Block premature FINAL when agent hasn't explored enough."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        step = ctx.get("step_count", 0)
        max_steps = ctx.get("max_steps", 10)
        # Don't block near budget exhaustion
        if step >= max_steps - 2:
            return False
        search_count = ctx.get("search_count", 0)
        read_count = ctx.get("read_count", 0)
        # Block if no exploration at all
        if search_count == 0 and read_count == 0:
            return True
        # Block if searched but never read (and not near end)
        if search_count > 0 and read_count == 0 and step < max_steps - 3:
            return True
        return False

    def intervene(self, ctx, action_type, arg):
        search_count = ctx.get("search_count", 0)
        if search_count == 0:
            return Intervention(
                type=InterventionType.MODIFY_ACTION,
                new_action_type="SEARCH",
                new_action_arg=ctx.get("question", arg),
                reason="No search performed yet; forcing search",
            )
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="READ",
            new_action_arg="doc_0",
            reason="Searched but never read; forcing read",
        )


@register_pf("retrieval_failure")
class RetrievalFailurePF(ProgramFunction):
    """Reformulate overly long search queries using PF helper or smart truncation.

    When PF helper is available: asks PF helper to extract a concise search query
    from the model's verbose output. Falls back to smart keyword extraction.
    """
    needs_helper = True

    # Rate limit: max 3 fires per episode to avoid over-intervention
    _MAX_FIRES = 3

    def should_activate(self, ctx, action_type, arg):
        if action_type != "SEARCH":
            return False
        words = arg.split()
        # Only activate for genuinely long queries (>15 words)
        if len(words) > 15:
            fires = ctx.get("_retrieval_failure_fires", 0)
            if fires >= self._MAX_FIRES:
                return False
            return True
        return False

    def intervene(self, ctx, action_type, arg, helper=None):
        words = arg.split()
        question = ctx.get("question", "")

        # Track fires
        ctx["_retrieval_failure_fires"] = ctx.get("_retrieval_failure_fires", 0) + 1

        # Helper-backed: ask PF helper to extract a good search query
        if helper is not None:
            try:
                result = helper.generate(
                    messages=[{"role": "user", "content": (
                        f"A search agent is trying to answer this question:\n"
                        f"Question: {question}\n\n"
                        f"The agent generated this overly long search query:\n"
                        f"\"{arg[:500]}\"\n\n"
                        f"Extract a concise, effective search query (5-12 words) that "
                        f"captures the key search intent. Reply with ONLY the query, "
                        f"no explanation."
                    )}],
                    max_tokens=50,
                    temperature=0.0,
                )
                if result and result.strip():
                    clean = result.strip().strip('"').strip("'").strip()
                    # Accept any reasonable length (1-20 words)
                    if 1 < len(clean.split()) <= 20:
                        logger.info(
                            f"[PF:retrieval_failure] PF helper reformulated: "
                            f"'{arg[:50]}...' → '{clean}'"
                        )
                        return Intervention(
                            type=InterventionType.MODIFY_ACTION,
                            new_action_type="SEARCH",
                            new_action_arg=clean,
                            reason=f"PF helper reformulated query ({len(words)} words → {len(clean.split())})",
                        )
                    else:
                        logger.warning(
                            f"[PF:retrieval_failure] PF helper returned bad length "
                            f"({len(clean.split())} words): '{clean[:80]}'"
                        )
                else:
                    logger.warning(f"[PF:retrieval_failure] PF helper returned empty result")
            except Exception as e:
                if note_api_error(e): raise
                logger.warning(f"[PF:retrieval_failure] PF helper call failed: {e}")

        # Smart fallback: extract key noun phrases rather than blind truncation
        # Try to find the core question keywords
        shortened = self._smart_shorten(arg, question)
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="SEARCH",
            new_action_arg=shortened,
            reason=f"Smart-shortened query ({len(words)} words → {len(shortened.split())})",
        )

    @staticmethod
    def _smart_shorten(query: str, question: str) -> str:
        """Extract the model's actual search intent from its verbose output.

        The Qwen model typically outputs: "actual search terms")\\n\\nreasoning..."
        The real search query is at the very beginning, often in quotes.
        We should extract THAT, not replace with the original question.
        """
        # Strategy 1: Extract quoted string at the start of the query
        # Pattern: "search terms") or "search terms"
        # This is the model's actual intended search query
        quote_match = re.match(r'^["\u201c](.+?)["\u201d]\s*\)?', query)
        if quote_match:
            extracted = quote_match.group(1).strip()
            if 1 < len(extracted.split()) <= 15:
                return extracted

        # Strategy 2: Extract content before first ")
        # Pattern: search terms")\n\nreasoning...
        paren_match = re.match(r'^(.+?)["\u201d]\s*\)', query)
        if paren_match:
            extracted = paren_match.group(1).strip().strip('"').strip()
            if 1 < len(extracted.split()) <= 15:
                return extracted

        # Strategy 3: Take the first line if it's short enough
        first_line = query.split('\n')[0].strip().strip('"').strip("'").rstrip(')')
        if 1 < len(first_line.split()) <= 12:
            return first_line

        # Strategy 4: Extract quoted strings anywhere in the query
        quoted = re.findall(r'"([^"]{3,60})"', query)
        if quoted:
            # Take the first quoted string that looks like a search query
            for q in quoted:
                q = q.strip()
                if 1 < len(q.split()) <= 12:
                    return q

        # Strategy 5: keep what discriminates, not what comes first. The
        # first-10-words fallback dropped every proper noun and number sitting
        # later in the sentence ("...U2's Joshua Tree", "...1993 Nobel"), which
        # is precisely the part a search engine needs. Collect capitalised
        # runs, numbers, and quoted spans from question+query, in order.
        src = f"{question} {query}"
        keep, seen = [], set()
        for m in re.finditer(r'"([^"]{2,60})"'
                             r"|\b((?:[A-Z][\w'\u2019.-]*)(?:\s+[A-Z][\w'\u2019.-]*)*)"
                             r"|\b(\d{2,4})\b", src):
            tok = next(g for g in m.groups() if g)
            tok = re.sub(r"^(?:The|A|An|In|On|At|Which|What|Who|Where|When|How|Did|Was|Is)\b\s*",
                         "", tok).strip()
            if tok and tok.lower() not in seen:
                seen.add(tok.lower()); keep.append(tok)
        cand = " ".join(keep)
        if 1 < len(cand.split()) <= 12:
            return cand
        if question and len(question.split()) <= 12:
            return question
        return " ".join((cand or question or query).split()[:10])


@register_pf("hallucination")
class HallucinationPF(ProgramFunction):
    """Block FINAL if answer makes claims without any read content."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        step = ctx.get("step_count", 0)
        max_steps = ctx.get("max_steps", 10)
        if step >= max_steps - 1:
            return False
        evid = (str(ctx.get("all_read_contents", "")) + " "
                + str(ctx.get("last_search_results_text", ""))).lower()
        if not evid.strip():
            # nothing gathered at all is insufficient_exploration's case,
            # and double-claiming it made this skill a duplicate that missed
            # every real hallucination in the audit
            return False
        # A hallucination is a SPECIFIC claim absent from the evidence:
        # a proper noun or a year in the answer that no gathered text contains.
        toks = re.findall(r"\b[A-Z][\w'\u2019-]+(?:\s+[A-Z][\w'\u2019-]+)*|\b(?:1[5-9]|20)\d{2}\b",
                          str(arg))
        novel = [t for t in toks
                 if t.lower() not in {"the", "a", "an"} and t.lower() not in evid]
        return bool(novel)

    def intervene(self, ctx, action_type, arg):
        search_count = ctx.get("search_count", 0)
        if search_count == 0:
            return Intervention(
                type=InterventionType.MODIFY_ACTION,
                new_action_type="SEARCH",
                new_action_arg=ctx.get("question", arg),
                reason="Answering without any source; forcing search",
            )
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="READ",
            new_action_arg="doc_0",
            reason="Answering without reading any doc; forcing read",
        )


@register_pf("adversarial_distraction")
class AdversarialDistractionPF(ProgramFunction):
    """Inject warning when search results show genuinely conflicting information.

    Only activates for questions with multi-hop complexity (2+ hop signals)
    to avoid injecting unnecessary warnings on simple factoid questions where
    "however"/"actually" are common in natural text without real conflict.
    """

    _CONFLICT_WORDS = {
        "however", "contrary", "incorrect", "not true",
        "actually", "in fact", "disputed", "false", "disagree",
        "contradicts", "conflicting",
    }

    # Rate limit: max 1 fire per episode to avoid warning fatigue
    _MAX_FIRES = 1

    def should_activate(self, ctx, action_type, arg):
        # Only after SEARCH results arrive
        if action_type != "SEARCH":
            return False
        results_text = ctx.get("last_search_results_text", "")
        if not results_text:
            return False

        # Rate limit
        fires = ctx.get("_adversarial_distraction_fires", 0)
        if fires >= self._MAX_FIRES:
            return False

        # Skip for simple questions — conflict words appear naturally in text
        # without indicating real adversarial distraction
        question = ctx.get("question", "")
        q_lower = question.lower()
        hop_signals = (
            question.count("'s")
            + sum(1 for w in ["whose", "which", "that"] if f" {w} " in f" {q_lower} ")
            + q_lower.count(" of the ")
        )
        if hop_signals < 1 and len(question.split()) < 15:
            return False  # Simple question, skip

        text_lower = results_text.lower()
        conflict_count = sum(1 for w in self._CONFLICT_WORDS if w in text_lower)
        # Require higher threshold (3+) to reduce false positives
        return conflict_count >= 3

    def intervene(self, ctx, action_type, arg, helper=None):
        ctx["_adversarial_distraction_fires"] = ctx.get("_adversarial_distraction_fires", 0) + 1
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[Note: Conflicting Sources] Search results contain "
                "contradictory claims. Cross-reference before answering."
            ),
            reason="Detected conflicting information in search results",
        )


@register_pf("temporal_confusion")
class TemporalConfusionPF(ProgramFunction):
    """On FINAL, warn if years in the answer are not found in read docs.

    Only injects a warning — never forces SEARCH, since the model may
    have correctly inferred or computed a year not literally in the text.
    """

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        all_read = ctx.get("all_read_contents", "")
        if not all_read:
            return False
        answer_years = set(re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', arg))
        if not answer_years:
            return False
        doc_years = set(re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', all_read))
        unsupported = answer_years - doc_years
        return len(unsupported) > 0

    def intervene(self, ctx, action_type, arg):
        all_read = ctx.get("all_read_contents", "")
        answer_years = set(re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", arg))
        doc_years = set(re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", all_read))
        unsupported = sorted(answer_years - doc_years)
        alternatives = sorted(doc_years - answer_years)
        # One wrong year against exactly one evidence year: the swap target is
        # determined, not chosen. More than one alternative and the pick would
        # be a guess -- state the mismatch instead.
        if len(unsupported) == 1 and len(alternatives) == 1:
            fixed = arg.replace(unsupported[0], alternatives[0])
            if fixed != arg:
                return Intervention(
                    type=InterventionType.MODIFY_ACTION,
                    new_action_type="FINAL", new_action_arg=fixed,
                    reason=(f"the evidence dates this {alternatives[0]}, not "
                            f"{unsupported[0]}"))
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(f"\n[TEMPORAL CHECK] The year(s) {', '.join(unsupported)} in your answer "
                          f"appear in no retrieved text"
                          + (f"; the evidence mentions {', '.join(alternatives)}"
                             if alternatives else "")
                          + ". Re-check the date against what was actually read."),
            reason="answer year unsupported by evidence")

class NumericalReasoningPF(ProgramFunction):
    """On FINAL, check if numbers in the answer are supported by read docs."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        all_read = ctx.get("all_read_contents", "")
        if not all_read:
            return False
        answer_nums = set(re.findall(r'\b(\d{2,})\b', arg))
        # Exclude years
        year_re = re.compile(r'^(1[0-9]{3}|20[0-9]{2})$')
        answer_nums = {n for n in answer_nums if not year_re.match(n)}
        if not answer_nums:
            return False
        doc_nums = set(re.findall(r'\b(\d{2,})\b', all_read))
        unsupported = answer_nums - doc_nums
        return len(unsupported) > 0

    def intervene(self, ctx, action_type, arg):
        all_read = ctx.get("all_read_contents", "")
        year = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")
        answer_nums = {n for n in re.findall(r"\b(\d{2,})\b", arg) if not year.match(n)}
        doc_nums = {n for n in re.findall(r"\b(\d{2,})\b", all_read) if not year.match(n)}
        unsupported = sorted(answer_nums - doc_nums)
        alternatives = sorted(doc_nums - answer_nums)
        if len(unsupported) == 1 and len(alternatives) == 1:
            fixed = arg.replace(unsupported[0], alternatives[0])
            if fixed != arg:
                return Intervention(
                    type=InterventionType.MODIFY_ACTION,
                    new_action_type="FINAL", new_action_arg=fixed,
                    reason=(f"the evidence gives {alternatives[0]}, not "
                            f"{unsupported[0]}"))
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(f"\n[NUMBER CHECK] The number(s) {', '.join(unsupported)} in your answer "
                          f"appear in no retrieved text. Quote the figure from the "
                          f"document, or re-read before finalizing."),
            reason="answer number unsupported by evidence")

class NegationOversightPF(ProgramFunction):
    """Inject reminder when question has negation but reasoning doesn't address it."""

    _NEGATION_WORDS = {"not", "never", "none", "neither", "except", "without",
                       "other than", "besides", "excluding"}

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        question = ctx.get("question", "")
        q_lower = question.lower()
        if not any(neg in q_lower for neg in self._NEGATION_WORDS):
            return False
        thought = ctx.get("thought", "")
        t_lower = thought.lower()
        return not any(neg in t_lower for neg in self._NEGATION_WORDS)

    def intervene(self, ctx, action_type, arg):
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[NEGATION REMINDER] The question contains a negation "
                "(not/never/except/...) but your reasoning doesn't address it. "
                "Re-read the question carefully."
            ),
            reason="Negation in question not reflected in reasoning",
        )


@register_pf("citation_mismatch")
class CitationMismatchPF(ProgramFunction):
    """On FINAL, check if the CORE entity in the answer exists in read documents.

    Only triggers when the answer contains a proper noun that is clearly the
    main answer AND that name is nowhere in any read document. Avoids false
    positives from peripheral mentions.
    """

    _STARTERS = {"The", "This", "That", "These", "Those", "Some", "Each",
                  "Every", "Many", "Most", "Both", "All", "Any", "In", "On",
                  "At", "By", "For", "To", "It", "He", "She", "We", "They",
                  "A", "An", "As", "So", "Or", "If", "My", "His", "Her"}

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        all_read = ctx.get("all_read_contents", "")
        if not all_read or len(all_read) < 100:
            return False
        # Only check short answers (likely a name/entity answer)
        if len(arg.split()) > 10:
            return False
        # Extract proper nouns from answer
        # Two-word-minimum made single-word inventions ("Moselle") invisible;
        # a single capitalised word mid-answer is still a checkable citation.
        entities = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', arg)
        if not entities:
            return False
        cleaned = []
        for phrase in entities:
            words = phrase.split()
            while words and words[0] in self._STARTERS:
                words = words[1:]
            if len(words) >= 2:
                cleaned.append(" ".join(words))
        if not cleaned:
            return False
        all_read_lower = all_read.lower()
        # Also check in search results text
        search_text = ctx.get("last_search_results_text", "").lower()
        combined = all_read_lower + " " + search_text
        unsupported = [e for e in set(cleaned) if e.lower() not in combined]
        # Only trigger if the MAJORITY of entities are unsupported
        return len(unsupported) > 0 and len(unsupported) >= len(set(cleaned))

    def intervene(self, ctx, action_type, arg):
        all_read = ctx.get("all_read_contents", "")
        ents = re.findall(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", arg)
        doc_ents = set(re.findall(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", all_read))
        for bad in ents:
            if bad.lower() in all_read.lower():
                continue
            bad_toks = set(bad.split())
            # An invented name usually keeps half of a real one ("Ann Anderson"
            # from the page's "Mary Anderson"). If exactly ONE document entity
            # shares a token with it, the swap target is determined; several
            # sharing, or none, and naming one would be a guess.
            near = {d for d in doc_ents
                    if d.lower() != bad.lower() and bad_toks & set(d.split())}
            if len(near) == 1:
                good = next(iter(near))
                fixed = arg.replace(bad, good)
                if fixed != arg:
                    return Intervention(
                        type=InterventionType.MODIFY_ACTION,
                        new_action_type="FINAL", new_action_arg=fixed,
                        reason=(f"{bad!r} appears in no read text; the evidence "
                                f"names {good!r}"))
            return Intervention(
                type=InterventionType.INJECT_CONTEXT,
                context_text=(f"\n[CITATION CHECK] {bad!r} in your answer appears in "
                              f"none of the documents you read. Quote the name from "
                              f"the text, or re-read before finalizing."),
                reason="answer entity unsupported by evidence")
        return _noop(self.skill_id, reason="entities_supported")

class OutdatedInformationPF(ProgramFunction):
    """Warn when all source documents are older than 5 years."""

    _RECENCY_KEYWORDS = {"current", "currently", "now", "today", "latest", "recent",
                         "present", "2023", "2024", "2025", "2026"}

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        # Only trigger if the question asks about current/recent info
        question = ctx.get("question", "").lower()
        if not any(kw in question for kw in self._RECENCY_KEYWORDS):
            return False
        all_read = ctx.get("all_read_contents", "")
        if not all_read:
            return False
        doc_years = [int(y) for y in re.findall(r'\b(19[5-9]\d|20[0-9]\d)\b', all_read)]
        if not doc_years:
            return False
        return max(doc_years) < 2021

    def intervene(self, ctx, action_type, arg):
        all_read = ctx.get("all_read_contents", "")
        doc_years = [int(y) for y in re.findall(r'\b(19[5-9]\d|20[0-9]\d)\b', all_read)]
        max_year = max(doc_years)
        # Only inject a warning — do NOT force SEARCH
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                f"\n[Note] Your newest source is from {max_year}. "
                f"The question asks about current info — consider searching for more recent data."
            ),
            reason=f"All sources older than 2021 (newest: {max_year})",
        )


@register_pf("format_extraction_error")
class FormatExtractionPF(ProgramFunction):
    """Clean up FINAL answer formatting issues.

    When PF helper is available, uses it to extract the precise answer
    from messy output. Falls back to regex-based cleanup otherwise.
    """
    needs_helper = True  # Use PF helper for precise answer extraction

    # Patterns to strip from the answer (prefix patterns)
    _PREFIX_PATTERNS = [
        r'^\**\s*Answers?\s*\**\s*:\s*',
        r'^(?:Final|Short)\s+answer\s*:\s*',
        r'^The answer is:?\s*',
        r'^Based on [\w\s]+,\s*',
        r'^Based on my research,?\s*',
        r'^After reviewing[\w\s]*,\s*',
        r'^From the[\w\s]+,\s*',
        r'^According to [\w\s]+,\s*',
        r'^In summary,?\s*',
    ]
    # Trailing reference/note patterns
    _SUFFIX_PATTERNS = [
        r'\s*\[\d+\]\s*$',
        r'\s*\(note:.*?\)\s*$',
        r'\s*\[note:.*?\]\s*$',
        r'\s*\(Source:.*?\)\s*$',
    ]
    # Markdown formatting patterns
    _MARKDOWN_BOLD_RE = re.compile(r'\*\*(.+?)\*\*')
    _MARKDOWN_HEADER_RE = re.compile(r'^#+\s*')
    # Quoted answer extraction
    _QUOTED_ANSWER_RE = re.compile(r'^["\'](.+?)["\'](.*)$', re.DOTALL)

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        if not arg:
            return False
        for pattern in self._PREFIX_PATTERNS:
            if re.search(pattern, arg, re.IGNORECASE):
                return True
        for pattern in self._SUFFIX_PATTERNS:
            if re.search(pattern, arg, re.IGNORECASE):
                return True
        # Check for markdown formatting artifacts
        if self._MARKDOWN_BOLD_RE.search(arg):
            return True
        if self._MARKDOWN_HEADER_RE.search(arg):
            return True
        # Check for quoted answer with trailing meta-text
        quote_match = self._QUOTED_ANSWER_RE.match(arg)
        if quote_match:
            rest = quote_match.group(2).strip()
            if rest and re.match(r'^[\s,.]*(which|note|based|according|source|however)', rest, re.IGNORECASE):
                return True
        return False

    def _clean_regex(self, arg: str) -> str:
        """Regex-based answer cleanup (code-only fallback)."""
        cleaned = arg
        # markdown wrappers come OFF first, or "**Answer:**" hides the prefix
        cleaned = self._MARKDOWN_BOLD_RE.sub(r'\1', cleaned)
        cleaned = self._MARKDOWN_HEADER_RE.sub('', cleaned).strip()
        for pattern in self._PREFIX_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        for pattern in self._SUFFIX_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        cleaned = self._MARKDOWN_BOLD_RE.sub(r'\1', cleaned)
        cleaned = self._MARKDOWN_HEADER_RE.sub('', cleaned).strip()
        quote_match = self._QUOTED_ANSWER_RE.match(cleaned)
        if quote_match:
            quoted_part = quote_match.group(1).strip()
            rest = quote_match.group(2).strip()
            if rest and re.match(r'^[\s,.]*(which|note|based|according|source|however)', rest, re.IGNORECASE):
                cleaned = quoted_part
        return cleaned

    def intervene(self, ctx, action_type, arg, helper=None):
        # Try PF helper-based extraction first (more accurate for complex cases)
        if helper is not None and len(arg) > 30:
            try:
                question = ctx.get("question", "")
                extracted = helper.generate(
                    messages=[{"role": "user", "content": (
                        f"Extract the precise, concise answer from this raw output. "
                        f"Return ONLY the answer entity/value, nothing else.\n\n"
                        f"Question: {question}\nRaw answer: {arg}"
                    )}],
                    max_tokens=100,
                    temperature=0.0,
                )
                if extracted and extracted.strip() and len(extracted.strip()) < len(arg):
                    cleaned = extracted.strip()
                    # Sanity: PF helper answer should be shorter and non-empty
                    if 0 < len(cleaned) < len(arg) * 0.9:
                        logger.info(f"[PF:format_extraction_error] PF helper extracted: {cleaned[:80]}")
                        return Intervention(
                            type=InterventionType.MODIFY_ACTION,
                            new_action_type="FINAL",
                            new_action_arg=cleaned,
                            reason="Teacher model extracted clean answer",
                        )
            except Exception as e:
                if note_api_error(e): raise
                logger.warning(f"[PF:format_extraction_error] PF helper call failed: {e}")

        # Fallback: regex-based cleanup
        cleaned = self._clean_regex(arg)
        if cleaned and cleaned != arg:
            return Intervention(
                type=InterventionType.MODIFY_ACTION,
                new_action_type="FINAL",
                new_action_arg=cleaned,
                reason="Cleaned formatting artifacts from answer",
            )
        return Intervention(type=InterventionType.NOOP, reason="No cleanup needed")


@register_pf("wrong_entity_confusion")
class WrongEntityConfusionPF(ProgramFunction):
    """After search, inject warning if results contain similarly-named entities."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "SEARCH":
            return False
        results_text = ctx.get("last_search_results_text", "")
        if not results_text:
            return False
        # Rate limit: at most once per episode (use ctx, not class attribute)
        if ctx.get("_wrong_entity_fired", False):
            return False
        # Only activate if the question mentions a named entity (multi-word proper noun)
        question = ctx.get("question", "")
        # middle initials ("Michael I. Jordan") must not break the name
        q_names = re.findall(r'([A-Z][a-z]+(?:\s+(?:[A-Z]\.\s+)?[A-Z][a-z]+)+)', question)
        if not q_names:
            return False
        # Look for names that share a word with the question entity. Two
        # distinct ones suffice: the confusion is a property of the match,
        # not of the page being crowded.
        names = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', results_text)
        if len(set(n.lower() for n in names)) < 2:
            return False
        # Check for similar names (shared word with question entity)
        q_words = set()
        for qn in q_names:
            q_words.update(w.lower() for w in qn.split())
        similar_count = 0
        seen_names = set()
        for name in names:
            name_lower = name.lower()
            if name_lower in seen_names:
                continue
            seen_names.add(name_lower)
            name_words = set(name_lower.split())
            if name_words & q_words and name_words != q_words:
                similar_count += 1
        return similar_count >= 2

    def intervene(self, ctx, action_type, arg):
        ctx["_wrong_entity_fired"] = True
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[ENTITY WARNING] Search results contain similarly-named entities. "
                "READ documents carefully to identify the correct one. "
                "Pay attention to disambiguating details (dates, locations, roles)."
            ),
            reason="Similar entity names detected in search results",
        )


@register_pf("reading_comprehension_error")
class ReadingComprehensionPF(ProgramFunction):
    """After READ, inject a reminder to extract carefully if content is dense."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "READ":
            return False
        all_read = ctx.get("all_read_contents", "")
        if not all_read:
            return False
        # Only fire once (first READ with dense content)
        read_count = ctx.get("read_count", 0)
        if read_count > 1:
            return False
        # Dense content: VERY high entity/number density
        recent = all_read[-3000:]
        nums = len(re.findall(r'\b\d{2,}\b', recent))
        names = len(re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', recent))
        return nums >= 10 or names >= 15

    def intervene(self, ctx, action_type, arg):
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[READ CAREFULLY] This document contains many entities/numbers. "
                "Extract the specific information relevant to the question. "
                "Don't confuse similar entities or misread numbers."
            ),
            reason="Dense document with many entities/numbers",
        )


@register_pf("multi_hop_reasoning_failure")
class MultiHopReasoningPF(ProgramFunction):
    """On FINAL, check if a multi-hop question was answered with single source."""

    _MULTI_HOP_INDICATORS = [
        "who", "what", "where", "when",  # Basic question words
    ]

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        question = ctx.get("question", "")
        # Multi-hop: question has multiple clauses or relative pronouns
        q_lower = question.lower()
        has_relative = any(w in q_lower for w in ["which", "whose", "that", "who"])
        # Check for compound questions
        has_compound = " and " in q_lower or "?" in question[:-1]
        if not (has_relative or has_compound):
            return False
        step = ctx.get("step_count", 0)
        max_steps = ctx.get("max_steps", 10)
        if step >= max_steps - 2:
            return False
        # Nothing read: clearly the later hops were never retrieved. But the
        # audit's misses (web_06, web_12) READ the first hop and answered from
        # it -- so also fire when evidence was gathered yet none of it
        # contains the committed answer, i.e. the answer came from somewhere
        # other than the retrieval chain.
        if ctx.get("read_count", 0) == 0:
            return True
        evid = (str(ctx.get("all_read_contents", "")) + " "
                + str(ctx.get("last_search_results_text", ""))).lower()
        a = str(arg).strip().lower()
        return bool(a) and a not in evid

    def intervene(self, ctx, action_type, arg):
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="SEARCH",
            new_action_arg=ctx.get("question", arg),
            reason="Multi-hop question answered with <=1 doc read; forcing more exploration",
        )


@register_pf("answer_completeness")
class AnswerCompletenessPF(ProgramFunction):
    """On FINAL, check if multi-part questions have all parts addressed."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        question = ctx.get("question", "")
        q_lower = question.lower()
        # Multi-part indicators
        has_and = " and " in q_lower
        has_multi_q = q_lower.count("?") > 1
        has_list = any(w in q_lower for w in ["both", "each", "all of", "list"])
        if not (has_and or has_multi_q or has_list):
            return False
        # Answer is suspiciously short for a multi-part question
        if len(arg.split()) < 3:
            return True
        return False

    def intervene(self, ctx, action_type, arg):
        # Name the parts, don't wave at them: split the question on its
        # conjunctions and say which part the short answer leaves out.
        q = ctx.get("question", "")
        parts = [p.strip(" ?") for p in re.split(
            r"\s+and\s+(?=(?:how|what|when|where|who|which|why|in\s+what)\b)|;\s*",
            q, flags=re.I) if p.strip(" ?")]
        listing = "".join(f"\n  ({i+1}) {p}?" for i, p in enumerate(parts[:3]))
        # The uncovered part is a concrete retrieval target: on the e2e loop
        # naming the parts repaired 0/6 while the missing evidence stayed
        # missing -- so when budget allows, GO GET the part the answer skips
        # (same mechanism query_decomposition proved out).
        step = ctx.get("step_count", 0); max_steps = ctx.get("max_steps", 10)
        answered = str(arg).lower()
        def covered(part):
            pl = part.lower()
            # a bare year answers a when/what-year part even with no word overlap
            if re.search(r"\bwhat year\b|\bwhen\b", pl) and re.search(r"\b(1[89]|20)\d{2}\b", answered):
                return True
            return any(w in answered for w in re.findall(r"[a-z]{4,}", pl)[:3])
        # check EVERY part -- the un-covered one may be the first
        missing = [p2 for p2 in parts if not covered(p2)]
        if missing and step < max_steps - 2:
            return Intervention(
                type=InterventionType.MODIFY_ACTION,
                new_action_type="SEARCH", new_action_arg=missing[0][:80],
                reason=f"multi-part question; part not yet evidenced: {missing[0][:60]!r}")
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(f"\n[COMPLETENESS] This question asks {len(parts)} things:"
                          f"{listing}\nYour answer {str(arg)[:40]!r} covers at most one. "
                          f"Answer every part, separated by ';'."),
            reason="multi-part question, single-part answer",
        )

class ReasoningErrorPF(ProgramFunction):
    """On FINAL, check for logical inconsistencies in reasoning.

    Only activates when the reasoning shows STRONG contradiction signals
    (multiple conflicting claims), not just "but/however" which are common
    in normal reasoning. Uses PF helper to verify before intervening.

    When PF helper available: asks PF helper to judge if there's a real flaw
    and whether the answer is correct. Only intervenes if PF helper finds
    a genuine error.
    Code-only fallback: only fires on very strong contradiction patterns.
    """
    needs_helper = True

    # Rate limit: max 1 fire per episode
    _MAX_FIRES = 1

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        thought = ctx.get("thought", "")
        if not thought or len(thought) < 150:
            return False

        # Rate limit
        if ctx.get("_reasoning_error_fired", False):
            return False

        t_lower = thought.lower()

        # Require STRONG contradiction: "but actually" or "however ... not"
        # Simple "but" or "however" alone is normal reasoning
        strong_patterns = [
            "but actually" in t_lower,
            "however, " in t_lower and " not " in t_lower.split("however")[1][:100] if "however" in t_lower else False,
            "i was wrong" in t_lower,
            "correction:" in t_lower,
            "wait, " in t_lower and (any(w in t_lower.split("wait")[1][:60]
                for w in ("no", "actually", "that")) if "wait" in t_lower else False),
            "contradict" in t_lower,
            bool(re.search(r"\bactually,?\s+(?:it|the|that|no)\b", t_lower)),
            "earlier i said" in t_lower or "i said earlier" in t_lower,
        ]
        if sum(bool(p) for p in strong_patterns) < 1:
            return False

        # A detected strong contradiction IS the signal; the old short-question
        # gate blocked real "wait, that contradicts..." on 7-word questions.
        question = ctx.get("question", "")
        if len(question.split()) < 5:
            return False

        return True

    def intervene(self, ctx, action_type, arg, helper=None):
        ctx["_reasoning_error_fired"] = True
        thought = ctx.get("thought", "")
        question = ctx.get("question", "")

        if helper is not None:
            try:
                all_read = ctx.get("all_read_contents", "")
                evidence = all_read[-1500:] if all_read else "(no documents)"
                result = helper.generate(
                    messages=[{"role": "user", "content": (
                        f"A web search agent is answering a question. Its reasoning "
                        f"may contain contradictions. Check if the proposed answer is "
                        f"actually correct despite the messy reasoning.\n\n"
                        f"Question: {question}\n"
                        f"Agent's reasoning (last part): {thought[-400:]}\n"
                        f"Agent's proposed answer: {arg}\n"
                        f"Evidence from docs: ...{evidence}\n\n"
                        f"Is the answer '{arg}' correct based on the evidence? "
                        f"Reply with:\n"
                        f"CORRECT: if the answer is right (even if reasoning is messy)\n"
                        f"WRONG: <brief explanation of the flaw and what the answer should be, with the correct answer in double quotes>"
                    )}],
                    max_tokens=100,
                    temperature=0.0,
                )
                if result:
                    result_upper = result.strip().upper()
                    if result_upper.startswith("CORRECT"):
                        logger.info(f"[PF:reasoning_error] PF helper confirmed answer is correct")
                        return Intervention(type=InterventionType.NOOP,
                                            reason="Teacher confirmed answer correct despite messy reasoning")
                    elif "WRONG" in result_upper:
                        correction = result.strip()
                        # containment gate: if the helper names a correction
                        # that appears verbatim in the read evidence, commit
                        # it; a value the evidence never states stays advice.
                        evid_l = (all_read or "").lower()
                        for tup in re.findall(
                                r'"([^"]{2,60})"'
                                r"|answer\s+(?:should\s+be|is)[:\s]+([^.;\n\x22]{2,60})",
                                correction):
                            cand = next(g for g in tup if g).strip().strip(".")
                            if (cand and cand.lower() != str(arg).strip().lower()
                                    and cand.lower() in evid_l):
                                return Intervention(
                                    type=InterventionType.MODIFY_ACTION,
                                    new_action_type="FINAL",
                                    new_action_arg=cand,
                                    reason=f"evidence states {cand!r}: {correction[:120]}")
                        logger.info(f"[PF:reasoning_error] PF helper found error: {correction[:100]}")
                        return Intervention(
                            type=InterventionType.INJECT_CONTEXT,
                            context_text=f"\n[REASONING CHECK] {correction}",
                            reason="Teacher identified genuine reasoning error",
                        )
            except Exception as e:
                if note_api_error(e): raise
                logger.warning(f"[PF:reasoning_error] PF helper call failed: {e}")

        # Code-only fallback: only inject a mild warning, never block
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[Note] Double-check that your answer matches the strongest evidence."
            ),
            reason="Contradictory reasoning pattern detected (code-only)",
        )


@register_pf("language_barrier")
class LanguageBarrierPF(ProgramFunction):
    """After READ, inject warning if document contains significant non-ASCII text."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "READ":
            return False
        all_read = ctx.get("all_read_contents", "")
        if not all_read:
            return False
        # Check last chunk only
        recent = all_read[-3000:]
        non_ascii = sum(1 for c in recent if ord(c) > 127)
        return non_ascii > 50

    def intervene(self, ctx, action_type, arg):
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[LANGUAGE NOTE] This document contains non-English text. "
                "Be careful with transliteration and name variants. "
                "The same entity may appear in different scripts."
            ),
            reason="Significant non-ASCII content in document",
        )


# ============================================================================
# NEW SKILLS: Reasoning / Information Distillation (3)
# ============================================================================

@register_pf("decompose_complex_question")
class DecomposeComplexQuestionPF(ProgramFunction):
    """At episode start, detect multi-hop questions and inject decomposition hints."""

    _RELATIVE_PRONOUNS = {"which", "whose", "that", "who", "whom", "where"}

    def should_activate(self, ctx, action_type, arg):
        # Only inject on early SEARCH steps (step 0-1) for multi-hop questions
        if action_type != "SEARCH":
            return False
        step = ctx.get("step_count", 0)
        if step > 1:
            return False
        question = ctx.get("question", "")
        q_lower = question.lower()
        # Detect multi-hop patterns
        # Pattern 1: possessive chain ("X's Y's Z")
        possessive_count = question.count("'s")
        if possessive_count >= 2:
            return True
        # Pattern 2: relative clause chains
        relative_count = sum(1 for w in self._RELATIVE_PRONOUNS if f" {w} " in f" {q_lower} ")
        if relative_count >= 1 and len(question.split()) > 10:
            return True
        # Pattern 3: "of the" chains indicating nested references
        of_the_count = q_lower.count(" of the ")
        if of_the_count >= 2:
            return True
        return False

    def intervene(self, ctx, action_type, arg):
        # Always inject guidance hint — do NOT modify the search query here
        # (retrieval_failure PF handles query shortening if needed)
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[DECOMPOSITION HINT] This is a multi-hop question. "
                "Search for each piece of information separately. "
                "Find intermediate entities first, then search for the final answer."
            ),
            reason="Multi-hop question detected; injecting decomposition guidance",
        )


@register_pf("evidence_synthesis")
class EvidenceSynthesisPF(ProgramFunction):
    """On FINAL, verify that key entities from question are found in read docs."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        question = ctx.get("question", "")
        all_read = ctx.get("all_read_contents", "")
        if not all_read or len(all_read) < 100:
            return False
        # Extract key entities from the question (multi-word proper nouns only)
        q_entities = [e for e in re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', question)
                      if len(e) > 3]
        if not q_entities:
            return False
        # Check if key question entities appear in read docs
        all_read_lower = all_read.lower()
        missing = [e for e in q_entities if e.lower() not in all_read_lower]
        # The trigger is "A question entity that never appears" -- singular.
        # Requiring ALL of >=2 to be missing made the skill structurally
        # silent on every one-entity question.
        if missing:
            step = ctx.get("step_count", 0)
            max_steps = ctx.get("max_steps", 10)
            return step < max_steps // 2
        return False

    def intervene(self, ctx, action_type, arg):
        question = ctx.get("question", "")
        all_read = ctx.get("all_read_contents", "")
        q_entities = [e for e in re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', question)
                      if len(e) > 3]
        all_read_lower = all_read.lower()
        missing = [e for e in q_entities if e.lower() not in all_read_lower]
        search_target = missing[0] if missing else ctx.get("question", arg)
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="SEARCH",
            new_action_arg=search_target,
            reason=f"Key entity '{search_target}' from question not found in read docs",
        )


@register_pf("comparison_analyzer")
class ComparisonAnalyzerPF(ProgramFunction):
    """For comparison questions, ensure both entities are researched before FINAL."""

    _COMPARISON_WORDS = {
        "first", "earlier", "later", "before", "after", "older", "younger",
        "more", "less", "bigger", "smaller", "longer", "shorter",
        "which", "compare", "difference", "vs", "versus",
    }

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        question = ctx.get("question", "")
        q_lower = question.lower()
        # Check if this is a comparison question
        is_comparison = any(w in q_lower for w in self._COMPARISON_WORDS)
        if not is_comparison:
            return False
        # Check if question mentions 2+ named entities
        q_entities = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', question)
        unique_entities = set(e.lower() for e in q_entities)
        if len(unique_entities) < 2:
            return False
        # Check if both entities appear in read content
        all_read = ctx.get("all_read_contents", "")
        if not all_read:
            return True  # No docs read, definitely incomplete
        all_read_lower = all_read.lower()
        entities_found = sum(1 for e in unique_entities if e in all_read_lower)
        # Block if not all entities are found in read docs (only in first half)
        step = ctx.get("step_count", 0)
        max_steps = ctx.get("max_steps", 10)
        return entities_found < len(unique_entities) and step < max_steps // 2

    def intervene(self, ctx, action_type, arg):
        question = ctx.get("question", "")
        all_read = ctx.get("all_read_contents", "")
        q_entities = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', question)
        unique_entities = list(set(e for e in q_entities))
        all_read_lower = (all_read or "").lower()
        missing = [e for e in unique_entities if e.lower() not in all_read_lower]
        search_target = missing[0] if missing else unique_entities[0]
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="SEARCH",
            new_action_arg=search_target,
            reason=f"Comparison question: entity '{search_target}' not yet researched",
        )


# ============================================================================
# NEW SKILLS: Search Optimization (3)
# ============================================================================

@register_pf("query_decomposition")
class QueryDecompositionPF(ProgramFunction):
    """Break overly complex search queries into focused sub-queries.

    NOTE: Defers to retrieval_failure PF for long queries (>15 words).
    This PF only handles the case of multiple question words in short queries.
    """

    def should_activate(self, ctx, action_type, arg):
        if action_type != "SEARCH":
            return False
        words = arg.split()
        # Do NOT activate for long queries — retrieval_failure handles those
        if len(words) > 15:
            return False
        # Only activate: multiple question words in 8-15 word queries
        q_words = {"who", "what", "where", "when", "which", "how"}
        q_count = sum(1 for w in words if w.lower() in q_words)
        return q_count >= 2 and len(words) >= 8

    def intervene(self, ctx, action_type, arg):
        # Decompose means DECOMPOSE: re-searching the whole compound question
        # is the failure this skill exists to fix. Take the first
        # sub-question; the second hop follows once its evidence is in.
        src = (arg or ctx.get("question", "")).strip().rstrip("?")
        first = re.split(r"\s+and\s+(?:how|what|when|where|who|why|which)\b|\?\s+",
                         src, 1)[0].strip()
        if len(first.split()) < 2:
            first = " ".join(src.split()[:8])
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="SEARCH",
            new_action_arg=first,
            reason="compound query; searching the first sub-question first",
        )


@register_pf("search_depth_controller")
class SearchDepthControllerPF(ProgramFunction):
    """Enforce minimum search/read depth based on question complexity."""

    _COMPLEX_INDICATORS = {"whose", "which", "that", "'s", "of the", "born", "died",
                           "director", "author", "spouse", "father", "mother"}

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        step = ctx.get("step_count", 0)
        max_steps = ctx.get("max_steps", 10)
        if step >= max_steps - 1:
            return False
        question = ctx.get("question", "")
        q_lower = question.lower()
        # Determine minimum depth
        complexity = sum(1 for ind in self._COMPLEX_INDICATORS if ind in q_lower)
        search_count = ctx.get("search_count", 0)
        read_count = ctx.get("read_count", 0)
        if complexity >= 3:
            # Complex: need at least 2 searches + 1 read
            return search_count < 2 or read_count < 1
        elif complexity >= 1:
            # Medium: need at least 1 search + 1 read
            return search_count < 1 or read_count < 1
        return False

    def intervene(self, ctx, action_type, arg):
        read_count = ctx.get("read_count", 0)
        search_count = ctx.get("search_count", 0)
        if search_count < 1:
            return Intervention(
                type=InterventionType.MODIFY_ACTION,
                new_action_type="SEARCH",
                new_action_arg=ctx.get("question", arg),
                reason="Complex question needs more searches before answering",
            )
        if read_count < 1:
            return Intervention(
                type=InterventionType.MODIFY_ACTION,
                new_action_type="READ",
                new_action_arg="doc_0",
                reason="Complex question needs at least 1 document read",
            )
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="SEARCH",
            new_action_arg=ctx.get("question", arg),
            reason="Complex question needs more exploration",
        )


# ============================================================================
# NEW SKILLS: Adversarial Defense (2)
# ============================================================================

@register_pf("claim_triangulation")
class ClaimTriangulationPF(ProgramFunction):
    """On FINAL, prefer multi-source confirmation; warn if single-source answer."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        read_count = ctx.get("read_count", 0)
        step = ctx.get("step_count", 0)
        max_steps = ctx.get("max_steps", 10)
        # Only trigger when we've read exactly 1 doc and have budget
        if read_count != 1 or step >= max_steps - 2:
            return False
        # Only for questions that might have adversarial content
        # or when the answer contains specific facts (names, numbers, dates)
        has_specific_facts = bool(
            re.findall(r'\b\d{2,}\b', arg) or
            re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', arg)
        )
        return has_specific_facts

    def intervene(self, ctx, action_type, arg):
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[VERIFICATION HINT] Your answer is based on a single source. "
                "Consider reading one more document to cross-verify the key facts "
                "before providing your final answer."
            ),
            reason="Single-source answer with specific facts; suggest cross-verification",
        )


@register_pf("misinformation_detector")
class MisinformationDetectorPF(ProgramFunction):
    """After READ, detect if newly read doc contradicts previously read docs."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "READ":
            return False
        read_count = ctx.get("read_count", 0)
        if read_count < 2:
            return False  # Need 2+ docs to compare
        all_read = ctx.get("all_read_contents", "")
        if not all_read:
            return False
        # Check for contradiction indicators in the combined read content
        text_lower = all_read.lower()
        # Look for conflicting years/numbers for the same entity
        years = re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', all_read)
        # two conflicting years across two read documents is already the
        # trigger ("two or more documents that disagree")
        if len(set(years)) >= 2:
            # Many different years — potential for confusion
            return True
        # Look for explicit contradiction language
        contradiction_phrases = [
            "however", "in contrast", "on the other hand",
            "contradicts", "incorrect", "actually",
        ]
        contra_count = sum(1 for p in contradiction_phrases if p in text_lower)
        return contra_count >= 2

    def intervene(self, ctx, action_type, arg):
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[CROSS-CHECK WARNING] Your read documents may contain conflicting "
                "information. Before answering, carefully compare the key facts across "
                "sources. Prefer information that is consistent across multiple documents."
            ),
            reason="Potential contradictions detected across read documents",
        )


@register_pf("constraint_search")
class ConstraintSearchPF(ProgramFunction):
    """Decompose multi-constraint questions (BrowseComp-style) into focused
    constraint-based search strategies.

    BrowseComp questions describe an entity through 5-10 independent constraints
    (date ranges, attributes, relationships). Standard search fails because
    cramming all constraints into one query returns nothing useful.

    Strategy:
    - Step 0-1: Detect multi-constraint question, extract the most searchable
      constraint, and modify the SEARCH query to focus on it.
    - Later steps: Inject context reminding agent to use "search → get candidates
      → verify constraints" strategy.
    - Never modify FINAL — only guide search behavior.
    """

    # Patterns that indicate constraint-heavy questions
    _DATE_RANGE_RE = re.compile(
        r'between\s+\d{4}\s+and\s+\d{4}|'
        r'\d{4}\s*[-–]\s*\d{4}|'
        r'\d{4}\s+(?:and|to)\s+\d{4}\s+inclusive',
        re.IGNORECASE,
    )
    # Constraint signal phrases
    _CONSTRAINT_PHRASES = [
        "born between", "born in", "formed between", "founded between",
        "published between", "released between", "played between",
        "joined", "scored", "attended", "located in", "based in",
        "starred", "directed by", "written by", "created by",
        "more than", "less than", "fewer than", "at least",
        "who was", "who is", "whose", "that was", "that is",
        "an actor", "a player", "a person", "a particular",
    ]

    def _count_constraints(self, question: str) -> int:
        """Estimate the number of independent constraints in a question."""
        q_lower = question.lower()
        count = 0
        # Date ranges
        count += len(self._DATE_RANGE_RE.findall(question))
        # Constraint phrases
        count += sum(1 for p in self._CONSTRAINT_PHRASES if p in q_lower)
        # Commas separating constraint clauses
        comma_count = question.count(",")
        if comma_count >= 3:
            count += comma_count - 2
        # Sentence count (each sentence often = one constraint)
        sentences = [s.strip() for s in re.split(r'[.!?]', question) if s.strip()]
        if len(sentences) >= 3:
            count += len(sentences) - 2
        return count

    # Words to strip from beginning of extracted queries
    _STRIP_PREFIXES = {"the", "a", "an", "of", "in", "for", "by", "on", "at",
                       "its", "their", "his", "her", "this", "that"}

    def _extract_searchable_constraint(self, question: str) -> Optional[str]:
        """Pick the most distinctive/searchable constraint from the question.

        Prefers constraints with named entities (places, events, titles) over
        generic date-range constraints. Returns a focused 4-8 word query.
        """
        sentences = [s.strip() for s in re.split(r'[.!?]', question) if len(s.strip()) > 10]

        best_sentence = None
        best_score = -1

        # Entity-type keywords that make a sentence highly searchable
        _ENTITY_KW = {"cup", "championship", "award", "academy", "series",
                      "university", "school", "college", "film", "movie", "manga",
                      "team", "league", "competition", "wembley", "olympics",
                      "nobel", "grammy", "oscar", "emmy", "pulitzer",
                      "institute", "museum", "cathedral", "festival"}

        for sent in sentences:
            score = 0
            s_lower = sent.lower()
            words = sent.split()

            # Proper nouns (capitalized words, skip sentence-initial)
            proper_nouns = [w.strip(",.;:()") for w in words[1:]
                          if w[0].isupper() and len(w) > 2]
            score += len(proper_nouns) * 3

            # Entity-type keyword boost
            for kw in _ENTITY_KW:
                if kw in s_lower:
                    score += 3

            # Penalize generic constraint-only sentences
            date_ranges = len(self._DATE_RANGE_RE.findall(sent))
            if date_ranges > 0 and len(proper_nouns) == 0:
                score -= 3

            # Penalize very short sentences (less informative)
            if len(words) < 5:
                score -= 1

            if score > best_score:
                best_score = score
                best_sentence = sent

        if not best_sentence or best_score <= 0:
            return None

        # Extract a focused 4-8 word query from the best sentence
        words = best_sentence.split()

        # Find key content words (proper nouns + entity keywords)
        key_indices = []
        for i, w in enumerate(words):
            w_clean = w.strip(",.;:()'\"")
            if not w_clean:
                continue
            if w_clean[0].isupper() and len(w_clean) > 2 and i > 0:
                key_indices.append(i)
            elif w_clean.lower() in _ENTITY_KW:
                key_indices.append(i)

        if key_indices:
            # Build query around key words with +-1 context
            start = max(0, min(key_indices) - 1)
            end = min(len(words), max(key_indices) + 2)
            query_words = words[start:end]
        else:
            # Fallback: first 6 content words
            query_words = words[:6]

        # Clean: strip leading articles/prepositions, limit to 8 words
        while query_words and query_words[0].lower().strip(",.") in self._STRIP_PREFIXES:
            query_words = query_words[1:]

        query = " ".join(query_words[:8]).strip(",.;: ")
        return query if len(query) > 5 else None

    def should_activate(self, ctx, action_type, arg):
        # Only on SEARCH actions
        if action_type != "SEARCH":
            return False

        question = ctx.get("question", "")
        constraint_count = self._count_constraints(question)

        # Need at least 4 constraints to be a BrowseComp-style question
        if constraint_count < 4:
            return False

        step = ctx.get("step_count", 0)

        # Step 0-1: Modify search query to focus on best constraint
        if step <= 1:
            # Check if query is too long (cramming all constraints)
            if len(arg.split()) > 8:
                return True

        # Steps 2-5: Inject strategy guidance if agent seems stuck
        if 2 <= step <= 5:
            action_history = ctx.get("action_history", [])
            search_count = sum(1 for a in action_history if a.get("action_type") == "SEARCH")
            read_count = ctx.get("read_count", 0)
            # Agent has searched multiple times but hasn't read — still fishing
            if search_count >= 2 and read_count == 0:
                return True

        return False

    def intervene(self, ctx, action_type, arg):
        step = ctx.get("step_count", 0)
        question = ctx.get("question", "")

        if step <= 1:
            # Modify first search to focus on best constraint
            focused_query = self._extract_searchable_constraint(question)
            if focused_query:
                return Intervention(
                    type=InterventionType.MODIFY_ACTION,
                    new_action_type="SEARCH",
                    new_action_arg=focused_query,
                    reason=f"Multi-constraint question ({self._count_constraints(question)} constraints); "
                           f"focusing search on most distinctive constraint",
                )

        # Inject strategy guidance
        constraint_count = self._count_constraints(question)
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                f"\n[CONSTRAINT SEARCH STRATEGY] This question has ~{constraint_count} constraints. "
                "Do NOT try to search for all constraints at once. Instead:\n"
                "1. Search for the most unique/specific constraint (a named event, place, or title)\n"
                "2. READ a result to find candidate entities\n"
                "3. Verify candidates against the other constraints\n"
                "4. Narrow down until only one entity matches ALL constraints"
            ),
            reason=f"Multi-constraint question; guiding search strategy (step {step})",
        )


# ============================================================================
# Answer Confidence Guard — prevent over-refinement regressions
# ============================================================================

@register_pf("answer_confidence_guard")
class AnswerConfidenceGuardPF(ProgramFunction):
    """Track candidate answers across steps and warn against over-refinement.

    Prevents the common regression pattern where the agent correctly identifies
    an answer early on, but then second-guesses itself after additional (often
    irrelevant) search/read steps, changing to a worse answer.

    When PF helper available: asks PF helper to judge which answer is better
    supported by the evidence. May revert to the original answer.
    """
    needs_helper = True

    # Patterns that indicate the model has settled on a candidate answer
    _ANSWER_PATTERNS = [
        re.compile(r'the answer (?:is|should be|would be)\s+["\']?(.+?)["\']?\s*[.,;]', re.IGNORECASE),
        re.compile(r'["\']?(.+?)["\']?\s+is the answer', re.IGNORECASE),
        re.compile(r'so the answer should be\s+["\']?(.+?)["\']?\s*[.,;]', re.IGNORECASE),
        re.compile(r'therefore,?\s+["\']?(.+?)["\']?\s*[.,;$]', re.IGNORECASE),
        re.compile(r'I (?:can|will) conclude (?:that |the answer is )\s*["\']?(.+?)["\']?\s*[.,;$]', re.IGNORECASE),
    ]

    def _extract_candidate(self, thought: str) -> Optional[str]:
        """Try to extract a candidate answer from reasoning text."""
        for pattern in self._ANSWER_PATTERNS:
            m = pattern.search(thought)
            if m:
                candidate = m.group(1).strip()
                # Filter out very long or very short extractions
                if 1 < len(candidate) < 100:
                    return candidate
        return None

    def should_activate(self, ctx, action_type, arg):
        thought = ctx.get("thought", "")
        candidates = ctx.get("_candidate_answers", [])

        if action_type == "FINAL":
            # Activate if there's a previous candidate and the current answer differs
            if candidates and arg:
                last_candidate = candidates[-1]
                old_answer = last_candidate.get("answer", "")
                # Only warn if answers are meaningfully different
                if old_answer and old_answer.lower().strip() != arg.lower().strip():
                    # Check if the old answer was from a step with read evidence
                    if last_candidate.get("has_read", False):
                        return True
            return False

        # On non-FINAL actions: check if thought contains a confident answer statement
        if thought and action_type in ("SEARCH", "READ", "SUMMARY"):
            candidate = self._extract_candidate(thought)
            if candidate:
                # Store the candidate (side-effect in should_activate, but
                # this runs deterministically and is the simplest approach)
                has_read = ctx.get("has_read", False)
                step = ctx.get("step_count", 0)
                if "_candidate_answers" not in ctx:
                    ctx["_candidate_answers"] = []
                ctx["_candidate_answers"].append({
                    "answer": candidate,
                    "step": step,
                    "has_read": has_read,
                })
                return False  # Don't intervene on non-FINAL, just track

        return False

    def intervene(self, ctx, action_type, arg, helper=None):
        candidates = ctx.get("_candidate_answers", [])
        if not candidates:
            return Intervention(type=InterventionType.NOOP, reason="No prior candidates")

        last_candidate = candidates[-1]
        old_answer = last_candidate.get("answer", "")
        question = ctx.get("question", "")

        # Helper-backed: ask which answer is better supported
        if helper is not None:
            try:
                all_read = ctx.get("all_read_contents", "")
                # Truncate evidence to fit in context
                evidence_snippet = all_read[-2000:] if all_read else "(no documents read)"
                result = helper.generate(
                    messages=[{"role": "user", "content": (
                        f"A search agent is answering a question. It first concluded "
                        f"'{old_answer}' after reading sources, but now wants to change "
                        f"to '{arg}'. Which answer is better supported by the evidence?\n\n"
                        f"Question: {question}\n"
                        f"Evidence (recent): ...{evidence_snippet}\n\n"
                        f"Reply with ONLY one of:\n"
                        f"KEEP_OLD: {old_answer}\n"
                        f"USE_NEW: {arg}\n"
                    )}],
                    max_tokens=50,
                    temperature=0.0,
                )
                if result and "KEEP_OLD" in result.upper():
                    logger.info(
                        f"[PF:answer_confidence_guard] PF helper chose old answer: "
                        f"'{old_answer}' over '{arg}'"
                    )
                    return Intervention(
                        type=InterventionType.MODIFY_ACTION,
                        new_action_type="FINAL",
                        new_action_arg=old_answer,
                        reason=f"PF helper judged '{old_answer}' better supported than '{arg}'",
                    )
                else:
                    logger.info(f"[PF:answer_confidence_guard] PF helper accepted new answer: '{arg}'")
                    return Intervention(type=InterventionType.NOOP, reason="Teacher approved answer change")
            except Exception as e:
                if note_api_error(e): raise
                logger.warning(f"[PF:answer_confidence_guard] PF helper call failed: {e}")

        # Fallback: warning injection
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                f"\n[CONFIDENCE WARNING] You previously concluded '{old_answer}' "
                f"based on source evidence (step {last_candidate.get('step', '?')}). "
                f"Only change your answer if you have strong new evidence."
            ),
            reason=f"Answer changed from '{old_answer}' to '{arg}'; prior answer had read support",
        )



# ============================================================================
# Helpers used by the override + the two new skills below
# ============================================================================

def _recent_searches(ctx, n=5):
    hist = ctx.get("action_history", []) or []
    return [a.get("arg", "") for a in hist[-n:]
            if a.get("action_type") == "SEARCH" and a.get("arg")]


def _question_entities(question):
    return re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", question or "")


def _jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ============================================================================
# OVERRIDE: iterative_refinement  (INJECT_CONTEXT -> MODIFY_ACTION)
# Original logic only injected a "your searches look similar" warning. The
# override redirects the next SEARCH to a sensible new query when one exists
# (original question, or the question's key entity), and falls back to the
# original soft warning when no safe reformulation is available.
# ============================================================================

@register_pf("iterative_refinement")
class IterativeRefinementPF(ProgramFunction):
    """When the current SEARCH overlaps a recent one, redirect to a better query
    instead of only warning. Safe deterministic reformulation: fall back to the
    original question (a sensible query) when it differs from what's being
    repeated; otherwise NOOP rather than risk a worse query."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "SEARCH":
            return False
        recent = _recent_searches(ctx)
        if len(recent) < 2:
            return False
        return any(_jaccard(arg, prev) > 0.5 for prev in recent)

    def intervene(self, ctx, action_type, arg, helper=None):
        question = (ctx.get("question") or "").strip()
        recent = _recent_searches(ctx)
        # If the original question is meaningfully different from what we keep
        # repeating, search it directly.
        if question and _jaccard(arg, question) < 0.6 and all(
                _jaccard(question, p) < 0.8 for p in recent):
            return Intervention(
                type=InterventionType.MODIFY_ACTION, new_action_type="SEARCH",
                new_action_arg=question, skill_id="iterative_refinement",
                reason="Repeated similar search; redirecting to the original question",
            )
        # Otherwise focus on the most specific named entity in the question.
        ents = _question_entities(question)
        if ents:
            target = max(ents, key=len)
            if _jaccard(target, arg) < 0.6:
                return Intervention(
                    type=InterventionType.MODIFY_ACTION, new_action_type="SEARCH",
                    new_action_arg=target, skill_id="iterative_refinement",
                    reason="Repeated similar search; focusing on the key entity",
                )
        # No safe reformulation — fall back to the original soft hint.
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=("\n[SEARCH REFINEMENT] Your recent searches are very similar. "
                          "Try specific names you've found, synonyms, or a different angle."),
            reason="Repeated similar searches (no safe rewrite)", skill_id="iterative_refinement",
        )


# ============================================================================
# NEW: search_stall_reformulate  (MODIFY_ACTION)
# Targets degenerate exact-repeat loops (HotpotQA) and empty-result stalls.
# ============================================================================

@register_pf("search_stall_reformulate")
class SearchStallReformulatePF(ProgramFunction):
    skill_id = "search_stall_reformulate"

    def should_activate(self, ctx, action_type, arg):
        if action_type != "SEARCH":
            return False
        recent = _recent_searches(ctx)
        # ≥2 prior searches and the current one exactly repeats a recent query
        norm = " ".join((arg or "").lower().split())
        repeats = sum(1 for p in recent if " ".join(p.lower().split()) == norm)
        if repeats >= 1 and len(recent) >= 2:
            return True
        # Empty results across the last couple of searches
        if (ctx.get("last_search_results_text", "") or "").strip() == "" and len(recent) >= 2:
            return True
        return False

    def intervene(self, ctx, action_type, arg, helper=None):
        question = (ctx.get("question") or "").strip()
        ents = _question_entities(question)
        recent_norms = {" ".join(p.lower().split()) for p in _recent_searches(ctx)}
        # Try, in order: a distinct named entity, the original question, then a
        # keyword subset — pick the first that isn't itself a repeat.
        candidates = []
        if ents:
            candidates.append(max(ents, key=len))
        if question:
            candidates.append(question)
            qwords = question.split()
            if len(qwords) > 6:
                candidates.append(" ".join(qwords[:6]))
        for cand in candidates:
            if cand and " ".join(cand.lower().split()) not in recent_norms:
                return Intervention(
                    type=InterventionType.MODIFY_ACTION, new_action_type="SEARCH",
                    new_action_arg=cand, skill_id=self.skill_id,
                    reason="Search stalled (repeat/empty); reformulating query",
                )
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=("\n[SEARCH STALLED] This query has not produced new results. "
                          "Break the question into sub-questions and search a bridging entity."),
            reason="Search stalled; no distinct reformulation available", skill_id=self.skill_id,
        )


# ============================================================================
# NEW: entity_constraint_check  (INJECT_CONTEXT — no safe deterministic rewrite)
# Targets the dominant 49% "final synthesis" bucket: the gold answer was in
# the reasoning but the committed answer dropped a constraint the question
# states. Restored 2026-08-26 after a strip regex truncated this file.
# ============================================================================

@register_pf("entity_constraint_check")
class EntityConstraintCheckPF(ProgramFunction):
    skill_id = "entity_constraint_check"

    _COMPARISON = {"first", "earlier", "later", "before", "after", "older",
                   "younger", "more", "less", "longer", "shorter", "which", "compare"}

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        if ctx.get("_entity_constraint_fired", False):
            return False
        question = (ctx.get("question") or "")
        q_lower = question.lower()
        # multi-hop / comparison signals where wrong-hop errors concentrate
        is_multihop = ("'s" in question) or any(
            f" {w} " in f" {q_lower} " for w in ["whose", "which", "that", "who"])
        is_comparison = any(w in q_lower for w in self._COMPARISON)
        return is_multihop or is_comparison

    def intervene(self, ctx, action_type, arg, helper=None):
        ctx["_entity_constraint_fired"] = True
        question = ctx.get("question") or ""
        ents = _question_entities(question)
        hint_ent = f" The question is about {ents[0]}" if ents else ""
        # In the efficacy test the templated advice repaired 1/6 while a plain
        # "double-check" repaired 3/6 -- the template was steering the model
        # wrong, chiefly by calling every "which" question a comparison. The
        # DIRECTION text now requires an actual comparative; otherwise the
        # inject is a concrete checklist built from THIS question, not advice.
        truly_cmp = bool(re.search(
            r"\b(earlier|earliest|later|latest|older|oldest|younger|first|last"
            r"|more|most|less|least|larger|longer|higher|lower|before|after)\b",
            question, re.I))
        if truly_cmp:
            msg = ("This is a comparison question. Re-check the DIRECTION "
                   "(earlier/later, more/less) and that BOTH entities' values were "
                   "compared, not just one.")
        else:
            constraints = "; ".join(_question_entities(question)[:3]) or "the question's entities"
            msg = (f"Before finalizing, verify against the question itself: the answer "
                   f"{arg!r} must be the quantity/person the question asks FOR, and must "
                   f"be consistent with: {constraints}. If the evidence names a different "
                   f"final-hop entity, use that one.")
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=f"\n[VERIFY ANSWER ENTITY]{hint_ent}. {msg}",
            reason="Multi-hop/comparison FINAL — verify the answer entity vs constraints",
            skill_id=self.skill_id,
        )

