"""
LLM-as-Judge (MBE) for semantic equivalence evaluation.

Uses GPT-4o to judge whether predicted answers are semantically
equivalent to gold answers. Runs as a standalone post-hoc process
on existing episode JSONL files.

Prompt and parsing logic follows ASearcher's DefaultJudge.
"""

import asyncio
import json
import ast
import re
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


JUDGE_PROMPT = """You are an evaluation assistant. Below is the model's full response to a question, including its reasoning process. Please determine if the model's final answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {gt_answer}

Model Response:
{pred_answer}

Based on the model's full response above, did the model arrive at an answer **equivalent** to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent.

The output should in the following json format:
```json
{{
    "rationale": your rationale for the judgement, as a text,
    "judgement": your judgement result, can only be "Correct" or "Incorrect",
}}
```
"""


@dataclass
class JudgeResult:
    """Result of a single LLM-as-Judge evaluation."""
    sample_id: str = ""
    question: str = ""
    gold_answer: str = ""
    pred_answer: str = ""
    judgement: bool = False
    rationale: str = ""
    raw_response: str = ""
    mbe_score: float = 0.0


class LLMJudge:
    """
    LLM-based judge for semantic answer equivalence (MBE metric).

    Supports both OpenAI and Anthropic providers.
    """

    def __init__(
        self,
        model: str = "",
        provider: str = "openai",
        api_key: Optional[str] = None,
        max_concurrent: int = 10,
    ):
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self._client = None

    def _get_client(self):
        """Lazy-initialize async API client based on provider."""
        if self._client is None:
            import os
            if self.provider == "anthropic":
                from anthropic import AsyncAnthropic
                api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
                self._client = AsyncAnthropic(api_key=api_key)
            elif self.provider == "google":
                from google import genai
                api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
                self._client = genai.Client(api_key=api_key)
            else:
                from openai import AsyncOpenAI
                api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
                self._client = AsyncOpenAI(api_key=api_key)
        return self._client

    @staticmethod
    def parse_judgement(raw_response: str) -> Tuple[bool, str]:
        """
        Parse judgement from LLM response.

        Tries JSON parsing, then ast.literal_eval, then regex fallback.
        Follows ASearcher's DefaultJudge.cal_metrics() logic.

        Returns:
            (judgement_bool, rationale_str)
        """
        mbe = None
        # Try to extract JSON block
        json_text = raw_response.split("```json")[-1].split("```")[0].strip()

        for parse_fn in [json.loads, ast.literal_eval]:
            try:
                mbe = parse_fn(json_text)
                break
            except Exception:
                pass

        if mbe is None and '"judgement": "incorrect"' in raw_response.lower():
            return False, ""
        if mbe is None and (
            '"judgement": "correct"' in raw_response.lower()
            or '"judgement": correct' in raw_response.lower()
        ):
            return True, ""
        if mbe is None:
            # Last resort: regex
            match = re.search(r'"judgement"\s*:\s*"?(correct|incorrect)"?', raw_response, re.IGNORECASE)
            if match:
                return match.group(1).lower() == "correct", ""
            return False, ""

        rationale = mbe.get("rationale", "") if isinstance(mbe, dict) else ""
        is_correct = (
            isinstance(mbe, dict)
            and "judgement" in mbe
            and mbe["judgement"].lower() == "correct"
        )
        return is_correct, rationale

    async def _call_api(self, prompt: str) -> str:
        """Call the appropriate API and return the raw text response."""
        client = self._get_client()

        if self.provider == "anthropic":
            response = await client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text if response.content else ""
        elif self.provider == "google":
            from google.genai import types
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )],
                config=types.GenerateContentConfig(max_output_tokens=512),
            )
            return response.text or ""
        else:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""

    async def judge_single(
        self,
        question: str,
        gold: str,
        pred: str,
    ) -> JudgeResult:
        """Judge a single prediction against gold answer."""
        prompt = JUDGE_PROMPT.format(
            question=question,
            gt_answer=gold,
            pred_answer=pred,
        )

        raw = await self._call_api(prompt)
        judgement, rationale = self.parse_judgement(raw)

        return JudgeResult(
            question=question,
            gold_answer=gold,
            pred_answer=pred,
            judgement=judgement,
            rationale=rationale,
            raw_response=raw,
            mbe_score=float(judgement),
        )

    async def judge_batch(
        self,
        episodes: List[dict],
        resume: bool = False,
    ) -> List[JudgeResult]:
        """
        Judge a batch of episodes concurrently with semaphore limiting.

        Args:
            episodes: List of episode dicts with 'question', 'gold_answers'/'answer',
                      'pred_answer'/'final_answer', and optionally 'sample_id'/'id'.
            resume: If True, skip episodes that already have llm_as_judge.status == "success".

        Returns:
            List of JudgeResult objects (same order as input).
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: List[Optional[JudgeResult]] = [None] * len(episodes)

        async def _judge_with_retry(idx: int, ep: dict):
            # Check resume
            if resume:
                judge_info = ep.get("llm_as_judge", {})
                if isinstance(judge_info, dict) and judge_info.get("status") == "success":
                    # Already judged
                    return JudgeResult(
                        sample_id=str(ep.get("sample_id", ep.get("id", str(idx)))),
                        question=ep.get("question", ""),
                        gold_answer=_get_gold(ep),
                        pred_answer=_get_pred(ep),
                        judgement=judge_info.get("judgement", "").lower() == "correct",
                        raw_response=judge_info.get("raw_response", ""),
                        mbe_score=float(ep.get("MBE", 0.0)),
                    )

            question = ep.get("question", "")
            gold = _get_gold(ep)
            pred = _get_pred(ep)

            for attempt in range(3):
                try:
                    async with semaphore:
                        result = await self.judge_single(question, gold, pred)
                    result.sample_id = str(ep.get("sample_id", ep.get("id", str(idx))))
                    return result
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"Judge retry {attempt+1}/3 for sample {idx}: {e}")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        logger.error(f"Judge failed for sample {idx}: {e}")
                        return JudgeResult(
                            sample_id=str(ep.get("sample_id", ep.get("id", str(idx)))),
                            question=question,
                            gold_answer=gold,
                            pred_answer=pred,
                            raw_response=f"ERROR: {e}",
                        )

        tasks = [_judge_with_retry(i, ep) for i, ep in enumerate(episodes)]
        results = await asyncio.gather(*tasks)
        return list(results)


def create_judge_from_config(config_path: str) -> LLMJudge:
    """Create an LLMJudge from agent_eval.yaml config.

    Reads roles.judge → api_models → provider/model_name/api_key.
    """
    import os
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    roles = cfg.get("roles", {})
    api_models = cfg.get("api_models", {})
    api_keys = cfg.get("api_keys", {})

    judge_key = roles.get("judge", "gpt")
    model_cfg = api_models.get(judge_key, {})

    provider = model_cfg.get("provider", "openai")
    model_name = model_cfg.get("model_name", "")
    max_concurrent = model_cfg.get("max_concurrent", 10)

    if provider == "anthropic":
        api_key = api_keys.get("anthropic_key") or os.environ.get("ANTHROPIC_API_KEY")
    elif provider == "openai":
        api_key = api_keys.get("openai_key") or os.environ.get("OPENAI_API_KEY")
    elif provider == "google":
        api_key = api_keys.get("google_key") or os.environ.get("GOOGLE_API_KEY")
    else:
        api_key = None

    return LLMJudge(
        model=model_name,
        provider=provider,
        api_key=api_key,
        max_concurrent=max_concurrent,
    )


def _get_gold(ep: dict) -> str:
    """Extract gold answer string from episode dict."""
    gold = ep.get("gold_answers", ep.get("answer", ep.get("gold_answer", "")))
    if isinstance(gold, list):
        return gold[0] if gold else ""
    return str(gold)


def _get_pred(ep: dict) -> str:
    """Extract predicted answer string from episode dict.

    Uses ``final.answer`` (the extracted short answer) as the primary source.
    Falls back to ``final.raw_output`` only if answer is empty.

    Note: raw_output contains the model's full reasoning which is often
    truncated during storage, causing the judge to miss the actual answer.
    Always prefer the extracted answer field.
    """
    # Try nested final dict — prefer answer (short, clean)
    final = ep.get("final", {})
    if isinstance(final, dict):
        answer = final.get("answer", "")
        if answer:
            return str(answer)
        # Fallback to raw_output only if answer is missing
        raw = final.get("raw_output", "")
        if raw:
            return str(raw)

    # Legacy flat fields
    pred = ep.get("pred_answer", ep.get("final_answer", ""))
    return str(pred)
