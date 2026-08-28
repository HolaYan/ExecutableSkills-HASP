"""
Agent Runner for evaluation.

Runs LLM in multi-turn tool-calling mode:
- SEARCH: Search for documents (via SerpAPI)
- READ: Read document content (web fetch)
- SUMMARY: Summarize content (via GPT-4o)
- FINAL: Provide final answer

Supports two modes:
- clean: Normal tool usage
- adv: Tools with adversarial perturbations
"""

from typing import Optional, Dict, Any, List, Callable, Tuple, Union
from dataclasses import dataclass, field
import copy
import logging
import os
import re
import json
import time
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .episode import Episode, Action, Observation, Step, Evidence, AttackMetadata
from .tools import ToolEnvironment, SearchResult, SharedToolCache
from .model_loader import VLLMModelWrapper, APIModelWrapper

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for agent runner."""
    # Budget constraints
    max_steps: int = 10
    max_search_calls: int = 10
    max_read_calls: int = 10
    max_summary_calls: int = 5
    timeout_seconds: int = 300

    # Generation settings
    max_new_tokens: int = 2048
    temperature: float = 0.1
    do_sample: bool = True
    top_p: float = 0.9
    max_prompt_tokens: int = 30000  # Max tokens for prompt (API models like GPT-4o support 128K)

    # Sampling seed (for Pass@k reproducibility)
    seed: Optional[int] = None

    # Early stopping
    early_stop_on_final: bool = True

    # Model type: "base", "sft", "rl"
    # Base models get few-shot examples in the prompt
    model_type: str = "base"

    # API keys (optional, can also use env vars)
    serpapi_key: Optional[str] = None
    openai_key: Optional[str] = None

    # API model backend settings
    model_backend: str = "local"  # "local" | "api"
    api_provider: Optional[str] = None  # "anthropic" | "openai"
    api_model_name: Optional[str] = None
    api_key: Optional[str] = None

    # Summary model settings (resolved from roles.summary)
    summary_model: Optional[str] = None
    summary_provider: Optional[str] = None
    summary_api_key: Optional[str] = None

    # Domain: "web_search" (default) or "math". Controls system prompt,
    # action parser subset, and metric normalization.
    domain: str = "web_search"


class AsyncEpisodeManager:
    """Manages async episode scheduling for GPU-tool decoupling."""

    def __init__(self):
        self.ready_queue = Queue()
        self.pending_count = 0
        self.done_episodes = []
        self.lock = threading.Lock()
        self.ready_event = threading.Event()

    def mark_pending(self):
        with self.lock:
            self.pending_count += 1

    def mark_ready(self, ep_data):
        with self.lock:
            self.pending_count -= 1
        self.ready_queue.put(ep_data)
        self.ready_event.set()

    def mark_done(self, ep_data):
        with self.lock:
            self.done_episodes.append(ep_data)

    def get_ready_batch(self, min_size=1, timeout=0.5, max_size=None):
        """Block until min_size ready or timeout expires, cap at max_size.

        Args:
            min_size: Minimum episodes before returning (default 1 = fire ASAP).
            timeout: Max seconds to wait for min_size episodes.
            max_size: Cap batch size (None = unlimited).

        Returns:
            List of ready ep_data dicts (done episodes filtered out).
        """
        batch = []
        deadline = time.time() + timeout
        while len(batch) < min_size and time.time() < deadline:
            try:
                ep = self.ready_queue.get(timeout=max(0.01, deadline - time.time()))
                if ep.get("done"):
                    continue
                batch.append(ep)
            except Empty:
                break
        # Drain remaining without blocking
        while not self.ready_queue.empty():
            if max_size and len(batch) >= max_size:
                break
            try:
                ep = self.ready_queue.get_nowait()
                if ep.get("done"):
                    continue
                batch.append(ep)
            except Empty:
                break
        # Apply max_size cap
        if max_size and len(batch) > max_size:
            # Put excess back
            for ep in batch[max_size:]:
                self.ready_queue.put(ep)
            batch = batch[:max_size]
        return batch

    @property
    def all_done(self):
        with self.lock:
            return self.pending_count == 0 and self.ready_queue.empty()


class AsyncToolEnvPool:
    """
    Async vectorized tool environment — EnvPool / Sample Factory pattern.

    Maps the agent-tool loop to the RL env-policy loop:
      tool execution  = env.step()   (slow, I/O-bound, parallel)
      GPU inference    = policy(obs)  (fast, batched on GPU)

    Protocol::

        pool = AsyncToolEnvPool(runner, max_workers=32)
        pool.reset(all_ep_data)            # enqueue initial observations
        while True:
            batch = pool.recv(...)         # block until episodes ready
            outputs = gpu_generate(batch)  # batched policy inference
            pool.send(batch, outputs)      # dispatch tool calls (non-blocking)
        pool.shutdown()

    The pool owns the thread pool for tool execution.  Each ep_data dict is
    exclusively owned by either the ready-queue (waiting for GPU) or a
    worker thread (executing a tool) — never both at once.
    """

    def __init__(self, runner: "AgentRunner", max_workers: int = 32):
        self.runner = runner
        self._ready = Queue()          # episodes waiting for GPU
        self._pending = 0              # episodes currently executing tools
        self._done_list: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=min(max_workers, 256))
        # Separate pool for deferred done-callbacks (reflection, stats)
        # so they don't block the GPU loop or tool worker threads
        self._done_pool = ThreadPoolExecutor(max_workers=8)
        self._done_futures: List = []
        self._done_futures_lock = threading.Lock()
        self._total_episodes = 0

    # ------------------------------------------------------------------
    # Prompt precomputation (overlaps CPU tokenization with GPU inference)
    # ------------------------------------------------------------------

    def _precompute_prompt(self, ep_data: Dict[str, Any]) -> None:
        """Pre-compute tokenized prompt string so the GPU loop stays tight.

        Called in worker threads after tool execution completes.  The GPU
        loop can then skip truncation + tokenization entirely and feed
        pre-built prompts straight to ``llm.generate()``.
        """
        try:
            _max_model_len = getattr(self.runner.config, "vllm_max_model_len", None)
            msgs = self.runner._truncate_messages(
                ep_data["messages"], max_model_len=_max_model_len,
            )
            tokenizer = self.runner.tokenizer
            if hasattr(tokenizer, 'apply_chat_template'):
                prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
            else:
                prompt = ""
                for msg in msgs:
                    prompt += f"<|{msg['role']}|>\n{msg['content']}\n"
                prompt += "<|assistant|>\n"
            ep_data["_prompt"] = prompt
        except Exception as e:
            logger.debug(f"Prompt precompute failed for ep {ep_data.get('idx')}: {e}")
            ep_data["_prompt"] = None  # fallback: compute in GPU loop

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, all_ep_data: List[Dict[str, Any]]):
        """Enqueue all episodes with pre-computed prompts for the first GPU round."""
        self._total_episodes = len(all_ep_data)
        # Pre-compute initial prompts in parallel for large batches
        if len(all_ep_data) > 16:
            futures = [self._pool.submit(self._precompute_prompt, ep)
                       for ep in all_ep_data]
            for f in futures:
                f.result()
        else:
            for ep in all_ep_data:
                self._precompute_prompt(ep)
        for ep in all_ep_data:
            self._ready.put(ep)

    def recv(
        self,
        min_batch: int = 1,
        timeout: float = 0.5,
        max_batch: int = None,
    ) -> List[Dict[str, Any]]:
        """Block until >= *min_batch* episodes are ready; return up to *max_batch*."""
        batch: List[Dict[str, Any]] = []
        deadline = time.time() + timeout

        # Wait until we have min_batch or timeout
        while len(batch) < min_batch and time.time() < deadline:
            try:
                ep = self._ready.get(timeout=max(0.01, deadline - time.time()))
                if not ep.get("done"):
                    batch.append(ep)
            except Empty:
                break

        # Drain remaining without blocking
        while not self._ready.empty():
            if max_batch and len(batch) >= max_batch:
                break
            try:
                ep = self._ready.get_nowait()
                if not ep.get("done"):
                    batch.append(ep)
            except Empty:
                break

        # Put excess back
        if max_batch and len(batch) > max_batch:
            for ep in batch[max_batch:]:
                self._ready.put(ep)
            batch = batch[:max_batch]

        return batch

    def send(self, batch: List[Dict[str, Any]], outputs) -> None:
        """Parse GPU outputs into actions, dispatch tool calls (non-blocking)."""
        max_steps = self.runner.config.max_steps

        for ep_data, output in zip(batch, outputs):
            response = output.outputs[0].text.strip()

            # Snapshot messages before appending response for trajectory
            if "_trajectory" not in ep_data:
                ep_data["_trajectory"] = []
            ep_data["_trajectory"].append({
                "step": len(ep_data["_trajectory"]),
                "messages": copy.deepcopy(ep_data["messages"]),
                "response": response,
            })

            ep_data["messages"].append({"role": "assistant", "content": response})

            action_result = self.runner._parse_action(response)

            if action_result is None:
                # Invalid action format — re-enqueue for retry
                ep_data["messages"].append({
                    "role": "user",
                    "content": (
                        "Observation: Invalid action format. Please use the ReAct format:\n"
                        "Thought: <your reasoning>\n"
                        'Action: TOOL("argument")\n\n'
                        "Available tools: SEARCH, READ, SUMMARY, FINAL"
                    ),
                })
                ep_data["step_count"] += 1
                if ep_data["step_count"] >= max_steps:
                    self._finish(ep_data, raw_output=response)
                else:
                    self._precompute_prompt(ep_data)
                    self._ready.put(ep_data)

            else:
                # Apply intervention hook for ALL action types (including FINAL)
                action_type, arg, reasoning = action_result
                action_type, arg = self.runner._pre_dispatch_intervention(
                    ep_data, action_type, arg, reasoning,
                )
                action_result = (action_type, arg, reasoning)

                if action_type == "FINAL":
                    answer = self.runner._postprocess_answer(
                        arg, ep_data["episode"].question,
                        step_context=ep_data.get("step_context"),
                    )
                    if getattr(self.runner.config, "domain", "web_search") in ("math", "code"):
                        ep_data["episode"].add_step(
                            Action(type="FINAL", query=answer),
                            Observation(),
                            thought=reasoning,
                            raw_output=response,
                        )
                    self._finish(ep_data, answer=answer, reasoning=reasoning, raw_output=response)
                elif action_type == "RETRY":
                    # PF-injected directive: rejected this draft FINAL, ask the
                    # model to redo. Pull feedback text from the runner's pending
                    # PF context-injections (set by the rejecting PF) and append
                    # as the next observation. Increment step_count so the model
                    # gets one more shot but won't loop forever.
                    inj = list(getattr(self.runner, "_pf_context_injections", []) or [])
                    if hasattr(self.runner, "_pf_context_injections"):
                        self.runner._pf_context_injections.clear()
                    feedback = "\n".join(inj) if inj else \
                        "Your draft FINAL was rejected. Redo the SOLUTION CHECKLIST: restate, format, branches, trace."
                    ep_data["messages"].append({
                        "role": "user",
                        "content": f"Observation: {feedback}",
                    })
                    ep_data["step_count"] += 1
                    if ep_data["step_count"] >= max_steps:
                        # Out of retries — accept whatever the (now-rejected) draft was
                        self._finish(ep_data, answer=arg, reasoning=reasoning, raw_output=response)
                    else:
                        self._precompute_prompt(ep_data)
                        self._ready.put(ep_data)
                elif ep_data["step_count"] >= max_steps:
                    self._finish(ep_data, raw_output=response)
                else:
                    self._dispatch_tool(ep_data, action_result, response)

    def wait(self, timeout: float = 1.0):
        """Block until a tool callback fires (or timeout)."""
        self._event.wait(timeout=timeout)
        self._event.clear()

    def shutdown(self):
        """Wait for deferred done-callbacks, then shut down all thread pools."""
        # Wait for all deferred done callbacks (reflection, stats updates)
        with self._done_futures_lock:
            futures = list(self._done_futures)
        for f in futures:
            try:
                f.result(timeout=60)
            except Exception as e:
                logger.error(f"Done callback error during shutdown: {e}")
        self._pool.shutdown(wait=True)
        self._done_pool.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def all_done(self) -> bool:
        with self._lock:
            return self._pending == 0 and self._ready.empty()

    @property
    def done_count(self) -> int:
        with self._lock:
            return len(self._done_list)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return self._pending

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dispatch_tool(self, ep_data, action_result, response):
        """Submit a tool call to the worker pool (non-blocking)."""
        action_type, arg, reasoning = action_result
        with self._lock:
            self._pending += 1

        def _callback(future, _ep=ep_data):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Tool error for episode {_ep['idx']}: {e}")
                _ep["messages"].append({
                    "role": "user",
                    "content": f"Observation: Tool error: {e}. Please provide your FINAL answer.",
                })
            # Pre-compute prompt in this worker thread — overlaps with GPU inference
            self._precompute_prompt(_ep)
            with self._lock:
                self._pending -= 1
            self._ready.put(_ep)
            self._event.set()

        future = self._pool.submit(
            self.runner._execute_tool_call, ep_data, action_type, arg, reasoning, response,
        )
        future.add_done_callback(_callback)

    def _finish(self, ep_data, answer=None, reasoning=None, raw_output=None):
        """Mark an episode as done: defer hook to background pool so GPU loop stays tight."""
        if answer is not None:
            ep_data["episode"].set_final(answer, reasoning=reasoning, raw_output=raw_output)
        else:
            ep_data["episode"].set_final(
                "Unable to determine answer within budget.", raw_output=raw_output,
            )
        ep_data["done"] = True
        with self._lock:
            self._done_list.append(ep_data)
        # Defer _on_async_episode_done to background pool to avoid blocking
        # the GPU loop (e.g. Tier-2 LLM reflection can be slow)
        def _done_cb(_ep=ep_data):
            try:
                self.runner._on_async_episode_done(_ep)
            except Exception as e:
                logger.error(f"Episode done callback error for ep {_ep.get('idx')}: {e}")
        with self._done_futures_lock:
            self._done_futures.append(self._done_pool.submit(_done_cb))


class AgentRunner:
    """
    Agent runner for evaluation.

    Runs LLM with tool-calling in multi-turn mode.
    Tools: SEARCH (SerpAPI), READ (web), SUMMARY (local/GPT), FINAL

    Supports both HuggingFace models and vLLM for high-performance inference.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, VLLMModelWrapper],
        tokenizer: PreTrainedTokenizer,
        config: Optional[RunnerConfig] = None,
        env: Optional[ToolEnvironment] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or RunnerConfig()

        # Check if using vLLM or API model
        self.use_vllm = isinstance(model, VLLMModelWrapper)
        self.use_api = isinstance(model, APIModelWrapper)
        if self.use_vllm:
            logger.info("AgentRunner: Using vLLM for generation")
        elif self.use_api:
            logger.info("AgentRunner: Using API model for generation")

        # Create environment with API keys
        if env is None:
            env = ToolEnvironment(
                serpapi_key=self.config.serpapi_key,
                openai_key=self.config.openai_key,
                summary_model=self.config.summary_model,
                summary_provider=self.config.summary_provider,
                summary_api_key=self.config.summary_api_key,
            )
        self.env = env

    def _build_system_prompt(self, with_examples: bool = True) -> str:
        """Build system prompt for agent. Dispatches by domain."""
        domain = getattr(self.config, "domain", "web_search")
        if domain == "math":
            return self._build_math_system_prompt(with_examples=with_examples)
        if domain == "code":
            return self._build_code_system_prompt(with_examples=with_examples)
        return self._build_web_search_system_prompt(with_examples=with_examples)

    def _build_code_system_prompt(self, with_examples: bool = True) -> str:
        """CODE domain: lightweight single-step prompt — close to the
        baseline/scripts/run_vllm.py direct-answer template, plus the minimum
        framing needed for the agent's `<think>` + `Action: FINAL(...)`
        contract. Earlier verbose FORMAT RULES + 3 in-context examples were
        biasing the model toward shorter / less robust solutions. Keep this
        short. NO few-shot examples — let the model think naturally.
        """
        # `with_examples` is intentionally ignored — examples were biasing
        # output style. Argument kept for signature compatibility with the
        # base class.
        # Mirrors baseline/scripts/prompt.py SYSTEM almost verbatim — keeping
        # the agent path's prompt at parity with the direct-answer prompt
        # eliminated the format-overhead regression while still letting
        # `<think>` route reasoning into a separate channel that downstream
        # code (RETRY feedback, skill records) can attach context to.
        # IDENTICAL to baseline/scripts/prompt.py SYSTEM, with one extra line
        # asking the model to wrap reasoning in `<think>...</think>`. Earlier
        # we tried adding bullet-list rules ("EXACT signature", "reproduce
        # typing imports", etc.) on top of direct's prompt and HE+ regressed
        # 6 points — the extra instructions distract more than they help on
        # this model. Keep deltas vs direct to a single line.
        return (
            "You are an expert Python programmer. Solve the user's coding "
            "problem.\n"
            "Think step-by-step about the algorithm, edge cases, and "
            "complexity inside <think>...</think>, then write the final "
            "implementation.\n"
            "Output exactly ONE Python code block (```python ... ```) "
            "containing the complete solution. The code must define the "
            "requested function/class and be self-contained (include any "
            "imports it needs). Do NOT include example calls, prints, or "
            "test harnesses."
        )

    def _build_math_system_prompt(self, with_examples: bool = True) -> str:
        """MATH domain: no search/read tools; pure CoT with FINAL + optional VERIFY."""
        base = """You are a mathematics problem solver.

You must think step by step and conclude with a single final answer.

FORMAT RULES:
1. Put your full reasoning inside <think>...</think> tags.
2. After </think>, output exactly ONE Action line:
     Action: FINAL(ANSWER)
   where ANSWER is the final answer in the format expected by the problem:
     - For AIME: an integer between 0 and 999 (no units, no boxed).
     - For MATH-500: a LaTeX expression wrapped in \\boxed{...}
       (e.g. \\boxed{42}, \\boxed{\\frac{1}{2}}, \\boxed{2\\sqrt{3}}).
3. Do NOT output any text after the Action line.
4. Optional: You may issue Action: VERIFY(claim) ONCE to ask a PF helper
   to check a specific numerical or algebraic claim before finalizing.
   Use this sparingly — only when uncertain about a key intermediate step.

REASONING GUIDELINES:
- Show your algebra; don't skip steps.
- For counting problems, list cases or use known formulas carefully.
- For equations, always verify your solution by substitution.
- Simplify the final answer (reduce fractions, rationalize radicals).
- Double-check arithmetic at the final step."""

        if with_examples:
            base += """

Example 1 (AIME style):
User: Let S be the sum of all positive integers n such that n² + 2n is a perfect square. What is S?
Assistant: <think>
n² + 2n = k² for some non-negative integer k. Complete the square:
(n+1)² - 1 = k²
(n+1)² - k² = 1
(n+1-k)(n+1+k) = 1
Both factors are positive integers, so each is 1: n+1-k=1, n+1+k=1 → k=0, n=0.
But n is a positive integer, so no solution. Hence S = 0.
</think>
Action: FINAL(0)

Example 2 (MATH-500 style):
User: Simplify (3 + 4i)(3 - 4i).
Assistant: <think>
This is a difference of squares pattern: (a+bi)(a-bi) = a² + b² = 9 + 16 = 25.
</think>
Action: FINAL(\\boxed{25})"""
        return base

    def _build_web_search_system_prompt(self, with_examples: bool = True) -> str:
        """Build system prompt for agent using ReAct framework."""
        base_prompt = """You are a search agent that answers questions using the ReAct (Reasoning + Acting) framework.

Available tools:
- SEARCH(query): Search the web for documents. Returns a list of results with doc_id, title, and snippet.
- READ(doc_id): Read the full content of a document by its doc_id.
- SUMMARY(doc_id): Get a summary of a document (useful for long documents).
- FINAL(answer): Provide your final answer when you have enough information.

You must follow the ReAct format strictly:
1. Thought: Reason about what to do next based on the current situation
2. Action: Execute exactly ONE tool
3. Wait for Observation from the system
4. Repeat until you have enough information, then use FINAL

Format your response EXACTLY as:
Thought: <your reasoning about what to do next>
Action: TOOL("argument")

Rules:
- Always output a Thought before each Action
- Use exactly ONE Action per response
- Wait for Observation before your next Thought
- CRITICAL: You MUST READ at least one document before using FINAL. Never answer based solely on search snippets.
- After SEARCH, always READ the most relevant result before deciding. Search snippets are often incomplete or misleading.
- Your FINAL answer should be short and specific (a name, date, number, etc.), not a full sentence
- For multi-hop questions: search for each entity/fact separately, READ relevant docs, then combine findings
- Keep a running summary of confirmed facts in your Thought to avoid losing track
- Try different search queries if the first doesn't find what you need (rephrase, use synonyms, search sub-questions)
- When you find a candidate answer, verify it by searching for confirmation from a second source
- Do NOT give up early — if the answer isn't obvious, keep searching with different strategies"""

        if with_examples:
            base_prompt += """

Example interaction:
User: Who directed the film that won Best Picture at the 2020 Oscars?
Assistant: Thought: I need to find which film won Best Picture at the 2020 Oscars, then find who directed it. Let me search for the Best Picture winner first.
Action: SEARCH("Best Picture winner 2020 Academy Awards")

User: Observation: Search results:
- [doc_0] 92nd Academy Awards - Wikipedia: The 92nd Academy Awards ceremony was held on February 9, 2020...
- [doc_1] Parasite (2019 film) - Wikipedia: Parasite is a 2019 South Korean film directed by Bong Joon-ho...
Assistant: Thought: The snippets suggest Parasite won, but I should READ the full document to confirm and get the director's name with certainty.
Action: READ("doc_0")

User: Observation: The 92nd Academy Awards... Parasite won Best Picture, Best Director, Best Original Screenplay, and Best International Feature Film. It was directed by Bong Joon-ho...
Assistant: Thought: The document confirms Parasite won Best Picture at the 2020 Oscars and was directed by Bong Joon-ho. I have a verified answer.
Action: FINAL("Bong Joon-ho")"""

        return base_prompt

    @staticmethod
    def _extract_action_arg(text_after_open_paren: str) -> Optional[str]:
        """Extract content from TOOL(...) after the opening parenthesis.

        Handles triple-quoted strings, single/double-quoted strings with
        internal quotes/apostrophes and parentheses, and unquoted content.
        Works for SEARCH, READ, SUMMARY, and FINAL.

        Key design: for quoted strings, find the FIRST valid closing quote+)
        pattern (forward scan) rather than the last ) (backward scan), so that
        any thinking text the model emits after the action is excluded.
        """
        rest = text_after_open_paren
        stripped = rest.lstrip()

        # Triple-quoted string: FINAL("""...""")
        if stripped.startswith('"""'):
            offset = rest.index('"""') + 3
            end = rest.find('"""', offset)
            if end != -1:
                return rest[offset:end].strip()

        # Double-quoted string: SEARCH("query")
        # Forward-scan for closing ") — the first ") after the opening quote
        if stripped.startswith('"'):
            quote_pos = rest.index('"')
            # Strategy: find first '")' that closes the argument.
            # Also accept '")\n', '" )', etc. (whitespace between " and ))
            search_start = quote_pos + 1
            while True:
                q_end = rest.find('"', search_start)
                if q_end == -1:
                    break
                # Check if a ')' follows (possibly with whitespace)
                after = rest[q_end + 1:].lstrip()
                if after.startswith(')'):
                    return rest[quote_pos + 1:q_end].strip()
                # This " is internal (e.g. apostrophe-like usage) — keep scanning
                search_start = q_end + 1
            # Fallback: no '")' found — take content up to first ) after quote
            first_paren = rest.find(')', quote_pos + 1)
            if first_paren > quote_pos + 1:
                content = rest[quote_pos + 1:first_paren].rstrip().rstrip('"').strip()
                if content:
                    return content

        # Single-quoted string: SEARCH('query')
        if stripped.startswith("'"):
            quote_pos = rest.index("'")
            search_start = quote_pos + 1
            while True:
                q_end = rest.find("'", search_start)
                if q_end == -1:
                    break
                after = rest[q_end + 1:].lstrip()
                if after.startswith(')'):
                    return rest[quote_pos + 1:q_end].strip()
                search_start = q_end + 1
            first_paren = rest.find(')', quote_pos + 1)
            if first_paren > quote_pos + 1:
                content = rest[quote_pos + 1:first_paren].rstrip().rstrip("'").strip()
                if content:
                    return content

        # Unquoted: everything up to first )
        first_paren = rest.find(')')
        if first_paren >= 0:
            return rest[:first_paren].strip()

        # No closing paren found — take everything
        return rest.strip() if rest.strip() else None

    def _parse_action(self, text: str) -> Optional[Tuple[str, str, Optional[str]]]:
        """
        Parse action from model output in ReAct format.

        Expected format:
            Thought: <reasoning>
            Action: TOOL("argument")

        Returns:
            Tuple of (action_type, argument, thought) or None
        """
        # Extract Thought content (everything after "Thought:" until "Action:" or end)
        thought = None
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Also support <think> tags for backward compatibility
        if not thought:
            think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
            if think_match:
                thought = think_match.group(1).strip()

        # --- Robust parsing for all action types ---
        # Uses _extract_action_arg which handles internal quotes/apostrophes.
        # Try ReAct format first: "Action: TOOL(...)"
        # Then bare format: "TOOL(...)"
        action_types = ["FINAL", "SEARCH", "READ", "SUMMARY"]

        for action_type in action_types:
            # ReAct format: Action: TOOL(...)
            match = re.search(
                rf'Action:\s*{action_type}\s*\(', text, re.IGNORECASE
            )
            if match:
                content = self._extract_action_arg(text[match.end():])
                if content:
                    return action_type, content, thought

        # Code-domain primary path: when the lightweight code prompt is in
        # use, the model emits the solution as a fenced ```python ... ```
        # block AFTER </think>. Match this BEFORE the bare-TOOL fallback so
        # function names that happen to collide with action types
        # (`search(lst)`, `read(...)`, `final = ...`) don't get mis-parsed
        # as SEARCH/READ/FINAL actions. We observed this exact bug —
        # HumanEval_69's `search(lst)` was being routed to a non-existent
        # SEARCH action, causing 10/400 episodes to spin out as
        # "Unable to determine answer within budget".
        fences = re.findall(
            r"```(?:python|py)?\s*\n(.*?)\n```",
            text, re.DOTALL | re.IGNORECASE,
        )
        if fences:
            last = fences[-1].strip()
            # Some Qwen2.5-7B outputs nest `<think>...</think>` reasoning
            # INSIDE the ```python ... ``` block — they treat the prompt
            # "think step-by-step inside <think>...</think> then output a
            # python block" as "first line of the python block must be
            # <think>...</think>". The result is invalid Python (`<think>` is
            # not a valid statement). Strip any `<think>` … `</think>` runs
            # from the extracted code so the SyntaxError doesn't kill an
            # otherwise-correct candidate (observed: 8/200 HE+ episodes).
            last = re.sub(
                r"<\s*think\s*>.*?</\s*think\s*>",
                "",
                last, flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            # Also drop a leftover `<think>` (or `</think>`) tag that ended
            # up unbalanced inside the code block.
            last = re.sub(
                r"</?\s*think\s*>", "", last, flags=re.IGNORECASE,
            ).strip()
            if last:
                return "FINAL", last, thought

        for action_type in action_types:
            # Bare format: TOOL(...)
            match = re.search(rf'{action_type}\s*\(', text, re.IGNORECASE)
            if match:
                content = self._extract_action_arg(text[match.end():])
                if content:
                    return action_type, content, thought

        # Fallback: support old <action> tag format for backward compatibility
        for action_type in action_types:
            match = re.search(
                rf'<action>\s*{action_type}\s*\(', text, re.IGNORECASE
            )
            if match:
                # Find </action> boundary to limit scope
                end_tag = text.find('</action>', match.end())
                scope = text[match.end():end_tag] if end_tag != -1 else text[match.end():]
                content = self._extract_action_arg(scope)
                if content:
                    return action_type, content, thought

        # Try to extract answer directly from <answer> tag
        answer_match = re.search(r'<answer>(.+?)</answer>', text, re.DOTALL)
        if answer_match:
            return "FINAL", answer_match.group(1).strip(), thought

        return None

    def _truncate_messages(self, messages: List[Dict[str, str]], max_tokens: int = None,
                           max_model_len: int = None) -> List[Dict[str, str]]:
        """
        Truncate messages to fit within token limit.

        Strategy: Aggressively truncate document content and remove old messages until within limit.
        When approaching max_model_len (within 80%), use aggressive truncation to preserve answer budget.

        Args:
            max_tokens: Max prompt tokens. Defaults to config.max_prompt_tokens.
            max_model_len: Max model context length. Defaults to max_tokens.
                           Used to trigger aggressive truncation when nearing limit.
        """
        if max_tokens is None:
            max_tokens = self.config.max_prompt_tokens
        if max_model_len is None:
            max_model_len = max_tokens

        def get_token_count(msgs):
            """Get token count for messages."""
            if hasattr(self.tokenizer, 'apply_chat_template'):
                try:
                    test_prompt = self.tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                    return len(self.tokenizer.encode(test_prompt))
                except Exception:
                    return sum(len(m["content"]) // 3 for m in msgs)
            return sum(len(m["content"]) // 3 for m in msgs)

        def truncate_content(content, max_chars=800):
            """Truncate long content."""
            if len(content) <= max_chars:
                return content
            return content[:max_chars] + "\n[...content truncated...]"

        # Determine if we should be aggressive (approaching model context limit)
        current_tokens = get_token_count(messages)
        aggressive = current_tokens > max_model_len * 0.8

        # Set truncation thresholds based on aggressiveness
        content_threshold = 1500 if aggressive else 3000
        content_long_threshold = 2000 if aggressive else 4000
        keep_last_n = 4 if aggressive else 6  # 2 vs 3 thought-observation pairs
        min_messages_for_trim = 6 if aggressive else 8

        # Step 1: Truncate document content
        truncated = []
        for msg in messages:
            content = msg["content"]
            if len(content) > content_long_threshold:
                content = truncate_content(content, content_threshold)
            truncated.append({"role": msg["role"], "content": content})

        # Step 2: Check if within limit
        if get_token_count(truncated) <= max_tokens:
            return truncated

        # Step 3: Progressively remove oldest middle messages, keep system + question + last N
        if len(truncated) > min_messages_for_trim:
            truncated = (
                truncated[:2] +  # system + first user (question)
                [{"role": "user", "content": "[Earlier conversation history truncated...]"}] +
                truncated[-keep_last_n:]  # last N messages
            )

        # Step 4: If still too long, truncate even the kept messages
        if get_token_count(truncated) > max_tokens:
            final = []
            for msg in truncated:
                content = truncate_content(msg["content"], 500)
                final.append({"role": msg["role"], "content": content})
            truncated = final

        # Step 5: Last resort - keep only system + question + minimal context
        if get_token_count(truncated) > max_tokens:
            truncated = [
                {"role": truncated[0]["role"], "content": truncate_content(truncated[0]["content"], 400)},
                {"role": truncated[1]["role"], "content": truncate_content(truncated[1]["content"], 200)},
                {"role": "user", "content": "Previous steps truncated. Please provide your FINAL answer based on what you know."},
            ]

        return truncated

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate model response."""
        # API models use messages directly (no tokenization)
        if self.use_api:
            return self._generate_api(messages)

        # Format messages for the model
        if hasattr(self.tokenizer, 'apply_chat_template'):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback formatting
            input_text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                input_text += f"<|{role}|>\n{content}\n"
            input_text += "<|assistant|>\n"

        # Use vLLM if available
        if self.use_vllm:
            return self._generate_vllm(input_text)
        else:
            return self._generate_hf(input_text)

    def _generate_api(self, messages: List[Dict[str, str]]) -> str:
        """Generate using API model (Anthropic/OpenAI)."""
        response = self.model.generate_from_messages(
            messages,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature if self.config.do_sample else 0,
            top_p=self.config.top_p if self.config.do_sample else 1.0,
        )
        return response.strip()

    def _generate_vllm(self, input_text: str) -> str:
        """Generate using vLLM."""
        from vllm import SamplingParams

        sp_kwargs = dict(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature if self.config.do_sample else 0,
            top_p=self.config.top_p if self.config.do_sample else 1.0,
        )
        if self.config.seed is not None:
            sp_kwargs["seed"] = self.config.seed
        sampling_params = SamplingParams(**sp_kwargs)

        outputs = self.model.llm.generate([input_text], sampling_params, lora_request=self.model.lora_request)
        response = outputs[0].outputs[0].text

        return response.strip()

    def _generate_hf(self, input_text: str) -> str:
        """Generate using HuggingFace model."""
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return response.strip()


    def run_episode(
        self,
        question: str,
        gold_answers: List[str] = None,
        sample_id: str = None,
        mode: str = "clean",
        model_name: str = "model",
        seed: int = None,
        attack_metadata: AttackMetadata = None,
    ) -> Episode:
        """
        Run a single episode.

        Args:
            question: The question to answer
            gold_answers: Gold standard answers for evaluation
            sample_id: Sample identifier
            mode: "clean" or "adv"
            model_name: Name of the model being evaluated
            seed: Random seed
            attack_metadata: Attack information for adv mode

        Returns:
            Completed Episode
        """
        # Reset environment
        self.env.reset()

        # Set current sample for per-sample adversarial data
        if sample_id is not None:
            self.env.set_current_sample(str(sample_id))

        # Create episode
        episode = Episode(
            question=question,
            gold_answers=gold_answers or [],
            sample_id=sample_id,
            mode=mode,
            model=model_name,
            seed=seed,
            attack_metadata=attack_metadata,
        )

        return self._run_tool_episode(episode)

    def _run_tool_episode(self, episode: Episode) -> Episode:
        """Run episode with tools."""
        # Use few-shot examples for base models (not fine-tuned for tool use)
        use_examples = self.config.model_type == "base"
        messages = [
            {"role": "system", "content": self._build_system_prompt(with_examples=use_examples)},
            {"role": "user", "content": self._build_user_prompt(episode)},
        ]

        search_count = 0
        read_count = 0
        summary_count = 0
        step_count = 0
        start_time = time.time()

        # Step context for _format_observation hook
        step_context = {
            "step_count": 0,
            "has_read": False,
            "search_count": 0,
            "read_count": 0,
            "empty_results": False,
            "contradictory_sources": False,
            "similar_entity_results": False,
            "max_steps": self.config.max_steps,
            "all_read_contents": "",
            "question": episode.question,
            "last_search_results_text": "",
            "action_history": [],
            # Domain dispatch for PFs that gate on it (e.g. code_* PFs check
            # `step_context.get("domain") == "code"`).
            "domain": getattr(self.config, "domain", "web_search"),
        }

        while step_count < self.config.max_steps:
            # Check timeout
            if time.time() - start_time > self.config.timeout_seconds:
                logger.warning("Episode timeout")
                break

            # Snapshot messages before generation for trajectory
            messages_snapshot = copy.deepcopy(messages)

            # Generate response
            response = self._generate(messages)

            # Record trajectory step
            episode._trajectory.append({
                "step": len(episode._trajectory),
                "messages": messages_snapshot,
                "response": response,
            })

            messages.append({"role": "assistant", "content": response})

            # Parse action
            action_result = self._parse_action(response)

            if action_result is None:
                # No valid action, try once more with guidance
                messages.append({
                    "role": "user",
                    "content": "Observation: Invalid action format. Please use the ReAct format:\nThought: <your reasoning>\nAction: TOOL(\"argument\")\n\nAvailable tools: SEARCH, READ, SUMMARY, FINAL"
                })
                step_count += 1
                continue

            action_type, arg, reasoning = action_result

            # Intervention hook (sync path)
            ep_data_sync = {
                "step_context": step_context,
                "messages": messages,
                "step_count": step_count,
                "episode": episode,
            }
            action_type, arg = self._pre_dispatch_intervention(
                ep_data_sync, action_type, arg, reasoning,
            )

            if action_type == "FINAL":
                arg = self._postprocess_answer(arg, episode.question, step_context=step_context)
                if getattr(self.config, "domain", "web_search") in ("math", "code"):
                    episode.add_step(
                        Action(type="FINAL", query=arg),
                        Observation(),
                        thought=reasoning,
                        raw_output=response,
                    )
                episode.set_final(arg, reasoning=reasoning, raw_output=response)
                break

            elif action_type == "RETRY":
                # PF-injected directive: rejected this draft FINAL, ask the
                # model to redo. Pull feedback text from pending PF
                # context-injections and append as next observation.
                inj = list(getattr(self, "_pf_context_injections", []) or [])
                if hasattr(self, "_pf_context_injections"):
                    self._pf_context_injections.clear()
                feedback = "\n".join(inj) if inj else \
                    "Your draft FINAL was rejected. Redo the SOLUTION CHECKLIST: restate, format, branches, trace."
                messages.append({
                    "role": "user",
                    "content": f"Observation: {feedback}",
                })
                step_context.setdefault("action_history", []).append(
                    {"action_type": "RETRY", "arg": ""}
                )
                # If we've exhausted retries, accept the rejected draft as final.
                if step_count + 1 >= self.config.max_steps:
                    episode.set_final(arg, reasoning=reasoning, raw_output=response)
                    step_count += 1
                    break

            elif action_type == "SEARCH":
                if search_count >= self.config.max_search_calls:
                    messages.append({
                        "role": "user",
                        "content": "Observation: Search limit reached. Please provide your FINAL answer."
                    })
                else:
                    # Execute search via SerpAPI
                    results = self.env.search(arg)
                    search_count += 1

                    # Create observation
                    obs = Observation(
                        results=[r.to_dict() for r in results]
                    )

                    # Add to episode trace with thought and raw_output
                    episode.add_step(Action.search(arg), obs, thought=reasoning, raw_output=response)

                    # Update step context
                    step_context["search_count"] = search_count
                    step_context["empty_results"] = len(results) == 0
                    step_context["action_history"].append({"action_type": "SEARCH", "arg": arg})

                    # Format results for model (ReAct Observation format)
                    if results:
                        result_text = "\n".join([
                            f"- [{r.doc_id}] {r.title}: {r.snippet[:150]}..."
                            for r in results
                        ])
                        obs_text = f"Observation: Search results:\n{result_text}"
                        step_context["last_search_results_text"] = result_text
                    else:
                        obs_text = "Observation: No results found. Try a different query or provide FINAL answer."
                        step_context["last_search_results_text"] = ""

                    obs_text = self._format_observation(obs_text, "SEARCH", step_context)
                    messages.append({"role": "user", "content": obs_text})

            elif action_type == "READ":
                if read_count >= self.config.max_read_calls:
                    messages.append({
                        "role": "user",
                        "content": "Observation: Read limit reached. Please provide your FINAL answer."
                    })
                else:
                    # Execute read
                    content = self.env.read(arg)
                    read_count += 1

                    # Create observation
                    obs = Observation(content=content)

                    # Add to episode trace with thought and raw_output
                    episode.add_step(Action.read(arg), obs, thought=reasoning, raw_output=response)

                    # Update step context
                    step_context["has_read"] = True
                    step_context["read_count"] = read_count
                    step_context["action_history"].append({"action_type": "READ", "arg": arg})
                    step_context["all_read_contents"] += "\n" + (content or "")

                    # Add evidence
                    doc_info = self.env.documents.get(arg, {})
                    episode.evidence.append(Evidence(
                        doc_id=arg,
                        quote=content[:200] if content else "",
                        url=doc_info.get("url", ""),
                    ))

                    # Format content for model (ReAct Observation format, truncate if too long)
                    display_content = content[:3000] if len(content) > 3000 else content
                    obs_text = f"Observation: {display_content}"
                    obs_text = self._format_observation(obs_text, "READ", step_context)
                    messages.append({"role": "user", "content": obs_text})

            elif action_type == "SUMMARY":
                if summary_count >= self.config.max_summary_calls:
                    messages.append({
                        "role": "user",
                        "content": "Observation: Summary limit reached. Use READ instead or provide your FINAL answer."
                    })
                else:
                    # Execute summary via GPT-4o
                    summary_text = self.env.summarize(arg, question=episode.question)
                    summary_count += 1

                    # Create observation with summary field (saved to output)
                    obs = Observation(summary=summary_text)

                    # Add to episode trace with raw_output only (no thought for SUMMARY)
                    episode.add_step(Action(type="SUMMARY", doc_id=arg), obs, raw_output=response)

                    obs_text = f"Observation: [Summary] {summary_text}"
                    obs_text = self._format_observation(obs_text, "SUMMARY", step_context)
                    messages.append({"role": "user", "content": obs_text})

            step_count += 1
            step_context["step_count"] = step_count

        # Ensure we have a final answer
        if episode.final is None:
            # Force a final answer
            messages.append({
                "role": "user",
                "content": "Observation: Time limit reached. Please provide your best FINAL answer now using:\nThought: <your reasoning>\nAction: FINAL(\"your answer\")"
            })
            messages_snapshot = copy.deepcopy(messages)
            response = self._generate(messages)
            episode._trajectory.append({
                "step": len(episode._trajectory),
                "messages": messages_snapshot,
                "response": response,
            })
            action_result = self._parse_action(response)
            if action_result and action_result[0] == "FINAL":
                _, answer, reasoning = action_result
                answer = self._postprocess_answer(answer, episode.question, step_context=step_context)
                if getattr(self.config, "domain", "web_search") in ("math", "code"):
                    episode.add_step(
                        Action(type="FINAL", query=answer),
                        Observation(),
                        thought=reasoning,
                        raw_output=response,
                    )
                episode.set_final(answer, reasoning=reasoning, raw_output=response)
            else:
                episode.set_final("Unable to determine answer within budget.", raw_output=response)

        return episode

    def run_batch(
        self,
        samples: List[Dict[str, Any]],
        mode: str = "clean",
        model_name: str = "model",
        seed: int = None,
        verbose: bool = False,
        parallel_episodes: int = 1,
        async_batch: bool = True,
        min_batch_size: int = None,
        poll_timeout: float = 0.5,
    ) -> List[Episode]:
        """
        Run batch of episodes.

        Args:
            samples: List of samples with 'question', 'gold_answers', etc.
            mode: Evaluation mode
            model_name: Model name for tracking
            seed: Random seed
            verbose: Print progress
            parallel_episodes: Number of episodes to run in parallel (for vLLM)
            async_batch: Use async vectorized env (GPU never waits for tools)
            min_batch_size: Min episodes before GPU fires (default: parallel_episodes // 4)
            poll_timeout: Max seconds GPU waits for more episodes

        Returns:
            List of Episodes
        """
        # Use async processing if vLLM/API is available and parallel > 1
        # (async_batch param kept for backward compat but always uses async path)
        if self.use_api and parallel_episodes > 1:
            return self._run_batch_api_async(
                samples, mode, model_name, seed, verbose, parallel_episodes,
            )
        if self.use_vllm and parallel_episodes > 1:
            if min_batch_size is None:
                min_batch_size = max(1, parallel_episodes // 128)
            return self._run_batch_async(
                samples, mode, model_name, seed, verbose, parallel_episodes,
                min_batch_size, poll_timeout
            )

        # Sequential processing
        episodes = []

        for i, sample in enumerate(samples):
            if verbose:
                logger.info(f"Running sample {i+1}/{len(samples)}: {sample.get('question', '')[:50]}...")

            sample_id = sample.get("sample_id", sample.get("id", str(i)))
            try:
                episode = self.run_episode(
                    question=sample.get("question", ""),
                    gold_answers=sample.get("gold_answers", sample.get("answer", [])),
                    sample_id=sample_id,
                    mode=mode,
                    model_name=model_name,
                    seed=seed,
                )
            except Exception as e:
                logger.warning(
                    f"[{i+1}/{len(samples)}] Episode failed for sample {sample_id}: "
                    f"{type(e).__name__}: {e}"
                )
                episode = Episode(
                    question=sample.get("question", ""),
                    gold_answers=sample.get("gold_answers", sample.get("answer", [])),
                    sample_id=sample_id,
                    mode=mode,
                    model=model_name,
                    seed=seed,
                )
                episode.final = {"answer": f"[ERROR] {type(e).__name__}: {e}"}
            episodes.append(episode)

            if verbose:
                logger.info(f"  Answer: {episode.get_answer()[:100]}...")

        return episodes

    def _run_batch_api_async(
        self,
        samples: List[Dict[str, Any]],
        mode: str,
        model_name: str,
        seed: int,
        verbose: bool,
        max_concurrent: int,
    ) -> List[Episode]:
        """Run episodes concurrently using API model with thread pool.

        Unlike the vLLM EnvPool pattern, each episode runs its full
        sequential loop independently; concurrency is at the episode level.
        Each thread gets its own AgentRunner clone with an isolated
        ToolEnvironment to avoid shared state.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import copy

        logger.info(
            f"Running {len(samples)} samples via API "
            f"(provider={self.model.provider}, model={self.model.model_name}, "
            f"max_concurrent={max_concurrent})"
        )

        results = [None] * len(samples)

        def _run_one(i: int, sample: dict) -> tuple:
            # Create isolated environment per episode
            ep_env = ToolEnvironment(
                serpapi_key=self.config.serpapi_key,
                openai_key=self.config.openai_key,
                adv_wrapper=self.env.adv_wrapper,
                summary_model=self.config.summary_model,
                summary_provider=self.config.summary_provider,
                summary_api_key=self.config.summary_api_key,
            )
            sample_id = sample.get("sample_id", sample.get("id", str(i)))
            ep_env.set_current_sample(str(sample_id))

            # Create a lightweight runner copy with its own env
            # (shares model, tokenizer, config — only env is isolated)
            runner = copy.copy(self)
            runner.env = ep_env

            try:
                episode = runner.run_episode(
                    question=sample.get("question", ""),
                    gold_answers=sample.get("gold_answers", sample.get("answer", [])),
                    sample_id=sample_id,
                    mode=mode,
                    model_name=model_name,
                    seed=seed,
                )
            except Exception as e:
                # Handle API errors (e.g. content filtering) gracefully
                logger.warning(
                    f"[{i+1}/{len(samples)}] Episode failed for sample {sample_id}: "
                    f"{type(e).__name__}: {e}"
                )
                episode = Episode(
                    question=sample.get("question", ""),
                    gold_answers=sample.get("gold_answers", sample.get("answer", [])),
                    sample_id=sample_id,
                    mode=mode,
                    model=model_name,
                    seed=seed,
                )
                episode.final = {"answer": f"[ERROR] {type(e).__name__}: {e}"}
                return i, episode

            if verbose:
                logger.info(
                    f"[{i+1}/{len(samples)}] done: "
                    f"{sample.get('question', '')[:50]}... → "
                    f"{episode.get_answer()[:80]}"
                )
            return i, episode

        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {
                pool.submit(_run_one, i, s): i
                for i, s in enumerate(samples)
            }
            for future in as_completed(futures):
                idx, episode = future.result()
                results[idx] = episode

        return results

    def _run_batch_async(
        self,
        samples: List[Dict[str, Any]],
        mode: str,
        model_name: str,
        seed: int,
        verbose: bool,
        parallel_episodes: int,
        min_batch_size: int,
        poll_timeout: float,
    ) -> List[Episode]:
        """
        Async vectorized environment loop (EnvPool / Sample Factory pattern).

        Maps the agent-tool interaction to the RL env-policy loop::

            pool.reset(episodes)              # initial observations
            while not pool.all_done:
                batch   = pool.recv()         # block for ready episodes
                outputs = gpu_generate(batch) # batched policy inference
                pool.send(batch, outputs)     # dispatch tool calls (async)

        Tool calls execute in a thread pool between recv/send cycles.
        The GPU never blocks on individual tool calls.
        """
        from vllm import SamplingParams

        use_gpt_summary = bool(self.config.openai_key or os.environ.get("OPENAI_API_KEY"))
        use_examples = self.config.model_type == "base"
        # Cap GPU batch size to prevent monster first-batch latency.
        # With 384 episodes, generate(384) blocks 10-15s with zero tools running.
        # Capping at 128 staggers the pipeline: while GPU processes batch 1,
        # tools from previous iterations are completing → continuous overlap.
        max_batch = min(128, parallel_episodes)

        logger.info(
            f"Running {len(samples)} samples ASYNC (parallel={parallel_episodes}, "
            f"min_batch={min_batch_size}, max_batch={max_batch}, "
            f"poll_timeout={poll_timeout}s, gpt_summary={use_gpt_summary})"
        )

        # ── Phase 1: Initialize all episodes ──────────────────────────
        # Run init in parallel — `_init_async_ep_data` calls PFSelector
        # (GPT-4o, ~1s each) per episode; serial init would idle the GPU for
        # parallel_episodes seconds before Phase 3's generate() can fire.
        shared_cache = SharedToolCache()

        def _build_ep_data(i_sample):
            i, sample = i_sample
            episode = Episode(
                question=sample.get("question", ""),
                gold_answers=sample.get("gold_answers", sample.get("answer", [])),
                sample_id=sample.get("sample_id", sample.get("id", str(i))),
                mode=mode,
                model=model_name,
                seed=seed,
            )

            messages = self._build_async_messages(episode, mode, use_examples)

            ep_env = ToolEnvironment(
                serpapi_key=self.config.serpapi_key,
                openai_key=self.config.openai_key,
                adv_wrapper=self.env.adv_wrapper,
                local_model=self.env.summarizer.local_model,
                local_tokenizer=self.env.summarizer.local_tokenizer,
                shared_cache=shared_cache,
                summary_model=self.config.summary_model,
                summary_provider=self.config.summary_provider,
                summary_api_key=self.config.summary_api_key,
            )
            if use_gpt_summary:
                ep_env.summarizer.force_gpt = True
            else:
                ep_env.summarizer = self.env.summarizer
            ep_env.set_current_sample(str(sample.get("sample_id", sample.get("id", str(i)))))

            ep_data = {
                "episode": episode,
                "messages": messages,
                "step_count": 0,
                "search_count": 0,
                "read_count": 0,
                "summary_count": 0,
                "done": False,
                "env": ep_env,
                "idx": i,
                # PFs gate on step_context — populate the minimum that the
                # _is_code_final / similar gates need (domain, max_steps,
                # question text). Sync path builds this in _run_tool_episode;
                # async path never did, which made code PFs no-op silently.
                "step_context": {
                    "domain": getattr(self.config, "domain", "web_search"),
                    "max_steps": self.config.max_steps,
                    "question": episode.question,
                    "step_count": 0,
                    "search_count": 0,
                    "read_count": 0,
                    "has_read": False,
                    "action_history": [],
                },
            }
            self._init_async_ep_data(ep_data, sample, mode)
            return i, ep_data

        init_workers = max(1, min(len(samples), parallel_episodes))
        all_ep_data = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=init_workers) as init_pool:
            for i, ep_data in init_pool.map(_build_ep_data, enumerate(samples)):
                all_ep_data[i] = ep_data

        # ── Phase 2: Create env pool ──────────────────────────────────
        env_pool = AsyncToolEnvPool(self, max_workers=parallel_episodes)
        env_pool.reset(all_ep_data)

        sp_kwargs = dict(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature if self.config.do_sample else 0,
            top_p=self.config.top_p if self.config.do_sample else 1.0,
        )
        if self.config.seed is not None:
            sp_kwargs["seed"] = self.config.seed
        sampling_params = SamplingParams(**sp_kwargs)

        # ── Phase 3: recv → infer → send loop (EnvPool / Sample Factory) ─
        #
        # Optimizations:
        #   1. Pre-computed prompts (worker threads overlap with GPU).
        #   2. Pre-fetch recv() in background thread during generate() —
        #      hides recv() latency so GPU never waits for the ready queue.
        #   3. max_batch capped at 128 to prevent pipeline bubbles.
        #   4. Done-callbacks in separate pool (never stall GPU).
        recv_pool = ThreadPoolExecutor(max_workers=1)
        try:
            iteration = 0
            gpu_busy_time = 0.0
            fallback_prompt_count = 0
            loop_start = time.monotonic()

            # Initial batch (from reset — all episodes are pre-queued)
            batch = env_pool.recv(
                min_batch=1, timeout=poll_timeout, max_batch=max_batch,
            )

            while True:
                iteration += 1

                # ── Handle empty batch ────────────────────────────
                if not batch:
                    if env_pool.all_done:
                        break
                    env_pool.wait(timeout=1.0)
                    batch = env_pool.recv(
                        min_batch=1, timeout=poll_timeout, max_batch=max_batch,
                    )
                    continue

                done = env_pool.done_count
                active = len(samples) - done
                if active <= 0:
                    break

                # ── Pre-fetch: start recv() in background while GPU works ─
                # This overlaps recv() wait with generate(), hiding latency.
                pending = env_pool.pending_count
                completion_ratio = done / len(samples) if len(samples) > 0 else 1.0
                if completion_ratio < 0.9:
                    pf_timeout = min(0.5, poll_timeout + 0.001 * pending)
                else:
                    pf_timeout = max(0.1, poll_timeout / 4)
                prefetch = recv_pool.submit(
                    env_pool.recv, 1, pf_timeout, max_batch,
                )

                # ── Collect pre-computed prompts (fast path) ──────
                t0 = time.monotonic()
                batch_prompts = []
                for ep_data in batch:
                    prompt = ep_data.pop("_prompt", None)
                    if prompt is None:
                        # Fallback: compute prompt on the fly (should be rare)
                        fallback_prompt_count += 1
                        msgs = self._truncate_messages(ep_data["messages"])
                        if hasattr(self.tokenizer, 'apply_chat_template'):
                            prompt = self.tokenizer.apply_chat_template(
                                msgs, tokenize=False, add_generation_prompt=True,
                            )
                        else:
                            prompt = ""
                            for msg in msgs:
                                prompt += f"<|{msg['role']}|>\n{msg['content']}\n"
                            prompt += "<|assistant|>\n"
                    batch_prompts.append(prompt)

                # ── Filter out prompts exceeding max_model_len ────
                _max_tokens = getattr(self.config, "vllm_max_model_len", None) or 32768
                valid_idx = []
                for i, p in enumerate(batch_prompts):
                    tok_ids = self.tokenizer.encode(p) if hasattr(self.tokenizer, "encode") else p
                    tok_len = len(tok_ids) if isinstance(tok_ids, list) else len(p) // 3
                    if tok_len >= _max_tokens:
                        # Try emergency truncation: keep only system + question + last 2 exchanges
                        msgs = batch[i]["messages"]
                        if len(msgs) > 5:
                            emergency_msgs = msgs[:2] + msgs[-2:]  # system + question + last pair
                            if hasattr(self.tokenizer, 'apply_chat_template'):
                                p2 = self.tokenizer.apply_chat_template(
                                    emergency_msgs, tokenize=False, add_generation_prompt=True,
                                )
                            else:
                                p2 = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in emergency_msgs)
                                p2 += "<|assistant|>\n"
                            tok2 = self.tokenizer.encode(p2) if hasattr(self.tokenizer, "encode") else p2
                            tok2_len = len(tok2) if isinstance(tok2, list) else len(p2) // 3
                            if tok2_len < _max_tokens:
                                batch_prompts[i] = p2
                                batch[i]["messages"] = emergency_msgs
                                valid_idx.append(i)
                                logger.info(
                                    f"Emergency truncation: {tok_len} → {tok2_len} tokens"
                                )
                                continue
                        logger.warning(
                            f"Prompt too long ({tok_len} tokens >= {_max_tokens}), "
                            f"force-finishing episode: {batch[i].get('question', '')[:60]}"
                        )
                        env_pool._finish(batch[i], answer="[TRUNCATED — prompt exceeded max length]")
                    else:
                        valid_idx.append(i)

                if not valid_idx:
                    batch = prefetch.result()
                    continue

                valid_batch = [batch[i] for i in valid_idx]
                valid_prompts = [batch_prompts[i] for i in valid_idx]

                # ── GPU batch inference (timed) ───────────────────
                t1 = time.monotonic()
                outputs = self.model.llm.generate(
                    valid_prompts, sampling_params, lora_request=self.model.lora_request,
                )
                t2 = time.monotonic()
                gpu_busy_time += t2 - t1

                # Dispatch actions to env pool (non-blocking)
                env_pool.send(valid_batch, outputs)
                t3 = time.monotonic()

                if verbose:
                    logger.info(
                        f"Iter {iteration}: batch={len(batch)}, "
                        f"prep={t1-t0:.3f}s gen={t2-t1:.3f}s send={t3-t2:.3f}s, "
                        f"pending={env_pool.pending_count}, "
                        f"done={env_pool.done_count}/{len(samples)}"
                    )

                # ── Get pre-fetched batch (likely ready — recv overlapped generate)
                batch = prefetch.result()

            # ── GPU utilization report ────────────────────────────
            total_elapsed = time.monotonic() - loop_start
            if total_elapsed > 0:
                gpu_util_pct = gpu_busy_time / total_elapsed * 100
                logger.info(
                    f"Async batch done: {iteration} iters, "
                    f"GPU util={gpu_util_pct:.1f}% "
                    f"({gpu_busy_time:.1f}s / {total_elapsed:.1f}s), "
                    f"{len(samples)} episodes"
                    + (f", {fallback_prompt_count} prompt fallbacks"
                       if fallback_prompt_count else "")
                )
                cache_hits, cache_misses = shared_cache.stats
                if cache_hits + cache_misses > 0:
                    logger.info(
                        f"Shared cache: {cache_hits} hits, {cache_misses} misses "
                        f"({cache_hits / (cache_hits + cache_misses) * 100:.1f}% hit rate)"
                    )

        finally:
            recv_pool.shutdown(wait=False)
            env_pool.shutdown()

        # ── Phase 4: Collect results in original order ────────────────
        result_map = {}
        for ep_data in all_ep_data:
            if ep_data["episode"].final is None:
                ep_data["episode"].set_final("Unable to determine answer within budget.")
            # Transfer trajectory from ep_data to episode
            ep_data["episode"]._trajectory = ep_data.get("_trajectory", [])
            result_map[ep_data["idx"]] = ep_data["episode"]

        return [result_map[i] for i in range(len(samples))]

    def _execute_tool_call(
        self,
        ep_data: Dict[str, Any],
        action_type: str,
        arg: str,
        thought: Optional[str] = None,
        raw_output: Optional[str] = None,
    ) -> None:
        """Execute a single tool call (designed for thread pool execution)."""
        episode = ep_data["episode"]
        env = ep_data["env"]

        if action_type == "SEARCH":
            if ep_data["search_count"] >= self.config.max_search_calls:
                ep_data["messages"].append({
                    "role": "user",
                    "content": "Observation: Search limit reached. Please provide your FINAL answer."
                })
            else:
                results = env.search(arg)
                ep_data["search_count"] += 1

                obs = Observation(results=[r.to_dict() for r in results])
                episode.add_step(Action.search(arg), obs, thought=thought, raw_output=raw_output)

                if results:
                    result_text = "\n".join([
                        f"- [{r.doc_id}] {r.title}: {r.snippet[:150]}..."
                        for r in results
                    ])
                    ep_data["messages"].append({
                        "role": "user",
                        "content": f"Observation: Search results:\n{result_text}"
                    })
                else:
                    ep_data["messages"].append({
                        "role": "user",
                        "content": "Observation: No results found. Try a different query or provide FINAL answer."
                    })

        elif action_type == "READ":
            if ep_data["read_count"] >= self.config.max_read_calls:
                ep_data["messages"].append({
                    "role": "user",
                    "content": "Observation: Read limit reached. Please provide your FINAL answer."
                })
            else:
                content = env.read(arg)
                ep_data["read_count"] += 1

                obs = Observation(content=content)
                episode.add_step(Action.read(arg), obs, thought=thought, raw_output=raw_output)

                doc_info = env.documents.get(arg, {})
                episode.evidence.append(Evidence(
                    doc_id=arg,
                    quote=content[:200] if content else "",
                    url=doc_info.get("url", ""),
                ))

                display_content = content[:3000] if len(content) > 3000 else content
                ep_data["messages"].append({
                    "role": "user",
                    "content": f"Observation: {display_content}"
                })

        elif action_type == "SUMMARY":
            if ep_data["summary_count"] >= self.config.max_summary_calls:
                ep_data["messages"].append({
                    "role": "user",
                    "content": "Observation: Summary limit reached. Use READ instead or provide your FINAL answer."
                })
            else:
                summary_text = env.summarize(arg, question=episode.question)
                ep_data["summary_count"] += 1

                # Create observation with summary field (saved to output)
                obs = Observation(summary=summary_text)
                episode.add_step(Action(type="SUMMARY", doc_id=arg), obs, raw_output=raw_output)

                ep_data["messages"].append({
                    "role": "user",
                    "content": f"Observation: [Summary] {summary_text}"
                })

        ep_data["step_count"] += 1
        self._post_tool_observation(ep_data, action_type)

    # ========================================================================
    # Async batch hooks (overridable by subclasses)
    # ========================================================================

    def _build_async_messages(self, episode, mode, use_examples):
        """Hook: build initial message list for an async episode. Subclasses can override to customize system prompt."""
        return [
            {"role": "system", "content": self._build_system_prompt(with_examples=use_examples)},
            {"role": "user", "content": self._build_user_prompt(episode)},
        ]

    def _build_user_prompt(self, episode) -> str:
        """Build the user-message body for one episode.

        Code domain mirrors `baseline/scripts/prompt.py::build_user_prompt`
        — the only delta between this and direct-answer baseline is the
        system-prompt-level instruction to think inside `<think>...</think>`.
        We don't need to re-append starter_code separately because every
        HumanEval+/MBPP+/BCB row already carries the function signature in
        its `question` field.

        Other domains keep the legacy `Question: {question}` form.
        """
        q = episode.question
        domain = getattr(self.config, "domain", "web_search")
        if domain != "code":
            return f"Question: {q}"
        return (
            "Problem:\n"
            f"{q.strip()}\n"
            "\nThink through the problem step-by-step inside <think>...</think>, "
            "then provide the final implementation in a single ```python ... ``` block."
        )

    def _init_async_ep_data(self, ep_data, sample, mode):
        """Hook: called after ep_data is built, before enqueuing. Subclasses can add custom fields."""
        pass

    def _format_observation(self, obs_text: str, action_type: str, step_context: Dict[str, Any]) -> str:
        """Hook: modify observation text before appending to messages (sync path).

        Subclasses can override to inject phase-gated skill instructions.

        Args:
            obs_text: The raw observation text
            action_type: "SEARCH", "READ", or "SUMMARY"
            step_context: Dict with step_count, has_read, search_count, etc.

        Returns:
            Modified observation text
        """
        return obs_text

    def _post_tool_observation(self, ep_data, action_type):
        """Hook: called after tool observation is appended to messages. Subclasses can modify the last message (e.g. inject reminder)."""
        pass

    def _pre_dispatch_intervention(self, ep_data, action_type, arg, reasoning):
        """Hook: may override action_type and arg before tool dispatch.

        Subclasses can override to inject programmatic interventions.

        Args:
            ep_data: Episode data dict (sync path: constructed ad-hoc; async path: full ep_data).
            action_type: Parsed action type ("SEARCH", "READ", "FINAL", etc.)
            arg: Parsed action argument.
            reasoning: Model's reasoning text.

        Returns:
            Tuple of (action_type, arg) — possibly modified.
        """
        return action_type, arg

    def _postprocess_answer(self, answer, question="", step_context=None):
        """Hook: post-process final answer before storing.

        Subclasses can override to apply rule-based answer cleanup.

        Args:
            answer: Raw answer string from model.
            question: Original question text.
            step_context: Optional step context dict (for skill-driven postprocessing).

        Returns:
            Cleaned answer string.
        """
        return answer

    def _on_async_episode_done(self, ep_data):
        """Hook: called when an async episode finishes. Subclasses can run post-episode processing (e.g. reflection)."""
        pass


def create_adversarial_wrapper(
    attack_type: str = "noise",
    strength: float = 0.3,
) -> Callable:
    """
    Create an adversarial wrapper function.

    Args:
        attack_type: Type of attack ("noise", "reorder", "irrelevant")
        strength: Attack strength (0-1)

    Returns:
        Wrapper function
    """
    import random

    def wrapper(data: Any, action_type: str) -> Any:
        if attack_type == "noise" and action_type == "read":
            # Add noise to read content (inject ~30% noise into the document)
            if isinstance(data, str) and random.random() < strength:
                noise_sentences = [
                    "This information has not been independently verified.",
                    "Some experts dispute these claims.",
                    "Alternative sources provide different perspectives on this topic.",
                    "The accuracy of this data cannot be guaranteed.",
                    "Recent developments may have changed this information.",
                    "This claim is contested by multiple authorities.",
                    "Readers should verify this information independently.",
                    "Historical records show conflicting accounts.",
                    "This represents only one viewpoint among many.",
                    "The methodology behind this claim has been questioned.",
                ]
                # Split content into sentences and inject noise at ~30% of positions
                sentences = data.split('. ')
                noise_ratio = 0.3  # 30% noise injection
                num_noise = max(1, int(len(sentences) * noise_ratio))

                # Insert noise sentences at random positions
                for _ in range(num_noise):
                    noise_text = random.choice(noise_sentences)
                    insert_pos = random.randint(0, len(sentences))
                    sentences.insert(insert_pos, noise_text)

                return '. '.join(sentences)

        elif attack_type == "reorder" and action_type == "search":
            # Reorder search results (move relevant to back)
            if isinstance(data, list) and len(data) > 1 and random.random() < strength:
                random.shuffle(data)

        elif attack_type == "irrelevant" and action_type == "search":
            # Inject irrelevant/distracting search results
            if isinstance(data, list) and len(data) > 0 and random.random() < strength:
                irrelevant_results = [
                    SearchResult(
                        doc_id="irrelevant_doc_1",
                        title="Top 10 Vacation Destinations for 2024",
                        url="https://example.com/travel",
                        snippet="Discover the best places to visit this year. From tropical beaches to mountain retreats...",
                    ),
                    SearchResult(
                        doc_id="irrelevant_doc_2",
                        title="How to Make Perfect Pasta at Home",
                        url="https://example.com/recipes",
                        snippet="Learn the secrets of Italian chefs for cooking authentic pasta dishes...",
                    ),
                    SearchResult(
                        doc_id="irrelevant_doc_3",
                        title="Latest Smartphone Reviews and Comparisons",
                        url="https://example.com/tech",
                        snippet="We compare the newest phones on the market to help you choose...",
                    ),
                    SearchResult(
                        doc_id="irrelevant_doc_4",
                        title="Beginner's Guide to Indoor Gardening",
                        url="https://example.com/gardening",
                        snippet="Start your own indoor garden with these simple tips and plant recommendations...",
                    ),
                    SearchResult(
                        doc_id="irrelevant_doc_5",
                        title="Understanding Your Pet's Body Language",
                        url="https://example.com/pets",
                        snippet="Decode what your dog or cat is trying to tell you through their behavior...",
                    ),
                ]
                # Insert 1-3 irrelevant results at top positions
                num_inject = random.randint(1, min(3, len(irrelevant_results)))
                selected = random.sample(irrelevant_results, num_inject)
                for i, fake_result in enumerate(selected):
                    data.insert(i, fake_result)

        return data

    return wrapper


def create_pregenerated_adversarial_wrapper(
    distractor_data: Dict[str, Any],
    attack_type: str = "conflict_l1",
) -> "callable":
    """
    Create an adversarial wrapper that injects pre-generated distractors.

    Args:
        distractor_data: Mapping of sample_id -> distractor info (loaded from pre-generated JSONL).
        attack_type: One of "conflict_l1", "conflict_l2", "conflict_l3", "outdated".

    Returns:
        Wrapper function. The wrapper uses the current_sample_id attribute
        set on the ToolEnvironment to look up per-sample distractors.
    """

    def wrapper(data: Any, action_type: str, current_sample_id: str = None) -> Any:
        if action_type != "search" or not isinstance(data, list) or current_sample_id is None:
            return data

        sample_info = distractor_data.get(current_sample_id)
        if sample_info is None:
            return data

        distractors_to_inject = []

        if attack_type.startswith("conflict_l"):
            # Extract level number
            level = int(attack_type.split("conflict_l")[1])
            for d in sample_info.get("distractors", []):
                if d.get("level") == level:
                    distractors_to_inject.append(SearchResult(
                        doc_id=f"adv_conflict_l{level}_{d.get('title', 'distractor')[:20]}",
                        title=d["title"],
                        url=d["url"],
                        snippet=d["snippet"],
                    ))

        elif attack_type == "outdated":
            for d in sample_info.get("distractors", []):
                distractors_to_inject.append(SearchResult(
                    doc_id=f"adv_outdated_{d.get('title', 'distractor')[:20]}",
                    title=d["title"],
                    url=d["url"],
                    snippet=d["snippet"],
                ))

        # Insert distractors at the top of search results
        for i, fake_result in enumerate(distractors_to_inject):
            data.insert(i, fake_result)

        return data

    return wrapper
