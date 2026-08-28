"""
Tool implementations for evaluation.

Provides real tool calls:
- SEARCH: Use SerpAPI for Google search
- READ: Fetch and extract content from URLs
- SUMMARY: Use local HuggingFace model or GPT-4o API for summarization
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging
import os
import re
import threading
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    doc_id: str
    title: str
    url: str
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }


class SharedToolCache:
    """Thread-safe cross-episode cache for SerpAPI/HTTP results.

    Avoids redundant network calls when multiple episodes search for the same
    query or read the same URL.  Designed to be created once per batch run and
    passed to every per-episode ``ToolEnvironment``.
    """

    def __init__(self):
        self._search_cache: Dict[str, List["SearchResult"]] = {}
        self._web_cache: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # -- search ----------------------------------------------------------------

    def get_search(self, key: str):
        with self._lock:
            result = self._search_cache.get(key)
            if result is not None:
                self._hits += 1
            else:
                self._misses += 1
            return result

    def put_search(self, key: str, results):
        with self._lock:
            self._search_cache[key] = results

    # -- web -------------------------------------------------------------------

    def get_web(self, url: str):
        with self._lock:
            result = self._web_cache.get(url)
            if result is not None:
                self._hits += 1
            else:
                self._misses += 1
            return result

    def put_web(self, url: str, content: str):
        with self._lock:
            self._web_cache[url] = content

    # -- stats -----------------------------------------------------------------

    @property
    def stats(self):
        with self._lock:
            return self._hits, self._misses


class TeacherSimulatedSearch:
    """
    Search tool that uses a PF helper (gpt-4o) to generate plausible,
    fact-grounded "search results" — each with embedded page content — in
    a single API call. Intended for TRAINING rollouts so the GPU doesn't
    sit idle waiting on real SerpAPI / URL fetches.

    Returned SearchResult carries a synthetic `url` like
    "teacher://doc_0". The full content for each result is stashed in
    `self.content_cache[url]` so a subsequent READ(doc_id) returns it
    without any extra network / API call.

    Activated when env var USE_TEACHER_SEARCH is set (and OPENAI_API_KEY
    is available). Eval path leaves it off so real SerpAPI keeps working.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2000,
        snippet_mode: Optional[bool] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("TEACHER_SEARCH_MODEL", "")
        # Snippet mode: return 1 result with a ~200-token self-contained snippet
        # in the `snippet` field (no separate long `content`). Activated for
        # cheap single-step training rollout. The `max_tokens` budget is
        # cut down to match.
        if snippet_mode is None:
            snippet_mode = os.environ.get("TEACHER_SEARCH_SNIPPET_MODE", "").lower() in ("1", "true", "yes")
        self.snippet_mode = snippet_mode
        self.max_tokens = 300 if snippet_mode else max_tokens
        # url → full page content, populated during search(); consumed by WebReader shim
        self.content_cache: Dict[str, str] = {}
        if not self.api_key:
            logger.warning(
                "TeacherSimulatedSearch: OPENAI_API_KEY not set — search will return empty"
            )
        if self.snippet_mode:
            logger.info("TeacherSimulatedSearch: snippet_mode=True (1 result, ~200-token snippet)")

    def search(self, query: str, max_results: int = 8) -> List[SearchResult]:
        if not self.api_key:
            return []
        if self.snippet_mode:
            # Single snippet, all factual content packed into `snippet` field.
            # Cheap path for training rollout: agent gets enough context to
            # reach a FINAL answer in one more step without needing READ.
            prompt = (
                f'Answer the following query with a concise factual snippet (≤200 tokens) '
                f'containing specific facts — dates, names, numbers — that would help a '
                f'QA agent answer questions about it. Plain prose, no lists/markup.\n\n'
                f'Query: "{query}"\n\n'
                f'Output JSON: {{"snippet": "…"}}. If unsure about any fact, omit it rather '
                f'than fabricate.'
            )
        else:
            prompt = (
                f'You are simulating Google search results for a multi-hop QA agent.\n'
                f'Query: "{query}"\n\n'
                f'Return {max_results} search results as a JSON object with a single key '
                f'"results" holding an array. Each item must have:\n'
                f'  - "title": realistic article/page title\n'
                f'  - "url":   plausible url (wikipedia.org, official site, news outlet, etc.)\n'
                f'  - "snippet": 1-2 sentence summary (≤40 words)\n'
                f'  - "content": 200-500 word page content with specific facts the agent '
                f'could READ to answer questions about this query. Include dates, names, '
                f'numbers, and source-style phrasing. NO placeholder text.\n\n'
                f'Prefer accuracy over creativity. If unsure, omit the item rather than '
                f'fabricate. Output JSON only, no prose.'
            )
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You simulate factual search results. Output strict JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            import json as _json
            raw = response.choices[0].message.content or "{}"
            data = _json.loads(raw)

            if self.snippet_mode:
                # One result with the whole factual snippet in-line. No READ
                # needed afterwards — the agent can go directly to FINAL.
                snippet_text = str(data.get("snippet") or "").strip()
                if not snippet_text:
                    return []
                doc_id = "doc_0"
                url = f"PF helper://{doc_id}"
                results = [SearchResult(
                    doc_id=doc_id, title=query, url=url, snippet=snippet_text,
                )]
                # Also cache as "content" so any accidental READ(doc_0)
                # returns the same text instead of an empty string.
                self.content_cache[url] = snippet_text
                self.content_cache[doc_id] = snippet_text
                logger.info(
                    f"TeacherSimulatedSearch (snippet_mode) returned 1 result "
                    f"({len(snippet_text.split())} words) for: {query}"
                )
                return results

            items = data.get("results") if isinstance(data, dict) else None
            if not isinstance(items, list):
                items = []
            results: List[SearchResult] = []
            for i, item in enumerate(items[:max_results]):
                if not isinstance(item, dict):
                    continue
                doc_id = f"doc_{i}"
                url = str(item.get("url") or f"PF helper://{doc_id}").strip()
                title = str(item.get("title") or "").strip()
                snippet = str(item.get("snippet") or "").strip()
                content = str(item.get("content") or "").strip()
                if not title and not snippet:
                    continue
                results.append(SearchResult(
                    doc_id=doc_id, title=title, url=url, snippet=snippet,
                ))
                # Stash full content under the url AND the synthetic doc_id
                # — the shim reader below will look up either.
                if content:
                    self.content_cache[url] = content
                    self.content_cache[doc_id] = content
            logger.info(
                f"TeacherSimulatedSearch returned {len(results)} results for: {query}"
            )
            return results
        except Exception as e:
            logger.warning(f"TeacherSimulatedSearch failed ({e}); returning empty")
            return []


class TeacherSimulatedWebReader:
    """Shim reader that pulls content from TeacherSimulatedSearch.content_cache.

    Takes precedence over WebReader when PF helper-sim mode is on: the URL
    is only synthetic so we never hit the network — the content was already
    generated during SEARCH.
    """

    def __init__(self, teacher_search: "TeacherSimulatedSearch"):
        self._teacher = teacher_search
        self._cache: Dict[str, str] = {}

    def read(self, url: str) -> str:
        content = self._teacher.content_cache.get(url, "")
        if content:
            return content
        # Fallback: if the url isn't in cache, return an explicit empty-ish
        # signal rather than fabricating more via another API call.
        return f"[PF helper-sim] no content cached for {url}"


class SerpAPISearch:
    """
    Search tool using SerpAPI.

    Requires SERPAPI_API_KEY environment variable.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY")
        if not self.api_key:
            logger.warning("SERPAPI_API_KEY not set. Search will use mock results.")

    def search(self, query: str, max_results: int = 8) -> List[SearchResult]:
        """
        Execute Google search via SerpAPI.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        if not self.api_key:
            return self._mock_search(query, max_results)

        try:
            from serpapi import GoogleSearch

            params = {
                "q": query,
                "api_key": self.api_key,
                "num": max_results,
            }

            search = GoogleSearch(params)
            results = search.get_dict()

            search_results = []
            organic_results = results.get("organic_results", [])

            for i, item in enumerate(organic_results[:max_results]):
                search_results.append(SearchResult(
                    doc_id=f"doc_{i}",
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                ))

            logger.info(f"SerpAPI search returned {len(search_results)} results for: {query}")
            return search_results

        except ImportError:
            logger.warning("serpapi not installed. Using mock search.")
            return self._mock_search(query, max_results)
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            return self._mock_search(query, max_results)

    def _mock_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Return mock search results when API is unavailable."""
        return [
            SearchResult(
                doc_id="mock_0",
                title=f"Mock result for: {query}",
                url="https://example.com/mock",
                snippet=f"This is a mock search result for query: {query}. Set SERPAPI_API_KEY for real results.",
            )
        ]


class WebReader:
    """
    Read and extract content from web URLs.
    """

    def __init__(self, timeout: int = 10, max_content_length: int = 20000,
                 shared_cache: "SharedToolCache | None" = None):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self._cache: Dict[str, str] = {}
        self._shared_cache = shared_cache

    def read(self, url: str) -> str:
        """
        Fetch and extract text content from URL.

        Args:
            url: URL to fetch

        Returns:
            Extracted text content
        """
        # Check per-instance cache
        if url in self._cache:
            return self._cache[url]

        # Check shared cross-episode cache
        if self._shared_cache is not None:
            cached = self._shared_cache.get_web(url)
            if cached is not None:
                self._cache[url] = cached
                return cached

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ASearcher/1.0)"
            }

            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            # Extract text content
            content = self._extract_text(response.text)

            # Truncate if too long
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "..."

            # Cache result (per-instance + shared)
            self._cache[url] = content
            if self._shared_cache is not None:
                self._shared_cache.put_web(url, content)

            logger.info(f"Read {len(content)} chars from: {url}")
            return content

        except Exception as e:
            logger.error(f"Failed to read URL {url}: {e}")
            return f"Error reading URL: {e}"

    def _extract_text(self, html: str) -> str:
        """Extract plain text from HTML."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Get text
            text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.split("\n")]
            text = "\n".join(line for line in lines if line)

            return text

        except ImportError:
            # Fallback: simple regex-based extraction
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()


class GPTSummarizer:
    """
    Summarization tool using API (OpenAI or Anthropic) or local HuggingFace model.

    Can use:
    - OpenAI API (requires OPENAI_API_KEY)
    - Anthropic API (requires ANTHROPIC_API_KEY)
    - Local HuggingFace model (if provided)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "",
        provider: str = "openai",
        max_tokens: int = 500,
        local_model: Optional[Any] = None,
        local_tokenizer: Optional[Any] = None,
        force_gpt: bool = False,
    ):
        self.provider = provider
        _env_vars = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}
        self.api_key = api_key or os.environ.get(_env_vars.get(provider, "OPENAI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.local_model = local_model
        self.local_tokenizer = local_tokenizer
        self.force_gpt = force_gpt
        self._generate_lock = threading.Lock()

        if self.local_model is not None:
            logger.info("Using local model for summarization")
        elif not self.api_key:
            logger.warning("No API key set and no local model provided. Summarization will use truncation.")

    def set_local_model(self, model: Any, tokenizer: Any) -> None:
        """Set local model and tokenizer for summarization."""
        self.local_model = model
        self.local_tokenizer = tokenizer
        logger.info("Local model set for summarization")

    def summarize(
        self,
        content: str,
        question: Optional[str] = None,
        max_length: int = 500,
    ) -> str:
        """
        Summarize content using local model, GPT-4o, or truncation.

        Args:
            content: Content to summarize
            question: Optional question to focus the summary
            max_length: Maximum summary length

        Returns:
            Summary text
        """
        # When force_gpt is set, skip local model to avoid vLLM lock contention
        if self.force_gpt and self.api_key:
            return self._summarize_with_api(content, question, max_length)
        # Priority: local model > GPT API > truncation
        if self.local_model is not None and self.local_tokenizer is not None:
            return self._summarize_with_local_model(content, question, max_length)
        elif self.api_key:
            return self._summarize_with_api(content, question, max_length)
        else:
            return self._simple_truncate(content, max_length)

    def _summarize_with_local_model(
        self,
        content: str,
        question: Optional[str] = None,
        max_length: int = 500,
    ) -> str:
        """Summarize using local HuggingFace model or vLLM wrapper."""
        try:
            if question:
                prompt = f"""Summarize the following content, focusing on information relevant to answering this question: "{question}"

Content:
{content[:4000]}

Provide a concise summary (max {max_length} words) that captures the key information relevant to the question."""
            else:
                prompt = f"""Summarize the following content concisely (max {max_length} words):

{content[:4000]}"""

            messages = [
                {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
                {"role": "user", "content": prompt},
            ]

            # Apply chat template
            if hasattr(self.local_tokenizer, 'apply_chat_template'):
                input_text = self.local_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                input_text = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"

            # Check if this is a vLLM model wrapper
            from .model_loader import VLLMModelWrapper
            if isinstance(self.local_model, VLLMModelWrapper):
                from vllm import SamplingParams
                sampling_params = SamplingParams(
                    max_tokens=self.max_tokens,
                    temperature=0.3,
                    top_p=0.9,
                )
                # vLLM LLM.generate() is not thread-safe; serialize with lock
                with self._generate_lock:
                    outputs = self.local_model.llm.generate(
                        [input_text], sampling_params,
                        lora_request=self.local_model.lora_request,
                    )
                summary = outputs[0].outputs[0].text.strip()
            else:
                # HuggingFace model path
                import torch
                inputs = self.local_tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                ).to(self.local_model.device)

                with torch.no_grad():
                    outputs = self.local_model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=0.3,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.local_tokenizer.pad_token_id,
                        eos_token_id=self.local_tokenizer.eos_token_id,
                    )

                summary = self.local_tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()

            logger.info(f"Local model summarized {len(content)} chars to {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Local model summarization failed: {e}")
            return self._simple_truncate(content, max_length)

    def _summarize_with_api(
        self,
        content: str,
        question: Optional[str] = None,
        max_length: int = 500,
    ) -> str:
        """Summarize using API (OpenAI or Anthropic)."""
        if question:
            prompt = (
                f'Summarize the following content, focusing on information relevant '
                f'to answering this question: "{question}"\n\n'
                f'Content:\n{content[:8000]}\n\n'
                f'Provide a concise summary (max {max_length} words) that captures '
                f'the key information relevant to the question.'
            )
        else:
            prompt = (
                f'Summarize the following content concisely (max {max_length} words):'
                f'\n\n{content[:8000]}'
            )

        system_msg = "You are a helpful assistant that summarizes text concisely."

        try:
            if self.provider == "anthropic":
                from anthropic import Anthropic
                client = Anthropic(api_key=self.api_key)
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_msg,
                    messages=[{"role": "user", "content": prompt}],
                )
                summary = response.content[0].text.strip()
            elif self.provider == "google":
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model=self.model,
                    contents=[types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    )],
                    config=types.GenerateContentConfig(
                        system_instruction=system_msg,
                        max_output_tokens=self.max_tokens,
                        temperature=0.3,
                    ),
                )
                summary = response.text.strip()
            else:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.3,
                )
                summary = response.choices[0].message.content.strip()

            logger.info(f"{self.provider}/{self.model} summarized {len(content)} chars to {len(summary)} chars")
            return summary

        except ImportError:
            logger.warning(f"{self.provider} package not installed. Using simple truncation.")
            return self._simple_truncate(content, max_length)
        except Exception as e:
            logger.error(f"API summarization failed: {e}")
            return self._simple_truncate(content, max_length)

    def _simple_truncate(self, content: str, max_length: int) -> str:
        """Simple truncation fallback."""
        words = content.split()
        if len(words) <= max_length:
            return content
        return " ".join(words[:max_length]) + "..."


class ToolEnvironment:
    """
    Tool environment with real API calls.

    Supports:
    - SEARCH: SerpAPI Google search
    - READ: Web content extraction
    - SUMMARY: Local model or GPT-4o summarization
    """

    def __init__(
        self,
        serpapi_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        adv_wrapper: Optional[callable] = None,
        use_cache: bool = True,
        local_model: Optional[Any] = None,
        local_tokenizer: Optional[Any] = None,
        shared_cache: Optional["SharedToolCache"] = None,
        summary_model: Optional[str] = None,
        summary_provider: Optional[str] = None,
        summary_api_key: Optional[str] = None,
    ):
        """
        Initialize tool environment.

        Args:
            serpapi_key: SerpAPI key (or use SERPAPI_API_KEY env var)
            openai_key: OpenAI API key (or use OPENAI_API_KEY env var)
            adv_wrapper: Optional adversarial wrapper function
            use_cache: Whether to cache results
            local_model: Optional local HuggingFace model for summarization
            local_tokenizer: Optional tokenizer for local model
            shared_cache: Optional cross-episode shared cache
            summary_model: Model name for summarization (default: gpt-4o)
            summary_provider: Provider for summarization ("openai" | "anthropic")
            summary_api_key: API key for summary provider
        """
        self._shared_cache = shared_cache
        # Training rollouts set USE_TEACHER_SEARCH=1 to replace SerpAPI + URL
        # fetch with a single gpt-4o call that returns fact-grounded synthetic
        # search results + pre-populated page content. Eval (no env var) keeps
        # the real SerpAPI path.
        _use_teacher = os.environ.get("USE_TEACHER_SEARCH", "").lower() in ("1", "true", "yes")
        if _use_teacher:
            self._teacher_search = TeacherSimulatedSearch(
                api_key=summary_api_key or openai_key or os.environ.get("OPENAI_API_KEY"),
            )
            self.searcher = self._teacher_search
            self.reader = TeacherSimulatedWebReader(self._teacher_search)
            logger.info(
                "[ToolEnvironment] USE_TEACHER_SEARCH=1 → SEARCH/READ routed to "
                f"PF helper ({self._teacher_search.model}); SerpAPI not used"
            )
        else:
            self._teacher_search = None
            self.searcher = SerpAPISearch(api_key=serpapi_key)
            self.reader = WebReader(shared_cache=shared_cache)
        # Summary API key: explicit > openai_key fallback (backward compat)
        _sum_key = summary_api_key or openai_key
        self.summarizer = GPTSummarizer(
            api_key=_sum_key,
            model=summary_model or "",
            provider=summary_provider or "openai",
            local_model=local_model,
            local_tokenizer=local_tokenizer,
        )
        self.adv_wrapper = adv_wrapper

        # Current sample ID for per-sample adversarial injection
        self.current_sample_id: Optional[str] = None

        # Document cache: doc_id -> {url, title, content}
        self.documents: Dict[str, Dict[str, Any]] = {}

        # Search result cache
        self._search_cache: Dict[str, List[SearchResult]] = {}

    def set_current_sample(self, sample_id: str) -> None:
        """Set current sample ID for per-sample adversarial data lookup."""
        self.current_sample_id = sample_id

    def set_local_model(self, model: Any, tokenizer: Any) -> None:
        """
        Set local model for summarization (replaces GPT calls).

        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
        """
        self.summarizer.set_local_model(model, tokenizer)

    def search(self, query: str, max_results: int = 8) -> List[SearchResult]:
        """
        Execute search and cache results.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of SearchResult
        """
        # Three-tier cache: per-episode → shared → SerpAPI network call
        cache_key = f"{query}:{max_results}"
        if cache_key in self._search_cache:
            results = self._search_cache[cache_key]
        elif self._shared_cache is not None and (shared_hit := self._shared_cache.get_search(cache_key)) is not None:
            results = shared_hit
            self._search_cache[cache_key] = results
        else:
            results = self.searcher.search(query, max_results)
            self._search_cache[cache_key] = results
            if self._shared_cache is not None:
                self._shared_cache.put_search(cache_key, results)

        # Store documents for later READ
        for result in results:
            self.documents[result.doc_id] = {
                "url": result.url,
                "title": result.title,
                "snippet": result.snippet,
                "content": None,  # Lazy load
            }

        # Apply adversarial wrapper if present
        if self.adv_wrapper:
            import inspect
            if 'current_sample_id' in inspect.signature(self.adv_wrapper).parameters:
                results = self.adv_wrapper(results, "search", current_sample_id=self.current_sample_id)
            else:
                results = self.adv_wrapper(results, "search")

        return results

    def read(self, doc_id: str) -> str:
        """
        Read document content.

        Args:
            doc_id: Document ID from search results

        Returns:
            Document content
        """
        if doc_id not in self.documents:
            return f"Document {doc_id} not found. Please SEARCH first."

        doc = self.documents[doc_id]

        # Lazy load content
        if doc["content"] is None:
            url = doc["url"]
            if url:
                doc["content"] = self.reader.read(url)
            else:
                doc["content"] = doc.get("snippet", "No content available.")

        content = doc["content"]

        # Apply adversarial wrapper if present
        if self.adv_wrapper:
            content = self.adv_wrapper(content, "read")

        return content

    def summarize(self, doc_id: str, question: Optional[str] = None) -> str:
        """
        Summarize document content.

        Args:
            doc_id: Document ID
            question: Optional question to focus summary

        Returns:
            Summary text
        """
        content = self.read(doc_id)

        if content.startswith("Document") and "not found" in content:
            return content

        summary = self.summarizer.summarize(content, question)

        # Apply adversarial wrapper if present
        if self.adv_wrapper:
            summary = self.adv_wrapper(summary, "summary")

        return summary

    def reset(self):
        """Reset environment state."""
        self.documents.clear()
        self._search_cache.clear()
        self.reader._cache.clear()
        self.current_sample_id = None
