"""
Model loading utilities for evaluation.

Three loading modes:
1. HuggingFace base model: Load directly from HF hub
2. Checkpoint: Load from local checkpoint folder (SFT/RL trained)
3. vLLM: High-performance inference with vLLM engine
"""

from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


# ============================================================================
# vLLM Wrapper for unified interface
# ============================================================================

class VLLMModelWrapper:
    """
    Wrapper for vLLM model to provide HuggingFace-like interface.

    This allows vLLM to be used with the same code that uses HuggingFace models.
    """

    def __init__(self, llm, sampling_params_default=None, lora_request=None):
        """
        Args:
            llm: vLLM LLM instance
            sampling_params_default: Default sampling parameters
            lora_request: Optional vLLM LoRARequest for adapter models
        """
        self.llm = llm
        self.sampling_params_default = sampling_params_default
        self.lora_request = lora_request
        # NOTE: Do NOT call torch.cuda.is_available() here — it initializes CUDA
        # in the main process and breaks vLLM's multiprocessing workers (spawn).
        self.device = torch.device("cuda")

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        **kwargs,
    ):
        """
        Generate using vLLM with HuggingFace-like interface.

        Note: This is a simplified wrapper. For batch processing,
        use generate_batch() directly for better efficiency.
        """
        from vllm import SamplingParams

        # vLLM expects prompt tokens, not input_ids tensor
        # This wrapper is mainly for compatibility; direct vLLM usage is preferred
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0,
            top_p=top_p if do_sample else 1.0,
        )

        # Convert input_ids back to token list for vLLM
        if input_ids is not None:
            prompt_token_ids = input_ids[0].tolist()
            # Use TokensPrompt for newer vLLM versions
            try:
                from vllm import TokensPrompt
                prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]
                outputs = self.llm.generate(prompts, sampling_params=sampling_params, lora_request=self.lora_request)
            except ImportError:
                # Fallback for older vLLM versions
                outputs = self.llm.generate(
                    prompt_token_ids=[prompt_token_ids],
                    sampling_params=sampling_params,
                    lora_request=self.lora_request,
                )

            # Return in HuggingFace format (input + output tokens)
            output_ids = outputs[0].outputs[0].token_ids
            full_ids = prompt_token_ids + list(output_ids)
            return type('Outputs', (), {'__getitem__': lambda s, i: torch.tensor([full_ids])})()

        return None

    def eval(self):
        """No-op for compatibility."""
        return self

    def generate_text(self, prompts: list, sampling_params=None) -> list:
        """
        Direct vLLM generation for better efficiency.

        Args:
            prompts: List of prompt strings
            sampling_params: vLLM SamplingParams

        Returns:
            List of generated texts
        """
        params = sampling_params or self.sampling_params_default
        outputs = self.llm.generate(prompts, params, lora_request=self.lora_request)
        return [output.outputs[0].text for output in outputs]

    def generate_batch(self, prompts: list, sampling_params=None) -> list:
        """
        Batch generation for multiple prompts.

        Args:
            prompts: List of prompt strings
            sampling_params: vLLM SamplingParams

        Returns:
            List of vLLM outputs
        """
        params = sampling_params or self.sampling_params_default
        return self.llm.generate(prompts, params, lora_request=self.lora_request)


class VLLMTokenizerWrapper:
    """Wrapper for tokenizer to work with VLLMModelWrapper."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token = tokenizer.pad_token
        self.eos_token = tokenizer.eos_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.vocab_size = tokenizer.vocab_size

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)


class ModelType(Enum):
    """Model types for evaluation."""
    BASE = "base"  # Original HuggingFace model
    SFT = "sft"    # SFT-trained checkpoint
    RL = "rl"      # RL-trained checkpoint
    API = "api"    # API-based model (Claude, GPT-5, etc.)


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    # Model source
    model_name_or_path: str = "Qwen/Qwen3-30B-A3B"
    model_type: ModelType = ModelType.BASE

    # Quantization
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Other settings
    trust_remote_code: bool = True
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    use_flash_attention_2: bool = True  # Enable Flash Attention 2 for faster inference
    low_cpu_mem_usage: bool = True
    cache_dir: Optional[str] = None


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def detect_pre_quantized(model_name_or_path: str, trust_remote_code: bool = True) -> Optional[str]:
    """
    Detect if a model is already pre-quantized (FP8, GPTQ, AWQ, etc.).

    Returns:
        Quantization type string (e.g. "fp8", "gptq", "awq") if pre-quantized,
        None if not pre-quantized.
    """
    try:
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
    except Exception:
        return None

    quant_cfg = getattr(config, "quantization_config", None)
    if quant_cfg is None:
        return None

    # quant_cfg can be a dict or a QuantizationConfig object
    if isinstance(quant_cfg, dict):
        quant_method = quant_cfg.get("quant_method", "")
    else:
        quant_method = getattr(quant_cfg, "quant_method", "")
        if not quant_method:
            # Fallback: infer from class name
            cls_name = type(quant_cfg).__name__.lower()
            if "fp8" in cls_name:
                quant_method = "fp8"
            elif "gptq" in cls_name:
                quant_method = "gptq"
            elif "awq" in cls_name:
                quant_method = "awq"

    return str(quant_method) if quant_method else None


def get_quantization_config(
    config: ModelConfig, pre_quantized: Optional[str] = None
) -> Optional[BitsAndBytesConfig]:
    """Create quantization config if needed. Skips if model is already pre-quantized."""
    if pre_quantized:
        logger.info(
            f"Model is already pre-quantized ({pre_quantized}), "
            f"skipping BitsAndBytes quantization"
        )
        return None

    if not config.load_in_4bit and not config.load_in_8bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        bnb_4bit_compute_dtype=get_torch_dtype(config.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=True,
    )


def load_model_for_eval(
    model_name_or_path: str,
    model_type: Union[ModelType, str] = ModelType.BASE,
    load_in_4bit: bool = True,
    use_flash_attention_2: bool = True,
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer for evaluation.

    Args:
        model_name_or_path: HuggingFace model name or local checkpoint path
        model_type: Type of model (base/sft/rl)
        load_in_4bit: Whether to use 4-bit quantization
        device_map: Device mapping strategy
        cache_dir: Cache directory for downloads
        **kwargs: Additional arguments

    Returns:
        Tuple of (model, tokenizer)

    Examples:
        # Load base model from HuggingFace
        model, tokenizer = load_model_for_eval(
            "Qwen/Qwen3-30B-A3B",
            model_type=ModelType.BASE
        )

        # Load SFT checkpoint
        model, tokenizer = load_model_for_eval(
            "./ckpt/sft_policy/best",
            model_type=ModelType.SFT
        )

        # Load RL checkpoint
        model, tokenizer = load_model_for_eval(
            "./ckpt/rl_cispo/best",
            model_type=ModelType.RL
        )
    """
    if isinstance(model_type, str):
        model_type = ModelType(model_type)

    logger.info(f"Loading model: {model_name_or_path} (type: {model_type.value})")

    config = ModelConfig(
        model_name_or_path=model_name_or_path,
        model_type=model_type,
        load_in_4bit=load_in_4bit,
        use_flash_attention_2=use_flash_attention_2,
        device_map=device_map,
        cache_dir=cache_dir,
    )

    # Determine if this is a local checkpoint or HF model
    is_local = os.path.isdir(model_name_or_path)

    # Detect LoRA adapter: has adapter_config.json
    is_lora_adapter = False
    lora_base_model = None
    if is_local:
        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            import json
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            lora_base_model = adapter_cfg.get("base_model_name_or_path")
            if lora_base_model:
                is_lora_adapter = True
                logger.info(f"Detected LoRA adapter, base model: {lora_base_model}")

    # Detect pre-quantized models (FP8, GPTQ, AWQ, etc.)
    pre_quantized = detect_pre_quantized(
        model_name_or_path, trust_remote_code=config.trust_remote_code
    )
    if pre_quantized:
        logger.info(f"Detected pre-quantized model: {pre_quantized}")

    # Build model loading kwargs
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "device_map": config.device_map,
        "low_cpu_mem_usage": config.low_cpu_mem_usage,
    }

    # Quantization: skip BnB if model is already pre-quantized
    quant_config = get_quantization_config(config, pre_quantized=pre_quantized)
    if quant_config:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = get_torch_dtype(config.torch_dtype)

    # Flash attention configuration
    if config.use_flash_attention_2:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using FlashAttention2 for faster inference")
        except ImportError:
            logger.warning("FlashAttention2 requested but not installed, falling back to sdpa")
            model_kwargs["attn_implementation"] = "sdpa"
    else:
        # Use sdpa (scaled dot product attention) as default fallback
        model_kwargs["attn_implementation"] = "sdpa"

    # Cache directory
    if config.cache_dir:
        model_kwargs["cache_dir"] = config.cache_dir

    # Override with kwargs
    model_kwargs.update(kwargs)

    # Load model
    try:
        if is_lora_adapter:
            # Load base model first, then apply LoRA adapter
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                lora_base_model,
                **model_kwargs
            )
            model = PeftModel.from_pretrained(base_model, model_name_or_path)
            model = model.merge_and_unload()
            logger.info(f"LoRA adapter merged into base model: {lora_base_model}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **model_kwargs
            )
        logger.info(f"Model loaded: {model.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Set to eval mode
    model.eval()

    # Load tokenizer
    tokenizer_path = model_name_or_path
    if is_local:
        # For local checkpoints, try to find tokenizer
        # First check if tokenizer files exist in checkpoint
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
        has_tokenizer = any(
            os.path.exists(os.path.join(model_name_or_path, f))
            for f in tokenizer_files
        )

        if not has_tokenizer:
            # Fallback to base model tokenizer
            if is_lora_adapter and lora_base_model:
                tokenizer_path = lora_base_model
                logger.info(f"Using tokenizer from LoRA base model: {tokenizer_path}")
            else:
                config_path = os.path.join(model_name_or_path, "config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path) as f:
                        cfg = json.load(f)
                        base_model = cfg.get("_name_or_path", "Qwen/Qwen3-30B-A3B")
                        tokenizer_path = base_model
                        logger.info(f"Using tokenizer from base model: {tokenizer_path}")
                else:
                    tokenizer_path = "Qwen/Qwen3-30B-A3B"
                    logger.warning(f"No tokenizer found, using default: {tokenizer_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=config.trust_remote_code,
            cache_dir=config.cache_dir,
        )

        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    return model, tokenizer


def load_multiple_models(
    model_configs: Dict[str, Dict[str, Any]],
) -> Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]]:
    """
    Load multiple models for comparison.

    Args:
        model_configs: Dictionary mapping model names to their configs
            Example:
            {
                "base": {"model_name_or_path": "Qwen/Qwen3-30B-A3B", "model_type": "base"},
                "sft": {"model_name_or_path": "./ckpt/sft/best", "model_type": "sft"},
                "rl": {"model_name_or_path": "./ckpt/rl/best", "model_type": "rl"},
            }

    Returns:
        Dictionary mapping model names to (model, tokenizer) tuples
    """
    models = {}
    for name, config in model_configs.items():
        logger.info(f"Loading model: {name}")
        model, tokenizer = load_model_for_eval(**config)
        models[name] = (model, tokenizer)
    return models


# Convenience functions for common use cases
def load_base_model(
    model_name: str = "Qwen/Qwen3-30B-A3B",
    **kwargs,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load base model from HuggingFace."""
    return load_model_for_eval(model_name, model_type=ModelType.BASE, **kwargs)


def load_sft_model(
    checkpoint_path: str,
    **kwargs,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load SFT-trained model from checkpoint."""
    return load_model_for_eval(checkpoint_path, model_type=ModelType.SFT, **kwargs)


def load_rl_model(
    checkpoint_path: str,
    **kwargs,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load RL-trained model from checkpoint."""
    return load_model_for_eval(checkpoint_path, model_type=ModelType.RL, **kwargs)


# ============================================================================
# API Model Wrapper (Claude, GPT-5, etc.)
# ============================================================================

class APIModelWrapper:
    """Wrapper for API-based LLMs (Anthropic, OpenAI, Google) with unified interface.

    Provides the same ``generate_from_messages`` API regardless of provider,
    plus ``eval()`` and ``device`` stubs for compatibility with code that
    expects a HuggingFace-like model object.
    """

    SUPPORTED_PROVIDERS = ("anthropic", "openai", "google")

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ):
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Must be one of {self.SUPPORTED_PROVIDERS}")
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        # Lazy-init clients
        self._sync_client = None
        self._async_client = None
        # Compatibility stubs
        self.device = torch.device("cpu")

    def _get_sync_client(self):
        """Lazy-init synchronous API client."""
        if self._sync_client is not None:
            return self._sync_client

        if self.provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
            self._sync_client = Anthropic(api_key=self.api_key)
        elif self.provider == "google":
            try:
                from google import genai
            except ImportError:
                raise ImportError("google-genai package required. Install with: pip install google-genai")
            self._sync_client = genai.Client(api_key=self.api_key)
        else:  # openai
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
            self._sync_client = OpenAI(api_key=self.api_key)
        return self._sync_client

    def _get_async_client(self):
        """Lazy-init asynchronous API client."""
        if self._async_client is not None:
            return self._async_client

        if self.provider == "anthropic":
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
            self._async_client = AsyncAnthropic(api_key=self.api_key)
        elif self.provider == "google":
            try:
                from google import genai
            except ImportError:
                raise ImportError("google-genai package required. Install with: pip install google-genai")
            # google-genai Client supports both sync and async via aio
            self._async_client = genai.Client(api_key=self.api_key)
        else:  # openai
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
            self._async_client = AsyncOpenAI(api_key=self.api_key)
        return self._async_client

    def generate_from_messages(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Synchronous generation from chat messages.

        Args:
            messages: List of dicts with 'role' and 'content' keys.
            max_tokens: Override default max_tokens.
            temperature: Override default temperature.
            top_p: Override default top_p.

        Returns:
            Generated text string.
        """
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p

        client = self._get_sync_client()

        if self.provider == "anthropic":
            # Anthropic separates system message
            system_msg = ""
            api_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    api_messages.append(msg)
            kwargs = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "messages": api_messages,
            }
            if system_msg:
                kwargs["system"] = system_msg
            response = client.messages.create(**kwargs)
            return response.content[0].text

        elif self.provider == "google":
            from google.genai import types
            # Merge system + user messages into a single contents list
            system_msg = ""
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    contents.append(types.Content(
                        role="user" if msg["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=msg["content"])],
                    ))
            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            if system_msg:
                config.system_instruction = system_msg
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            return response.text

        else:  # openai
            # Newer OpenAI models (gpt-5, o-series) require max_completion_tokens
            # instead of max_tokens, and only support temperature=1.
            oai_kwargs = {
                "model": self.model_name,
                "messages": messages,
            }
            model_lower = self.model_name.lower()
            _is_new_oai = any(k in model_lower for k in ("gpt-5", "o1", "o3", "o4"))
            if _is_new_oai:
                oai_kwargs["max_completion_tokens"] = max_tokens
                # These models only support default temperature/top_p
            else:
                oai_kwargs["max_tokens"] = max_tokens
                oai_kwargs["temperature"] = temperature
                oai_kwargs["top_p"] = top_p
            response = client.chat.completions.create(**oai_kwargs)
            return response.choices[0].message.content

    async def agenerate_from_messages(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Asynchronous generation from chat messages.

        Same interface as generate_from_messages but uses async clients.
        """
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p

        client = self._get_async_client()

        if self.provider == "anthropic":
            system_msg = ""
            api_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    api_messages.append(msg)
            kwargs = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "messages": api_messages,
            }
            if system_msg:
                kwargs["system"] = system_msg
            response = await client.messages.create(**kwargs)
            return response.content[0].text

        elif self.provider == "google":
            from google.genai import types
            system_msg = ""
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    contents.append(types.Content(
                        role="user" if msg["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=msg["content"])],
                    ))
            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            if system_msg:
                config.system_instruction = system_msg
            response = await client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            return response.text

        else:  # openai
            oai_kwargs = {
                "model": self.model_name,
                "messages": messages,
            }
            model_lower = self.model_name.lower()
            _is_new_oai = any(k in model_lower for k in ("gpt-5", "o1", "o3", "o4"))
            if _is_new_oai:
                oai_kwargs["max_completion_tokens"] = max_tokens
            else:
                oai_kwargs["max_tokens"] = max_tokens
                oai_kwargs["temperature"] = temperature
                oai_kwargs["top_p"] = top_p
            response = await client.chat.completions.create(**oai_kwargs)
            return response.choices[0].message.content

    def generate(self, messages: list, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, **kwargs) -> str:
        """Alias for generate_from_messages — used by PFs and PF selector."""
        return self.generate_from_messages(
            messages=messages, max_tokens=max_tokens, temperature=temperature,
        )

    def eval(self):
        """No-op for compatibility with HF model interface."""
        return self


class APITokenizerStub:
    """Stub tokenizer for API models.

    Provides ``apply_chat_template()`` and ``encode()`` for compatibility
    with code that expects a tokenizer. Token counts are rough estimates.
    """

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.vocab_size = 100000  # Placeholder

    def apply_chat_template(
        self,
        messages: list,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        """Concatenate message contents for token count estimation."""
        parts = []
        for msg in messages:
            parts.append(f"<|{msg['role']}|>\n{msg['content']}")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "\n".join(parts)

    def encode(self, text: str, **kwargs) -> list:
        """Rough 4-chars-per-token estimate."""
        n_tokens = max(1, len(text) // 4)
        return list(range(n_tokens))

    def decode(self, token_ids, **kwargs) -> str:
        """Stub decode — returns empty string."""
        return ""

    def __call__(self, text, **kwargs):
        """Stub __call__ for compatibility."""
        encoded = self.encode(text)
        return {"input_ids": [encoded], "attention_mask": [[1] * len(encoded)]}


def load_model_api(
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.9,
    **kwargs,
) -> Tuple["APIModelWrapper", "APITokenizerStub"]:
    """Load an API-based model (Anthropic, OpenAI, or Google).

    Args:
        provider: "anthropic", "openai", or "google"
        model_name: Model ID (e.g. "claude-sonnet-4-20250514", "gpt-5", "gemini-2.5-flash")
        api_key: API key (falls back to env vars)
        max_tokens: Default max tokens for generation
        temperature: Default temperature
        top_p: Default top_p
        **kwargs: Additional keyword arguments (ignored, for forward compat)

    Returns:
        Tuple of (APIModelWrapper, APITokenizerStub)
    """
    # Fallback to environment variables
    if api_key is None:
        _ENV_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}
        env_var = _ENV_VARS.get(provider, "OPENAI_API_KEY")
        api_key = os.environ.get(env_var)
        if api_key:
            logger.info(f"Using API key from environment variable {env_var}")

    model = APIModelWrapper(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    tokenizer = APITokenizerStub()

    logger.info(f"API model loaded: provider={provider}, model={model_name}")
    return model, tokenizer


# ============================================================================
# vLLM Loading Functions
# ============================================================================

def get_available_gpu_count() -> int:
    """Get the number of available GPUs without initializing CUDA.

    Uses CUDA_VISIBLE_DEVICES env var or falls back to nvidia-smi to avoid
    calling torch.cuda.is_available() which initializes CUDA in the main
    process and breaks vLLM's spawn-based multiprocessing.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        if cuda_visible == "":
            return 0
        return len(cuda_visible.split(","))
    # Fallback: query nvidia-smi without initializing CUDA
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split("\n"))
    except Exception:
        pass
    return 0


def load_model_vllm(
    model_name_or_path: str,
    model_type: Union[ModelType, str] = ModelType.BASE,
    tensor_parallel_size: Optional[int] = None,  # None means auto-detect
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    dtype: str = "bfloat16",
    quantization: Optional[str] = None,
    max_num_seqs: int = 256,
    enable_chunked_prefill: bool = True,
    **kwargs,
) -> Tuple[VLLMModelWrapper, VLLMTokenizerWrapper]:
    """
    Load model using vLLM for high-performance inference.

    vLLM provides:
    - Continuous batching for higher throughput
    - PagedAttention for efficient memory usage
    - Optimized CUDA kernels

    Args:
        model_name_or_path: HuggingFace model name or local checkpoint path
        model_type: Type of model (base/sft/rl)
        tensor_parallel_size: Number of GPUs for tensor parallelism (None = auto-detect all GPUs)
        gpu_memory_utilization: Fraction of GPU memory to use (0-1)
        max_model_len: Maximum sequence length
        dtype: Data type ("float16", "bfloat16", "auto")
        quantization: Quantization method ("awq", "gptq", "fp8", None)
        max_num_seqs: Maximum number of sequences per batch (higher = better GPU util)
        enable_chunked_prefill: Enable chunked prefill for better batching
        **kwargs: Additional vLLM arguments

    Returns:
        Tuple of (VLLMModelWrapper, VLLMTokenizerWrapper)

    Example:
        model, tokenizer = load_model_vllm(
            "Qwen/Qwen3-4B-Instruct",
            gpu_memory_utilization=0.9,
            max_num_seqs=256,
        )
    """
    # CRITICAL: Set spawn method BEFORE importing vllm to avoid CUDA fork issues
    # with tensor_parallel > 1. Must happen before any multiprocessing context is created.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        import multiprocessing as _mp
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vLLM not installed. Install with: pip install vllm"
        )

    if isinstance(model_type, str):
        model_type = ModelType(model_type)

    # Auto-detect GPU count if tensor_parallel_size not specified
    if tensor_parallel_size is None or tensor_parallel_size <= 0:
        tensor_parallel_size = get_available_gpu_count()
        if tensor_parallel_size == 0:
            raise RuntimeError("No GPUs available for vLLM")
        logger.info(f"Auto-detected {tensor_parallel_size} GPUs for tensor parallelism")

    logger.info(f"Loading model with vLLM: {model_name_or_path} (type: {model_type.value})")
    logger.info(f"  tensor_parallel_size: {tensor_parallel_size}")

    # Increase RPC/SHM timeout to avoid broadcast errors when GPU is idle
    # during async tool execution (default 60s is too short).
    os.environ.setdefault("VLLM_RPC_TIMEOUT", "300")

    # Set environment variables for better multi-GPU stability
    if tensor_parallel_size > 1:
        # Increase distributed timeout for large models
        os.environ.setdefault("VLLM_DISTRIBUTED_TIMEOUT_SECONDS", "300")
        # Disable chunked prefill for TP > 1 to avoid shared memory issues
        enable_chunked_prefill = False
        logger.info("  Disabled chunked_prefill for multi-GPU stability")

    # Detect LoRA adapter: has adapter_config.json but no config.json
    lora_request = None
    is_local = os.path.isdir(model_name_or_path)
    adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json") if is_local else None
    model_config_path = os.path.join(model_name_or_path, "config.json") if is_local else None

    if is_local and adapter_config_path and os.path.exists(adapter_config_path) and not os.path.exists(model_config_path):
        import json
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_model_path = adapter_cfg.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError(f"adapter_config.json at {model_name_or_path} missing 'base_model_name_or_path'")
        logger.info(f"Detected LoRA adapter at {model_name_or_path}, base model: {base_model_path}")

        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(
            lora_name=model_type.value if isinstance(model_type, ModelType) else model_type,
            lora_int_id=1,
            lora_path=os.path.abspath(model_name_or_path),
        )
        # Use base model as the main model
        actual_model_path = base_model_path
    else:
        actual_model_path = model_name_or_path

    # Build vLLM kwargs with optimizations for higher GPU utilization
    vllm_kwargs = {
        "model": actual_model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "dtype": dtype,
        "trust_remote_code": True,
        # Key parameters for higher GPU utilization
        "max_num_seqs": max_num_seqs,  # More concurrent sequences
        "enable_chunked_prefill": enable_chunked_prefill,  # Disabled for TP > 1
    }

    # Use multiprocessing backend for tensor parallelism
    if tensor_parallel_size > 1:
        vllm_kwargs["distributed_executor_backend"] = "mp"
        # L40S (compute cap 8.9) + TP=2 crashes in custom_all_reduce during warmup
        # (seen as `custom_all_reduce.cuh:455 'invalid argument'`). Match what
        # training/rejection_sampling/rollout.py and training/distill/train.py already do.
        vllm_kwargs.setdefault("disable_custom_all_reduce", True)

    # Auto-detect pre-quantized models for vLLM
    pre_quant = detect_pre_quantized(actual_model_path)
    if pre_quant:
        if quantization and quantization != pre_quant:
            logger.info(f"  Model is pre-quantized as {pre_quant}; overriding config quantization ({quantization})")
        quantization = pre_quant
    elif not quantization:
        quantization = None

    if quantization:
        vllm_kwargs["quantization"] = quantization

    # Enable LoRA if adapter detected
    if lora_request is not None:
        lora_rank = adapter_cfg.get("r", 16)
        vllm_kwargs["enable_lora"] = True
        vllm_kwargs["max_lora_rank"] = lora_rank
        logger.info(f"  LoRA adapter enabled: {lora_request.lora_path} (rank={lora_rank})")

    # Override with any additional kwargs
    vllm_kwargs.update(kwargs)

    # Fallback: if model architecture is not natively supported by vLLM,
    # use the generic Transformers backend (e.g. Qwen3.5 MoE models).
    try:
        from transformers import AutoConfig as _AC
        _arch = _AC.from_pretrained(actual_model_path, trust_remote_code=True).architectures or []
    except Exception:
        _arch = []
    if _arch:
        # Get supported architectures (compatible with vLLM 0.15+)
        try:
            from vllm.model_executor.models import ModelRegistry
            _supported = set(ModelRegistry.get_supported_archs())
        except (ImportError, AttributeError):
            try:
                from vllm.model_executor.models import _MODELS
                _supported = set(_MODELS.keys()) if isinstance(_MODELS, dict) else set(str(_MODELS))
            except (ImportError, AttributeError):
                _supported = set()
        if _supported and not any(a in _supported for a in _arch):
            # Pick the right generic backend based on model type
            _fallback = "TransformersMoEForCausalLM" if "moe" in _arch[0].lower() else "TransformersForCausalLM"
            logger.info(f"  Architecture {_arch} not natively supported; using generic backend: {_fallback}")
            vllm_kwargs["hf_overrides"] = {"architectures": [_fallback]}

    # Load vLLM model
    try:
        llm = LLM(**vllm_kwargs)
        logger.info(f"vLLM model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model with vLLM: {e}")
        raise

    # Create default sampling params
    default_sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.1,
        top_p=0.9,
    )

    # Wrap model
    model_wrapper = VLLMModelWrapper(llm, default_sampling_params, lora_request=lora_request)

    # Load tokenizer separately
    tokenizer_path = model_name_or_path
    is_local = os.path.isdir(model_name_or_path)

    if is_local:
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
        has_tokenizer = any(
            os.path.exists(os.path.join(model_name_or_path, f))
            for f in tokenizer_files
        )
        if not has_tokenizer:
            config_path = os.path.join(model_name_or_path, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path) as f:
                    cfg = json.load(f)
                    tokenizer_path = cfg.get("_name_or_path", "Qwen/Qwen3-4B-Instruct")
            else:
                tokenizer_path = "Qwen/Qwen3-4B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tokenizer_wrapper = VLLMTokenizerWrapper(tokenizer)

    logger.info(f"vLLM setup complete: {model_name_or_path}")

    return model_wrapper, tokenizer_wrapper


def is_vllm_available() -> bool:
    """Check if vLLM is available."""
    try:
        import vllm
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"vLLM import failed (non-ImportError): {type(e).__name__}: {e}")
        return False
