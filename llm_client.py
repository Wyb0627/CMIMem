"""
LLM Client for MemBuilder (Open Source Version)

This module provides a clean OpenAI-compatible interface for LLM operations.
It works with any OpenAI-compatible API endpoint.

Usage:
    # Standard OpenAI
    client = create_llm_client(base_url="https://api.openai.com/v1", api_key="sk-...")
    
    # Local proxy (e.g., vLLM, custom proxy)
    client = create_llm_client(base_url="http://localhost:8766/v1", api_key="any")
    
    # Other compatible services
    client = create_llm_client(base_url="https://your-service.com/v1", api_key="...")

Configuration:
    Set environment variables:
    - OPENAI_API_KEY: API key
    - OPENAI_BASE_URL: Base URL (optional, defaults to OpenAI)
"""

import json
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI


# =============================================================================
# Retry policy for transient API errors (429 rate-limit, 5xx, timeouts, conn).
# =============================================================================
# Tunable via env (set once per process; reads happen at retry time):
#     LLM_RETRY_MAX_ATTEMPTS   default 6   (1 initial + 5 retries)
#     LLM_RETRY_BASE_DELAY_S   default 1.0 base for exp backoff
#     LLM_RETRY_MAX_DELAY_S    default 60  cap per-attempt sleep
#     LLM_RETRY_JITTER_FRAC    default 0.3 random jitter as fraction of delay
# These bounds give a worst-case wall time of ~1+2+4+8+16+32 = 63s of sleeping
# before giving up, which is well below DashScope's typical RPM window. The
# outer _retry_with_json_validation in memory_system.py still wraps this with
# its own (now mostly dormant) retry budget, so a single 429 storm can't kill
# the whole conv.

def _is_retryable_api_error(exc: BaseException) -> bool:
    """Return True iff this exception represents a *transient* API failure
    that's worth retrying with backoff (rate-limit, 5xx, timeout, conn drop).

    We deliberately do NOT retry on 4xx other than 429 -- those (400 bad
    request, 401 auth, 404 model-not-found) are deterministic and retrying
    them just wastes quota.
    """
    # openai SDK raises a hierarchy under openai.OpenAIError; using duck
    # typing on status_code keeps us version-agnostic (1.x and 2.x).
    status = getattr(exc, "status_code", None)
    if status is None:
        # Some openai exception subclasses (APITimeoutError, APIConnectionError)
        # don't expose status_code -- fall back to class name matching.
        cls_name = type(exc).__name__
        if cls_name in {
            "RateLimitError",
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
            "APIConnectionTimeoutError",
        }:
            return True
        # Generic socket / urllib3 / httpx transport errors bubbling through.
        if cls_name in {"ConnectionError", "Timeout", "ReadTimeout"}:
            return True
        return False
    # Honor explicit status codes when present.
    if status == 429:
        return True
    if 500 <= int(status) < 600:
        return True
    return False


# Context-length error patterns. We match against str(exc) because:
# - openai SDK BadRequestError stores the server's JSON body in the message
# - vLLM returns: "'max_tokens' or 'max_completion_tokens' is too large: 8192"
# - DashScope/OpenAI return: "context_length_exceeded" or "maximum context length"
_CONTEXT_LENGTH_PATTERNS = (
    "max_completion_tokens' is too large",
    "context_length_exceeded",
    "maximum context length",
)


def _is_context_length_error(exc: BaseException) -> bool:
    """Return True iff this exception represents a context-length-exceeded
    error (prompt + max_tokens > model's capacity).

    We only treat 400-level (or unset status) errors as context-length errors;
    a 429 with a similar message would still be a rate-limit, not a sizing
    issue. Matching is on the rendered string because the OpenAI SDK embeds
    the upstream JSON body there.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return False
    msg = str(exc).lower()
    return any(pat.lower() in msg for pat in _CONTEXT_LENGTH_PATTERNS)


def _is_local_vllm_endpoint(base_url: Optional[str]) -> bool:
    """Heuristic: treat localhost / 127.0.0.1 / port 8000 as our local
    vLLM (memory-builder) endpoint. We deliberately skip context-length
    fallback there, because:
    - Truncating messages for the memory-construction prompt would feed
      the model an incomplete view and produce noisy/wrong memories
    - Memory build has its own, smarter top-k-degradation fallback in
      memory_system.py that's aware of prompt structure
    """
    if not base_url:
        return False
    b = base_url.lower()
    return ("localhost" in b) or ("127.0.0.1" in b) or (":8000" in b)


def _truncate_messages_middle(messages: List[Dict[str, str]]) -> Optional[List[Dict[str, str]]]:
    """Drop middle messages, keeping the first (system prompt) and the last
    (current user turn). Returns None if no shorter list is possible
    (i.e. already <=2 messages or already at minimum).

    The first message is preserved because it typically holds the system
    prompt that defines the role/task. The last message is preserved
    because it is the most recent user turn — usually the question that
    the model needs to answer. Middle history is what gets sacrificed.
    """
    if len(messages) <= 2:
        return None
    return [messages[0], messages[-1]]


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Extract Retry-After hint from a 429 response if the server provided one.

    DashScope (and OpenAI proper) populate the ``retry-after`` header on
    429s. Honor it as a *lower bound* on our sleep, capped at MAX_DELAY_S.
    """
    try:
        resp = getattr(exc, "response", None)
        if resp is None:
            return None
        headers = getattr(resp, "headers", {}) or {}
        # Header name is case-insensitive in HTTP; httpx.Headers handles both.
        val = headers.get("retry-after") or headers.get("Retry-After")
        if val is None:
            return None
        return max(0.0, float(val))
    except (TypeError, ValueError, AttributeError):
        return None


def _retry_with_backoff(fn, *, op_name: str = "llm_api"):
    """Invoke ``fn()`` with adaptive backoff on retryable errors.

    Backoff = min(MAX_DELAY, BASE * 2**attempt) * (1 + uniform(±JITTER));
    server's Retry-After header (if any) acts as a lower bound. Jitter is
    crucial when many workers hit the limit at the same instant -- without
    it they would synchronously retry and re-trigger the 429.
    """
    try:
        max_attempts = int(os.environ.get("LLM_RETRY_MAX_ATTEMPTS", "6"))
        base_delay = float(os.environ.get("LLM_RETRY_BASE_DELAY_S", "1.0"))
        max_delay = float(os.environ.get("LLM_RETRY_MAX_DELAY_S", "60"))
        jitter_frac = float(os.environ.get("LLM_RETRY_JITTER_FRAC", "0.3"))
    except ValueError:
        max_attempts, base_delay, max_delay, jitter_frac = 6, 1.0, 60.0, 0.3
    max_attempts = max(1, max_attempts)

    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_retryable_api_error(e) or attempt == max_attempts - 1:
                raise
            backoff = min(max_delay, base_delay * (2 ** attempt))
            # Symmetric jitter so the expected sleep equals `backoff`.
            jitter = backoff * jitter_frac * (2 * random.random() - 1)
            sleep_s = max(0.1, backoff + jitter)
            ra = _retry_after_seconds(e)
            if ra is not None:
                sleep_s = max(sleep_s, min(ra, max_delay))
            status = getattr(e, "status_code", None) or type(e).__name__
            # Only print on final retry attempt to reduce log noise
            if attempt == max_attempts - 2:
                print(
                    f"  ⚠️ {op_name} transient error ({status}), "
                    f"sleep {sleep_s:.1f}s then FINAL retry "
                    f"(attempt {attempt + 1}/{max_attempts}): {str(e)[:120]}",
                    flush=True,
                )
            time.sleep(sleep_s)
    # Unreachable: loop either returns or raises.
    raise last_exc  # pragma: no cover


# =============================================================================
# Local embedding backend (optional)
# =============================================================================
# When the chat provider is fine but the embedding API has tight rate/batch
# limits (e.g. DashScope text-embedding-v* allows only batch<=10), you can
# offload embeddings to a local GPU model via sentence-transformers.
#
# Activate by setting (or via configs/embedding.yaml + build_rl_data_*.sh):
#     EMBEDDING_PROVIDER=local
#     LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B   # ModelScope id, OR local path
#     LOCAL_EMBEDDING_DEVICE=cuda                     # optional, default auto
#     LOCAL_EMBEDDING_BATCH_SIZE=256                  # optional, default 64
#     LOCAL_EMBEDDING_AUTO_DOWNLOAD=1                 # optional, default 1
#     LOCAL_EMBEDDING_CACHE_DIR=/path/to/cache        # optional, ModelScope cache
#
# Model resolution order (when LOCAL_EMBEDDING_MODEL is given):
#   1. If it's an existing local directory -> load from disk directly.
#   2. Else, treat it as a ModelScope (魔搭) repo id and call snapshot_download
#      to fetch onto local disk, then load the resulting directory.
#      ModelScope is used instead of HuggingFace because the runtime hosts
#      cannot reach huggingface.co; ModelScope mirrors the same checkpoints
#      on Aliyun-internal infra.
#
# The model is loaded once per process (lazily on first use) and shared across
# all LLMClient instances to avoid wasting GPU memory.

_LOCAL_EMBEDDER = None
_LOCAL_EMBEDDER_LOCK = threading.Lock()


def _resolve_local_model_path(model_ref, auto_download=True, cache_dir=None):
    """Resolve a model reference (local dir or ModelScope id) to a local path.

    We never go through HuggingFace by default.
    """
    # Already a usable local directory? (e.g. user pre-downloaded snapshot)
    if model_ref and os.path.isdir(model_ref):
        return model_ref

    if not auto_download:
        raise FileNotFoundError(
            f"LOCAL_EMBEDDING_MODEL='{model_ref}' is not an existing directory "
            f"and auto_download is disabled. Either set auto_download=true or "
            f"point LOCAL_EMBEDDING_MODEL to a pre-downloaded snapshot path."
        )

    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise ImportError(
            "Auto-downloading local embedding models requires the `modelscope` "
            "package. "
            "Install via: pip install modelscope"
        ) from e

    print(f"[LocalEmbedder] Downloading {model_ref} from ModelScope "
          f"(cache_dir={cache_dir or 'default'})...", flush=True)
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    local_dir = snapshot_download(model_ref, **kwargs)
    print(f"[LocalEmbedder] Snapshot ready at: {local_dir}", flush=True)
    return local_dir


def _get_local_embedder(model_ref, device=None, auto_download=True, cache_dir=None):
    """Lazy-load and cache a local sentence-transformers model (process-wide).

    Before loading a new model, checks whether cmi_reward's dedicated
    ``_KERNEL_MI_EMBEDDERS`` already holds the same model_ref to avoid
    loading the same multi-GB model twice into GPU memory.
    """
    global _LOCAL_EMBEDDER
    if _LOCAL_EMBEDDER is not None and _LOCAL_EMBEDDER["model_ref"] == model_ref:
        return _LOCAL_EMBEDDER["model"]

    with _LOCAL_EMBEDDER_LOCK:
        if _LOCAL_EMBEDDER is not None and _LOCAL_EMBEDDER["model_ref"] == model_ref:
            return _LOCAL_EMBEDDER["model"]

        # Try to reuse an already-loaded model from cmi_reward's dedicated cache
        try:
            from training.reward_server.cmi_reward import _KERNEL_MI_EMBEDDERS
            if model_ref in _KERNEL_MI_EMBEDDERS:
                model = _KERNEL_MI_EMBEDDERS[model_ref]
                _LOCAL_EMBEDDER = {"model_ref": model_ref, "model": model}
                print(f"[LocalEmbedder] Reusing CMI kernel_mi embedder for {model_ref} "
                      f"(no duplicate GPU load)", flush=True)
                return model
        except (ImportError, AttributeError):
            pass

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Local embedding backend requires `sentence-transformers`. "
                "Install via: pip install sentence-transformers"
            ) from e

        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        local_path = _resolve_local_model_path(
            model_ref, auto_download=auto_download, cache_dir=cache_dir
        )

        print(f"[LocalEmbedder] Loading {model_ref} on {device} (path={local_path})...",
              flush=True)
        model = SentenceTransformer(local_path, device=device, trust_remote_code=True)
        try:
            dim = model.get_sentence_embedding_dimension()
        except Exception:
            dim = "?"
        print(f"[LocalEmbedder] Loaded. dim={dim}", flush=True)
        _LOCAL_EMBEDDER = {"model_ref": model_ref, "model": model}
        return model

class LLMClient:
    """
    OpenAI-compatible LLM client.
    
    Works with any OpenAI-compatible API endpoint including:
    - OpenAI API
    - Azure OpenAI
    - vLLM servers
    - Local proxy servers
    - Any other OpenAI-compatible service
    """

    @staticmethod
    def _normalize_message_content(content: Any) -> str:
        """Normalize OpenAI message content into plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _strip_think_tags(content: str) -> str:
        """Remove  blocks from model output.

        Some thinking models (DeepSeek-V4, Qwen3) embed chain-of-thought
        reasoning inside <think> tags. Strip them so downstream JSON
        extraction sees only the final answer.
        """
        import re
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()

    @staticmethod
    def _extract_json_text(content: str) -> str:
        """Extract JSON object text from mixed model output."""
        import re

        clean = (content or "").strip()
        if not clean:
            return clean

        # Remove markdown fences if present.
        clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', clean).strip()

        # Prefer the first JSON object block if model prepends analysis text.
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            return clean[start:end + 1].strip()
        return clean
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o",
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL (defaults to OPENAI_BASE_URL env var or OpenAI)
            model: Default model name
            embedding_model: Default embedding model name. If None, falls
                back to the ``EMBEDDING_MODEL`` env var (set by build scripts
                from configs/embedding.yaml). If still unset, uses
                ``text-embedding-v4`` (DashScope's current default), NOT the
                old OpenAI-only ``text-embedding-3-small`` -- the latter 404s
                on DashScope and was the silent default for callers that
                bypass build_rl_data_s_cleaned.sh.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model
        self.embedding_model = (
            embedding_model
            or os.environ.get("EMBEDDING_MODEL", "").strip()
            or "text-embedding-v4"
        )
        
        if not self.api_key:
            raise ValueError(
                "API key required. Set via api_key parameter or OPENAI_API_KEY env var."
            )
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        # TP=1 single-GPU inference can be slow for long prompts (30-90s);
        # set generous timeouts to avoid premature APITimeoutError.
        # LLM_CLIENT_TIMEOUT env var allows runtime override (in seconds).
        request_timeout = float(os.environ.get("LLM_CLIENT_TIMEOUT", "300"))
        client_kwargs["timeout"] = request_timeout

        self.client = OpenAI(**client_kwargs)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 8192,
        response_format: Optional[Dict] = None,
        model: str = None,
        **kwargs
    ) -> str:
        """
        Send chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})
            model: Model override (uses default if not specified)
            **kwargs: Additional OpenAI API parameters
        
        Returns:
            Response content as string
        """
        model = model or self.model
        
        request_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        if response_format:
            request_kwargs["response_format"] = response_format

        # Disable thinking mode for DashScope thinking models (DeepSeek-V4,
        # Qwen3, etc.) unless the caller explicitly opted in. Thinking mode
        # adds latency, cost, and can interfere with JSON output.
        if "extra_body" not in request_kwargs:
            base = (self.base_url or "").lower()
            if "dashscope" in base:
                request_kwargs["extra_body"] = {"enable_thinking": False}

        # Adaptive backoff on 429 / 5xx / timeouts / conn drops. The
        # response_format-fallback below is independent: it handles the
        # "this server doesn't accept response_format" case (a deterministic
        # 4xx) which is NOT retryable and shouldn't go through backoff.
        def _do_call(kw):
            return _retry_with_backoff(
                lambda: self.client.chat.completions.create(**kw),
                op_name="chat_completion",
            )

        try:
            response = _do_call(request_kwargs)
        except Exception as e:
            # Some OpenAI-compatible servers (e.g., vLLM) may not support
            # response_format -- they 4xx with a deterministic error. Strip
            # it and retry once. If that still fails, propagate.
            if "response_format" in request_kwargs and not _is_retryable_api_error(e):
                request_kwargs.pop("response_format", None)
                try:
                    response = _do_call(request_kwargs)
                except Exception as e2:
                    e = e2  # fall through to context-length check below
                else:
                    e = None

            if e is not None:
                # Context-length fallback: only for non-vLLM endpoints.
                # Memory-build vLLM gets its own structure-aware fallback
                # in memory_system.py.
                if (
                    _is_context_length_error(e)
                    and not _is_local_vllm_endpoint(self.base_url)
                ):
                    truncated = _truncate_messages_middle(request_kwargs["messages"])
                    if truncated is not None:
                        old_count = len(request_kwargs["messages"])
                        new_count = len(truncated)
                        print(
                            f"  ⚠️ context length exceeded, truncating messages "
                            f"from {old_count} to {new_count}",
                            flush=True,
                        )
                        request_kwargs["messages"] = truncated
                        response = _do_call(request_kwargs)
                    else:
                        raise e
                else:
                    raise e
        raw_content = response.choices[0].message.content
        content = self._normalize_message_content(raw_content)

        # Strip residual <think> blocks that some models emit even when
        # enable_thinking=False (or when the server ignores the flag).
        content = self._strip_think_tags(content)
        
        # Extract JSON text if response_format requires it, but do NOT
        # validate with json.loads() here -- the caller (memory_system's
        # _retry_with_json_validation) has its own _parse_json_with_repair
        # that can fix common model output errors (e.g. missing commas).
        # Validating here would short-circuit that repair logic.
        if response_format and response_format.get("type") == "json_object":
            content = self._extract_json_text(content)
            if not content:
                raise ValueError("Model returned empty content for json_object response.")
        
        return content
    
    def get_embeddings(
        self,
        texts: List[str],
        model: str = None,
        batch_size: int = None,
    ) -> List[List[float]]:
        """
        Get embeddings for a list of texts.

        Backend selection:
          - If ``EMBEDDING_PROVIDER=local``, embeddings are computed by a local
            sentence-transformers model (loaded once per process). Useful when
            chat uses a remote API but the embedding API has tight rate/batch
            limits (e.g. DashScope text-embedding-v* caps batch at 10).
          - Otherwise (default), embeddings are fetched from the OpenAI-compatible
            endpoint configured on this client, with auto-batching.

        Args:
            texts: List of text strings to embed
            model: Embedding model override (only used by remote backend)
            batch_size: Max texts per call. If None, uses provider-specific env
                var defaults: ``LOCAL_EMBEDDING_BATCH_SIZE`` (default 64) for
                local, ``OPENAI_EMBEDDINGS_BATCH_SIZE`` (default 10) for remote.

        Returns:
            List of embedding vectors (order preserved).
        """
        if not texts:
            return []

        provider = os.environ.get("EMBEDDING_PROVIDER", "").strip().lower()
        if provider == "local":
            return self._get_embeddings_local(texts, batch_size=batch_size)
        return self._get_embeddings_remote(texts, model=model, batch_size=batch_size)

    def _get_embeddings_remote(
        self,
        texts: List[str],
        model: str = None,
        batch_size: int = None,
    ) -> List[List[float]]:
        """Fetch embeddings via OpenAI-compatible HTTP endpoint, with auto-batching."""
        model = model or self.embedding_model
        if batch_size is None:
            try:
                batch_size = int(os.environ.get("OPENAI_EMBEDDINGS_BATCH_SIZE", "10"))
            except ValueError:
                batch_size = 10
        batch_size = max(1, batch_size)

        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            response = _retry_with_backoff(
                lambda c=chunk: self.client.embeddings.create(input=c, model=model, encoding_format="float"),
                op_name="embeddings",
            )
            if not response.data:
                raise ValueError(
                    f"Embedding service returned empty data "
                    f"(batch {start//batch_size + 1}, chunk_size={len(chunk)})"
                )
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings

    def _get_embeddings_local(
        self,
        texts: List[str],
        batch_size: int = None,
    ) -> List[List[float]]:
        """Compute embeddings via a local sentence-transformers model (GPU-friendly).

        Reads the following env vars (set by build scripts from configs/embedding.yaml):
          LOCAL_EMBEDDING_MODEL          ModelScope id or local snapshot path (required)
          LOCAL_EMBEDDING_DEVICE         e.g. 'cuda', 'cuda:1', 'cpu' (default: auto)
          LOCAL_EMBEDDING_BATCH_SIZE     int (default: 64)
          LOCAL_EMBEDDING_AUTO_DOWNLOAD  '1'/'true' to fetch via ModelScope (default: 1)
          LOCAL_EMBEDDING_CACHE_DIR      ModelScope cache dir (optional)
        """
        model_ref = os.environ.get("LOCAL_EMBEDDING_MODEL", "").strip()
        if not model_ref:
            raise ValueError(
                "EMBEDDING_PROVIDER=local but LOCAL_EMBEDDING_MODEL is not set. "
                "Example: export LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B"
            )
        device = os.environ.get("LOCAL_EMBEDDING_DEVICE") or None

        if batch_size is None:
            try:
                batch_size = int(os.environ.get("LOCAL_EMBEDDING_BATCH_SIZE", "64"))
            except ValueError:
                batch_size = 64
        batch_size = max(1, batch_size)

        auto_download = os.environ.get("LOCAL_EMBEDDING_AUTO_DOWNLOAD", "1").strip().lower()
        auto_download = auto_download in ("1", "true", "yes", "y")
        cache_dir = os.environ.get("LOCAL_EMBEDDING_CACHE_DIR", "").strip() or None

        embedder = _get_local_embedder(
            model_ref,
            device=device,
            auto_download=auto_download,
            cache_dir=cache_dir,
        )
        # Serialize GPU inference across threads to prevent CUDA OOM from
        # concurrent forward passes stacking intermediate activations.
        with _LOCAL_EMBEDDER_LOCK:
            vectors = embedder.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return vectors.tolist()


# Alias for backward compatibility
OpenAIClient = LLMClient

# Provider registry: what this client module supports
AVAILABLE_PROVIDERS = ['openai', 'vllm']
DEFAULT_PROVIDER = 'openai'


def create_llm_client(
    api_key: str = None,
    base_url: str = None,
    model: str = "gpt-4o",
    provider: str = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create LLM client.
    
    This public version always creates an OpenAI-compatible LLMClient.
    The ``provider`` parameter is accepted for API compatibility with the
    internal client module but does not change dispatch behavior here —
    use ``base_url`` to point at different backends (vLLM, proxies, etc.).
    
    Args:
        api_key: API key (defaults to OPENAI_API_KEY env var)
        base_url: Base URL (defaults to OPENAI_BASE_URL env var or OpenAI)
        model: Default model name
        provider: Accepted for interface compatibility; not used for dispatch
        **kwargs: Additional arguments passed to LLMClient
    
    Returns:
        LLMClient instance
    
    Examples:
        # Use OpenAI
        client = create_llm_client(api_key="sk-...")
        
        # Use local proxy
        client = create_llm_client(
            base_url="http://localhost:8766/v1",
            api_key="any",
            model="claude-sonnet-4-5-20250929"
        )
        
        # Use vLLM
        client = create_llm_client(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            model="Qwen/Qwen2-7B"
        )
    """
    return LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        **kwargs
    )
