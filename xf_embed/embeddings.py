import asyncio
import logging
from typing import List, Optional
import httpx
import numpy as np
import tiktoken
from openai import APIConnectionError, InternalServerError, RateLimitError, AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

try:
    import sys as _sys
    if "/web/fastapi_app" not in _sys.path:
        _sys.path.append("/web/fastapi_app")
    import direct_metering  # ledger row per direct OpenAI call; never load-bearing
except Exception:
    direct_metering = None

from xf_embed.cache import cache_query_vector, get_cached_query_vector
from xf_embed.config import settings

logger = logging.getLogger(__name__)

_openai_client: Optional[AsyncOpenAI] = None
_httpx_client: Optional[httpx.AsyncClient] = None
_tokenizer: Optional[tiktoken.Encoding] = None
_semaphore: Optional[asyncio.Semaphore] = None
_inflight = {}

MAX_TOKENS = 8191
MAX_BATCH_TOKENS = 300000
MAX_BATCH_INPUTS = 2048
_transient_errors = retry_if_exception_type((APIConnectionError, InternalServerError, RateLimitError))


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.OPENAI_REQUEST_TIMEOUT,
            # Tenacity owns retries so SDK retries cannot multiply them.
            max_retries=0,
        )
    return _openai_client


def get_httpx_client() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None or _httpx_client.is_closed:
        _httpx_client = httpx.AsyncClient(
            timeout=5.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
    return _httpx_client


async def close_httpx_client() -> None:
    global _httpx_client
    if _httpx_client is not None and not _httpx_client.is_closed:
        await _httpx_client.aclose()
        _httpx_client = None


async def close_embedding_clients() -> None:
    """Drain pending generation before closing both embedding transports."""
    global _openai_client
    tasks = list(_inflight.values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    try:
        await close_httpx_client()
    finally:
        client, _openai_client = _openai_client, None
        if client is not None:
            await client.close()


def get_tokenizer() -> tiktoken.Encoding:
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.encoding_for_model(settings.OPENAI_EMBEDDING_MODEL)
        except Exception:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def get_semaphore(concurrency: int = 10) -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(concurrency)
    return _semaphore


def count_tokens(text: str) -> int:
    """Returns token count for input text."""
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, disallowed_special=()))


def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncates text to max_tokens."""
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text, disallowed_special=())
    if len(tokens) > max_tokens:
        return tokenizer.decode(tokens[:max_tokens])
    return text


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalizes vector to unit length (L2 norm = 1.0) for Inner Product cosine similarity."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return v / norm


async def _generate_via_local_service(text: str) -> Optional[np.ndarray]:
    """Attempts to generate query vector via local xfai-embedding daemon on port 8765."""
    try:
        client = get_httpx_client()
        url = f"{settings.LOCAL_EMBEDDING_URL}/embed"
        resp = await client.post(url, json={"text": text, "type": "query"})
        if resp.status_code == 200:
            data = resp.json()
            if "embedding" in data:
                vector = np.asarray(data["embedding"], dtype=np.float32)
                if (vector.shape == (settings.OPENAI_DIMENSIONS,) and np.isfinite(vector).all()
                        and np.any(vector)):
                    return vector
    except Exception as e:
        logger.debug(f"Local embedding service request failed, falling back to direct API: {e}")
    return None


def _response_vectors(response, count: int) -> np.ndarray:
    """Map by the API's input indices and reject incomplete/malformed replies."""
    ordered = [None] * count
    for item in response.data:
        if not isinstance(item.index, int) or not 0 <= item.index < count or ordered[item.index] is not None:
            raise ValueError("Invalid embedding response index")
        vector = np.asarray(item.embedding, dtype=np.float32)
        if (vector.shape != (settings.OPENAI_DIMENSIONS,) or not np.isfinite(vector).all()
                or not np.any(vector)):
            raise ValueError("Invalid embedding response vector")
        ordered[item.index] = vector
    if any(vector is None for vector in ordered):
        raise ValueError("Incomplete embedding response")
    return np.stack(ordered)


def _meter_response(response, model: str, count: int = 1) -> None:
    if direct_metering is not None:
        try:
            direct_metering.fire(direct_metering.meter_embedding(
                response, model=model, request_source="xf_embed:fallback", n_inputs=count))
        except Exception:
            logger.warning("Embedding usage metering failed", exc_info=True)


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(settings.OPENAI_MAX_RETRIES + 1), retry=_transient_errors, reraise=True)
async def _generate_via_openai_api(text: str, model: Optional[str] = None) -> np.ndarray:
    """Generates embedding directly via OpenAI API."""
    client = get_openai_client()
    target_model = model or settings.OPENAI_EMBEDDING_MODEL
    response = await client.embeddings.create(
        input=[text],
        model=target_model,
        dimensions=settings.OPENAI_DIMENSIONS if "3-" in target_model else None,
    )
    _meter_response(response, target_model)
    return _response_vectors(response, 1)[0]


async def generate_embedding(
    text: str,
    model: Optional[str] = None,
    normalize: bool = True,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Generates embedding for a single text with multi-tier caching (Redis/memory)
    and local daemon fallback.
    """
    if not text.strip():
        raise ValueError("Embedding input must contain non-whitespace text")
    safe_text = truncate_text(text)

    # 1. Tier 1: Check Cache (0.1ms)
    if use_cache and not model:
        cached = await get_cached_query_vector(safe_text)
        if cached is not None:
            return normalize_vector(cached) if normalize else cached

    async def generate():
        raw_vec = None
        if (settings.USE_LOCAL_EMBEDDING_SERVICE and not model
                and settings.OPENAI_EMBEDDING_MODEL == settings.LOCAL_EMBEDDING_MODEL):
            raw_vec = await _generate_via_local_service(safe_text)
        if raw_vec is None:
            raw_vec = await _generate_via_openai_api(safe_text, model=model)
        if use_cache and not model:
            await cache_query_vector(safe_text, raw_vec)
        return raw_vec

    key = (asyncio.get_running_loop(), safe_text, model, use_cache)
    task = _inflight.get(key)
    if task is None:
        task = asyncio.create_task(generate())
        _inflight[key] = task

        def completed(done):
            if _inflight.get(key) is done:
                del _inflight[key]
            # A cancelled caller must not cancel shared work or leave an
            # unobserved exception when no other callers remain.
            if not done.cancelled():
                done.exception()

        task.add_done_callback(completed)
    raw_vec = await asyncio.shield(task)
    return normalize_vector(raw_vec) if normalize else raw_vec


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(settings.OPENAI_MAX_RETRIES + 1), retry=_transient_errors, reraise=True)
async def _request_batch(inputs, model):
    response = await get_openai_client().embeddings.create(
        input=inputs, model=model,
        dimensions=settings.OPENAI_DIMENSIONS if "3-" in model else None,
    )
    _meter_response(response, model, len(inputs))
    return _response_vectors(response, len(inputs))


async def generate_embeddings_batch(
    texts: List[str],
    model: Optional[str] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Generates embeddings for a batch of texts."""
    if not texts:
        return np.empty((0, settings.OPENAI_DIMENSIONS), dtype=np.float32)
    if any(not text.strip() for text in texts):
        raise ValueError("Embedding input must contain non-whitespace text")

    target_model = model or settings.OPENAI_EMBEDDING_MODEL
    tokenizer = get_tokenizer()
    batches, pending, total_tokens = [], [], 0
    for text in texts:
        tokens = tokenizer.encode(text, disallowed_special=())[:MAX_TOKENS]
        if not tokens:
            raise ValueError("Embedding input must contain at least one token")
        if pending and (len(pending) >= MAX_BATCH_INPUTS or total_tokens + len(tokens) > MAX_BATCH_TOKENS):
            batches.append(pending)
            pending, total_tokens = [], 0
        pending.append(tokens)
        total_tokens += len(tokens)
    if pending:
        batches.append(pending)
    arrays = [await _request_batch(batch, target_model) for batch in batches]
    vecs = np.concatenate(arrays)
    return normalize_vector(vecs) if normalize else vecs
