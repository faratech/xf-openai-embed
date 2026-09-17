import asyncio
import logging
from typing import List, Optional
import httpx
import numpy as np
import tiktoken
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from xf_embed.cache import cache_query_vector, get_cached_query_vector
from xf_embed.config import settings

logger = logging.getLogger(__name__)

_openai_client: Optional[AsyncOpenAI] = None
_httpx_client: Optional[httpx.AsyncClient] = None
_tokenizer: Optional[tiktoken.Encoding] = None
_semaphore: Optional[asyncio.Semaphore] = None

MAX_TOKENS = 8191


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.OPENAI_REQUEST_TIMEOUT,
            max_retries=settings.OPENAI_MAX_RETRIES,
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
    return len(tokenizer.encode(text))


def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncates text to max_tokens."""
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
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
            if "embedding" in data and len(data["embedding"]) == settings.OPENAI_DIMENSIONS:
                return np.array(data["embedding"], dtype=np.float32)
    except Exception as e:
        logger.debug(f"Local embedding service request failed, falling back to direct API: {e}")
    return None


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
async def _generate_via_openai_api(text: str, model: Optional[str] = None) -> np.ndarray:
    """Generates embedding directly via OpenAI API."""
    client = get_openai_client()
    target_model = model or settings.OPENAI_EMBEDDING_MODEL
    safe_text = truncate_text(text)

    response = await client.embeddings.create(
        input=[safe_text],
        model=target_model,
        dimensions=settings.OPENAI_DIMENSIONS if "3-" in target_model else None,
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


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
    safe_text = truncate_text(text)

    # 1. Tier 1: Check Cache (0.1ms)
    if use_cache and not model:
        cached = await get_cached_query_vector(safe_text)
        if cached is not None:
            return normalize_vector(cached) if normalize else cached

    # 2. Tier 2: Call local daemon (port 8765) if enabled
    raw_vec: Optional[np.ndarray] = None
    if settings.USE_LOCAL_EMBEDDING_SERVICE and not model:
        raw_vec = await _generate_via_local_service(safe_text)

    # 3. Tier 3: Direct OpenAI API fallback
    if raw_vec is None:
        raw_vec = await _generate_via_openai_api(safe_text, model=model)

    # Cache the warm vector for future queries
    if use_cache and not model:
        await cache_query_vector(safe_text, raw_vec)

    return normalize_vector(raw_vec) if normalize else raw_vec


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5), reraise=True)
async def generate_embeddings_batch(
    texts: List[str],
    model: Optional[str] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Generates embeddings for a batch of texts."""
    if not texts:
        return np.empty((0, settings.OPENAI_DIMENSIONS), dtype=np.float32)

    client = get_openai_client()
    target_model = model or settings.OPENAI_EMBEDDING_MODEL
    safe_texts = [truncate_text(t) for t in texts]

    response = await client.embeddings.create(
        input=safe_texts,
        model=target_model,
        dimensions=settings.OPENAI_DIMENSIONS if "3-" in target_model else None,
    )
    embeddings = [data.embedding for data in response.data]
    vecs = np.array(embeddings, dtype=np.float32)
    return normalize_vector(vecs) if normalize else vecs
