from contextlib import AsyncExitStack, asynccontextmanager
import logging
import time
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from xf_embed.cache import (
    cache_search_results,
    close_cache,
    compute_search_cache_key,
    get_cached_search_results,
    get_redis_client,
)
from xf_embed.config import settings
from xf_embed.db import close_db_pool, get_db_pool
from xf_embed.elastic_engine import close_es_client, get_es_client, search_elasticsearch
from xf_embed.embeddings import close_embedding_clients
from xf_embed.hybrid_search import perform_hybrid_search, perform_vector_search
from xf_embed.vector_engine import get_faiss_index
from xf_embed.visibility import VisibilityUnavailable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    max_results: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    algorithm: Optional[Literal["rrf", "weighted"]] = Field(None, description="Hybrid fusion algorithm: 'rrf' or 'weighted'")

    @field_validator("query")
    @classmethod
    def require_search_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Search query must contain non-whitespace text")
        return value


class SearchResultItem(BaseModel):
    type: str = Field("post", description="Item type: 'post' or 'thread'")
    post_id: Optional[int] = None
    thread_id: Optional[int] = None
    thread_title: Optional[str] = ""
    message: Optional[str] = ""
    post_date: Optional[int] = 0
    username: Optional[str] = ""
    score: float = 0.0
    ranks: Optional[Dict[str, int]] = None
    rrf_score: Optional[float] = None
    distance: Optional[float] = None


class VectorSearchResponse(BaseModel):
    faiss_results: List[SearchResultItem]
    count: int
    query: str
    cached: bool = False
    latency_ms: Optional[float] = None


class ElasticSearchResponse(BaseModel):
    elasticsearch_results: List[SearchResultItem]
    count: int
    query: str
    cached: bool = False
    latency_ms: Optional[float] = None


class CombinedSearchResponse(BaseModel):
    combined_results: List[SearchResultItem]
    vector_results: Optional[List[SearchResultItem]] = None
    elasticsearch_results: Optional[List[SearchResultItem]] = None
    count: int
    query: str
    algorithm: str
    cached: bool = False
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    database: bool
    elasticsearch: bool
    redis: bool
    faiss_index_loaded: bool
    faiss_total_vectors: int
    vector_backend: str


def _cached_response(response_model, data, query: str, started: float):
    if not data:
        return None
    try:
        return response_model(**{
            **data, "query": query, "cached": True,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        })
    except ValidationError:
        logger.warning("Ignoring invalid cached search response")
        return None


# ---------------------------------------------------------------------------
# Lifespan Management
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as resources:
        # Register before startup so partially initialized resources also close.
        # The stack runs every callback even if another callback raises. It
        # unwinds in reverse, so stop generation before clearing its cache.
        for closer in (close_db_pool, close_es_client, close_cache, close_embedding_clients):
            resources.push_async_callback(closer)

        logger.info("Starting up xf-openai-embed service...")
        # Initialize DB pool
        try:
            await get_db_pool()
        except Exception as e:
            logger.warning(f"Could not initialize DB pool at startup: {e}")

        # Check Redis cache
        try:
            await get_redis_client()
        except Exception as e:
            logger.warning(f"Could not initialize Redis client: {e}")

        # MariaDB vector searches do not need a resident FAISS index.
        if settings.VECTOR_BACKEND == "faiss":
            try:
                idx = await get_faiss_index()
                if idx:
                    logger.info(f"FAISS index loaded at startup with {idx.ntotal} vectors.")
                else:
                    logger.info("FAISS index file not found on disk. MariaDB native vector or build needed.")
            except Exception as e:
                logger.warning(f"Could not load FAISS index at startup: {e}")

        try:
            yield
        finally:
            logger.info("Shutting down xf-openai-embed service...")


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/faiss", tags=["Search"])


@router.post("/search/", response_model=VectorSearchResponse)
async def search_endpoint(request: QueryRequest):
    """Semantic vector search using FAISS or MariaDB native vector distance with caching."""
    t0 = time.perf_counter()
    cache_key = compute_search_cache_key(request.query, "vector", request.max_results)

    cached_data = await get_cached_search_results(cache_key)
    cached_response = _cached_response(VectorSearchResponse, cached_data, request.query, t0)
    if cached_response is not None:
        return cached_response

    try:
        results = await perform_vector_search(request.query, max_results=request.max_results)
        latency = (time.perf_counter() - t0) * 1000.0
        resp = VectorSearchResponse(
            faiss_results=[SearchResultItem(**r) for r in results],
            count=len(results),
            query=request.query,
            cached=False,
            latency_ms=round(latency, 2),
        )
        await cache_search_results(cache_key, resp.model_dump())
        return resp
    except VisibilityUnavailable:
        raise
    except Exception:
        logger.exception("Vector search failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector search failed"
        )


@router.post("/elastic/", response_model=ElasticSearchResponse)
async def elastic_endpoint(request: QueryRequest):
    """Keyword search using Elasticsearch BM25 with caching."""
    t0 = time.perf_counter()
    cache_key = compute_search_cache_key(request.query, "elastic", request.max_results)

    cached_data = await get_cached_search_results(cache_key)
    cached_response = _cached_response(ElasticSearchResponse, cached_data, request.query, t0)
    if cached_response is not None:
        return cached_response

    try:
        results = await search_elasticsearch(request.query, max_results=request.max_results)
        latency = (time.perf_counter() - t0) * 1000.0
        resp = ElasticSearchResponse(
            elasticsearch_results=[SearchResultItem(**r) for r in results],
            count=len(results),
            query=request.query,
            cached=False,
            latency_ms=round(latency, 2),
        )
        await cache_search_results(cache_key, resp.model_dump())
        return resp
    except VisibilityUnavailable:
        raise
    except Exception:
        logger.exception("Elasticsearch search failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Elasticsearch query failed"
        )


@router.post("/combined/", response_model=CombinedSearchResponse)
async def combined_endpoint(request: QueryRequest):
    """True Hybrid Search combining Vector Search + Elasticsearch BM25 via Reciprocal Rank Fusion."""
    t0 = time.perf_counter()
    cache_key = compute_search_cache_key(request.query, "combined", request.max_results, request.algorithm)

    cached_data = await get_cached_search_results(cache_key)
    cached_response = _cached_response(CombinedSearchResponse, cached_data, request.query, t0)
    if cached_response is not None:
        return cached_response

    try:
        fused, vec_hits, es_hits = await perform_hybrid_search(
            query=request.query,
            max_results=request.max_results,
            algorithm=request.algorithm,
        )
        latency = (time.perf_counter() - t0) * 1000.0
        resp = CombinedSearchResponse(
            combined_results=[SearchResultItem(**r) for r in fused],
            vector_results=[SearchResultItem(**r) for r in vec_hits],
            elasticsearch_results=[SearchResultItem(**r) for r in es_hits],
            count=len(fused),
            query=request.query,
            algorithm=request.algorithm or settings.HYBRID_ALGORITHM,
            cached=False,
            latency_ms=round(latency, 2),
        )
        await cache_search_results(cache_key, resp.model_dump())
        return resp
    except VisibilityUnavailable:
        raise
    except Exception:
        logger.exception("Hybrid search failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hybrid search failed"
        )


# ---------------------------------------------------------------------------
# FastAPI Application Factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="XenForo Semantic & Hybrid Search API",
        description="High-performance search API combining OpenAI embeddings, FAISS, and Elasticsearch",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.exception_handler(VisibilityUnavailable)
    async def visibility_unavailable(_request, _exc: VisibilityUnavailable):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Search visibility is temporarily unavailable"},
        )

    @app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
    async def healthcheck():
        db_ok = False
        es_ok = False
        redis_ok = False
        faiss_loaded = False
        faiss_total = 0

        # Check DB
        try:
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
            db_ok = True
        except Exception:
            pass

        # Check ES
        try:
            es = get_es_client()
            es_ok = await es.ping()
        except Exception:
            pass

        # Check Redis
        try:
            r = await get_redis_client()
            if r is not None:
                redis_ok = await r.ping()
        except Exception:
            pass

        # Only inspect the active backend; loading an unused index can be costly.
        if settings.VECTOR_BACKEND == "faiss":
            try:
                idx = await get_faiss_index()
                if idx:
                    faiss_loaded = True
                    faiss_total = idx.ntotal
            except Exception:
                pass

        overall_status = "healthy" if (db_ok and (faiss_loaded or settings.VECTOR_BACKEND == "mariadb")) else "degraded"

        return HealthResponse(
            status=overall_status,
            database=db_ok,
            elasticsearch=es_ok,
            redis=redis_ok,
            faiss_index_loaded=faiss_loaded,
            faiss_total_vectors=faiss_total,
            vector_backend=settings.VECTOR_BACKEND,
        )

    return app


app = create_app()
