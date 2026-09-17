"""
faiss_router.py: Backward-compatible FastAPI router and server entrypoint.

Exposes both `app` (FastAPI instance) and `router` (APIRouter) backed by the modernized
xf_embed package with async connection pooling, Reciprocal Rank Fusion, and IndexIDMap2.
"""

import logging
from xf_embed.api import (
    app,
    router,
    QueryRequest,
    SearchResultItem,
    VectorSearchResponse,
    ElasticSearchResponse,
    CombinedSearchResponse,
)
from xf_embed.config import settings
from xf_embed.vector_engine import load_faiss_index_from_disk, get_faiss_index, search_faiss
from xf_embed.elastic_engine import search_elasticsearch
from xf_embed.hybrid_search import perform_hybrid_search, perform_vector_search

logger = logging.getLogger(__name__)

# Legacy aliases for backward compatibility with external scripts
load_faiss_index = load_faiss_index_from_disk
FAISS_INDEX_PATH = settings.FAISS_INDEX_PATH

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("faiss_router:app", host=settings.HOST, port=settings.PORT, reload=True)
