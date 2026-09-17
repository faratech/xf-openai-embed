import asyncio
import json
import logging
import time
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="xf-search: Semantic and Hybrid Search CLI for XenForo")
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host address to bind to"),
    port: int = typer.Option(8000, help="Port number to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    workers: int = typer.Option(1, help="Number of worker processes"),
):
    """Start the FastAPI search server using Uvicorn."""
    import uvicorn
    console.print(f"[bold green]Starting xf-search server on {host}:{port}...[/bold green]")
    uvicorn.run(
        "xf_embed.api:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
    )


@app.command()
def index_faiss(
    batch_size: int = typer.Option(10000, help="Batch size for fetching vectors from MySQL"),
    output_path: Optional[str] = typer.Option(None, help="Output path for FAISS index file"),
):
    """Build or rebuild the FAISS vector index by streaming from MySQL."""
    from xf_embed.vector_engine import build_faiss_index
    from xf_embed.config import settings

    target_path = output_path or settings.FAISS_INDEX_PATH
    console.print(f"[bold blue]Building FAISS index (saving to: {target_path})...[/bold blue]")

    start_time = time.time()

    def progress(batch_count, total_count):
        console.print(f"  Indexed [bold cyan]{total_count}[/bold cyan] vectors...")

    async def run():
        await build_faiss_index(batch_size=batch_size, save_path=target_path, progress_callback=progress)

    asyncio.run(run())
    duration = time.time() - start_time
    console.print(f"[bold green]✓ Indexing complete in {duration:.2f} seconds![/bold green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query string"),
    mode: str = typer.Option("hybrid", help="Search mode: 'hybrid', 'vector', or 'elastic'"),
    limit: int = typer.Option(10, help="Maximum number of results to display"),
):
    """Execute a test search query directly from the terminal."""
    from xf_embed.hybrid_search import perform_hybrid_search, perform_vector_search
    from xf_embed.elastic_engine import search_elasticsearch

    console.print(f"[bold yellow]Searching for:[/bold yellow] '{query}' (mode: {mode}, limit: {limit})")

    async def run():
        t0 = time.time()
        try:
            if mode == "vector":
                results = await perform_vector_search(query, max_results=limit)
            elif mode == "elastic":
                results = await search_elasticsearch(query, max_results=limit)
            else:
                results, _, _ = await perform_hybrid_search(query, max_results=limit)
            elapsed = (time.time() - t0) * 1000.0
            return results, elapsed
        finally:
            from xf_embed.db import close_db_pool
            from xf_embed.elastic_engine import close_es_client
            await close_db_pool()
            await close_es_client()

    results, latency_ms = asyncio.run(run())

    console.print(f"[bold green]Retrieved {len(results)} results in {latency_ms:.1f}ms[/bold green]\n")

    table = Table(title=f"Search Results ({mode.upper()})")
    table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("ID", style="green")
    table.add_column("Score", justify="right", style="bold yellow")
    table.add_column("Thread Title / Snippet", style="white")

    for idx, r in enumerate(results, start=1):
        item_id = str(r.get("post_id") or r.get("thread_id") or "N/A")
        score_val = f"{r.get('score', 0.0):.4f}"
        title = r.get("thread_title") or "No title"
        msg = r.get("message", "")
        preview = f"[bold]{title}[/bold]\n{(msg[:120] + '...') if len(msg) > 120 else msg}".strip()
        table.add_row(str(idx), r.get("type", "post"), item_id, score_val, preview)

    console.print(table)


@app.command()
def benchmark(
    query: str = typer.Option("windows 11 update issue", help="Query string to benchmark"),
    iterations: int = typer.Option(10, help="Number of benchmark iterations"),
):
    """Benchmark latency of vector, elasticsearch, and hybrid search pipelines."""
    from xf_embed.hybrid_search import perform_hybrid_search, perform_vector_search
    from xf_embed.elastic_engine import search_elasticsearch
    import statistics

    console.print(f"[bold cyan]Benchmarking search pipelines with {iterations} iterations...[/bold cyan]")

    async def run():
        vec_times = []
        es_times = []
        hybrid_times = []

        for i in range(iterations):
            # Vector
            t0 = time.perf_counter()
            await perform_vector_search(query, max_results=10)
            vec_times.append((time.perf_counter() - t0) * 1000.0)

            # ES
            t0 = time.perf_counter()
            await search_elasticsearch(query, max_results=10)
            es_times.append((time.perf_counter() - t0) * 1000.0)

            # Hybrid
            t0 = time.perf_counter()
            await perform_hybrid_search(query, max_results=10)
            hybrid_times.append((time.perf_counter() - t0) * 1000.0)

        return vec_times, es_times, hybrid_times

    vec_times, es_times, hybrid_times = asyncio.run(run())

    table = Table(title="Search Latency Benchmark (ms)")
    table.add_column("Pipeline", style="bold cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("Median (ms)", justify="right")
    table.add_column("Min (ms)", justify="right")
    table.add_column("Max (ms)", justify="right")

    for name, times in [("Vector Search", vec_times), ("Elasticsearch (BM25)", es_times), ("Hybrid (RRF)", hybrid_times)]:
        table.add_row(
            name,
            f"{statistics.mean(times):.2f}",
            f"{statistics.median(times):.2f}",
            f"{min(times):.2f}",
            f"{max(times):.2f}",
        )

    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
