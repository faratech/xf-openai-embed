import os
import asyncio
import asyncmy.cursors
import json
import math
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tqdm import tqdm
import logging
import tiktoken
import argparse
from tenacity import retry, retry_if_exception_type, wait_random_exponential, stop_after_attempt

# Load environment variables
from xf_embed.config import settings
load_dotenv('.env')
if os.path.exists('/web/.env'):
    load_dotenv('/web/.env')
    load_dotenv('/web/.config')

# Database-only checks and --help must not require paid API credentials.
openai = None


def get_openai_client():
    global openai
    if openai is None:
        # Tenacity owns retries; SDK retries would multiply each attempt.
        openai = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.OPENAI_REQUEST_TIMEOUT,
            max_retries=0,
        )
    return openai


async def close_openai_client():
    global openai
    client, openai = openai, None
    if client is not None:
        await client.close()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

tokenizer = None


def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        try:
            tokenizer = tiktoken.encoding_for_model(settings.OPENAI_EMBEDDING_MODEL)
        except KeyError:
            tokenizer = tiktoken.get_encoding("cl100k_base")
    return tokenizer

# Constants
MAX_TOKENS = 8191
EXPECTED_EMBEDDING_LENGTH = settings.OPENAI_DIMENSIONS
BATCH_SIZE = 2048
MAX_CONCURRENT_REQUESTS = 10  # Adjust this based on your rate limits and system capabilities

# Create a semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

MAX_BATCH_TOKENS = 300_000
MAX_BATCH_INPUTS = 2048


@retry(
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)),
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(settings.OPENAI_MAX_RETRIES + 1), reraise=True,
)
async def _request_embeddings(inputs, model):
    kwargs = {"input": inputs, "model": model}
    if "3-" in model:
        kwargs["dimensions"] = settings.OPENAI_DIMENSIONS
    return await get_openai_client().embeddings.create(**kwargs)


async def get_embedding(text: str, model=None) -> list[float]:
    return (await get_embedding_batch([text], model=model))[0]


async def get_embedding_batch(texts, model=None) -> list[list[float]]:
    model = model or settings.OPENAI_EMBEDDING_MODEL
    # Token arrays avoid repeated tokenization and preserve exact truncation limits.
    inputs = [(get_tokenizer().encode(text, disallowed_special=()) if isinstance(text, str) else text)[:MAX_TOKENS]
              for text in texts]
    if any(not tokens for tokens in inputs):
        raise ValueError("Embedding inputs must not be empty")
    embeddings = []
    offset = 0
    while offset < len(inputs):
        end, tokens = offset, 0
        while end < len(inputs) and end - offset < MAX_BATCH_INPUTS:
            length = len(inputs[end])
            if tokens + length > MAX_BATCH_TOKENS:
                break
            tokens += length
            end += 1
        response = await _request_embeddings(inputs[offset:end], model)
        data = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in data] != list(range(end - offset)):
            raise ValueError("Embedding response indexes do not match the requested inputs")
        for item in data:
            valid, reason = validate_embedding(item.embedding)
            if not valid:
                raise ValueError(f"Invalid embedding response: {reason}")
        embeddings.extend(item.embedding for item in data)
        offset = end
    return embeddings

async def get_db_pool():
    logging.info("Attempting to create database connection pool")
    try:
        pool = await asyncmy.create_pool(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            db=settings.MYSQL_DATABASE,
            autocommit=True,
            pool_recycle=3600,
            maxsize=20
        )
        logging.info("Database connection pool created successfully")
        return pool
    except Exception as e:
        logging.error(f"Failed to create database connection pool: {e}")
        return None

async def rebuild_database(pool):
    logging.info("Rebuilding the entire embeddings database...")
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("DELETE FROM openai_embeddings WHERE thread_id IS NOT NULL AND post_id IS NULL")
            await conn.commit()
    logging.info("Database has been cleared. Now processing all thread titles...")
    await process_threads(pool)

def count_tokens(text):
    return len(get_tokenizer().encode(text, disallowed_special=()))

def validate_embedding(embedding):
    if embedding is None:
        return False, "Embedding is None"
    if isinstance(embedding, str):
        # JSON string from VEC_ToText
        try:
            embedding = json.loads(embedding)
        except json.JSONDecodeError:
            return False, "Invalid JSON format"
    if not isinstance(embedding, (list, tuple)):
        return False, f"Invalid type: {type(embedding)}"
    if len(embedding) != EXPECTED_EMBEDDING_LENGTH:
        return False, f"Invalid length: Expected {EXPECTED_EMBEDDING_LENGTH}, got {len(embedding)}"
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
               for value in embedding):
        return False, "Embedding coordinates must be finite numbers"
    if not any(embedding):
        return False, "Embedding must not be a zero vector"
    return True, "Valid"

async def process_single_thread(pool, thread_id):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT title FROM xf_thread WHERE thread_id = %s', (thread_id,))
            result = await cursor.fetchone()
    if not result:
        logging.error("Thread ID %s not found", thread_id)
        return
    await process_batch_with_semaphore(pool, [{'thread_id': thread_id, 'title': result[0]}])

async def process_batch_with_semaphore(pool, batch):
    return await _process_batch(pool, batch, repair=False)


async def _process_batch(pool, batch, repair=False):
    async with semaphore:
        prepared = []
        seen = set()
        for item in batch:
            if item['thread_id'] in seen:
                continue
            seen.add(item['thread_id'])
            tokens = get_tokenizer().encode(item['title'], disallowed_special=())
            if not tokens:
                logging.warning("Skipping empty title for thread_id %s", item['thread_id'])
                continue
            prepared.append((item, tokens[:MAX_TOKENS], len(tokens) > MAX_TOKENS))
        if not prepared:
            return 0
        embeddings = await get_embedding_batch([tokens for _, tokens, _ in prepared])
        if len(embeddings) != len(prepared):
            raise ValueError("Embedding response count does not match the requested inputs")
        for embedding in embeddings:
            valid, reason = validate_embedding(embedding)
            if not valid:
                raise ValueError(f"Invalid embedding response: {reason}")
        async with pool.acquire() as conn:
            # Reads use autocommit so pool release can reuse their connections;
            # writes retain one atomic transaction for the complete batch.
            try:
                await conn.begin()
                async with conn.cursor() as cursor:
                    for (item, _, is_truncated), embedding in zip(prepared, embeddings):
                        vector_json = json.dumps(embedding, allow_nan=False)
                        if repair:
                            await cursor.execute('''
                                UPDATE openai_embeddings
                                SET embedding = VEC_FromText(%s), embedding_length = %s,
                                    is_truncated = %s, last_updated = NOW()
                                WHERE thread_id = %s AND post_id IS NULL
                            ''', (vector_json, len(embedding), is_truncated, item['thread_id']))
                        else:
                            await cursor.execute('''
                                INSERT INTO openai_embeddings
                                (thread_id, embedding, embedding_length, is_truncated, section, last_updated)
                                VALUES (%s, VEC_FromText(%s), %s, %s, %s, NOW())
                                ON DUPLICATE KEY UPDATE
                                    embedding = VALUES(embedding),
                                    embedding_length = VALUES(embedding_length),
                                    is_truncated = VALUES(is_truncated),
                                    last_updated = NOW()
                            ''', (item['thread_id'], vector_json, len(embedding), is_truncated, 0))
                    await conn.commit()
            except BaseException:
                try:
                    # Includes cancellation: finish the transaction before the
                    # lease is returned so the connection remains reusable.
                    await conn.rollback()
                except BaseException:
                    conn.close()
                raise
        logging.info("Processed %s%s threads", len(prepared), " repair" if repair else "")
        return len(prepared)

async def process_threads(pool):
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('SELECT COUNT(*) FROM xf_thread')
                total_threads = (await cursor.fetchone())[0]
                await cursor.execute('SELECT COUNT(DISTINCT thread_id) FROM openai_embeddings WHERE post_id IS NULL')
                completed_threads = (await cursor.fetchone())[0]
        pending_threads = total_threads - completed_threads
        logging.info(f"Total threads: {total_threads}, Completed: {completed_threads}, Pending: {pending_threads}")
        last_processed_id = 0
        with tqdm(total=pending_threads, desc="Processing threads") as pbar:
            while True:
                threads, last_processed_id = await fetch_threads(pool, last_processed_id, BATCH_SIZE * MAX_CONCURRENT_REQUESTS)
                if not threads:
                    break  # No more threads to process

                batches = [threads[i:i + BATCH_SIZE] for i in range(0, len(threads), BATCH_SIZE)]
                tasks = [process_batch_with_semaphore(pool, batch) for batch in batches]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        raise result

                pbar.update(len(threads))

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

async def fetch_threads(pool, last_processed_id, batch_size):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('''
                SELECT t.thread_id, t.title
                FROM xf_thread t
                LEFT JOIN openai_embeddings e ON t.thread_id = e.thread_id AND e.post_id IS NULL
                WHERE e.thread_id IS NULL AND t.thread_id > %s
                ORDER BY t.thread_id
                LIMIT %s
            ''', (last_processed_id, batch_size))
            threads = await cursor.fetchall()
            if not threads:
                return [], last_processed_id
            return [{'thread_id': thread_id, 'title': title} for thread_id, title in threads], threads[-1][0]

async def _fetch_embedding_rows(pool, last_id, batch_size, with_text=False):
    text_column = ", x.title" if with_text else ""
    join = "JOIN xf_thread x ON e.thread_id = x.thread_id" if with_text else ""
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f'''
                SELECT e.id, e.thread_id, VEC_ToText(e.embedding), e.embedding_length{text_column}
                FROM openai_embeddings e {join}
                WHERE e.thread_id IS NOT NULL AND e.post_id IS NULL AND e.id > %s
                ORDER BY e.id
                LIMIT %s
            ''', (last_id, batch_size))
            return await cursor.fetchall()


async def _embedding_count(pool):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT COUNT(*) FROM openai_embeddings WHERE thread_id IS NOT NULL AND post_id IS NULL')
            return (await cursor.fetchone())[0]


async def check_embeddings(pool):
    total = await _embedding_count(pool)
    issues_count, last_id = 0, 0
    with tqdm(total=total, desc="Checking embeddings") as pbar:
        while True:
            batch = await _fetch_embedding_rows(pool, last_id, 100)
            if not batch:
                break
            for rec_id, entity_id, vec_text, stored_length in batch:
                valid, reason = validate_embedding(vec_text)
                if not valid or stored_length != EXPECTED_EMBEDDING_LENGTH:
                    issues_count += 1
                    logging.warning("Issue with thread_id %s: %s", entity_id, reason)
            last_id = batch[-1][0]
            pbar.update(len(batch))
    logging.info("Checked %s embeddings. Found %s issues.", total, issues_count)
    return issues_count


async def process_repair_batch_with_semaphore(pool, batch):
    return await _process_batch(pool, batch, repair=True)


async def repair_embeddings(pool):
    total = await _embedding_count(pool)
    repaired_count, last_id = 0, 0
    with tqdm(total=total, desc="Repairing embeddings") as pbar:
        while True:
            batch = await _fetch_embedding_rows(pool, last_id, BATCH_SIZE * MAX_CONCURRENT_REQUESTS, with_text=True)
            if not batch:
                break
            batch_to_repair = []
            for rec_id, entity_id, vec_text, stored_length, text in batch:
                valid, reason = validate_embedding(vec_text)
                if not valid or stored_length != EXPECTED_EMBEDDING_LENGTH:
                    logging.warning("Invalid embedding for thread_id %s: %s", entity_id, reason)
                    batch_to_repair.append({'thread_id': entity_id, 'title': text})
            tasks = [process_repair_batch_with_semaphore(pool, batch_to_repair[i:i+BATCH_SIZE])
                     for i in range(0, len(batch_to_repair), BATCH_SIZE)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
                repaired_count += result
            last_id = batch[-1][0]
            pbar.update(len(batch))
    logging.info("Repaired %s embeddings out of %s total.", repaired_count, total)
    return repaired_count

async def main(args):
    logging.info("Starting main function")
    pool = await get_db_pool()
    if pool is None:
        raise RuntimeError("Could not obtain database pool")

    try:
        if args.check:
            logging.info("Starting embedding check")
            await check_embeddings(pool)
        elif args.repair:
            logging.info("Starting embedding repair")
            await repair_embeddings(pool)
        elif args.rebuild:
            logging.info("Rebuilding database from scratch")
            await rebuild_database(pool)
        elif args.thread is not None:
            logging.info(f"Processing single thread ID: {args.thread}")
            await process_single_thread(pool, args.thread)
        else:
            logging.info("Processing all threads")
            await process_threads(pool)
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        raise
    finally:
        logging.info("Closing database pool")
        pool.close()
        try:
            await pool.wait_closed()
        finally:
            await close_openai_client()
        logging.info("Database pool closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thread title embedding generation and management script")
    parser.add_argument("--check", action="store_true", help="Check embeddings for issues")
    parser.add_argument("--repair", action="store_true", help="Attempt to repair problematic embeddings")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the embeddings database from scratch")
    parser.add_argument("--thread", type=int, help="Process a single thread ID")
    args = parser.parse_args()

    asyncio.run(main(args))
