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

# Initialize AsyncOpenAI with the API key
# Tenacity is the only retry layer; SDK retries would multiply each attempt.
openai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, max_retries=0)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize tokenizer for the specific model
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

# Constants
MAX_TOKENS = 8191
EXPECTED_EMBEDDING_LENGTH = 1536
BATCH_SIZE = 50  # Database write batch; API requests also enforce the token budget
MAX_CONCURRENT_REQUESTS = 10  # Adjust this based on your rate limits and system capabilities

# Create a semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

MAX_BATCH_TOKENS = 300_000
MAX_BATCH_INPUTS = 2048


@retry(
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)),
    wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), reraise=True,
)
async def _request_embeddings(inputs, model):
    return await openai.embeddings.create(input=inputs, model=model)


async def get_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    return (await get_embedding_batch([text], model=model))[0]


async def get_embedding_batch(texts, model="text-embedding-3-small") -> list[list[float]]:
    # Token arrays avoid repeated tokenization and preserve exact truncation limits.
    inputs = [(tokenizer.encode(text, disallowed_special=()) if isinstance(text, str) else text)[:MAX_TOKENS]
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
            autocommit=False,
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
            await cursor.execute("DELETE FROM openai_embeddings WHERE post_id IS NOT NULL AND section = 0")
            await conn.commit()
    logging.info("Database has been cleared. Now processing all posts...")
    await process_posts(pool)

def count_tokens(text):
    return len(tokenizer.encode(text, disallowed_special=()))

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

async def process_single_post(pool, post_id):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT message FROM xf_post WHERE post_id = %s', (post_id,))
            result = await cursor.fetchone()
    if not result:
        logging.error("Post ID %s not found", post_id)
        return
    await process_batch_with_semaphore(pool, [{'post_id': post_id, 'message': result[0]}])

async def process_batch_with_semaphore(pool, batch):
    return await _process_batch(pool, batch, repair=False)


async def _process_batch(pool, batch, repair=False):
    async with semaphore:
        prepared = []
        seen = set()
        for item in batch:
            if item['post_id'] in seen:
                continue
            seen.add(item['post_id'])
            tokens = tokenizer.encode(item['message'], disallowed_special=())
            if not tokens:
                logging.warning("Skipping empty message for post_id %s", item['post_id'])
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
            async with conn.cursor() as cursor:
                for (item, _, is_truncated), embedding in zip(prepared, embeddings):
                    vector_json = json.dumps(embedding, allow_nan=False)
                    if repair:
                        await cursor.execute('''
                            UPDATE openai_embeddings
                            SET embedding = VEC_FromText(%s), embedding_length = %s,
                                is_truncated = %s, last_updated = NOW()
                            WHERE post_id = %s AND section = 0
                        ''', (vector_json, len(embedding), is_truncated, item['post_id']))
                    else:
                        await cursor.execute('''
                            INSERT INTO openai_embeddings
                            (post_id, embedding, embedding_length, is_truncated, section, last_updated)
                            VALUES (%s, VEC_FromText(%s), %s, %s, %s, NOW())
                            ON DUPLICATE KEY UPDATE
                                embedding = VALUES(embedding),
                                embedding_length = VALUES(embedding_length),
                                is_truncated = VALUES(is_truncated),
                                last_updated = NOW()
                        ''', (item['post_id'], vector_json, len(embedding), is_truncated, 0))
                await conn.commit()
        logging.info("Processed %s%s posts", len(prepared), " repair" if repair else "")
        return len(prepared)

async def process_posts(pool):
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('SELECT COUNT(*) FROM xf_post')
                total_posts = (await cursor.fetchone())[0]
                await cursor.execute('SELECT COUNT(DISTINCT post_id) FROM openai_embeddings WHERE section = 0')
                completed_posts = (await cursor.fetchone())[0]
        pending_posts = total_posts - completed_posts
        logging.info(f"Total posts: {total_posts}, Completed: {completed_posts}, Pending: {pending_posts}")
        last_processed_id = 0
        with tqdm(total=pending_posts, desc="Processing posts") as pbar:
            while True:
                posts, last_processed_id = await fetch_posts(pool, last_processed_id, BATCH_SIZE * MAX_CONCURRENT_REQUESTS)
                if not posts:
                    break  # No more posts to process

                batches = [posts[i:i + BATCH_SIZE] for i in range(0, len(posts), BATCH_SIZE)]
                tasks = [process_batch_with_semaphore(pool, batch) for batch in batches]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        raise result

                pbar.update(len(posts))

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

async def fetch_posts(pool, last_processed_id, batch_size):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('''
                SELECT p.post_id, p.message
                FROM xf_post p
                LEFT JOIN openai_embeddings e ON p.post_id = e.post_id AND e.section = 0
                WHERE e.post_id IS NULL AND p.post_id > %s
                ORDER BY p.post_id
                LIMIT %s
            ''', (last_processed_id, batch_size))
            posts = await cursor.fetchall()
            if not posts:
                return [], last_processed_id
            return [{'post_id': post_id, 'message': message} for post_id, message in posts], posts[-1][0]

async def _fetch_embedding_rows(pool, last_id, batch_size, with_text=False):
    text_column = ", x.message" if with_text else ""
    join = "JOIN xf_post x ON e.post_id = x.post_id" if with_text else ""
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f'''
                SELECT e.id, e.post_id, VEC_ToText(e.embedding), e.embedding_length{text_column}
                FROM openai_embeddings e {join}
                WHERE e.post_id IS NOT NULL AND e.section = 0 AND e.id > %s
                ORDER BY e.id
                LIMIT %s
            ''', (last_id, batch_size))
            return await cursor.fetchall()


async def _embedding_count(pool):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT COUNT(*) FROM openai_embeddings WHERE post_id IS NOT NULL AND section = 0')
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
                    logging.warning("Issue with post_id %s: %s", entity_id, reason)
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
                    logging.warning("Invalid embedding for post_id %s: %s", entity_id, reason)
                    batch_to_repair.append({'post_id': entity_id, 'message': text})
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
        elif args.post is not None:
            logging.info(f"Processing single post ID: {args.post}")
            await process_single_post(pool, args.post)
        else:
            logging.info("Processing all posts")
            await process_posts(pool)
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        raise
    finally:
        logging.info("Closing database pool")
        pool.close()
        await pool.wait_closed()
        logging.info("Database pool closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding generation and management script")
    parser.add_argument("--check", action="store_true", help="Check embeddings for issues")
    parser.add_argument("--repair", action="store_true", help="Attempt to repair problematic embeddings")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the embeddings database from scratch")
    parser.add_argument("--post", type=int, help="Process a single post ID")
    args = parser.parse_args()

    asyncio.run(main(args))
