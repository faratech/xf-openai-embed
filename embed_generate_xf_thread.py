import os
import asyncio
import aiomysql
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm
import logging
import tiktoken
import argparse
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Load environment variables
load_dotenv('/web/.env')

# Initialize AsyncOpenAI with the API key
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize tokenizer for the specific model
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

# Constants
MAX_TOKENS = 8191
EXPECTED_EMBEDDING_LENGTH = 1536
BATCH_SIZE = 2048
MAX_CONCURRENT_REQUESTS = 10  # Adjust this based on your rate limits and system capabilities

# Create a semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def get_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    response = await openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def get_embedding_batch(texts: list[str], model="text-embedding-3-small") -> list[list[float]]:
    try:
        response = await openai.embeddings.create(
            input=texts,
            model=model
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        logging.error(f"Error in get_embedding_batch: {e}")
        raise

async def get_db_pool():
    logging.info("Attempting to create database connection pool")
    try:
        pool = await aiomysql.create_pool(
            host=os.getenv("MYSQL_HOST"),
            port=int(os.getenv("MYSQL_PORT")),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            db=os.getenv("MYSQL_DATABASE"),
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
            await cursor.execute("DELETE FROM openai_embeddings WHERE thread_id IS NOT NULL")
            await conn.commit()
    logging.info("Database has been cleared. Now processing all thread titles...")
    await process_threads(pool)

def count_tokens(text):
    return len(tokenizer.encode(text))

def validate_embedding(embedding):
    if embedding is None:
        return False, "Embedding is None"
    if isinstance(embedding, bytes):
        if len(embedding) % 4 != 0:  # Check if byte length is a multiple of 4 (size of float32)
            return False, f"Invalid buffer size: {len(embedding)} bytes is not a multiple of 4"
        embedding = np.frombuffer(embedding, dtype=np.float32)
    if len(embedding) != EXPECTED_EMBEDDING_LENGTH:
        return False, f"Invalid length: Expected {EXPECTED_EMBEDDING_LENGTH}, got {len(embedding)}"
    if not np.isfinite(embedding).all():
        return False, "Embedding contains NaNs or Infinities"
    if np.max(np.abs(embedding)) > 100:
        return False, f"Extreme values detected: max abs value = {np.max(np.abs(embedding))}"
    return True, "Valid"

async def process_single_thread(pool, thread_id):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT title FROM xf_thread WHERE thread_id = %s', (thread_id,))
            result = await cursor.fetchone()
            if not result:
                logging.error(f"Thread ID {thread_id} not found")
                return

            title = result[0]
            tokens = count_tokens(title)
            if tokens > MAX_TOKENS:
                truncated_title = tokenizer.decode(tokenizer.encode(title)[:MAX_TOKENS])
                logging.info(f"Processing truncated thread ID {thread_id}")
            else:
                truncated_title = title
                logging.info(f"Processing full thread ID {thread_id}")

            embedding = await get_embedding(truncated_title)
            is_valid, validation_message = validate_embedding(embedding)
            if not is_valid:
                logging.error(f"Invalid embedding for thread ID {thread_id}: {validation_message}")
                return

            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

            await cursor.execute('''
                INSERT INTO openai_embeddings
                (thread_id, embedding, embedding_length, is_truncated, section, last_updated)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                    embedding = VALUES(embedding),
                    embedding_length = VALUES(embedding_length),
                    is_truncated = VALUES(is_truncated)
            ''', (thread_id, embedding_blob, len(embedding), tokens > MAX_TOKENS, 0))
            await conn.commit()
            logging.info(f"Processed thread ID {thread_id}")

async def process_batch_with_semaphore(pool, batch):
    async with semaphore:
        titles = [thread['title'] for thread in batch]
        embeddings = await get_embedding_batch(titles)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                for thread, embedding in zip(batch, embeddings):
                    thread_id = thread['thread_id']
                    tokens = count_tokens(thread['title'])
                    is_truncated = tokens > MAX_TOKENS

                    embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

                    await cursor.execute('''
                        INSERT INTO openai_embeddings
                        (thread_id, embedding, embedding_length, is_truncated, section, last_updated)
                        VALUES (%s, %s, %s, %s, %s, NOW())
                        ON DUPLICATE KEY UPDATE
                            embedding = VALUES(embedding),
                            embedding_length = VALUES(embedding_length),
                            is_truncated = VALUES(is_truncated)
                    ''', (thread_id, embedding_blob, len(embedding), is_truncated, 0))

                await conn.commit()

        logging.info(f"Processed batch of {len(batch)} threads")

async def process_threads(pool):
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('SELECT COUNT(*) FROM xf_thread')
                total_threads = (await cursor.fetchone())[0]
                await cursor.execute('SELECT COUNT(DISTINCT thread_id) FROM openai_embeddings')
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
                await asyncio.gather(*tasks)

                pbar.update(len(threads))

    except Exception as e:
        logging.error(f"An error occurred: {e}")

async def fetch_threads(pool, last_processed_id, batch_size):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('''
                SELECT t.thread_id, t.title
                FROM xf_thread t
                LEFT JOIN (SELECT DISTINCT thread_id FROM openai_embeddings) e ON t.thread_id = e.thread_id
                WHERE e.thread_id IS NULL AND t.thread_id > %s
                ORDER BY t.thread_id
                LIMIT %s
            ''', (last_processed_id, batch_size))
            threads = await cursor.fetchall()
            if not threads:
                return [], last_processed_id
            return [{'thread_id': thread_id, 'title': title} for thread_id, title in threads], threads[-1][0]

async def check_embeddings(pool):
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('SELECT COUNT(*) FROM openai_embeddings WHERE thread_id IS NOT NULL')
                total_embeddings = (await cursor.fetchone())[0]
                issues_count = 0
                with tqdm(total=total_embeddings, desc="Checking embeddings") as pbar:
                    for offset in range(0, total_embeddings, 100):
                        await cursor.execute(f'''
                            SELECT thread_id, embedding, embedding_length
                            FROM openai_embeddings
                            WHERE thread_id IS NOT NULL
                            LIMIT {offset}, 100
                        ''')
                        batch = await cursor.fetchall()
                        for thread_id, embedding_blob, stored_length in batch:
                            if embedding_blob is None:
                                logging.warning(f"Embedding is None for thread_id {thread_id}")
                                issues_count += 1
                                continue
                            try:
                                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                                is_valid, message = validate_embedding(embedding)
                                if not is_valid or stored_length != EXPECTED_EMBEDDING_LENGTH:
                                    issues_count += 1
                                    logging.warning(f"Issue with thread_id {thread_id}: {message}")
                            except ValueError as e:
                                logging.error(f"Error processing embedding for thread_id {thread_id}: {e}")
                                issues_count += 1
                        pbar.update(len(batch))
                logging.info(f"Checked {total_embeddings} embeddings. Found {issues_count} issues.")
    except Exception as e:
        logging.error(f"An error occurred during check: {e}")

async def process_repair_batch_with_semaphore(pool, batch):
    async with semaphore:
        titles = [item['title'] for item in batch]
        try:
            embeddings = await get_embedding_batch(titles)
        except Exception as e:
            logging.error(f"Error getting embeddings from OpenAI: {e}")
            return

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                for item, embedding in zip(batch, embeddings):
                    thread_id = item['thread_id']
                    tokens = count_tokens(item['title'])
                    is_truncated = tokens > MAX_TOKENS

                    embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

                    await cursor.execute('''
                        UPDATE openai_embeddings
                        SET embedding = %s, embedding_length = %s, is_truncated = %s, last_updated = NOW()
                        WHERE thread_id = %s
                    ''', (embedding_blob, len(embedding), is_truncated, thread_id))

                await conn.commit()

        logging.info(f"Processed repair batch of {len(batch)} threads")

async def repair_embeddings(pool):
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('SELECT COUNT(*) FROM openai_embeddings WHERE thread_id IS NOT NULL')
                total_embeddings = (await cursor.fetchone())[0]
                repaired_count = 0
                with tqdm(total=total_embeddings, desc="Repairing embeddings") as pbar:
                    for offset in range(0, total_embeddings, BATCH_SIZE * MAX_CONCURRENT_REQUESTS):
                        await cursor.execute(f'''
                            SELECT openai_embeddings.thread_id, embedding, embedding_length, title
                            FROM openai_embeddings
                            JOIN xf_thread ON openai_embeddings.thread_id = xf_thread.thread_id
                            LIMIT {offset}, {BATCH_SIZE * MAX_CONCURRENT_REQUESTS}
                        ''')
                        batch = await cursor.fetchall()
                        batch_to_repair = []
                        for thread_id, embedding_blob, stored_length, title in batch:
                            is_valid, validation_message = validate_embedding(embedding_blob)
                            if not is_valid or stored_length != EXPECTED_EMBEDDING_LENGTH:
                                logging.warning(f"Invalid embedding for thread_id {thread_id}: {validation_message}")
                                tokens = count_tokens(title)
                                if tokens > MAX_TOKENS:
                                    title = tokenizer.decode(tokenizer.encode(title)[:MAX_TOKENS])
                                batch_to_repair.append({'thread_id': thread_id, 'title': title})
                                repaired_count += 1

                        if batch_to_repair:
                            repair_batches = [batch_to_repair[i:i + BATCH_SIZE] for i in range(0, len(batch_to_repair), BATCH_SIZE)]
                            tasks = [process_repair_batch_with_semaphore(pool, repair_batch) for repair_batch in repair_batches]
                            await asyncio.gather(*tasks)

                        pbar.update(len(batch))

                logging.info(f"Repaired {repaired_count} embeddings out of {total_embeddings} total.")
    except Exception as e:
        logging.error(f"An error occurred during repair: {e}")
        raise  # Re-raise the exception for higher-level handling

async def main(args):
    logging.info("Starting main function")
    pool = await get_db_pool()
    if pool is None:
        logging.error("Could not obtain database pool. Exiting.")
        return

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
    finally:
        logging.info("Closing database pool")
        pool.close()
        await pool.wait_closed()
        logging.info("Database pool closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thread title embedding generation and management script")
    parser.add_argument("--check", action="store_true", help="Check embeddings for issues")
    parser.add_argument("--repair", action="store_true", help="Attempt to repair problematic embeddings")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the embeddings database from scratch")
    parser.add_argument("--thread", type=int, help="Process a single thread ID")
    args = parser.parse_args()

    asyncio.run(main(args))
