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
            await cursor.execute("DELETE FROM openai_embeddings")
            await conn.commit()
    logging.info("Database has been cleared. Now processing all posts...")
    await process_posts(pool)

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

async def process_single_post(pool, post_id):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('SELECT message FROM xf_post WHERE post_id = %s', (post_id,))
            result = await cursor.fetchone()
            if not result:
                logging.error(f"Post ID {post_id} not found")
                return

            message = result[0]
            tokens = count_tokens(message)
            if tokens > MAX_TOKENS:
                truncated_message = tokenizer.decode(tokenizer.encode(message)[:MAX_TOKENS])
                logging.info(f"Processing truncated post ID {post_id}")
            else:
                truncated_message = message
                logging.info(f"Processing full post ID {post_id}")

            embedding = await get_embedding(truncated_message)
            is_valid, validation_message = validate_embedding(embedding)
            if not is_valid:
                logging.error(f"Invalid embedding for post ID {post_id}: {validation_message}")
                return

            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

            await cursor.execute('''
                INSERT INTO openai_embeddings
                (post_id, embedding, embedding_length, is_truncated, section, last_updated)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                    embedding = VALUES(embedding),
                    embedding_length = VALUES(embedding_length),
                    is_truncated = VALUES(is_truncated)
            ''', (post_id, embedding_blob, len(embedding), tokens > MAX_TOKENS, 0))
            await conn.commit()
            logging.info(f"Processed post ID {post_id}")

async def process_batch_with_semaphore(pool, batch):
    async with semaphore:
        texts = [post['message'] for post in batch]
        embeddings = await get_embedding_batch(texts)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                for post, embedding in zip(batch, embeddings):
                    post_id = post['post_id']
                    tokens = count_tokens(post['message'])
                    is_truncated = tokens > MAX_TOKENS

                    embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

                    await cursor.execute('''
                        INSERT INTO openai_embeddings
                        (post_id, embedding, embedding_length, is_truncated, section, last_updated)
                        VALUES (%s, %s, %s, %s, %s, NOW())
                        ON DUPLICATE KEY UPDATE
                            embedding = VALUES(embedding),
                            embedding_length = VALUES(embedding_length),
                            is_truncated = VALUES(is_truncated)
                    ''', (post_id, embedding_blob, len(embedding), is_truncated, 0))

                await conn.commit()

        logging.info(f"Processed batch of {len(batch)} posts")

async def process_posts(pool):
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('SELECT COUNT(*) FROM xf_post')
                total_posts = (await cursor.fetchone())[0]
                await cursor.execute('SELECT COUNT(DISTINCT post_id) FROM openai_embeddings')
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
                await asyncio.gather(*tasks)

                pbar.update(len(posts))

    except Exception as e:
        logging.error(f"An error occurred: {e}")

async def fetch_posts(pool, last_processed_id, batch_size):
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('''
                SELECT p.post_id, p.message
                FROM xf_post p
                LEFT JOIN (SELECT DISTINCT post_id FROM openai_embeddings) e ON p.post_id = e.post_id
                WHERE e.post_id IS NULL AND p.post_id > %s
                ORDER BY p.post_id
                LIMIT %s
            ''', (last_processed_id, batch_size))
            posts = await cursor.fetchall()
            if not posts:
                return [], last_processed_id
            return [{'post_id': post_id, 'message': message} for post_id, message in posts], posts[-1][0]

async def check_embeddings(pool):
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('SELECT COUNT(*) FROM openai_embeddings')
                total_embeddings = (await cursor.fetchone())[0]
                issues_count = 0
                with tqdm(total=total_embeddings, desc="Checking embeddings") as pbar:
                    for offset in range(0, total_embeddings, 100):
                        await cursor.execute(f'''
                            SELECT post_id, embedding, embedding_length
                            FROM openai_embeddings
                            LIMIT {offset}, 100
                        ''')
                        batch = await cursor.fetchall()
                        for post_id, embedding_blob, stored_length in batch:
                            if embedding_blob is None:
                                logging.warning(f"Embedding is None for post_id {post_id}")
                                issues_count += 1
                                continue
                            try:
                                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                                is_valid, message = validate_embedding(embedding)
                                if not is_valid or stored_length != EXPECTED_EMBEDDING_LENGTH:
                                    issues_count += 1
                                    logging.warning(f"Issue with post_id {post_id}: {message}")
                            except ValueError as e:
                                logging.error(f"Error processing embedding for post_id {post_id}: {e}")
                                issues_count += 1
                        pbar.update(len(batch))
                logging.info(f"Checked {total_embeddings} embeddings. Found {issues_count} issues.")
    except Exception as e:
        logging.error(f"An error occurred during check: {e}")

async def process_repair_batch_with_semaphore(pool, batch):
    async with semaphore:
        texts = [item['message'] for item in batch]
        try:
            embeddings = await get_embedding_batch(texts)
        except Exception as e:
            logging.error(f"Error getting embeddings from OpenAI: {e}")
            return

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                for item, embedding in zip(batch, embeddings):
                    post_id = item['post_id']
                    tokens = count_tokens(item['message'])
                    is_truncated = tokens > MAX_TOKENS

                    embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

                    await cursor.execute('''
                        UPDATE openai_embeddings
                        SET embedding = %s, embedding_length = %s, is_truncated = %s, last_updated = NOW()
                        WHERE post_id = %s
                    ''', (embedding_blob, len(embedding), is_truncated, post_id))
                
                await conn.commit()
        
        logging.info(f"Processed repair batch of {len(batch)} posts")

async def repair_embeddings(pool):
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('SELECT COUNT(*) FROM openai_embeddings')
                total_embeddings = (await cursor.fetchone())[0]
                repaired_count = 0
                with tqdm(total=total_embeddings, desc="Repairing embeddings") as pbar:
                    for offset in range(0, total_embeddings, BATCH_SIZE * MAX_CONCURRENT_REQUESTS):
                        await cursor.execute(f'''
                            SELECT openai_embeddings.post_id, embedding, embedding_length, message
                            FROM openai_embeddings
                            JOIN xf_post ON openai_embeddings.post_id = xf_post.post_id
                            LIMIT {offset}, {BATCH_SIZE * MAX_CONCURRENT_REQUESTS}
                        ''')
                        batch = await cursor.fetchall()
                        batch_to_repair = []
                        for post_id, embedding_blob, stored_length, message in batch:
                            is_valid, validation_message = validate_embedding(embedding_blob)
                            if not is_valid or stored_length != EXPECTED_EMBEDDING_LENGTH:
                                logging.warning(f"Invalid embedding for post_id {post_id}: {validation_message}")
                                tokens = count_tokens(message)
                                if tokens > MAX_TOKENS:
                                    message = tokenizer.decode(tokenizer.encode(message)[:MAX_TOKENS])
                                batch_to_repair.append({'post_id': post_id, 'message': message})
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
        elif args.post is not None:
            logging.info(f"Processing single post ID: {args.post}")
            await process_single_post(pool, args.post)
        else:
            logging.info("Processing all posts")
            await process_posts(pool)
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
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
