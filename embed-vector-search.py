#!/usr/bin/env python3
"""
Vector similarity search using MariaDB native VECTOR functions.

Uses VEC_DISTANCE() for in-database cosine similarity - no need to load
all embeddings into RAM.
"""

import os
import sys
import json
import openai
import mysql.connector
from dotenv import load_dotenv

# Load environment
load_dotenv('/web/.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """Get embedding from OpenAI."""
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def find_similar_posts(query: str, top_n: int = 10):
    """
    Find similar posts using MariaDB native vector distance.

    Uses VEC_DISTANCE with cosine similarity directly in SQL -
    much faster than loading all vectors into Python.
    """
    # Get query embedding
    print(f"Generating embedding for: {query}")
    query_embedding = get_embedding(query)

    # Convert to JSON for VEC_FromText
    query_vector_json = '[' + ','.join(str(f) for f in query_embedding) + ']'

    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset='utf8mb4',
        collation='utf8mb4_unicode_ci'
    )
    cursor = conn.cursor()

    # Use VEC_DISTANCE for in-database similarity search
    # cosine distance = 1 - cosine_similarity, so lower is better
    query = """
        SELECT
            e.post_id,
            e.thread_id,
            p.message,
            t.title,
            VEC_DISTANCE(e.embedding, VEC_FromText(%s)) AS distance
        FROM openai_embeddings e
        LEFT JOIN xf_post p ON e.post_id = p.post_id
        LEFT JOIN xf_thread t ON e.thread_id = t.thread_id
        WHERE e.embedding IS NOT NULL
          AND e.post_id IS NOT NULL
        ORDER BY distance ASC
        LIMIT %s
    """

    cursor.execute(query, (query_vector_json, top_n))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 embed-vector-search.py <query> [top_n]")
        print("Example: python3 embed-vector-search.py 'blue screen error' 10")
        sys.exit(1)

    query = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    results = find_similar_posts(query, top_n)

    print(f"\nTop {len(results)} similar posts (lower distance = more similar):\n")
    print("-" * 80)

    for post_id, thread_id, message, title, distance in results:
        similarity = 1 - distance  # Convert distance to similarity
        title_preview = (title[:60] + '...') if title and len(title) > 60 else (title or 'N/A')
        msg_preview = (message[:100] + '...') if message and len(message) > 100 else (message or 'N/A')
        msg_preview = msg_preview.replace('\n', ' ')

        print(f"Post ID: {post_id} | Thread: {thread_id} | Similarity: {similarity:.4f}")
        print(f"Title: {title_preview}")
        print(f"Message: {msg_preview}")
        print("-" * 80)


if __name__ == "__main__":
    main()
