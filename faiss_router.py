from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import os
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
import faiss
import numpy as np
import openai
import time
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv('/web/.env')

# Initialize Router
router = APIRouter()

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# FAISS index path
FAISS_INDEX_PATH = "faiss_index.bin"

# Elasticsearch setup
es = Elasticsearch(["http://localhost:9200"])

class QueryRequest(BaseModel):
    query: str
    max_results: int = 10

# Create a connection pool
db_config = {
    "host": os.getenv("MYSQL_HOST"),
    "port": int(os.getenv("MYSQL_PORT")),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
    "charset": 'utf8mb4',
    "collation": 'utf8mb4_unicode_ci'
}
connection_pool = MySQLConnectionPool(pool_name="mypool", pool_size=5, **db_config)

def get_db_connection():
    return connection_pool.get_connection()

@lru_cache(maxsize=1)
def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    else:
        ids, embeddings, _ = fetch_embeddings_from_db()
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
        return index

@lru_cache(maxsize=1)
def fetch_embeddings_from_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            oe.post_id,
            oe.thread_id,
            oe.embedding,
            COALESCE(xp.post_date, xt.last_post_date) as date
        FROM
            openai_embeddings oe
        LEFT JOIN
            xf_post xp ON oe.post_id = xp.post_id
        LEFT JOIN
            xf_thread xt ON oe.thread_id = xt.thread_id
    """)
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    ids = []
    embeddings = []
    dates = []
    for post_id, thread_id, embedding_blob, date in results:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        ids.append((post_id, thread_id))
        embeddings.append(embedding)
        dates.append(date)

    return np.array(ids), np.array(embeddings), np.array(dates)

def generate_query_embedding(query):
    response = openai.embeddings.create(input=[query], model="text-embedding-ada-002")
    return np.array(response.data[0].embedding)

def search_faiss_index(query_embedding, index, top_k=10):
    query_vector = np.array([query_embedding])
    D, I = index.search(query_vector, top_k)
    return I[0], D[0]

def fetch_details_from_db(ids):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    post_ids = [id[0] for id in ids if id[0]]
    thread_ids = [id[1] for id in ids if id[1]]

    results = []
    if post_ids:
        cursor.execute("""
            SELECT p.post_id, p.message, p.post_date, p.thread_id, t.title as thread_title
            FROM xf_post p
            JOIN xf_thread t ON p.thread_id = t.thread_id
            WHERE p.post_id IN (%s)
        """ % ','.join(['%s'] * len(post_ids)), post_ids)
        results.extend([{'type': 'post', **row} for row in cursor.fetchall()])

    if thread_ids:
        cursor.execute("""
            SELECT thread_id, title, last_post_date
            FROM xf_thread
            WHERE thread_id IN (%s)
        """ % ','.join(['%s'] * len(thread_ids)), thread_ids)
        results.extend([{'type': 'thread', **row} for row in cursor.fetchall()])

    cursor.close()
    conn.close()
    return results

def get_ids_and_details_from_indices(indices, ids):
    relevant_ids = [ids[i] for i in indices]
    return fetch_details_from_db(relevant_ids)

def build_query(query, max_results=10):
    now_timestamp = int(time.time())
    return {
        "query": {
            "function_score": {
                "query": {
                    "simple_query_string": {
                        "query": query,
                        "fields": ["title^3", "message"],
                        "default_operator": "and"
                    }
                },
                "functions": [
                    {
                        "exp": {
                            "date": {
                                "origin": now_timestamp,
                                "scale": "10d",
                                "decay": 0.5
                            }
                        }
                    }
                ],
                "boost_mode": "sum"
            }
        },
        "size": max_results,
        "sort": [
            {"_score": "desc"},
            {"date": "desc"}
        ],
        "_source": ["title", "message", "date", "user", "discussion_id", "node", "post_id"]
    }

def search_elasticsearch(query, max_results=10):
    search_body = build_query(query, max_results)
    try:
        response = es.search(index="wf_wf", body=search_body)
        return [hit['_source'] for hit in response['hits']['hits']]
    except Exception as e:
        print(f"An error occurred during the Elasticsearch query: {e}")
        return []

@router.post("/search/")
async def search(query_request: QueryRequest):
    query = query_request.query
    max_results = query_request.max_results

    index = load_faiss_index()
    ids, _, _ = fetch_embeddings_from_db()

    query_embedding = generate_query_embedding(query)
    top_k_indices, _ = search_faiss_index(query_embedding, index)
    top_faiss_results = get_ids_and_details_from_indices(top_k_indices, ids)

    return {"faiss_results": top_faiss_results}

@router.post("/elastic/")
async def elastic(query_request: QueryRequest):
    query = query_request.query
    max_results = query_request.max_results

    top_es_results = search_elasticsearch(query, max_results)

    return {"elasticsearch_results": top_es_results}

@router.post("/combined/")
async def combined(query_request: QueryRequest):
    query = query_request.query
    max_results = query_request.max_results

    with ThreadPoolExecutor(max_workers=2) as executor:
        faiss_future = executor.submit(search, query_request)
        es_future = executor.submit(elastic, query_request)

        faiss_results = faiss_future.result()["faiss_results"]
        es_results = es_future.result()["elasticsearch_results"]

    combined_results = faiss_results + es_results
    combined_results = sorted(combined_results, key=lambda x: x.get('post_date', x.get('last_post_date', 0)), reverse=True)

    return {"combined_results": combined_results}
