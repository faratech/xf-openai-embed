import os
import mysql.connector
import faiss
import numpy as np
import openai
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv('/web/.env')

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# FAISS index path
FAISS_INDEX_PATH = "faiss_index.bin"

# Elasticsearch setup
es = Elasticsearch(["http://localhost:9200"])

def get_mysql_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        charset='utf8mb4',
        collation='utf8mb4_unicode_ci'
    )

def fetch_embeddings_from_db():
    conn = get_mysql_connection()
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

def create_faiss_index(embeddings, d):
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def save_faiss_index(index, filepath):
    faiss.write_index(index, filepath)

def load_faiss_index(filepath):
    return faiss.read_index(filepath)

def generate_query_embedding(query):
    response = openai.embeddings.create(input=[query], model="text-embedding-ada-002")
    return np.array(response.data[0].embedding)

def search_faiss_index(query_embedding, index, top_k=1):
    query_vector = np.array([query_embedding])
    D, I = index.search(query_vector, top_k)
    return I[0], D[0]

def fetch_details_from_db(ids):
    conn = get_mysql_connection()
    cursor = conn.cursor()

    results = []
    for post_id, thread_id in ids:
        if post_id:
            cursor.execute("SELECT message, post_date, thread_id FROM xf_post WHERE post_id = %s", (post_id,))
            message, post_date, associated_thread_id = cursor.fetchone()
            cursor.execute("SELECT title FROM xf_thread WHERE thread_id = %s", (associated_thread_id,))
            thread_title = cursor.fetchone()[0]
            results.append({
                'type': 'post',
                'post_id': post_id,
                'thread_id': associated_thread_id,
                'message': message,
                'post_date': post_date,
                'thread_title': thread_title
            })
        elif thread_id:
            cursor.execute("SELECT title, last_post_date FROM xf_thread WHERE thread_id = %s", (thread_id,))
            title, last_post_date = cursor.fetchone()
            results.append({
                'type': 'thread',
                'thread_id': thread_id,
                'title': title,
                'last_post_date': last_post_date
            })

    cursor.close()
    conn.close()
    return results

def get_ids_and_details_from_indices(indices, ids):
    relevant_ids = [ids[i] for i in indices]
    return fetch_details_from_db(relevant_ids)

def build_query(query, max_results=1):
    # Convert "now" to an appropriate timestamp
    now_timestamp = int(time.time())

    # Building the query DSL
    query_dsl = {
        "simple_query_string": {
            "query": query,
            "fields": ["title^3", "message"],
            "default_operator": "and"
        }
    }

    # Applying function score for recency with decay
    function_score_dsl = {
        "function_score": {
            "query": query_dsl,
            "functions": [
                {
                    "exp": {
                        "date": {
                            "origin": now_timestamp,
                            "scale": "10d",  # Adjust decay scale as needed
                            "decay": 0.5
                        }
                    }
                }
            ],
            "boost_mode": "sum"
        }
    }

    # Assembling the final query DSL
    search_dsl = {
        "query": function_score_dsl,
        "size": max_results,
        "sort": [
            {"_score": "desc"},
            {"date": "desc"}
        ],
        "_source": ["title", "message", "date", "user", "discussion_id", "node", "post_id"]
    }

    return search_dsl

def search_elasticsearch(query, max_results=1):
    search_body = build_query(query, max_results)
    try:
        response = es.search(index="wf_wf", body=search_body)
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            results.append(source)
        return results
    except Exception as e:
        print(f"An error occurred during the Elasticsearch query: {e}")
        return []

def main():
    if os.path.exists(FAISS_INDEX_PATH):
        index = load_faiss_index(FAISS_INDEX_PATH)
        print(f"Loaded FAISS index from {FAISS_INDEX_PATH}")

        ids, _, _ = fetch_embeddings_from_db()
    else:
        ids, embeddings, dates = fetch_embeddings_from_db()

        d = embeddings.shape[1]
        index = create_faiss_index(embeddings, d)

        save_faiss_index(index, FAISS_INDEX_PATH)
        print(f"Created and saved new FAISS index to {FAISS_INDEX_PATH}")

    query = input("Enter your search query: ")
    query_embedding = generate_query_embedding(query)

    # FAISS search
    top_k_indices, distances = search_faiss_index(query_embedding, index)
    top_faiss_results = get_ids_and_details_from_indices(top_k_indices, ids)

    # Elasticsearch search
    top_es_results = search_elasticsearch(query)

    # Combine and rank the results
    combined_results = top_faiss_results + top_es_results

    # Safe sorting by date, ensuring no KeyError occurs
    def get_sort_key(result):
        if 'post_date' in result:
            return result['post_date']
        elif 'last_post_date' in result:
            return result['last_post_date']
        else:
            return 0  # Default value if both keys are missing

    combined_results = sorted(combined_results, key=get_sort_key, reverse=True)

    # Output results in JSON format
    output = {
        "faiss_results": top_faiss_results,
        "elasticsearch_results": top_es_results,
        "combined_results": combined_results
    }

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
