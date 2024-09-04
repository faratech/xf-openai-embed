import os
import sys
import openai
import numpy as np
import mysql.connector
from dotenv import load_dotenv

# Load the OpenAI API key and database credentials from the .env file
load_dotenv('/web/.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT"))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# Function to get embedding from OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding  # Access the embedding directly from the response object

# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def fetch_embeddings():
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset='utf8mb4',  # Specify charset
        collation='utf8mb4_unicode_ci'  # Specify a supported collation
    )
    cursor = conn.cursor()
    cursor.execute("SELECT post_id, embedding FROM openai_embeddings")
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    embeddings = [(post_id, np.frombuffer(embedding, dtype=np.float32)) for post_id, embedding in results]
    return embeddings

# Function to find the most similar posts
def find_most_similar_posts(query_embedding, embeddings, top_n=5):
    similarities = []
    for post_id, embedding in embeddings:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((post_id, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 search_posts.py <query>")
        sys.exit(1)

    # Extract the query from command-line arguments
    query = " ".join(sys.argv[1:])
    
    # Generate the embedding for the query
    query_embedding = get_embedding(query)
    
    # Fetch all embeddings from the database
    embeddings = fetch_embeddings()

    # Find the most similar posts
    top_similar_posts = find_most_similar_posts(query_embedding, embeddings)

    # Print the results
    print(f"Top {len(top_similar_posts)} most similar posts:")
    for post_id, similarity in top_similar_posts:
        print(f"Post ID: {post_id}, Similarity: {similarity:.4f}")
