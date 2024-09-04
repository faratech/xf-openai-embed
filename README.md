# OpenAI Embeddings for XenForo Search (with FAISS)

OpenAI Embeddings for XenForo Search (with FAISS) is an advanced search solution in development designed to enhance the search capabilities of XenForo forums. It leverages FAISS (Facebook AI Similarity Search) for efficient semantic similarity search, combined with Elasticsearch for traditional keyword-based search, providing a powerful hybrid search experience. It utilizes OpenAI embeddings.

## What is FAISS?

FAISS (Facebook AI Similarity Search) is a library developed by Facebook Research that enables efficient similarity search and clustering of dense vectors. In this project, FAISS is used to perform fast and accurate semantic searches based on vector representations (embeddings) of forum posts and thread titles.

## Components

1. **FAISS Router** (`faiss_router.py`): Handles API routes for search operations.
2. **Embedding Generator** (`embed_generate_xf_post.py` and `embed_generate_xf_thread.py`): Generates embeddings for forum posts and thread titles.
3. **Elasticsearch Indexer** (`embed-elastic.py`): Indexes forum data into Elasticsearch.
4. **Vector Search** (`embed-vector-search.py`): Performs vector similarity search using FAISS.
5. **Combined Search** (`embed-faiss-search-with-embed.py`): Implements the hybrid search combining FAISS and Elasticsearch results.

## Requirements

### Software Requirements

- Python 3.7 or higher
- XenForo and XenForo Enhanced Search add-on
- MySQL server (5.7+ or 8.0+)
- Elasticsearch server (7.x or higher)
- Git (for cloning the repository)
- FastAPI

### Account Requirements

- An OpenAI account with API access (for generating embeddings) - https://platform.openai.com/docs/guides/embeddings

### Hardware Requirements

- At least 8GB of RAM (16GB or more recommended for larger forums)
- Sufficient storage space for the database and index files (depends on forum size)

### Python Libraries

- FastAPI
- uvicorn
- aiomysql
- numpy
- faiss-cpu (or faiss-gpu for GPU support)
- openai
- python-dotenv
- elasticsearch
- tqdm
- tiktoken
- tenacity

## Prerequisites

Before installing FAISS for XenForo, ensure you have:

1. Set up and configured your XenForo forum.
2. Installed and configured MySQL server.
3. Installed and configured Elasticsearch server.
4. Obtained an OpenAI API key.
5. Set up FastAPI with web access.
6. Installed Python 3.7+ on your system.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-repo/xf-openai-embed.git
   cd xf-openai-embed
   ```

2. **Set up the environment**:

   Create a `.env` file in the root directory and add the following content:

   ```bash
   OPENAI_API_KEY=your_openai_api_key
   MYSQL_HOST=your_mysql_host
   MYSQL_PORT=3306
   MYSQL_USER=your_mysql_user
   MYSQL_PASSWORD=your_mysql_password
   MYSQL_DATABASE=your_xenforo_database
   ELASTICSEARCH_HOST=http://localhost:9200
   ```

3. **Install the required Python packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Create the database tables**:

   Run the `create_tables.py` script to create the necessary tables in your MySQL database:

   ```bash
   python create_tables.py
   ```

5. **Run the embedding generators**:

   To generate embeddings for XenForo posts and threads, run the following scripts:

   - For posts: `embed_generate_xf_post.py`
   - For threads: `embed_generate_xf_thread.py`

   ```bash
   python embed_generate_xf_post.py
   python embed_generate_xf_thread.py
   ```

6. **Start the FAISS API**:

   Run the FAISS API using FastAPI and uvicorn:

   ```bash
   uvicorn faiss_router:app --reload
   ```

7. **Access the search interface**:

   You can now access the FAISS-powered search interface at:

   ```bash
   http://localhost:8000/faiss/search
   ```

## Environmental Variables

This project requires several environmental variables to be set for proper operation. Here's a list of the required environmental variables:

- **OPENAI_API_KEY**: Your OpenAI API key for generating embeddings
- **MYSQL_HOST**: The hostname of your MySQL server
- **MYSQL_PORT**: The port number of your MySQL server
- **MYSQL_USER**: The username for accessing your MySQL database
- **MYSQL_PASSWORD**: The password for the MySQL user
- **MYSQL_DATABASE**: The name of your XenForo database
- **ELASTICSEARCH_HOST**: The URL of your Elasticsearch server

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

© 2024 Mike Fara, Fara Technologies LLC
