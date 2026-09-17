from elasticsearch import Elasticsearch
import time
from datetime import datetime

# Elasticsearch setup
es = Elasticsearch(["http://localhost:9200"])

def build_query(query, max_results=10):
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
        "_source": ["title", "message", "date", "user", "discussion_id", "node"]
    }

    return search_dsl

def search_elasticsearch(query, max_results=10):
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
    query = input("Enter your search query: ")
    top_es_results = search_elasticsearch(query)

    # Output the results
    if top_es_results:
        for result in top_es_results:
            print(result)
    else:
        print("No results found or an error occurred.")

if __name__ == "__main__":
    main()
