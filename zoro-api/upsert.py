import requests
import time
import os
from openai import OpenAI
from dotenv import load_dotenv


VECTOR_DB_URL = "http://localhost:6464"
COLLECTION_NAME = "new_collection"
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text: str):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def main():
    # Ensure DB is reachable
    status = requests.get(VECTOR_DB_URL).status_code
    if status != 200:
        print("Vector DB not reachable")
        return

    # Delete collection if exists
    requests.delete(f"{VECTOR_DB_URL}/collections/{COLLECTION_NAME}")

    # Create collection
    payload = {
        "collection_name": COLLECTION_NAME,
        "dimension": DIMENSION,
        "distance": "cosine"
    }
    response = requests.post(f"{VECTOR_DB_URL}/collections", json=payload)
    print("Create collection:", response.json())

    # Sample documents
    documents = [
        "Neural networks are the foundation of deep learning models.",
        "Transformers power most modern large language models.",
        "Vector databases enable fast similarity search over embeddings.",
        "Cosine similarity measures the angle between two vectors.",
        "Python is the most popular language for machine learning.",
        "FastAPI is used to build high-performance backend APIs.",
        "Docker containers simplify application deployment.",
        "Kubernetes manages containerized workloads at scale.",
        "PostgreSQL is a reliable relational database system.",
        "Redis is commonly used for caching and session storage.",
        "Football players train daily to improve stamina and skills.",
        "The FIFA World Cup is watched by millions of fans worldwide.",
        "Classical music improves focus and cognitive performance.",
        "Stock markets react strongly to economic news.",
        "Healthy eating and exercise improve overall fitness."
    ]

    # Upsert documents
    for idx, doc in enumerate(documents):
        embedding = embed_text(doc)

        payload = {
            "vectors": [embedding],
            "ids": [idx],
            "payload": [{"document": doc}]
        }

        response = requests.post(
            f"{VECTOR_DB_URL}/collections/{COLLECTION_NAME}/points",
            json=payload
        )

        if response.status_code != 200:
            print("Insert failed:", response.text)
            break

        print(f"Inserted document {idx}")

    print("Done inserting documents.")


if __name__ == "__main__":
    main()
