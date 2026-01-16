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
    query = "Which programming language is best for AI and machine learning?"

    # Embed query
    query_embedding = embed_text(query)

    payload = {
        "vectors": query_embedding,
        "limit": 3
    }

    start = time.time()
    response = requests.post(
        f"{VECTOR_DB_URL}/collections/{COLLECTION_NAME}/points/search",
        json=payload
    )
    end = time.time()

    print(f"Search time: {end - start:.3f} seconds")
    print("Results:")
    print(response.json())


if __name__ == "__main__":
    main()
