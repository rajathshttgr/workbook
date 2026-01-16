from store import VectorStore
from openai import OpenAI
from dotenv import load_dotenv
import os

# embedding configs
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536

# sample document for testing
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
    "Healthy eating and exercise improve overall fitness.",
]

# initialize openai embeddings model
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_text(text: str):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


# initialize vector store / faiss instance
store = VectorStore(dim=DIMENSION)


# upsert documents to faiss - store

for idx, doc in enumerate(documents):
    vectors = embed_text(doc)
    payload_metadata = {"doc_id": idx, "document": doc}

    # add to faiss
    store.add(vectors, payload_metadata)


# Search Query
query_doc = "Which programming language is best for AI and machine learning?"
query_vectors = embed_text(query_doc)

results = store.search(query_vectors, top_k=3)

for result in results:
    print(result)

# expected result
# Python is the most popular language for machine learning.
# Transformers power most modern large language models.
# Neural networks are the foundation of deep learning models.
