from dotenv import load_dotenv
import os

load_dotenv()  

import time
from pinecone import Pinecone

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("API_KEY"))

# Check if the index exists, and create it if it doesn't
if not pc.has_index("spirited-juniper"):
    pc.create_index_for_model(
        name="spirited-juniper",
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "content"}
        }
    )

# Access the index
index = pc.Index("spirited-juniper")

# upsert some records into the index
index.upsert_records(
    namespace="docs",
    records=[
        {"_id": "rec1", "content": "Refund requests must be submitted within 30 days.", "text": "policy"},
        {"_id": "rec2", "content": "Enterprise support responds within 4 hours.", "text": "policy"},
        {"_id": "rec3", "content": "New employees receive 15 days PTO in year one.", "text": "hr"},
        {"_id": "rec4", "content": "Production deployments require team lead approval.", "text": "ops"},
        {"_id": "rec5", "content": "API rate limit: 1000 requests/minute on Pro tier.", "text": "specs"},
    ]
)

# print the index stats
print(index.describe_index_stats())

# search for the top 5 records that match the query "what is the refund policy" and rerank them using the bge-reranker-v2-m3 model
results = index.search(
    namespace="docs",
    query={"top_k": 5, "inputs": {"text": "what is the refund policy"}},
    rerank={
        "model": "bge-reranker-v2-m3",
        "top_n": 3,
        "rank_fields": ["content"]
    }
)

for hit in results["result"]["hits"]:
    print(f"{hit.score:.2f}  {hit.fields['content']}")
