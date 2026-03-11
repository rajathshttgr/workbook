from zoro_client import ZoroClient
from zoro_client import VectorConfig, Distance
import numpy as np

client = ZoroClient(host="localhost", port=6464)
# or
# client = ZoroClient(url="http://localhost:6478")


# Create collection
client.recreate_collection(
    collection_name="test",
    vector_config=VectorConfig(size=100, distance=Distance.COSINE),
)

# Upsert points

import numpy as np

vectors = np.random.rand(5, 100).tolist()

payloads = [
    {"document": "LangChain integrationx"},
    {"document": "LlamaIndex integrationx"},
    {"document": "Hybrid searchx"},
    {"document": "Fast ANN searchx"},
    {"document": "Python for Machine Learningx"},
]

client.upsert_points(
    collection_name="test",
    vectors=vectors,
    ids=[12, 4, 34, 23, 2],
    payloads=payloads,
)

# search query

results = client.search(
    collection_name="test", query_vector=np.random.rand(100).tolist(), limit=2
)


print(client.list_collections())
