import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Config
N = 100_000
DIM = 100
BATCH_SIZE = 10_000

COLLECTION_NAME = "benchmark"

# Connect (use gRPC for fairness/performance)
client = QdrantClient(host="localhost", port=6333)

# Recreate collection
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
)

# Generate data
vectors = np.random.random((N, DIM)).astype("float32")

# -----------------------
# INSERT BENCHMARK
# -----------------------
print("Starting Qdrant batch insertion benchmark...")

start_time = time.time()

for i in range(0, N, BATCH_SIZE):
    batch_vectors = vectors[i : i + BATCH_SIZE]

    points = [
        PointStruct(id=i + j, vector=vec.tolist())
        for j, vec in enumerate(batch_vectors)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)

end_time = time.time()

total_time = end_time - start_time
throughput = N / total_time

print(f"\n[QDRANT RESULTS]")
print(f"Inserted {N} vectors in {total_time:.2f} sec")
print(f"Throughput: {throughput:.2f} vectors/sec")
