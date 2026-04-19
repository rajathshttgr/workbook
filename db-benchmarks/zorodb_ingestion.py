import time
import numpy as np
from zoro_client import ZoroClient, VectorConfig, Distance

# Config
N = 100_000
DIM = 100
BATCH_SIZE = 10_000
COLLECTION_NAME = "benchmark"

# Local server
client = ZoroClient(host="localhost", port=6464)

# Recreate collection
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vector_config=VectorConfig(size=DIM, distance=Distance.COSINE),
)

vectors = np.random.random((N, DIM)).astype("float32")

print("Starting Zoro batch insertion benchmark...")

start_time = time.time()

total_inserted = 0

for i in range(0, N, BATCH_SIZE):
    batch_vectors = vectors[i : i + BATCH_SIZE]

    batch_vectors_list = batch_vectors.tolist()
    batch_ids = list(range(i, i + len(batch_vectors_list)))

    resp = client.upsert_points(
        collection_name=COLLECTION_NAME,
        vectors=batch_vectors_list,
        ids=batch_ids,
        payloads=None,
    )

    total_inserted += len(batch_vectors_list)

end_time = time.time()

total_time = end_time - start_time
throughput = total_inserted / total_time

print(f"\n[ZORO-DB RESULTS")
print(f"Inserted {total_inserted} vectors in {total_time:.2f} sec")
print(f"Throughput: {throughput:.2f} vectors/sec")
