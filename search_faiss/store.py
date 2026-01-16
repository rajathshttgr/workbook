import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # CPU, Inner Product (IP) Distance
        self.payloads = []  # keeps metadata aligned with vectors

    def add(self, vectors, payloads):
        """
        vectors: List[List[float]]
        payloads: List[Any] (same length as vectors)
        """
        vectors = np.array([vectors]).astype("float32")

        if vectors.shape[1] != self.dim:
            raise ValueError("Vector dimension mismatch")

        self.index.add(vectors)
        self.payloads.append(payloads)

    def search(self, query_vector, top_k=2):
        """
        query_vector: List[float]
        """
        query_vector = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append({"payload": self.payloads[idx], "distance": float(dist)})

        return results
