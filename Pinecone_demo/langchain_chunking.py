from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv() 

import time
from pinecone import Pinecone

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("API_KEY"))

# Check if the index exists, and create it if it doesn't
if not pc.has_index("zeta-bedrock"):
    pc.create_index_for_model(
        name="zeta-bedrock",
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "content"}
        }
    )

# Access the index
index = pc.Index("zeta-bedrock")

# initialize the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

# load the text from a file and split it into chunks
chunks = splitter.split_text("nips 2017 attention is all you need paper x2 dghw part1 17: of GPUs used, and an estimate of the sustained single-precision \ufb02oating-point capacity of each GPU 5. 6.2 Model Variations To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3. In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads. 5We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively. 8 Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. nips 2017 attention is all you need paper x2 dghw part1 17: of GPUs used, and an estimate of the sustained single-precision \ufb02oating-point capacity of each GPU 5. 6.2 Model Variations To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3. In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads. 5We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively. 8 Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model")

records = []

for i, chunk in enumerate(chunks):
    records.append({
        "_id": f"doc1-{i}",
        "content": chunk,
        "document": "employee_handbook.pdf"
    })

index.upsert_records(
    namespace="docs",
    records=records
)

# print the index stats
print(index.describe_index_stats())

# search for the top 5 records that match the query "what is the refund policy" and rerank them using the bge-reranker-v2-m3 model
results = index.search(
    namespace="docs",
    query={"top_k": 5, "inputs": {"text": "How many GPUs were used in the experiments and what was the sustained single-precision floating-point capacity of each GPU?"}},
    rerank={
        "model": "bge-reranker-v2-m3",
        "top_n": 3,
        "rank_fields": ["content"]
    }
)

for hit in results["result"]["hits"]:
    print(f"{hit.score:.2f}  {hit.fields['content'][0:200]}")