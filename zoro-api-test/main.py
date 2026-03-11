import requests
import time
import numpy as np

url = "http://localhost:6464"


def main():

    status = requests.get(url).status_code

    # ensure connection is established
    if status != 200:
        return 0

    # create collection

    requests.delete(
        url + "/collections" + "/test"
    )  # make sure collection already not created

    payload = {"collection_name": "test", "dimension": 1536, "distance": "cosine"}
    response = requests.post(url + "/collections", json=payload)
    print(response.json())

    ## upsert 10 point to collection
    start = time.time()

    for i in range(10):

        embedding = np.random.rand(1536).tolist()

        metadata = f"some random text {embedding[0:2]}"

        payload = {
            "vectors": [embedding],
            "ids": [i],
            "payload": [{"document": metadata}],
        }

        response = requests.post(url + "/collections/test/points", json=payload)
        if response.status_code != 200:
            break

    # search
    embedding = np.random.rand(1536).tolist()
    payload = {
        "vectors": embedding,
        "limit": 3,
    }

    response = requests.post(url + "/collections/test/points/search", json=payload)

    end = time.time()
    print(f"Elapsed: {end - start:.2f} seconds")
    print(response.json())

    # count points
    # response=requests.get(url+"/collections/test/count")
    # print(response.json())


if __name__ == "__main__":
    main()
