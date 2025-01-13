import requests
import os

url = "https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
output_path = "msmarco_dataset.tar.gz"

response = requests.get(url, stream=True)
with open(output_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        file.write(chunk)

print("Download complete!")