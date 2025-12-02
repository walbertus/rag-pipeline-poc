from langchain_core.embeddings import Embeddings
from config.config import EmbeddingsConfig
import requests


class ModelGarden(Embeddings):
    def __init__(self, config: EmbeddingsConfig):
        self.config = config

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            self.config.url + "/embed",
            headers={"Content-Type": "application/json"},
            json={
                "model": self.config.model,
                "input": texts,
                "encoding_format": "float",
            },
        )
        response.raise_for_status()

        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
