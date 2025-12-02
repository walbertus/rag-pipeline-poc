from config.config import EmbeddingsConfig
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings


class EmbeddingsFactory:
    @staticmethod
    def get_embeddings(config: EmbeddingsConfig) -> Embeddings:
        if config.source == "ollama":
            return OllamaEmbeddings(model=config.model)
        # Add other embedding sources as needed
        raise ValueError(f"Unsupported embeddings source: {config.source}")
