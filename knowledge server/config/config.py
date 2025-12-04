import yaml


def load_config(filepath):
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
    return config


class Config:
    vector_store: "VectorStoreConfig"
    log_level: str
    chunk_size: int
    chunk_overlap: int
    embeddings: "EmbeddingsConfig"

    def __init__(self, filepath):
        config = load_config(filepath)
        if config is None:
            raise ValueError("Failed to load configuration.")
        self.vector_store = VectorStoreConfig(config)
        self.log_level = config.get("log_level", "INFO")
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.embeddings = EmbeddingsConfig(config)
        self.lark = LarkConfig(config)


class EmbeddingsConfig:
    source: str
    model: str

    def __init__(self, config: dict):
        embeddings_config = config.get("embeddings", None)
        if embeddings_config is None:
            raise ValueError("Embeddings configuration is missing in the config file.")

        self.source = embeddings_config.get("source", None)
        self.model = embeddings_config.get("model", None)


class LarkConfig:
    domain: str
    app_id: str
    app_secret: str

    def __init__(self, config: dict):
        lark_config = config.get("lark", None)
        if lark_config is None:
            raise ValueError("Lark configuration is missing in the config file.")

        self.domain = lark_config.get("domain", None)
        self.app_id = lark_config.get("app_id", None)
        self.app_secret = lark_config.get("app_secret", None)


class VectorStoreConfig:
    type: str
    url: str
    collection_name: str
    reset_collection: bool
    enable_full_text_search: bool

    def __init__(self, config: dict):
        vector_store_config = config.get("vector_store", None)
        if vector_store_config is None:
            raise ValueError(
                "Vector store configuration is missing in the config file."
            )

        self.type = vector_store_config.get("type", None)
        self.url = vector_store_config.get("url", None)
        self.collection_name = vector_store_config.get("collection_name", None)
        self.reset_collection = vector_store_config.get("reset_collection", False)
        self.enable_full_text_search = vector_store_config.get(
            "enable_full_text_search", False
        )
