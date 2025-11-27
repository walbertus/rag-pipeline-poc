import yaml


def load_config(filepath):
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
    return config


class Config:
    vector_store: "VectorStoreConfig"
    log_level: str
    dataset_path: str
    chunk_size: int
    chunk_overlap: int

    def __init__(self, filepath):
        config = load_config(filepath)
        if config is None:
            raise ValueError("Failed to load configuration.")
        self.vector_store = VectorStoreConfig(config)
        self.log_level = config.get("log_level", "INFO")
        self.dataset_path = config.get("dataset_path", None)
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)


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
