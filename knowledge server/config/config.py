import yaml

def load_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config

class Config:
    def __init__(self, filepath):
        config = load_config(filepath)
        if config is None:
            raise ValueError("Failed to load configuration.")
        self.vector_store = VectorStoreConfig(config)
        self.log_level = config.get('log_level', 'INFO')

class VectorStoreConfig:
    def __init__(self, config: dict):
        vector_store_config = config.get('vector_store', None)
        if vector_store_config is None:
            raise ValueError("Vector store configuration is missing in the config file.")
        
        self.type = vector_store_config.get('type', None)
        self.host = vector_store_config.get('host', None)
        self.port = vector_store_config.get('port', None)
        self.collection_name = vector_store_config.get('collection_name', None)
