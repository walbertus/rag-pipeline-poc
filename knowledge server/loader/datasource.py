import logging

from langchain_core.document_loaders.base import BaseLoader
from loader.directory import DirectoryLoader


class Datasource:
    type: str
    path: str
    url: str

    def __init__(self, type: str, path: str = "", url: str = ""):
        self.type = type
        self.path = path
        self.url = url


class DatasourceLoader(BaseLoader):
    loader: BaseLoader

    def __init__(self, datasource: Datasource, logger: logging.Logger):
        if datasource.type == "directory":
            self.loader = DirectoryLoader(datasource.path, logger)
        else:
            raise ValueError(f"Unsupported source type: {datasource.type}")

    def lazy_load(self):
        return self.loader.lazy_load()

    def load(self):
        return self.loader.load()
