import logging


from loader.lark import LarkSuiteDocLoader, LarkSuiteWikiLoader


from langchain_core.document_loaders.base import BaseLoader
from loader.directory import DirectoryLoader

import lark_oapi as lark


class Datasource:
    type: str
    path: str
    url: str
    id: str

    def __init__(self, type: str, path: str = "", url: str = "", id: str = ""):
        self.type = type
        self.path = path
        self.url = url
        self.id = id


class LoaderFactory:
    logger: logging.Logger
    lark_client: lark.Client

    def __init__(self, lark_client: lark.Client, logger: logging.Logger) -> None:
        self.lark_client = lark_client
        self.logger = logger

    def get_loader(self, datasource: Datasource) -> BaseLoader:
        if datasource.type == "directory":
            return DirectoryLoader(datasource.path, self.logger)
        elif datasource.type == "lark-doc":
            return LarkSuiteDocLoader(
                client=self.lark_client,
                document_id=datasource.id,
            )
        elif datasource.type == "lark-wiki":
            return LarkSuiteWikiLoader(
                client=self.lark_client,
                wiki_id=datasource.id,
            )
        else:
            raise ValueError(f"Unsupported source type: {datasource.type}")
