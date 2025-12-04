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
        if not type:
            raise ValueError("Document source type is missing.")

        self.type = type
        self.path = path
        self.url = url
        self.id = id

        if self.type == "directory" and not self.path:
            raise ValueError("Directory source path is missing.")
        elif self.type == "lark-doc" and not self.id:
            raise ValueError("Lark document source id is missing.")
        elif self.type == "lark-wiki" and not self.id:
            raise ValueError("Lark wiki source id is missing.")
        elif self.type not in ["directory", "lark-doc", "lark-wiki"]:
            raise ValueError(f"Unsupported document source type: {self.type}")


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
