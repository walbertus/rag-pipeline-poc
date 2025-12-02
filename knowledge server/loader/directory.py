from collections.abc import Iterator
from langchain_core.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders import (
    FileSystemBlobLoader,
    PyPDFDirectoryLoader,
)
from langchain_core.documents import Document
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.generic import GenericLoader
from langchain_core.document_loaders.base import BaseLoader


class TextParser(BaseBlobParser):
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        content = blob.as_string()
        metadata = {"source": blob.source}
        yield Document(page_content=content, metadata=metadata)

    def parse(self, blob: Blob) -> list[Document]:
        return list(self.lazy_parse(blob=blob))


class DirectoryLoader(BaseLoader):
    pdf_loader: PyPDFDirectoryLoader
    md_loader: GenericLoader

    def __init__(self, path: str, logger) -> None:
        self.pdf_loader = PyPDFDirectoryLoader(
            path, recursive=False, mode="single", extraction_mode="layout"
        )
        self.md_loader = GenericLoader(
            blob_loader=FileSystemBlobLoader(
                path=path,
                glob="**/*.md",
            ),
            blob_parser=TextParser(),
        )
        self.logger = logger

    def lazy_load(self) -> Iterator[Document]:
        self.logger.debug("Loading PDF documents from %s", self.pdf_loader.path)
        for doc in self.pdf_loader.lazy_load():
            yield doc
        self.logger.debug(
            "Loading Markdown documents from %s", self.md_loader.blob_loader.path
        )
        for doc in self.md_loader.lazy_load():
            yield doc

    def load(self) -> list[Document]:
        return list(self.lazy_load())
