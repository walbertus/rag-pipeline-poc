from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
import lark_oapi as lark
from lark_oapi.api.docx.v1 import RawContentDocumentRequest, GetDocumentRequest
from lark_oapi.api.wiki.v2 import GetNodeSpaceRequest

from typing import Iterator


class LarkSuiteDocLoader(BaseLoader):
    client: lark.Client
    document_id: str

    def __init__(self, client: lark.Client, document_id: str):
        self.client = client
        self.document_id = document_id

    def lazy_load(self) -> Iterator[Document]:
        request_raw = (
            RawContentDocumentRequest.builder().document_id(self.document_id).build()
        )

        response_raw = self.client.docx.v1.document.raw_content(request_raw)
        if not response_raw.success():
            raise RuntimeError(
                f"Failed to fetch document raw content: {response_raw.msg}"
            )

        request_metadata = (
            GetDocumentRequest.builder().document_id(self.document_id).build()
        )

        response_metadata = self.client.docx.v1.document.get(request_metadata)
        if not response_metadata.success():
            raise RuntimeError(
                f"Failed to fetch document metadata: {response_metadata.msg}"
            )

        metadata = {
            "document_id": self.document_id,
            "revision_id": response_metadata.data.document.revision_id,
            "title": response_metadata.data.document.title,
            "source": f"lark-doc://{self.document_id}",
        }

        content = response_raw.data.content

        if content is None:
            content = ""

        yield Document(page_content=str(content), metadata=metadata)


class LarkSuiteWikiLoader(LarkSuiteDocLoader):
    def __init__(self, client: lark.Client, wiki_id: str):
        request = GetNodeSpaceRequest.builder().token(wiki_id).obj_type("wiki").build()

        response = client.wiki.v2.space.get_node(request)
        if not response.success():
            raise RuntimeError(f"Failed to fetch wiki node space: {response.msg}")

        document_id = response.data.node.obj_token
        if not document_id:
            raise RuntimeError("Wiki node space does not contain a valid document ID.")
        super().__init__(client=client, document_id=str(document_id))
