from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)
import logging
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.config import VectorStoreConfig


class MilvusVectorStore:
    def __init__(
        self,
        config: VectorStoreConfig,
        chunk_size: int,
        chunk_overlap: int,
        embeddings: Embeddings,
        logger: logging.Logger,
    ):
        self.client = MilvusClient(uri=config.url)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_dim = self._get_embedding_dimension(embeddings)
        self.config = config
        self.embeddings = embeddings
        self.logger = logger

        if config.reset_collection:
            self._reset_collection()
        else:
            self._ensure_collection_exists()

        if not self.client.has_collection(self.config.collection_name):
            raise ValueError(
                f"Collection {self.config.collection_name} does not exist in Milvus."
            )

    def _get_embedding_dimension(self, embeddings: Embeddings) -> int:
        sample_text = "sample"
        embedding = embeddings.embed_query(sample_text)
        return len(embedding)

    def _ensure_collection_exists(self) -> None:
        if not self.client.has_collection(self.config.collection_name):
            self.__create_collection()

    def _reset_collection(self) -> None:
        if self.client.has_collection(self.config.collection_name):
            self.client.drop_collection(self.config.collection_name)
        self.__create_collection()

    def __create_collection(self) -> None:
        schema = self.client.create_schema(auto_id=True)
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            description="document id",
            auto_id=True,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=int(self.chunk_size * 1.1),  # give 110% chunk size for safety
            enable_analyzer=self.config.enable_full_text_search,  # toggle full-text search
            description="document chunked text",
        )
        schema.add_field(
            field_name="text_vector_dense",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.vector_dim,
            enable_analyzer=True,
            description="dense vector embedding of the document text",
        )
        schema.add_field(
            field_name="metadata",
            datatype=DataType.JSON,
            description="document metadata",
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="text_vector_dense",
            index_name="text_vector_dense_index",
            index_type="AUTOINDEX",  # Need to compare with IVF_FLAT
            metric_type="IP",  # Need to compare with L2 (Euclidean)
        )

        if self.config.enable_full_text_search:
            schema.add_field(
                field_name="text_vector_sparse",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
                description="sparse vector embedding of the document text for full-text search. auto generated with BM25",
            )
            schema.add_function(
                Function(
                    name="text_bm25_emb",
                    input_field_names=["text"],
                    output_field_names=["text_vector_sparse"],
                    function_type=FunctionType.BM25,  # currently the only function type for sparse in Milvus
                )
            )
            index_params.add_index(
                field_name="text_vector_sparse",
                index_name="text_vector_sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE"
                },  # need to compare another algo
            )

        self.client.create_collection(
            collection_name=self.config.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def __document_to_milvus_format(self, document: Document) -> dict:
        return {
            "text": document.page_content,
            "text_vector_dense": self.embeddings.embed_query(document.page_content),
            "metadata": document.metadata,
        }

    def add_documents(self, documents: list[Document]) -> None:
        self.logger.debug("Adding %d documents to the collection", len(documents))
        data = [self.__document_to_milvus_format(doc) for doc in documents]
        self.client.insert(
            collection_name=self.config.collection_name,
            data=data,
        )
        self.client.flush(collection_name=self.config.collection_name)

    def search(self, query: str, top_k: int = 4) -> list[Document]:
        vector_search = AnnSearchRequest(
            data=[self.embeddings.embed_query(query)],
            anns_field="text_vector_dense",
            param={"nprobe": 10},
            limit=top_k * 2,  # retrieve more to allow reranking
        )
        full_text_search = AnnSearchRequest(
            data=[query],
            anns_field="text_vector_sparse",
            param={"drop_ratio_search": 0.2},
            limit=top_k * 2,  # retrieve more to allow reranking
        )
        searchs = [vector_search]
        if self.config.enable_full_text_search:
            searchs.append(full_text_search)

        search_results = self.client.hybrid_search(
            collection_name=self.config.collection_name,
            reqs=searchs,
            ranker=RRFRanker(),
            limit=top_k,
            output_fields=["text", "metadata"],
        )

        results = []
        for hits in search_results:
            for hit in hits:
                if hit.entity is None:
                    self.logger.warning("Hit entity is unexpected None, skipping.")
                    continue
                else:
                    doc = Document(page_content=hit.entity.get("text"))
                    doc.metadata = hit.entity.get("metadata")
                    results.append(doc)

        return results
