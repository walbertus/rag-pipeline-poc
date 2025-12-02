import logging
from config.config import Config
from loader.datasource import Datasource, DatasourceLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp.server.fastmcp import FastMCP

from model.factory import EmbeddingsFactory
from vector_store.milvus import MilvusVectorStore

CONFIG_FILE_PATH = "config.yaml"


def build_logger() -> logging.Logger:
    logger = logging.getLogger("knowledge_server")
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def read_datasource(logger: logging.Logger) -> list[Datasource]:
    """
    This is demo function to gather datasource from yaml.
    TODO: document sources should be from database and user input later on.
    """
    import yaml

    with open("datasource.yaml", "r") as f:
        from_yaml = yaml.safe_load(f)

    datasources = []
    for source in from_yaml.get("datasource", []):
        if "type" not in source:
            raise ValueError("Document source type is missing.")
        if source["type"] == "directory" and "path" not in source:
            raise ValueError("Directory source path is missing.")
        datasources.append(
            Datasource(
                source["type"], source.get("path", None), source.get("url", None)
            )
        )

    return datasources


def main():
    logger = build_logger()
    logger.info("Loading configuration from %s", CONFIG_FILE_PATH)
    config = Config(CONFIG_FILE_PATH)
    logger.setLevel(config.log_level.upper())
    logger.debug(config)

    datasources = read_datasource(logger)
    loaders = [DatasourceLoader(datasource, logger) for datasource in datasources]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    for doc in docs:
        logger.info("Loaded document from %s", doc.metadata.get("source", "unknown"))
        logger.debug("Document content: %s", doc.page_content[:100])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    embeddings = EmbeddingsFactory.get_embeddings(config.embeddings)

    vector_store = MilvusVectorStore(
        config.vector_store,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        embeddings=embeddings,
        logger=logger,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Adding %d document chunks to the vector store", len(chunks))
    vector_store.add_documents(chunks)

    logger.debug(
        "Searching for query: %s", "What is Barito project name is inspired from?"
    )
    results = vector_store.search(
        query="What is Barito project name is inspired from?", top_k=4
    )
    logger.debug("Search results total: %s", len(results))
    for i, result in enumerate(results):
        logger.info("Result %d: %s", i + 1, result.page_content[:200])

    logger.info(
        "Searching for query: %s",
        "Who is the goto financial head of consumer payment infrastructure?",
    )
    results = vector_store.search(
        query="Who is the goto financial head of consumer payment infrastructure?",
        top_k=4,
    )
    logger.debug("Search results total: %s", len(results))
    for i, result in enumerate(results):
        logger.info("Result %d: %s", i + 1, result.page_content[:200])

    mcp_server = FastMCP("KnowledgeServer")

    @mcp_server.tool()
    def query_knowledge_base(query: str, top_k: int = 4) -> list[str]:
        """Query the knowledge base to gather relevant information."""
        logger.info("Received query: %s", query)
        results = vector_store.search(query=query, top_k=top_k)
        logger.info("Returning %d results", len(results))
        return [str(result) for result in results]

    mcp_server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
