import logging
from config.config import Config
from loader.factory import Datasource, LoaderFactory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp.server.fastmcp import FastMCP

from model.factory import EmbeddingsFactory
from vector_store.milvus import MilvusVectorStore

import lark_oapi as lark


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

    if not from_yaml or "datasource" not in from_yaml:
        logger.error("No datasource found in datasource.yaml")
        return []

    datasources = []
    for source in from_yaml.get("datasource", []):
        datasources.append(
            Datasource(
                type=source.get("type", ""),
                path=source.get("path", ""),
                url=source.get("url", ""),
                id=source.get("id", ""),
            )
        )

    return datasources


def main():
    logger = build_logger()
    logger.info("Loading configuration from %s", CONFIG_FILE_PATH)
    config = Config(CONFIG_FILE_PATH)
    logger.setLevel(config.log_level.upper())
    logger.debug(config)

    lark_log_level = getattr(
        lark.LogLevel, config.log_level.upper(), lark.LogLevel.INFO
    )
    lark_client = (
        lark.Client.builder()
        .domain(config.lark.domain)
        .app_id(config.lark.app_id)
        .app_secret(config.lark.app_secret)
        .log_level(lark_log_level)
        .build()
    )

    datasources = read_datasource(logger)
    loaderFactory = LoaderFactory(lark_client=lark_client, logger=logger)
    loaders = [loaderFactory.get_loader(datasource) for datasource in datasources]

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

    for loader in loaders:
        for doc in loader.lazy_load():
            document = doc  # make a copy from iterator to single Document
            logger.debug("Document content: %s", document.page_content[:20])
            logger.info(
                "Loaded document from %s", document.metadata.get("source", "unknown")
            )
            chunks = splitter.split_documents([document])
            logger.info("Adding %d document chunks to the vector store", len(chunks))
            vector_store.add_documents(chunks)

    queries = [
        "What is Barito project name is inspired from?",
        "Who is the goto financial head of consumer payment infrastructure?",
        "How to query pod metrics?",
    ]
    for query in queries:
        logger.debug("Searching for query: %s", query)
        results = vector_store.search(query=query, top_k=4)
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
