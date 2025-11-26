import logging
from config.config import Config
from loader.loader import DirectoryLoader

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


def main():
    logger = build_logger()
    logger.info("Loading configuration from %s", CONFIG_FILE_PATH)
    config = Config(CONFIG_FILE_PATH)
    logger.setLevel(config.log_level.upper())
    logger.debug(config)

    logger.info("Loading documents from %s", config.dataset_path)
    loader = DirectoryLoader(config.dataset_path, logger)
    for doc in loader.lazy_load():
        logger.info("Loaded document from %s", doc.metadata.get("source", "unknown"))
        logger.debug("Document content: %s", doc.page_content[:100])


if __name__ == "__main__":
    main()
