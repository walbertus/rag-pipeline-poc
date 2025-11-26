CONFIG_FILE_PATH = "config.yaml"
import logging
from config.config import Config

# from pprint import p


def main():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info("Loading configuration from %s", CONFIG_FILE_PATH)
    config = Config(CONFIG_FILE_PATH)
    print("Hello from knowledge-server!")
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    logger.debug(config)


if __name__ == "__main__":
    main()
