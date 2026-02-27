import logging


def configure_app_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.WARNING)
    root.setLevel(logging.WARNING)

    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    for logger_name in ("agent", "llm_router", "api"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        if logger.handlers:
            continue

        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
