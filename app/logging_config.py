import logging


def configure_app_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.WARNING)
    root.setLevel(logging.WARNING)

    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    for logger_name in ("service", "llm_router"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if logger.handlers:
            continue

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
