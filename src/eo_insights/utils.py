import logging


class EOInsightsException(Exception):
    """Custom exception class for EO Insights errors."""

    pass


def get_logger(name: str = "eo_insights") -> logging.Logger:
    """Set up a simple logger"""
    console = logging.StreamHandler()
    time_format = "%Y-%m-%d %H:%M:%S"
    console.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s: %(message)s",
            datefmt=time_format,
        )
    )

    log = logging.getLogger(name)
    if not log.hasHandlers():
        log.addHandler(console)
    log.setLevel(logging.INFO)

    return log
