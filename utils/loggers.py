import logging
from tqdm import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""
    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def setup_tqdm_logger(name, level):
    """Set up a TQDM logger to avoid TQDM progress bar being interrupted by logger's output to console

    Args:
        name:   Name of the logger
        level:  The logging level

    Returns:
        logger: The tqdm logger
    """
    # Get logger and handlers
    logger = logging.getLogger(name)
    handler = TqdmLoggingHandler()
    
    # Setup logger
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger