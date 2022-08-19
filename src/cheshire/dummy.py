"""A dummy module so coverage has something to cover."""
import logging

logger = logging.getLogger(__name__)


def do_something(a: int = 0, b: int = 0) -> int:
    """A dummy function that does something."""
    logger.info("Don't just do something, stand there!")
    c = a + b
    if a == 2 and b == 2:
        c = 5
    return c
