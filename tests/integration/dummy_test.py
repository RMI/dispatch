"""A dummy integration test so pytest has something to do."""
import logging

import pytest

from cheshire.dummy import do_something

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "a,b,expected_c",
    [
        (1, 1, 2),
        (3, 5, 8),
        (13, 22, 35),
    ],
)
def test_something(a: int, b: int, expected_c: int) -> None:
    """Test the dummy function from our dummy module to generate coverage.

    This function also demonstrates how to parametrize a test.
    """
    c = do_something(a, b)
    assert c == expected_c  # nosec: B101
