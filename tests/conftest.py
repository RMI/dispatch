"""PyTest configuration module. Defines useful fixtures, command line args."""
import logging
from pathlib import Path

import pandas as pd
import pytest

logger = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add package-specific command line options to pytest.

    This is slightly magical -- pytest has a hook that will run this function
    automatically, adding any options defined here to the internal pytest options that
    already exist.
    """
    parser.addoption(
        "--sandbox",
        action="store_true",
        default=False,
        help="Flag to indicate that the tests should use a sandbox.",
    )


@pytest.fixture(scope="session")
def test_dir() -> Path:
    """Return the path to the top-level directory containing the tests.

    This might be useful if there's test data stored under the tests directory that
    you need to be able to access from elsewhere within the tests.

    Mostly this is meant as an example of a fixture.
    """
    return Path(__file__).parent


@pytest.fixture(scope="session")
def re_profiles(test_dir) -> pd.DataFrame:
    """RE Profiles."""
    return pd.read_parquet(test_dir / "data/re_profiles.parquet")


@pytest.fixture(scope="session")
def fossil_profiles(test_dir) -> pd.DataFrame:
    """Fossil Profiles."""
    return pd.read_parquet(test_dir / "data/fossil_profiles.parquet")


@pytest.fixture(scope="session")
def fossil_specs(test_dir) -> pd.DataFrame:
    """Fossil Profiles."""
    df = pd.read_parquet(test_dir / "data/plant_specs.parquet")
    df.columns = [pd.Timestamp(x) if x[0] == "2" else x for x in df.columns]
    return df
