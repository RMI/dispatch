"""PyTest configuration module. Defines useful fixtures, command line args."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from etoolbox.datazip import DataZip

from dispatch import DispatchModel

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
    return df


@pytest.fixture(scope="session")
def fossil_cost(test_dir) -> pd.DataFrame:
    """Fossil Profiles."""
    return pd.read_parquet(test_dir / "data/fossil_cost.parquet")


@pytest.fixture
def ent_fresh(test_dir) -> dict:
    """Fossil Profiles."""
    return DataZip.dfs_from_zip(test_dir / "data/8fresh.zip")


@pytest.fixture
def ent_redispatch(test_dir) -> dict:
    """Fossil Profiles."""
    return DataZip.dfs_from_zip(test_dir / "data/8redispatch.zip")


@pytest.fixture(scope="session", params=["8fresh", "8redispatch"])
def ent_dm(test_dir, request) -> DispatchModel:
    """Fossil Profiles."""
    return DispatchModel(
        **DataZip.dfs_from_zip(test_dir / f"data/{request.param}.zip")
    )()


@pytest.fixture(scope="session")
def ent_out_for_excl_test(test_dir):
    """Dispatchable_summary with excluded generator."""
    ent_redispatch = DataZip.dfs_from_zip(test_dir / "data/8redispatch.zip")

    ent_redispatch["dispatchable_specs"] = ent_redispatch["dispatchable_specs"].assign(
        exclude=lambda x: np.where(x.index == (55380, "CTG1"), True, False),
    )
    self = DispatchModel(**ent_redispatch)
    self()
    df = self.dispatchable_summary(by=None)
    return df.groupby(level=[0, 1]).sum()


@pytest.fixture(scope="session")
def ent_out_for_test(test_dir):
    """Dispatchable_summary without excluded generator."""
    ent_redispatch = DataZip.dfs_from_zip(test_dir / "data/8redispatch.zip")

    self = DispatchModel(**ent_redispatch)
    self()
    df = self.dispatchable_summary(by=None)
    return df.groupby(level=[0, 1]).sum()
