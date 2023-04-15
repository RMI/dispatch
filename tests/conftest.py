"""PyTest configuration module.

Defines useful fixtures, command line args.
"""
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dispatch import DispatchModel, zero_profiles_outside_operating_dates
from etoolbox.datazip import DataZip

logger = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add package-specific command line options to pytest.

    This is slightly magical -- pytest has a hook that will run this
    function automatically, adding any options defined here to the
    internal pytest options that already exist.
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

    This might be useful if there's test data stored under the tests
    directory that you need to be able to access from elsewhere within
    the tests.

    Mostly this is meant as an example of a fixture.
    """
    return Path(__file__).parent


@pytest.fixture(scope="session")
def temp_dir(test_dir) -> Path:
    """Return the path to a temp directory that gets deleted on teardown."""
    out = test_dir / "temp"
    out.mkdir(exist_ok=True)
    yield out
    shutil.rmtree(out)


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
    return dict(DataZip(test_dir / "data/8fresh.zip").items())


@pytest.fixture
def ent_redispatch(test_dir) -> dict:
    """Fossil Profiles."""
    return dict(DataZip(test_dir / "data/8redispatch.zip").items())


@pytest.fixture(scope="session", params=["8fresh", "8redispatch"])
def ent_dm(test_dir, request) -> tuple[str, DispatchModel]:
    """Fossil Profiles."""
    indicator = {"8fresh": "f", "8redispatch": "r"}
    data = dict(DataZip(test_dir / f"data/{request.param}.zip").items())
    return (
        indicator[request.param],
        DispatchModel(**data)(),
    )


@pytest.fixture(scope="session")
def ent_out_for_excl_test(test_dir):
    """Dispatchable_summary with excluded generator."""
    ent_redispatch = dict(DataZip(test_dir / "data/8redispatch.zip").items())

    ent_redispatch["dispatchable_specs"] = ent_redispatch["dispatchable_specs"].assign(
        exclude=lambda x: np.where(x.index == (55380, "CTG1"), True, False),
    )
    self = DispatchModel(**ent_redispatch)
    self()
    df = self.dispatchable_summary(by=None)
    return df.groupby(level=[0, 1]).sum()


@pytest.fixture(scope="session")
def ent_out_for_no_limit_test(test_dir):
    """Dispatchable_summary with excluded generator."""
    ent_redispatch = dict(DataZip(test_dir / "data/8redispatch.zip").items())

    ent_redispatch["dispatchable_specs"] = ent_redispatch["dispatchable_specs"].assign(
        no_limit=lambda x: np.where(x.index == (55380, "CTG1"), True, False),
    )
    self = DispatchModel(**ent_redispatch)
    self()
    df = self.dispatchable_summary(by=None)
    return df.groupby(level=[0, 1]).sum()


@pytest.fixture(scope="session")
def ent_out_for_test(test_dir):
    """Dispatchable_summary without excluded generator."""
    ent_out_for_test = dict(DataZip(test_dir / "data/8redispatch.zip").items())

    self = DispatchModel(**ent_out_for_test)
    self()
    df = self.dispatchable_summary(by=None)
    return df.groupby(level=[0, 1]).sum()


@pytest.fixture(scope="session")
def mini_dm(fossil_profiles, fossil_specs, fossil_cost, re_profiles):
    """Setup `DispatchModel`."""
    re = np.array([5000.0, 5000.0, 0.0, 0.0])

    fossil_profiles.columns = fossil_specs.index
    fossil_profiles = zero_profiles_outside_operating_dates(
        fossil_profiles, fossil_specs.operating_date, fossil_specs.retirement_date
    )
    dm = DispatchModel(
        load_profile=fossil_profiles.sum(axis=1) - re_profiles @ re,
        dispatchable_profiles=fossil_profiles,
        dispatchable_cost=fossil_cost,
        dispatchable_specs=fossil_specs,
        storage_specs=pd.DataFrame(
            [
                (-1, "es", 5000, 4, 0.9, fossil_profiles.index.min()),
                (-2, "es", 2000, 8760, 0.5, fossil_profiles.index.min()),
            ],
            columns=[
                "plant_id_eia",
                "generator_id",
                "capacity_mw",
                "duration_hrs",
                "roundtrip_eff",
                "operating_date",
            ],
        ).set_index(["plant_id_eia", "generator_id"]),
    )()
    return dm
