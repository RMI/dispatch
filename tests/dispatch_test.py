"""Where dispatch tests will go."""
import numpy as np
import pandas as pd

from dispatch import DispatchModel


def setup_dm(fossil_profiles, fossil_specs, re_profiles, re, storage):
    """Setup `DispatchModel`."""
    fossil_profiles.columns = fossil_specs.index
    dm = DispatchModel.from_patio(
        fossil_profiles.sum(axis=1) - re_profiles @ re,
        fossil_profiles=fossil_profiles,
        plant_data=fossil_specs,
        storage_specs=storage,
    )
    return dm


def test_from_patio(fossil_profiles, re_profiles, fossil_specs):
    """Dummy test to quiet pytest."""
    dm = setup_dm(
        fossil_profiles,
        fossil_specs,
        re_profiles,
        np.array([5000.0, 5000.0, 0.0, 0.0]),
        pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
    )
    assert dm


def test_new(fossil_profiles, re_profiles, fossil_specs):
    """Dummy test to quiet pytest."""
    fossil_specs.iloc[
        0, fossil_specs.columns.get_loc("retirement_date")
    ] = fossil_profiles.index.max() - pd.Timedelta(weeks=15)
    self = DispatchModel.new(
        net_load_profile=fossil_profiles.sum(axis=1),
        fossil_plant_specs=fossil_specs,
        storage_specs=pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
    )
    assert self


def test_new_with_dates(fossil_profiles, re_profiles, fossil_specs):
    """Test operating and retirement dates for fossil and storage."""
    fossil_specs.iloc[
        0, fossil_specs.columns.get_loc("retirement_date")
    ] = fossil_profiles.index.max() - pd.Timedelta(weeks=15)
    fossil_specs.loc[8066, "retirement_date"] = pd.Timestamp(
        year=2018, month=12, day=31
    )
    self = DispatchModel.new(
        net_load_profile=fossil_profiles.sum(axis=1),
        fossil_plant_specs=fossil_specs,
        storage_specs=pd.DataFrame(
            [
                (5000, 4, 0.9, pd.Timestamp(year=2016, month=1, day=1)),
                (2000, 8760, 0.5, pd.Timestamp(year=2019, month=1, day=1)),
            ],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff", "operating_date"],
        ),
        jit=True,
    )
    self()
    assert self


def test_low_lost_load(fossil_profiles, re_profiles, fossil_specs):
    """Dummy test to quiet pytest."""
    fossil_profiles.columns = pd.MultiIndex.from_tuples(fossil_specs.index)
    dm = setup_dm(
        fossil_profiles,
        fossil_specs,
        re_profiles,
        np.array([5000.0, 5000.0, 0.0, 0.0]),
        pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
    )
    dm()
    assert (dm.lost_load() / dm.lost_load().sum()).iloc[0] > 0.9999


def test_write_and_read(fossil_profiles, re_profiles, fossil_specs, test_dir):
    """Test that DispatchModel can be written and read."""
    fossil_profiles.columns = fossil_specs.index
    dm = setup_dm(
        fossil_profiles,
        fossil_specs,
        re_profiles,
        np.array([5000.0, 5000.0, 0.0, 0.0]),
        pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
    )
    file = test_dir / "test_obj.zip"
    try:
        dm.to_disk(file)
        x = DispatchModel.from_disk(file)
        x()
        x.to_disk(file, clobber=True)
    except Exception as exc:
        raise exc
    else:
        assert True
    finally:
        if file.exists():
            file.unlink()


def test_marginal_cost(fossil_profiles, re_profiles, fossil_specs):
    """Setup for testing cost and grouper methods."""
    fossil_profiles.columns = fossil_specs.index
    self = setup_dm(
        fossil_profiles,
        fossil_specs,
        re_profiles,
        np.array([5000.0, 5000.0, 0.0, 0.0]),
        pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
    )
    x = self.grouper(self.historical_cost, "technology_description")
    assert not x.empty