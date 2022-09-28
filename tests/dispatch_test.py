"""Where dispatch tests will go."""
import numpy as np
import pandas as pd

from dispatch import DispatchModel, apply_op_ret_date


def setup_dm(fossil_profiles, fossil_specs, fossil_cost, re_profiles, re, storage):
    """Setup `DispatchModel`."""
    fossil_profiles.columns = fossil_specs.index
    fossil_profiles = apply_op_ret_date(
        fossil_profiles, fossil_specs.operating_date, fossil_specs.retirement_date
    )
    dm = DispatchModel.from_patio(
        fossil_profiles.sum(axis=1) - re_profiles @ re,
        dispatchable_profiles=fossil_profiles,
        cost_data=fossil_cost,
        plant_data=fossil_specs,
        storage_specs=storage,
    )
    return dm


def test_from_patio(fossil_profiles, re_profiles, fossil_specs, fossil_cost):
    """Dummy test to quiet pytest."""
    dm = setup_dm(
        fossil_profiles,
        fossil_specs,
        fossil_cost,
        re_profiles,
        np.array([5000.0, 5000.0, 0.0, 0.0]),
        pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
    )
    assert dm


def test_new(fossil_profiles, re_profiles, fossil_specs, fossil_cost):
    """Dummy test to quiet pytest."""
    fossil_specs.iloc[
        0, fossil_specs.columns.get_loc("retirement_date")
    ] = fossil_profiles.index.max() - pd.Timedelta(weeks=15)
    self = DispatchModel.from_fresh(
        net_load_profile=fossil_profiles.sum(axis=1),
        dispatchable_specs=fossil_specs,
        dispatchable_cost=fossil_cost,
        storage_specs=pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
        jit=True,
    )
    self()
    assert self


def test_new_with_dates(fossil_profiles, re_profiles, fossil_specs, fossil_cost):
    """Test operating and retirement dates for fossil and storage."""
    fossil_specs.iloc[
        0, fossil_specs.columns.get_loc("retirement_date")
    ] = fossil_profiles.index.max() - pd.Timedelta(weeks=15)
    fossil_specs.loc[8066, "retirement_date"] = pd.Timestamp(
        year=2018, month=12, day=31
    )
    self = DispatchModel.from_fresh(
        net_load_profile=fossil_profiles.sum(axis=1),
        dispatchable_specs=fossil_specs,
        dispatchable_cost=fossil_cost,
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


def test_low_lost_load(fossil_profiles, re_profiles, fossil_specs, fossil_cost):
    """Dummy test to quiet pytest."""
    fossil_profiles.columns = pd.MultiIndex.from_tuples(fossil_specs.index)
    dm = setup_dm(
        fossil_profiles,
        fossil_specs,
        fossil_cost,
        re_profiles,
        np.array([5000.0, 5000.0, 0.0, 0.0]),
        pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
    )
    dm()
    assert (dm.lost_load() / dm.lost_load().sum()).iloc[0] > 0.998


def test_write_and_read(
    fossil_profiles, re_profiles, fossil_specs, test_dir, fossil_cost
):
    """Test that DispatchModel can be written and read."""
    fossil_profiles.columns = fossil_specs.index
    dm = setup_dm(
        fossil_profiles,
        fossil_specs,
        fossil_cost,
        re_profiles,
        np.array([5000.0, 5000.0, 0.0, 0.0]),
        pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
    )
    file = test_dir / "test_obj.zip"
    try:
        dm.to_file(file)
        x = DispatchModel.from_file(file)
        x()
        x.to_file(file, clobber=True, include_output=True)
    except Exception as exc:
        raise exc
    else:
        assert True
    finally:
        if file.exists():
            file.unlink()


def test_marginal_cost(fossil_profiles, re_profiles, fossil_specs, fossil_cost):
    """Setup for testing cost and grouper methods."""
    fossil_profiles.columns = fossil_specs.index
    self = setup_dm(
        fossil_profiles,
        fossil_specs,
        fossil_cost,
        re_profiles,
        np.array([5000.0, 5000.0, 0.0, 0.0]),
        pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
    )
    x = self.grouper(self.historical_cost, "technology_description")
    assert not x.empty


def test_operations_summary(fossil_profiles, re_profiles, fossil_specs, fossil_cost):
    """Setup for testing cost and grouper methods."""
    fossil_profiles.columns = fossil_specs.index
    self = setup_dm(
        fossil_profiles,
        fossil_specs,
        fossil_cost,
        re_profiles,
        np.array([5000.0, 5000.0, 0.0, 0.0]),
        pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
    )
    self()
    x = self.operations_summary(by=None)
    assert not x.empty
