"""Where dispatch tests will go."""
import numpy as np
import pandas as pd
import pytest

from dispatch import DispatchModel, apply_op_ret_date, dfs_from_zip
from dispatch.helpers import dfs_to_zip, idfn


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
        file.unlink(missing_ok=True)


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
    x = self.dispatchable_summary(by=None)
    assert not x.empty


def test_storage_summary(fossil_profiles, re_profiles, fossil_specs, fossil_cost):
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
    x = self.storage_summary(by=None)
    assert not x.empty


def test_full_output(test_dir):
    """Test full_output."""
    patio = dfs_from_zip(test_dir / "data/8fresh.zip")
    self = DispatchModel(**patio)
    self()
    df = self.full_output()
    assert not df.empty


def test_plotting(fossil_profiles, re_profiles, fossil_specs, fossil_cost, test_dir):
    """Testing plotting function."""
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
    self.plot_year(2015)
    x = self.plot_period("2015-01-01", "2015-01-05")
    img_path = test_dir / "plot.pdf"
    try:
        x.write_image(str(img_path))
    except Exception as exc:
        raise AssertionError("unable to write image") from exc
    finally:
        img_path.unlink(missing_ok=True)


def test_dfs_to_from_zip(test_dir):
    """Dfs are same after being written and read back."""
    df_dict = {
        "a": pd.DataFrame(
            [[0, 1], [2, 3]],
            columns=pd.MultiIndex.from_tuples([(0, "a"), (1, "b")]),
        ),
        "b": pd.Series([1, 2, 3, 4]),
    }
    try:
        dfs_to_zip(df_dict, test_dir / "df_test")
        df_load = dfs_from_zip(test_dir / "df_test")
        for a, b in zip(df_dict.values(), df_load.values()):
            assert a.compare(b).empty
    except Exception as exc:
        raise AssertionError("Something broke") from exc
    finally:
        (test_dir / "df_test.zip").unlink(missing_ok=True)


@pytest.mark.parametrize("existing", ["existing", "additions"], ids=idfn)
def test_redispatch_different(test_dir, existing):
    """Test that redispatch and historical are not the same."""
    patio = dfs_from_zip(test_dir / "data/8redispatch.zip")
    self = DispatchModel(**patio)
    self()
    if existing == "existing":
        cols = [tup for tup in self.dispatchable_profiles.columns if tup[0] > 0]
    else:
        cols = [tup for tup in self.dispatchable_profiles.columns if tup[0] < 0]
    comp = (
        self.redispatch.loc[:, cols]
        .round(0)
        .compare(self.dispatchable_profiles.loc[:, cols].round(0))
    )
    assert not comp.empty


@pytest.mark.parametrize("existing", ["existing", "additions"], ids=idfn)
def test_fresh_different(test_dir, existing):
    """Test that dispatch and full capacity profiles (fresh) are not the same."""
    patio = dfs_from_zip(test_dir / "data/8fresh.zip")
    self = DispatchModel(**patio)
    self()
    if existing == "existing":
        cols = [tup for tup in self.dispatchable_profiles.columns if tup[0] > 0]
    else:
        cols = [tup for tup in self.dispatchable_profiles.columns if tup[0] < 0]
    comp = (
        self.redispatch.loc[:, cols]
        .round(0)
        .compare(self.dispatchable_profiles.loc[:, cols].round(0))
    )
    assert not comp.empty, f"dispatch of {existing} failed"


@pytest.mark.skip(reason="for debugging only")
def test_ent(test_dir):
    """Harness for testing dispatch."""
    patio = dfs_from_zip(test_dir / "data/8fresh.zip")
    self = DispatchModel(**patio, jit=False)
    self()
    assert False
