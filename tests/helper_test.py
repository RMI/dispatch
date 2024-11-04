"""Tests for :mod:`dispatch.helpers`."""

import numpy as np
import pandas as pd

from dispatch.helpers import copy_profile, dispatch_key
from src.dispatch.helpers import zero_profiles_outside_operating_dates


def zero_profiles_outside_operating_dates_slow(
    profiles: pd.DataFrame,
    operating_date: pd.Series,
    retirement_date: pd.Series,
    capacity_mw: pd.Series | None = None,
) -> pd.DataFrame:
    """Zero profile unless it is between operating and retirement date.

    Args:
        profiles: profiles of plants with a DatetimeIndex
        operating_date: in service date for each plant, the index of operating date is
            used throughout
        retirement_date: retirement date for each plant
        capacity_mw: capacity of each plant (only used when ``profiles`` are
            normalized)

    Returns:
        Profiles reflecting operating and retirement dates.
    """
    if not isinstance(profiles.index, pd.DatetimeIndex):
        raise AssertionError("profiles.index must be a pd.DatetimeIndex")
    if capacity_mw is None:
        capacity_mw = pd.Series(1, index=operating_date.index, name="capacity_mw")
    if profiles.shape[1] == len(operating_date) == len(retirement_date):
        pass
    else:
        raise AssertionError(
            "`profiles` must have same number of columns as lengths of "
            "`op_date` and `ret_date`"
        )
    # duplicate the DatetimeIndex so it is the same shape as `profiles`
    dt_idx = pd.concat(
        [profiles.index.to_series()] * profiles.shape[1],
        axis=1,
    ).to_numpy(dtype=np.datetime64)
    return pd.DataFrame(
        (
            (
                dt_idx
                <= retirement_date.fillna(profiles.index.max()).to_numpy(
                    dtype=np.datetime64
                )
            )
            & (
                dt_idx
                >= operating_date.fillna(profiles.index.min()).to_numpy(
                    dtype=np.datetime64
                )
            )
        )
        * profiles.to_numpy()
        * capacity_mw.to_numpy(),
        index=profiles.index,
        columns=operating_date.index,
    )


def test_zero_profiles_outside_operating_dates(fossil_specs, fossil_profiles):
    """Test impact of total_var_mwh.

    Test that changing total_var_mwh changes dispatch but not cost calculations.
    """
    fossil_profiles.columns = fossil_specs.index
    fast = zero_profiles_outside_operating_dates(
        fossil_profiles.copy(),
        fossil_specs.operating_date.copy(),
        fossil_specs.retirement_date.copy(),
    )
    slow = zero_profiles_outside_operating_dates_slow(
        fossil_profiles, fossil_specs.operating_date, fossil_specs.retirement_date
    )
    assert np.all(np.isclose(fast, slow))


def test_copy_profile(ent_fresh):
    """Test copy_profile."""
    load = ent_fresh["load_profile"]
    yr_range = range(2050, 2055)
    out = copy_profile(load.loc["2025"], yr_range)
    assert tuple(out.index.year.unique()) == tuple(yr_range)


def test_dispatch_key_series():
    """Test dispatch key func on :class:`pandas.Series`."""
    idx = (
        pd.Series(["May", "April", "Solar", "Gas CC", "March"])
        .sort_values(key=dispatch_key)
        .reset_index(drop=True)
    )
    pd.testing.assert_series_equal(
        idx, pd.Series(["Gas CC", "Solar", "March", "April", "May"])
    )


def test_dispatch_key_idx():
    """Test dispatch key func on :class:`pandas.Index`."""
    idx = pd.Index(["May", "April", "Solar", "Gas CC", "March"]).sort_values(
        key=dispatch_key
    )
    pd.testing.assert_index_equal(
        idx, pd.Index(["Gas CC", "Solar", "March", "April", "May"])
    )


def test_dispatch_key_list():
    """Test dispatch key func on :class:`list`."""
    idx = sorted(["May", "April", "Solar", "Gas CC", "March"], key=dispatch_key)
    assert idx == ["Gas CC", "Solar", "March", "April", "May"]
