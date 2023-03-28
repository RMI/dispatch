"""Tests for :mod:`dispatch.helpers`."""

import pandas as pd
from dispatch.helpers import copy_profile, dispatch_key


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
