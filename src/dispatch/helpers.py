"""Some helpers for profiles and such."""

from __future__ import annotations

import pandas as pd


def copy_profile(
    profiles: pd.DataFrame | pd.Series, years: range | tuple
) -> pd.DataFrame | pd.Series:
    """Create multiple 'years' of hourly profile data.

    Args:
        profiles: the profile to make copies of
        years: the years, each of which will be a copy
            of ``profile``.

    Returns: Copied profiles.

    """
    dfs = []
    assert isinstance(profiles.index, pd.DatetimeIndex)
    if isinstance(profiles, pd.Series):
        profiles = profiles.to_frame()
    if len(profiles.index.year.unique()) > 1:
        raise AssertionError("`profile` must be for a single year")
    for yr in years:
        dfs.append(
            profiles.assign(
                datetime=lambda x: x.index.map(lambda y: y.replace(year=yr)),
            ).set_index("datetime")
        )
    return pd.concat(dfs, axis=0).squeeze()


def apply_op_ret_date(
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

    Returns: Profiles reflecting operating and retirement dates.

    """
    assert isinstance(profiles.index, pd.DatetimeIndex)
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
    ).to_numpy()
    return pd.DataFrame(
        (
            (dt_idx <= retirement_date.fillna(profiles.index.max()).to_numpy())
            & (dt_idx >= operating_date.fillna(profiles.index.min()).to_numpy())
        )
        * profiles.to_numpy()
        * capacity_mw.to_numpy(),
        index=profiles.index,
        columns=operating_date.index,
    )


def _str_cols(df, *args):
    return df.set_axis(list(map(str, range(df.shape[1]))), axis="columns")


def _to_frame(df, n):
    return df.to_frame(name=n)


def _null(df, *args):
    return df
