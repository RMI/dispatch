"""Some helpers for profiles and such."""

from __future__ import annotations

import logging
import warnings

import pandas as pd

from dispatch.constants import ORDERING

LOGGER = logging.getLogger(__name__)


def copy_profile(
    profiles: pd.DataFrame | pd.Series, years: range | tuple
) -> pd.DataFrame | pd.Series:
    """Create multiple 'years' of hourly profile data.

    Args:
        profiles: the profile to make copies of
        years: the years, each of which will be a copy
            of ``profile``.

    Returns:
        Copied profiles.
    """
    dfs = []
    if not isinstance(profiles.index, pd.DatetimeIndex):
        raise AssertionError("profiles.index must be a pd.DatetimeIndex")
    if isinstance(profiles, pd.Series):
        profiles = profiles.to_frame()
    if len(profiles.index.year.unique()) > 1:
        raise AssertionError("`profile` must be for a single year")
    for yr in years:
        dfs.append(
            profiles.assign(
                datetime=lambda x: x.index.map(
                    lambda y: y.replace(year=yr)  # noqa: B023
                ),
            ).set_index("datetime")
        )
    return pd.concat(dfs, axis=0).squeeze()


def zero_profiles_outside_operating_dates(
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
    if (profiles.shape[1] == len(operating_date) == len(retirement_date)) and (
        (capacity_mw is None) or (len(capacity_mw) == profiles.shape[1])
    ):
        pass
    else:
        raise AssertionError(
            "`profiles` must have same number of columns as lengths of "
            "`op_date` and `ret_date`"
        )
    update_operating = operating_date[
        operating_date.fillna(profiles.index.min()) > profiles.index.min()
    ]
    update_retiring = retirement_date[
        retirement_date.fillna(profiles.index.max()) < profiles.index.max()
    ]
    for ix, dt in update_operating.items():
        profiles.loc[profiles.index < dt, ix] = 0.0
    for ix, dt in update_retiring.items():
        profiles.loc[profiles.index > dt, ix] = 0.0
    if capacity_mw is not None:
        return profiles * capacity_mw.to_numpy()
    return profiles


def apply_op_ret_date(
    profiles: pd.DataFrame,
    operating_date: pd.Series,
    retirement_date: pd.Series,
    capacity_mw: pd.Series | None = None,
) -> pd.DataFrame:
    """Renamed to :func:`.zero_profiles_outside_operating_dates`."""
    warnings.warn(
        "This is now an alias for the better named "
        "`zero_profiles_outside_operating_dates` function, this version will be "
        "removed in future versions.",
        FutureWarning,
        stacklevel=2,
    )
    return zero_profiles_outside_operating_dates(
        profiles=profiles,
        operating_date=operating_date,
        retirement_date=retirement_date,
        capacity_mw=capacity_mw,
    )


def dispatch_key(item):
    """Key function for use sorting, including with :mod:`pandas` objects."""
    if isinstance(item, pd.Series):
        return item.astype(str).str.casefold().replace(ORDERING)
    if isinstance(item, pd.Index):
        return pd.Index([ORDERING.get(x.casefold(), str(x)) for x in item])
    return ORDERING.get(item.casefold(), str(item))
