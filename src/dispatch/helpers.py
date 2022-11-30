"""Some helpers for profiles and such."""
from __future__ import annotations

import logging
import warnings
from io import BytesIO
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from etoolbox.datazip import DataZip as EtDataZip

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


def _str_cols(df, *args):
    return df.set_axis(list(map(str, range(df.shape[1]))), axis="columns")


def dispatch_key(item):
    """Key function for use sorting, including with :mod:`pandas` objects."""
    if isinstance(item, pd.Series):
        return item.str.casefold().replace(ORDERING)
    if isinstance(item, pd.Index):
        return pd.Index([ORDERING.get(x.casefold(), str(x)) for x in item])
    return ORDERING.get(item.casefold(), str(item))


class ObjMeta(NamedTuple):
    """NamedTuple for testing."""

    module: str
    qualname: str
    constructor: str | None = None


class DataZip(EtDataZip):
    """SubClass of :class:`ZipFile` with methods for easier use with :mod:`pandas`.

    z = DataZip(file, mode="r", compression=ZIP_STORED, allowZip64=True,
                compresslevel=None)

    """

    def __init__(self, file: str | Path | BytesIO, mode="r", *args, **kwargs):
        """Open the ZIP file.

        Args:
            file: Either the path to the file, or a file-like object.
                  If it is a path, the file will be opened and closed by ZipFile.
            mode: The mode can be either read 'r', write 'w', exclusive create 'x',
                  or append 'a'.
            compression: ZIP_STORED (no compression), ZIP_DEFLATED (requires zlib),
                         ZIP_BZIP2 (requires bz2) or ZIP_LZMA (requires lzma).
            allowZip64: if True ZipFile will create files with ZIP64 extensions when
                        needed, otherwise it will raise an exception when this would
                        be necessary.
            compresslevel: None (default for the given compression type) or an integer
                           specifying the level to pass to the compressor.
                           When using ZIP_STORED or ZIP_LZMA this keyword has no effect.
                           When using ZIP_DEFLATED integers 0 through 9 are accepted.
                           When using ZIP_BZIP2 integers 1 through 9 are accepted.
        """
        warnings.simplefilter("once")
        warnings.warn(
            "DataZip is now in rmi.etoolbox (https://github.com/rmi/etoolbox)"
            " and will be removed from rmi.dispatch",
            DeprecationWarning,
        )
        warnings.simplefilter("default")

        super().__init__(file, mode, *args, **kwargs)


def idfn(val):
    """ID function for pytest parameterization."""
    if isinstance(val, float):
        return None
    return str(val)
