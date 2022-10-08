"""Some helpers for profiles and such."""
from __future__ import annotations

import json
from collections.abc import Generator
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
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


def _to_frame(df, n):
    return df.to_frame(name=n)


def _null(df, *args):
    return df


def dfs_to_zip(df_dict: dict[str, pd.DataFrame], path: Path, clobber=False) -> None:
    """Create a zip of parquets.

    Args:
        df_dict: dict of dfs to put into a zip
        path: path for the zip
        clobber: if True, overwrite exiting file with same path

    Returns: None

    """
    bad_cols = {}
    other_stuff = {}
    path = path.with_suffix(".zip")
    if path.exists():
        if not clobber:
            raise FileExistsError(f"{path} exists, to overwrite set `clobber=True`")
        path.unlink()
    with ZipFile(path, "w") as z:
        for key, val in df_dict.items():
            if isinstance(val, pd.DataFrame) and not val.empty:
                try:
                    z.writestr(f"{key}.parquet", val.to_parquet())
                except ValueError:
                    bad_cols.update({key: (list(val.columns), list(val.columns.names))})
                    z.writestr(f"{key}.parquet", _str_cols(val).to_parquet())
            elif isinstance(val, pd.Series) and not val.empty:
                z.writestr(f"{key}.parquet", val.to_frame(name=key).to_parquet())
            elif isinstance(val, (float, int, str, tuple, dict, list)):
                other_stuff.update({key: val})
        z.writestr(
            "other_stuff.json", json.dumps(other_stuff, ensure_ascii=False, indent=4)
        )
        z.writestr("bad_cols.json", json.dumps(bad_cols, ensure_ascii=False, indent=4))


def dfs_from_zip(path: Path, lazy=False) -> dict | Generator:
    """Dict of dfs from a zip of parquets.

    Args:
        path: path of the zip to load
        lazy: if True, return a generator rather than a dict

    Returns: dict of dfs or Generator of name, df pairs

    """
    if lazy:
        return _lazy_load(path)
    out_dict = {}
    with ZipFile(path.with_suffix(".zip"), "r") as z:
        bad_cols = json.loads(z.read("bad_cols.json"))
        for name in z.namelist():
            if "parquet" in name:
                out_dict.update(
                    {
                        name.removesuffix(".parquet"): pd.read_parquet(
                            BytesIO(z.read(name))
                        ).squeeze()
                    }
                )
        out_dict = out_dict | json.loads(z.read("other_stuff.json"))

    for df_name, (cols, names) in bad_cols.items():
        if isinstance(names, (tuple, list)) and len(names) > 1:
            cols = pd.MultiIndex.from_tuples(cols, names=names)
        else:
            cols = pd.Index(cols, name=names[0])
        out_dict[df_name].columns = cols

    return out_dict


def _lazy_load(path: Path) -> Generator[tuple[str, pd.DataFrame]]:
    with ZipFile(path.with_suffix(".zip"), "r") as z:
        bad_cols = json.loads(z.read("bad_cols.json"))
        for name in z.namelist():
            if "parquet" in name:
                key = name.removesuffix(".parquet")
                df = pd.read_parquet(BytesIO(z.read(name))).squeeze()
                if key in bad_cols:
                    cols, names = bad_cols[key]
                    if isinstance(names, (tuple, list)) and len(names) > 1:
                        cols = pd.MultiIndex.from_tuples(cols, names=names)
                    else:
                        cols = pd.Index(cols, name=names[0])
                    df.columns = cols

                yield name, df


def idfn(val):
    """ID function for pytest parameterization."""
    if isinstance(val, float):
        return None
    return str(val)
