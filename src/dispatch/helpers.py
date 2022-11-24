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
        # if mode in ("a", "x"):
        #     raise ValueError("DataZip does not support modes 'a' or 'x'")
        #
        # if isinstance(file, str):
        #     file = Path(file)
        #
        # if isinstance(file, Path):
        #     file = file.with_suffix(".zip")
        #     if file.exists() and mode == "w":
        #         raise FileExistsError(
        #             f"{file} exists, you cannot write or append to an existing DataZip."
        #         )
        warnings.simplefilter("once")
        warnings.warn(
            "DataZip is now in rmi.etoolbox (https://github.com/rmi-electricity/etoolbox)"
            " and will be removed from rmi.dispatch",
            DeprecationWarning,
        )
        warnings.simplefilter("default")

        super().__init__(file, mode, *args, **kwargs)

    #     try:
    #         self.bad_cols = self._read_dict("bad_cols")
    #     except KeyError:
    #         self.bad_cols = {}
    #     try:
    #         self.obj_meta = self._read_dict("obj_meta")
    #     except KeyError:
    #         self.obj_meta = {}
    #
    # def _stemlist(self):
    #     return list(map(lambda x: x.partition(".")[0], self.namelist()))
    #
    # def read(
    #     self, name: str | ZipInfo, pwd: bytes | None = ..., super_=False
    # ) -> bytes | pd.DataFrame | pd.Series | dict:
    #     """Return obj or bytes for name."""
    #     stem, _, suffix = name.partition(".")
    #     if suffix == "parquet" or f"{stem}.parquet" in self.namelist():
    #         return self._read_df(stem)
    #     if suffix == "json" or f"{stem}.json" in self.namelist():
    #         if stem in self.obj_meta:
    #             return self._read_dictable(stem)
    #         else:
    #             return self._read_dict(stem)
    #     if (suffix == "zip" or f"{stem}.zip" in self.namelist()) and not super_:
    #         return self._read_obj(stem)
    #     return super().read(name)
    #
    # def read_dfs(self) -> Generator[tuple[str, pd.DataFrame | pd.Series]]:
    #     """Read all dfs lazily."""
    #     for name, *suffix in map(lambda x: x.split("."), self.namelist()):
    #         if "parquet" in suffix:
    #             yield name, self.read(name)
    #
    # def _read_obj(self, name) -> pd.DataFrame | pd.Series:
    #     temp = BytesIO(super().read(name + ".zip"))
    #     mod, qname, constructor = self.obj_meta[name]
    #     cls = getattr(import_module(mod), qname)
    #     return getattr(cls, constructor)(temp)
    #
    # def _read_dictable(self, name) -> pd.DataFrame | pd.Series:
    #     temp = json.loads(super().read(name + ".json"))
    #     mod, qname, constructor = self.obj_meta[name]
    #     cls = getattr(import_module(mod), qname)
    #     if isinstance(temp, dict):
    #         return cls(**temp)
    #     return cls(temp)
    #
    # def _read_df(self, name) -> pd.DataFrame | pd.Series:
    #     out = pd.read_parquet(BytesIO(super().read(name + ".parquet")))
    #
    #     if name in self.bad_cols:
    #         cols, names = self.bad_cols[name]
    #         if isinstance(names, (tuple, list)) and len(names) > 1:
    #             cols = pd.MultiIndex.from_tuples(cols, names=names)
    #         else:
    #             cols = pd.Index(cols, name=names[0])
    #         out.columns = cols
    #     return out.squeeze()
    #
    # def _read_dict(self, name) -> dict:
    #     return json.loads(super().read(name + ".json"))
    #
    # def writed(
    #     self,
    #     name: str,
    #     data: str | dict | pd.DataFrame | pd.Series | NamedTuple | Any,
    #     **kwargs,
    # ) -> None:
    #     """Write dict, df, str, to name."""
    #     if data is None:
    #         LOGGER.info("Unable to write data %s because it is None.", name)
    #         return None
    #     name = name.removesuffix(".json").removesuffix(".parquet").removesuffix(".zip")
    #     if isinstance(data, (pd.DataFrame, pd.Series)):
    #         self._write_df(name, data, **kwargs)
    #     elif isinstance(data, tuple) and hasattr(data, "_asdict"):
    #         self._write_nt(name, data)
    #     elif isinstance(data, (dict, list, tuple)):
    #         self._write_simple_json(name, data)
    #     elif hasattr(data, "to_file") and hasattr(data, "from_file"):
    #         self._write_obj(name, data, **kwargs)
    #     else:
    #         raise TypeError(
    #             "`data` must be a dict, pd.DataFrame, pd.Series or implement a `to_file` method"
    #         )
    #
    # def _write_obj(self, name, obj, **kwargs):
    #     if name not in self._stemlist():
    #         obj.to_file(temp := BytesIO(), **kwargs)
    #         self.writestr(f"{name}.zip", temp.getvalue())
    #         self.obj_meta.update(
    #             {
    #                 name: (
    #                     obj.__class__.__module__,
    #                     obj.__class__.__qualname__,
    #                     "from_file",
    #                 )
    #             }
    #         )
    #     else:
    #         raise FileExistsError(f"{name} already in {self.filename}")
    #
    # def _write_nt(self, name, obj: NamedTuple) -> None:
    #     """Write a namedtuple in the ZIP as json."""
    #     if name not in self._stemlist():
    #         self.writestr(
    #             f"{name}.json", json.dumps(obj._asdict(), ensure_ascii=False, indent=4)
    #         )
    #         self.obj_meta.update(
    #             {name: (obj.__class__.__module__, obj.__class__.__qualname__, None)}
    #         )
    #     else:
    #         raise FileExistsError(f"{name} already in {self.filename}")
    #
    # def _write_df(self, name: str, df: pd.DataFrame | pd.Series, **kwargs) -> None:
    #     """Write a df in the ZIP as parquet."""
    #     if df.empty:
    #         LOGGER.info("Unable to write df %s because it is empty.", name)
    #         return None
    #     if name not in self._stemlist():
    #         if isinstance(df, pd.Series):
    #             df = df.to_frame(name=name)
    #         try:
    #             self.writestr(f"{name}.parquet", df.to_parquet())
    #         except ValueError:
    #             self.bad_cols.update({name: (list(df.columns), list(df.columns.names))})
    #             self.writestr(f"{name}.parquet", _str_cols(df).to_parquet())
    #     else:
    #         raise FileExistsError(f"{name} already in {self.filename}")
    #
    # def _write_simple_json(
    #     self,
    #     name,
    #     obj: dict[int | str, list | tuple | dict | str | float | int],
    #     **kwargs,
    # ) -> None:
    #     """Write a dict in the ZIP as json."""
    #     if name not in self._stemlist():
    #         self.writestr(f"{name}.json", json.dumps(obj, ensure_ascii=False, indent=4))
    #         if isinstance(obj, tuple):
    #             self.obj_meta.update({name: self._objinfo(obj)})
    #     else:
    #         raise FileExistsError(f"{name} already in {self.filename}")
    #
    # def _objinfo(self, obj):
    #     return obj.__class__.__module__, obj.__class__.__qualname__, None
    #
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     if self.mode == "w":
    #         self._write_simple_json("bad_cols", self.bad_cols)
    #         self._write_simple_json("obj_meta", self.obj_meta)
    #     super().__exit__(exc_type, exc_val, exc_tb)
    #
    # @classmethod
    # def dfs_to_zip(cls, path: Path, df_dict: dict[str, pd.DataFrame], clobber=False):
    #     """Create a zip of parquets.
    #
    #     Args:
    #         df_dict: dict of dfs to put into a zip
    #         path: path for the zip
    #         clobber: if True, overwrite exiting file with same path
    #
    #     Returns: None
    #
    #     """
    #     path = path.with_suffix(".zip")
    #     if path.exists():
    #         if not clobber:
    #             raise FileExistsError(f"{path} exists, to overwrite set `clobber=True`")
    #         path.unlink()
    #     with cls(path, "w") as z:
    #         other_stuff = {}
    #         for key, val in df_dict.items():
    #             if isinstance(val, (pd.Series, pd.DataFrame, dict)):
    #                 z.writed(key, val)
    #             elif isinstance(val, (float, int, str, tuple, dict, list)):
    #                 other_stuff.update({key: val})
    #         z.writed("other_stuff", other_stuff)
    #
    # @classmethod
    # def dfs_from_zip(cls, path: Path) -> dict:
    #     """Dict of dfs from a zip of parquets.
    #
    #     Args:
    #         path: path of the zip to load
    #
    #     Returns: dict of dfs
    #
    #     """
    #     with cls(path, "r") as z:
    #         out_dict = dict(z.read_dfs())
    #         try:
    #             other = z.read("other_stuff")
    #         except KeyError:
    #             other = {}
    #         out_dict = out_dict | other
    #
    #     return out_dict


def idfn(val):
    """ID function for pytest parameterization."""
    if isinstance(val, float):
        return None
    return str(val)
