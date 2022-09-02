"""Simple dispatch model interface."""


from __future__ import annotations

import inspect
import json
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pkg_resources

__all__ = ["DispatchModel"]


from dispatch.engine import dispatch_engine, dispatch_engine_compiled

LOGGER = logging.getLogger(__name__)
__version__ = pkg_resources.get_distribution("rmi.dispatch").version
MTDF = pd.DataFrame()
"""An empty :py:class:`pd.DataFrame`."""


class DispatchModel:
    """Class to contain the core dispatch model functionality.

    - allow the core dispatch model to accept data set up for different uses
    - provide a nicer API that accepts pandas objects, rather than the core numba/numpy engine
    - methods for common analysis of dispatch results
    """

    __slots__ = (
        "net_load_profile",
        "fossil_plant_specs",
        "fossil_profiles",
        "storage_specs",
        "re_profiles",
        "re_plant_specs",
        "jit",
        "name",
        "dt_idx",
        "yrs_idx",
        "fossil_redispatch",
        "storage_dispatch",
        "system_data",
        "starts",
        "__meta__",
    )

    def __init__(
        self,
        net_load_profile: pd.Series[float],
        fossil_plant_specs: pd.DataFrame,
        fossil_profiles: pd.DataFrame,
        storage_specs: pd.DataFrame | None = None,
        re_profiles: pd.DataFrame | None = None,
        re_plant_specs: pd.DataFrame | None = None,
        jit: bool = True,
        name: str = "",
    ):
        """Initialize DispatchModel.

        Args:
            net_load_profile: net load, as in net of RE generation, negative net load means
                excess renewable generation
            fossil_plant_specs: rows are fossil generators, columns must contain:
                capacity_mw: generator nameplate/operating capacity
                ramp_rate: max 1-hr increase in output, in MW
                operating_date: the date the plant entered or will enter service
                retirement_date: the date the plant will retire
                startup_cost: cost to start up the generator
                datetime(freq='YS'): a column for each year with marginal cost data
            fossil_profiles: set the maximum output of each generator in each hour
            storage_specs: rows are types of storage, columns must contain:
                capacity_mw: max charge/discharge capacity in MW
                duration_hrs: storage duration in hours
                roundtrip_eff: roundtrip efficiency
                operating_date: datetime unit starts operating
            re_profiles: ??
            re_plant_specs: ??
            jit: if True, use numba to compile the dispatch engine, False is mostly for debugging
            name: a name, only used in the repr
        """
        self.net_load_profile = net_load_profile
        self.jit = jit
        self.__meta__ = {
            "version": __version__,
            "created": datetime.now().strftime("%c"),
        }

        self.dt_idx = self.net_load_profile.index
        self.yrs_idx = self.dt_idx.to_series().groupby([pd.Grouper(freq="YS")]).first()

        # make sure we have all the `fossil_plant_specs` columns we need
        for col in ("capacity_mw", "ramp_rate", "startup_cost"):
            if col not in fossil_plant_specs:
                raise AssertionError(f"`fossil_plant_specs` requires `{col}` column")
        if not all(x in fossil_plant_specs for x in self.yrs_idx):
            raise AssertionError(
                "`fossil_plant_specs` requires columns for plant cost with 'YS' datetime names"
            )
        self.fossil_plant_specs: pd.DataFrame = fossil_plant_specs

        if not name and "balancing_authority_code_eia" in self.fossil_plant_specs:
            self.name = (
                self.fossil_plant_specs.balancing_authority_code_eia.mode().iloc[0]
            )
        else:
            self.name = name

        # validate `storage_specs`
        if storage_specs is None:
            LOGGER.warning("Careful, dispatch without storage is untested")
            self.storage_specs = pd.DataFrame(
                [0.0, 0, 1.0, self.net_load_profile.index.max()],
                columns=[
                    "capacity_mw",
                    "duration_hrs",
                    "roundtrip_eff",
                    "operating_date",
                ],
            )
        else:
            for col in (
                "capacity_mw",
                "duration_hrs",
                "roundtrip_eff",
                "operating_date",
            ):
                if col not in storage_specs:
                    raise AssertionError(f"`storage_specs` requires `{col}` column")
            self.storage_specs = storage_specs

        if len(fossil_profiles) != len(self.net_load_profile):
            raise AssertionError(
                "`fossil_profiles` and `net_load_profile` must be the same length"
            )
        if fossil_profiles.shape[1] != len(self.fossil_plant_specs):
            raise AssertionError(
                "`fossil_profiles` columns and `fossil_plant_specs` rows must match"
            )
        self.fossil_profiles = fossil_profiles

        self.re_plant_specs = re_plant_specs
        self.re_profiles = re_profiles

        # create vars with correct column names that will be replaced after dispatch
        self.fossil_redispatch = MTDF.reindex(columns=self.fossil_plant_specs.index)
        self.storage_dispatch = MTDF.reindex(
            columns=[
                col
                for i in range(self.storage_specs.shape[0])
                for col in (f"charge_{i}", f"discharge_{i}", f"soc_{i}")
            ]
        )
        self.system_data = MTDF.reindex(
            columns=["deficit", "dirty_charge", "curtailment"]
        )
        self.starts = MTDF.reindex(columns=self.fossil_plant_specs.index)

    @classmethod
    def from_disk(cls, path: Path | str):
        """Recreate an instance of `DispatchModel` from disk."""
        if not isinstance(path, Path):
            path = Path(path)
        data_dict = {}
        with ZipFile(path.with_suffix(".zip"), "r") as z:
            metadata = json.loads(z.read("metadata.json"))
            for x in z.namelist():
                if "parquet" in x:
                    data_dict[x.removesuffix(".parquet")] = pd.read_parquet(
                        BytesIO(z.read(x))
                    )
        if metadata["__qualname__"] != cls.__qualname__:
            raise TypeError(
                f"{path.name} represents a `{metadata['__qualname__']}` which "
                f"is not compatible with `{cls.__qualname__}.from_disk()`"
            )

        # have to fix columns and types
        data_dict["fossil_profiles"].columns = data_dict["fossil_plant_specs"].index
        data_dict["fossil_redispatch"].columns = data_dict["fossil_plant_specs"].index
        data_dict["fossil_plant_specs"].columns = [
            pd.Timestamp(x) if x[0] == "2" else x
            for x in data_dict["fossil_plant_specs"].columns
        ]
        data_dict["net_load_profile"] = data_dict["net_load_profile"].squeeze()

        sig = inspect.signature(cls).parameters
        self = cls(
            **{k: v for k, v in data_dict.items() if k in sig},
            **{k: v for k, v in metadata.items() if k in sig},
        )
        for k, v in data_dict.items():
            if k not in sig:
                setattr(self, k, v)
        self.__meta__ = {k: v for k, v in metadata.items() if k not in sig}
        return self

    @classmethod
    def from_patio(
        cls,
        net_load: pd.Series[float],
        fossil_profiles: pd.DataFrame,
        plant_data: pd.DataFrame,
        storage_specs: pd.DataFrame,
        jit: bool = True,
    ) -> DispatchModel:
        """Create DispatchModel with data from patio.BAScenario."""
        if "operating_date" not in storage_specs:
            storage_specs = storage_specs.assign(operating_date=net_load.index.min())
        return cls(
            net_load_profile=net_load,
            fossil_plant_specs=plant_data,
            fossil_profiles=fossil_profiles,
            storage_specs=storage_specs,
            jit=jit,
        )

    @classmethod
    def new(
        cls,
        net_load_profile: pd.Series[float],
        fossil_plant_specs: pd.DataFrame,
        storage_specs: pd.DataFrame,
        jit: bool = True,
    ) -> DispatchModel:
        """Run dispatch without historical hourly operating constraints."""
        if "operating_date" not in storage_specs:
            storage_specs = storage_specs.assign(
                operating_date=net_load_profile.index.min()
            )

        # duplicate the DatetimeIndex so it is the same shape as `fossil_profiles`
        dt_ix = pd.concat(
            [net_load_profile.index.to_series()] * len(fossil_plant_specs),
            axis=1,
        ).to_numpy()
        wk = pd.Timedelta(weeks=1)

        # insert an `operating_date` column if it doesn't exist and fill missing values
        # with a date before the dispatch period
        if "operating_date" not in fossil_plant_specs:
            fossil_plant_specs = fossil_plant_specs.assign(
                operating_date=net_load_profile.index.min()
            )

        # insert a `retirement_date` column if it doesn't exist and fill missing values
        # with a date after the dispatch period
        if "retirement_date" not in fossil_plant_specs:
            if "planned_retirement_date" not in fossil_plant_specs:
                fossil_plant_specs = fossil_plant_specs.assign(
                    retirement_date=net_load_profile.index.max() + wk
                )
            else:
                fossil_plant_specs = fossil_plant_specs.rename(
                    columns={"planned_retirement_date": "retirement_date"}
                )
        fossil_plant_specs = fossil_plant_specs.fillna(
            {"retirement_date": net_load_profile.index.max() + wk}
        )

        fossil_profiles = pd.DataFrame(
            # make a boolean array for whether a particular hour comes between
            # a generator's `operating_date` and `retirement_date` or not
            (
                (dt_ix < fossil_plant_specs.retirement_date.to_numpy())
                & (dt_ix >= fossil_plant_specs.operating_date.to_numpy())
            )
            * fossil_plant_specs.capacity_mw.to_numpy(),
            columns=fossil_plant_specs.index,
            index=net_load_profile.index,
        )

        return cls(
            net_load_profile=net_load_profile,
            fossil_plant_specs=fossil_plant_specs,
            fossil_profiles=fossil_profiles,
            storage_specs=storage_specs,
            jit=jit,
        )

    @property
    def dispatch_func(self):
        """Appropriate dispatch engine depending on ``self.jit``."""
        return dispatch_engine_compiled if self.jit else dispatch_engine

    @property
    def is_redispatch(self):
        """True if this is redispatch, i.e. has meaningful historical dispatch."""
        # more than 2 unique values are required because any plant that begins
        # operation during the period will have both 0 and its capacity
        return self.fossil_profiles.nunique().max() > 2

    @property
    def historical_cost(self) -> pd.DataFrame:
        """Total hourly historical cost by generator."""
        if self.is_redispatch:
            return self._cost(self.fossil_profiles)
        else:
            out = self.fossil_profiles.copy()
            out.loc[:, :] = np.nan
            return out

    @property
    def historical_dispatch(self) -> pd.DataFrame:
        """Total hourly historical cost by generator."""
        if self.is_redispatch:
            return self.fossil_profiles
        else:
            out = self.fossil_profiles.copy()
            out.loc[:, :] = np.nan
            return out

    @property
    def redispatch_cost(self) -> pd.DataFrame:
        """Total hourly redispatch cost by generator."""
        return self._cost(self.fossil_redispatch)

    # TODO probably a bad idea to use __call__, but nice to not have to think of a name
    def __call__(self) -> None:
        """Run dispatch model."""
        fos_prof, storage, deficits, starts = self.dispatch_func(
            net_load=self.net_load_profile.to_numpy(dtype=np.float_),  # type: ignore
            hr_to_cost_idx=(
                self.net_load_profile.index.year  # type: ignore
                - self.net_load_profile.index.year.min()  # type: ignore
            ).to_numpy(dtype=np.int64),
            fossil_profiles=self.fossil_profiles.to_numpy(dtype=np.float_),
            fossil_ramp_mw=self.fossil_plant_specs.ramp_rate.to_numpy(dtype=np.float_),
            fossil_startup_cost=self.fossil_plant_specs.startup_cost.to_numpy(
                dtype=np.float_
            ),
            fossil_marginal_cost=self.fossil_plant_specs[self.yrs_idx].to_numpy(
                dtype=np.float_
            ),
            storage_mw=self.storage_specs.capacity_mw.to_numpy(dtype=np.float_),
            storage_hrs=self.storage_specs.duration_hrs.to_numpy(dtype=np.int64),
            storage_eff=self.storage_specs.roundtrip_eff.to_numpy(dtype=np.float_),
            # determine the index of the first hour that each storage resource could operate
            storage_op_hour=np.argmax(
                pd.concat(
                    [self.net_load_profile.index.to_series()] * len(self.storage_specs),
                    axis=1,
                ).to_numpy()
                >= self.storage_specs.operating_date.to_numpy(),
                axis=0,
            ),
        )
        self.fossil_redispatch = pd.DataFrame(
            fos_prof.astype(np.float32),
            index=self.dt_idx,
            columns=self.fossil_profiles.columns,
        )
        self.storage_dispatch = pd.DataFrame(
            np.hstack([storage[:, :, x] for x in range(storage.shape[2])]).astype(
                np.float32
            ),
            index=self.dt_idx,
            columns=[
                col
                for i in range(storage.shape[2])
                for col in (f"charge_{i}", f"discharge_{i}", f"soc_{i}")
            ],
        )
        self.system_data = pd.DataFrame(
            deficits.astype(np.float32),
            index=self.dt_idx,
            columns=["deficit", "dirty_charge", "curtailment"],
        )
        self.starts = (
            pd.DataFrame(
                starts.T,
                columns=self.fossil_plant_specs.index,
                index=self.yrs_idx,
            )
            .stack([0, 1])  # type: ignore
            .reorder_levels([1, 2, 0])
            .sort_index()
        )

    def _cost(self, profiles: pd.DataFrame) -> pd.DataFrame:
        """Determine total cost based on hourly production and starts."""
        profs = profiles.to_numpy()
        marginal_cost = profs * self.fossil_plant_specs[self.yrs_idx].T.reindex(
            index=self.net_load_profile.index, method="ffill"
        )
        start_cost = self.fossil_plant_specs.startup_cost.to_numpy() * np.where(
            (profs == 0) & (np.roll(profs, -1, axis=0) > 0), 1, 0
        )
        return marginal_cost + start_cost

    def grouper(
        self,
        df: pd.DataFrame,
        by: str | None = "technology_description",
        freq: str = "YS",
        col_name: str | None = None,
    ) -> pd.DataFrame:
        """Aggregate a df of generator profiles.

        Columns are grouped using `by` column from
        `self.fossil_plant_specs` and `freq` determines
        the output time resolution.

        Args:
            df: dataframe to apply grouping to
            by: column from `self.fossil_plant_specs` to use for grouping df columns,
                if None, no column grouping
            freq: output time resolution
            col_name: if specified, stack the output and use this as the column name

        """
        if by is None:
            out = df.groupby([pd.Grouper(freq=freq)]).sum()
        else:
            df = df.copy()
            col_grouper = self.fossil_plant_specs[by].to_dict()
            df.columns = list(df.columns)
            out = (
                df.rename(columns=col_grouper)
                .groupby(level=0, axis=1)
                .sum()
                .groupby([pd.Grouper(freq=freq)])
                .sum()
            )
            out.columns.name = by
        if col_name is None:
            return out
        return (
            out.stack(level=out.columns.names, dropna=False)
            .reorder_levels(order=[*out.columns.names, "datetime"])
            .to_frame(name=col_name)
            .sort_index()
        )

    def lost_load(
        self, comparison: pd.Series[float] | np.ndarray | float | None = None
    ) -> pd.Series[int]:
        """Number of hours during which deficit was in various duration bins."""
        if comparison is None:
            durs = self.system_data.deficit / self.net_load_profile
        else:
            durs = self.system_data.deficit / comparison
        bins = map(
            float,
            "0.0, 0.0001, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0".split(
                ", "
            ),
        )
        return pd.value_counts(
            pd.cut(durs, list(bins), include_lowest=True)
        ).sort_index()

    def hrs_to_check(
        self, cutoff: float = 0.01, comparison: pd.Series[float] | float | None = None
    ) -> list[pd.Timestamp]:
        """Hours from dispatch to look at more closely.

        Hours with positive deficits are ones where not all of net load was served
        we want to be able to easily check the two hours immediately before these
        positive deficit hours.
        """
        if comparison is None:
            comparison = self.net_load_profile.groupby(
                [pd.Grouper(freq="YS")]
            ).transform(
                "max"
            )  # type: ignore
        td_1h = np.timedelta64(1, "h")
        return sorted(
            {
                hr
                for dhr in self.system_data[
                    self.system_data.deficit / comparison > cutoff
                ].index
                for hr in (dhr - 2 * td_1h, dhr - td_1h, dhr)
                if hr in self.net_load_profile.index
            }
        )

    def storage_capacity(self) -> pd.DataFrame:
        """Number of hours when storage charge or discharge was in various bins."""
        rates = self.storage_dispatch.filter(like="charge")
        # a mediocre way to define the bins...
        d_max = int(np.ceil(rates.max().max()))
        g_bins = [
            y
            for x in range(1, 6)
            for y in (1.0 * 10**x, 2.5 * 10.0**x, 5.0 * 10**x)
        ]
        bins = [0, 0.01] + [x for x in g_bins if x < d_max] + [d_max]
        return pd.concat(
            [pd.value_counts(pd.cut(rates[col], bins)) for col in rates],
            axis=1,
        ).sort_index()

    def storage_durations(self) -> pd.DataFrame:
        """Number of hours during which state of charge was in various duration bins."""
        df = self.storage_dispatch.filter(like="soc")
        durs = df / self.storage_specs.capacity_mw.to_numpy(dtype=float)
        # a mediocre way to define the bins...
        d_max = int(np.ceil(durs.max().max()))
        g_bins = [0.0, 0.01, 2.0, 4.0] + [
            y
            for x in range(1, 6)
            for y in (1.0 * 10**x, 2.5 * 10.0**x, 5.0 * 10**x)
        ]
        bins = [x for x in g_bins if x < d_max] + [d_max]
        return pd.concat(
            [pd.value_counts(pd.cut(durs[col], bins)).sort_index() for col in durs],
            axis=1,
        )

    def system_level_summary(self, freq="YS"):
        """Create system and storage summary metrics."""
        out = pd.concat(
            [
                # mwh deficit, curtailment, dirty charge
                self.system_data.groupby(pd.Grouper(freq=freq))
                .sum()
                .rename(columns={c: f"{c}_mwh" for c in self.system_data}),
                # max deficit pct of net load
                self.system_data[["deficit"]]
                .groupby(pd.Grouper(freq=freq))
                .max()
                .rename(columns={"deficit": "deficit_max_pct_net_load"})
                / self.net_load_profile.max(),
                # count of deficit greater than 2%
                pd.Series(
                    self.system_data[
                        self.system_data / self.net_load_profile.max() > 0.02
                    ]
                    .groupby(pd.Grouper(freq=freq))
                    .deficit.count(),
                    name="deficit_gt_2pct_count",
                ),
                # storage op max
                self.storage_dispatch.assign(
                    **{
                        f"storage_{i}_max_mw": self.storage_dispatch.filter(
                            like=f"e_{i}"
                        ).max(axis=1)
                        for i in self.storage_specs.index
                    },
                    **{
                        f"storage_{i}_max_hrs": self.storage_dispatch[f"soc_{i}"]
                        / self.storage_specs.loc[i, "capacity_mw"]
                        for i in self.storage_specs.index
                    },
                )
                .groupby(pd.Grouper(freq=freq))
                .max()
                .filter(like="max"),
            ],
            axis=1,
        )
        return out.assign(
            **{
                f"storage_{i}_mw_utilization": out[f"storage_{i}_max_mw"]
                / self.storage_specs.loc[i, "capacity_mw"]
                for i in self.storage_specs.index
            },
            **{
                f"storage_{i}_hrs_utilization": out[f"storage_{i}_max_hrs"]
                / self.storage_specs.loc[i, "duration_hrs"]
                for i in self.storage_specs.index
            },
        )

    def operations_summary(
        self,
        by: str | None = "technology_description",
        freq="YS",
    ):
        """Create granular summary of fossil plant metrics.

        Args:
            by: column from `self.fossil_plant_specs` to use for grouping fossil plants,
                if None, no column grouping
            freq: output time resolution
        """
        return (
            pd.concat(
                [
                    self.grouper(
                        self.historical_dispatch,
                        by=by,
                        freq=freq,
                        col_name="historical_fossil_mwh",
                    ),
                    self.grouper(
                        self.fossil_redispatch,
                        by=by,
                        freq=freq,
                        col_name="redispatch_fossil_mwh",
                    ),
                    self.grouper(
                        self.historical_cost,
                        by=by,
                        freq=freq,
                        col_name="historical_fossil_cost",
                    ),
                    self.grouper(
                        self.redispatch_cost,
                        by=by,
                        freq=freq,
                        col_name="redispatch_fossil_cost",
                    ),
                ],
                axis=1,
            )
            .assign(
                avoided_fossil_mwh=lambda x: np.maximum(
                    x.historical_fossil_mwh - x.redispatch_fossil_mwh, 0.0
                ),
                avoided_fossil_cost=lambda x: np.maximum(
                    x.historical_fossil_cost - x.redispatch_fossil_cost, 0.0
                ),
                pct_fossil_replaced=lambda x: np.maximum(
                    x.avoided_fossil_mwh / x.historical_fossil_mwh, 0.0
                ),
            )
            .sort_index()
        )

    def to_disk(self, path: Path | str, compression=ZIP_DEFLATED, clobber=False):
        """Save `DispatchModel` to disk.

        A very ugly process at the moment because of our goal not to use pickle
        and to try to keep the file small-ish. Also need to manage the fact that
        the parquet requirement for string column names causes some problems.
        """
        if not isinstance(path, Path):
            path = Path(path)
        path = path.with_suffix(".zip")
        if path.exists() and not clobber:
            raise FileExistsError(f"{path} exists, to overwrite set `clobber=True`")

        auto_parquet = (
            "re_profiles",
            "storage_dispatch",
            "system_data",
            "storage_specs",
        )
        metadata = {
            "name": self.name,
            "jit": self.jit,
            "__qualname__": self.__class__.__qualname__,
            **self.__meta__,
        }

        # need to make all column names strings
        fossil_plant_specs = self.fossil_plant_specs.copy()
        fossil_plant_specs.columns = list(map(str, self.fossil_plant_specs.columns))
        fossil_profiles = self.fossil_profiles.set_axis(
            list(map(str, range(self.fossil_profiles.shape[1]))), axis="columns"
        )
        fossil_redispatch = self.fossil_redispatch.set_axis(
            list(map(str, range(self.fossil_profiles.shape[1]))), axis="columns"
        )
        with ZipFile(path, "w", compression=compression) as z:
            for df_name in auto_parquet:
                df = getattr(self, df_name)
                if df is not None:
                    z.writestr(f"{df_name}.parquet", df.to_parquet())
            z.writestr(
                "net_load_profile.parquet",
                self.net_load_profile.to_frame(name="nl").to_parquet(),
            )
            z.writestr("fossil_plant_specs.parquet", fossil_plant_specs.to_parquet())
            z.writestr("fossil_profiles.parquet", fossil_profiles.to_parquet())
            z.writestr("fossil_redispatch.parquet", fossil_redispatch.to_parquet())
            z.writestr(
                "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=4)
            )

    def __repr__(self) -> str:
        return (
            self.__class__.__qualname__
            + f"({self.name=}, {self.jit=}, n_plants={len(self.fossil_plant_specs)}, ...)".replace(
                "self.", ""
            )
        )
