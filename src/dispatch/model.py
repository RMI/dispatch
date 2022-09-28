"""Simple dispatch model interface."""
from __future__ import annotations

import inspect
import json
import logging
from datetime import datetime
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

__all__ = ["DispatchModel"]


from dispatch.engine import dispatch_engine, dispatch_engine_compiled
from dispatch.helpers import _null, _str_cols, _to_frame, apply_op_ret_date
from dispatch.metadata import FOSSIL_SPECS_SCHEMA, NET_LOAD_SCHEMA, Validator

LOGGER = logging.getLogger(__name__)

MTDF = pd.DataFrame()
"""An empty :class:`pandas.DataFrame`."""


class DispatchModel:
    """Class to contain the core dispatch model functionality.

    - allow the core dispatch model to accept data set up for different uses
    - provide a nicer API that accepts pandas objects on top of :func:`.dispatch_engine`
    - methods for common analysis of dispatch results
    """

    __slots__ = (
        "net_load_profile",
        "fossil_plant_specs",
        "fossil_cost",
        "fossil_profiles",
        "storage_specs",
        "re_profiles",
        "re_plant_specs",
        "dt_idx",
        "yrs_idx",
        "fossil_redispatch",
        "storage_dispatch",
        "system_data",
        "starts",
        "__meta__",
    )
    _parquet_out = {
        "re_profiles": _null,
        "storage_dispatch": _null,
        "system_data": _null,
        "storage_specs": _null,
        "fossil_plant_specs": _null,
        "fossil_cost": _null,
        "fossil_profiles": _str_cols,
        "fossil_redispatch": _str_cols,
        "net_load_profile": _to_frame,
    }

    def __init__(
        self,
        net_load_profile: pd.Series[float],
        fossil_plant_specs: pd.DataFrame,
        fossil_profiles: pd.DataFrame,
        fossil_cost: pd.DataFrame,
        storage_specs: pd.DataFrame | None = None,
        re_profiles: pd.DataFrame | None = None,
        re_plant_specs: pd.DataFrame | None = None,
        jit: bool = True,
        name: str = "",
    ):
        """Initialize DispatchModel.

        Args:
            net_load_profile: net load, as in net of RE generation, negative net
                load means excess renewable generation
            fossil_plant_specs: rows are fossil generators, columns must include:

                -   capacity_mw: generator nameplate/operating capacity
                -   ramp_rate: max 1-hr increase in output, in MW
                -   operating_date: the date the plant entered or will enter service
                -   retirement_date: the date the plant will retire

            fossil_profiles: set the maximum output of each generator in each hour
            fossil_cost: cost metrics for each fossil generator in each year
                must be tidy with :class:`pandas.MultiIndex` of
                ``['plant_id_eia', 'generator_id', 'datetime']``, columns must
                include:

                -   vom_per_mwh: variable O&M (USD/MWh)
                -   fuel_per_mwh: fuel cost (USD/MWh)
                -   fom_per_kw: fixed O&M (USD/kW)
                -   start_per_kw: generator startup cost (USD/kW)

            storage_specs: rows are types of storage, columns must include:

                -   capacity_mw: max charge/discharge capacity in MW
                -   duration_hrs: storage duration in hours
                -   roundtrip_eff: roundtrip efficiency
                -   operating_date: datetime unit starts operating

            re_profiles: ??
            re_plant_specs: ??
            jit: if ``True``, use numba to compile the dispatch engine, ``False`` is
                mostly for debugging
            name: a name, only used in the ``repr``
        """
        if not name and "balancing_authority_code_eia" in fossil_plant_specs:
            name = fossil_plant_specs.balancing_authority_code_eia.mode().iloc[0]
        self.__meta__: dict[str, str] = {
            "name": name,
            "version": version("rmi.dispatch"),
            "created": datetime.now().strftime("%c"),
            "jit": jit,
        }

        self.net_load_profile: pd.Series = NET_LOAD_SCHEMA.validate(net_load_profile)
        self.fossil_plant_specs: pd.DataFrame = FOSSIL_SPECS_SCHEMA.validate(
            fossil_plant_specs
        )

        self.dt_idx = self.net_load_profile.index
        self.yrs_idx = self.dt_idx.to_series().groupby([pd.Grouper(freq="YS")]).first()

        validator = Validator(self)
        self.fossil_cost: pd.DataFrame = validator.fossil_cost(fossil_cost).pipe(
            self.add_total_costs
        )
        self.storage_specs: pd.DataFrame = validator.storage_specs(storage_specs)
        self.fossil_profiles: pd.DataFrame = apply_op_ret_date(
            validator.fossil_profiles(fossil_profiles),
            self.fossil_plant_specs.operating_date,
            self.fossil_plant_specs.retirement_date,
        )

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

    def add_total_costs(self, fossil_cost):
        """Add columns for total FOM and total startup from respective unit costs."""
        fossil_cost = (
            fossil_cost.reset_index()
            .merge(
                self.fossil_plant_specs.capacity_mw.reset_index(),
                on=["plant_id_eia", "generator_id"],
                validate="m:1",
            )
            .set_index(["plant_id_eia", "generator_id", "datetime"])
            .assign(fom=lambda x: x.capacity_mw * x.fom_per_kw * 1000)
        )
        if "startup_cost" not in fossil_cost:
            fossil_cost = fossil_cost.assign(
                startup_cost=lambda x: x.capacity_mw * x.start_per_kw * 1000
            )
        return fossil_cost.drop(columns=["capacity_mw"])

    @classmethod
    def from_file(cls, path: Path | str):
        """Recreate an instance of :class:`.DispatchModel` from disk."""
        if not isinstance(path, Path):
            path = Path(path)

        def _type_check(meta):
            if meta["__qualname__"] != cls.__qualname__:
                raise TypeError(
                    f"{path.name} represents a `{meta['__qualname__']}` which "
                    f"is not compatible with `{cls.__qualname__}.from_disk()`"
                )
            del meta["__qualname__"]

        data_dict = {}
        with ZipFile(path.with_suffix(".zip"), "r") as z:
            metadata = json.loads(z.read("metadata.json"))
            _type_check(metadata)
            plant_index = pd.MultiIndex.from_tuples(
                metadata.pop("plant_index"), names=["plant_id_eia", "generator_id"]
            )
            for df_name in cls._parquet_out:
                if (x := df_name + ".parquet") in z.namelist():
                    df_in = pd.read_parquet(BytesIO(z.read(x))).squeeze()
                    if df_name in ("fossil_profiles", "fossil_redispatch"):
                        df_in.columns = plant_index
                    data_dict[df_name] = df_in

        sig = inspect.signature(cls).parameters
        self = cls(
            **{k: v for k, v in (data_dict | metadata).items() if k in sig},
        )
        for k, v in data_dict.items():
            if k not in sig:
                setattr(self, k, v)
        self.__meta__.update({k: v for k, v in metadata.items() if k not in sig})
        return self

    @classmethod
    def from_patio(
        cls,
        net_load: pd.Series[float],
        fossil_profiles: pd.DataFrame,
        plant_data: pd.DataFrame,
        cost_data: pd.DataFrame,
        storage_specs: pd.DataFrame,
        jit: bool = True,
    ) -> DispatchModel:
        """Create :class:`.DispatchModel` with data from patio.BAScenario."""
        if "operating_date" not in storage_specs:
            storage_specs = storage_specs.assign(operating_date=net_load.index.min())
        return cls(
            net_load_profile=net_load,
            fossil_plant_specs=plant_data,
            fossil_cost=cost_data,
            fossil_profiles=fossil_profiles,
            storage_specs=storage_specs,
            jit=jit,
        )

    @classmethod
    def from_fresh(
        cls,
        net_load_profile: pd.Series[float],
        fossil_plant_specs: pd.DataFrame,
        fossil_marginal_cost: pd.DataFrame,
        storage_specs: pd.DataFrame,
        jit: bool = True,
    ) -> DispatchModel:
        """Run dispatch without historical hourly operating constraints."""
        if "operating_date" not in storage_specs:
            storage_specs = storage_specs.assign(
                operating_date=net_load_profile.index.min()
            )

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
                    retirement_date=net_load_profile.index.max()
                )
            else:
                fossil_plant_specs = fossil_plant_specs.rename(
                    columns={"planned_retirement_date": "retirement_date"}
                )
        fossil_plant_specs = fossil_plant_specs.fillna(
            {"retirement_date": net_load_profile.index.max()}
        )

        # make a boolean array for whether a particular hour comes between
        # a generator's `operating_date` and `retirement_date` or not
        fossil_profiles = apply_op_ret_date(
            pd.DataFrame(
                1, index=net_load_profile.index, columns=fossil_plant_specs.index
            ),
            fossil_plant_specs.operating_date,
            fossil_plant_specs.retirement_date,
            fossil_plant_specs.capacity_mw,
        )

        return cls(
            net_load_profile=net_load_profile,
            fossil_plant_specs=fossil_plant_specs,
            fossil_cost=fossil_marginal_cost,
            fossil_profiles=fossil_profiles,
            storage_specs=storage_specs,
            jit=jit,
        )

    @property
    def dispatch_func(self):
        """Appropriate dispatch engine depending on ``jit`` setting."""
        return dispatch_engine_compiled if self.__meta__["jit"] else dispatch_engine

    @property
    def is_redispatch(self):
        """True if this is redispatch, i.e. has meaningful historical dispatch."""
        # more than 2 unique values are required because any plant that begins
        # operation during the period will have both 0 and its capacity
        return self.fossil_profiles.nunique().max() > 2

    @property
    def historical_cost(self) -> dict[str, pd.DataFrame]:
        """Total hourly historical cost by generator."""
        if self.is_redispatch:
            return self._cost(self.fossil_profiles)
        else:
            out = self.fossil_profiles.copy()
            out.loc[:, :] = np.nan
            return {"fuel": out, "vom": out, "startup": out}

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
    def redispatch_cost(self) -> dict[str, pd.DataFrame]:
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
            fossil_startup_cost=self.fossil_cost.startup_cost.unstack().to_numpy(
                dtype=np.float_
            ),
            fossil_marginal_cost=self.fossil_cost.total_var_mwh.unstack().to_numpy(
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

    def _cost(self, profiles: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Determine total cost based on hourly production and starts."""
        profs = profiles.to_numpy()
        fuel_cost = profs * self.fossil_cost.fuel_per_mwh.unstack(
            level=("plant_id_eia", "generator_id")
        ).reindex(index=self.net_load_profile.index, method="ffill")
        vom_cost = profs * self.fossil_cost.vom_per_mwh.unstack(
            level=("plant_id_eia", "generator_id")
        ).reindex(index=self.net_load_profile.index, method="ffill")
        start_cost = np.where(
            (profs == 0) & (np.roll(profs, -1, axis=0) > 0), 1, 0
        ) * self.fossil_cost.startup_cost.unstack(
            level=("plant_id_eia", "generator_id")
        ).reindex(
            index=self.net_load_profile.index, method="ffill"
        )
        fom = (
            apply_op_ret_date(
                self.fossil_cost.fom.unstack(level=("plant_id_eia", "generator_id")),
                self.fossil_plant_specs.operating_date.apply(
                    lambda x: x.replace(day=1, month=1)
                ),
                self.fossil_plant_specs.retirement_date.apply(
                    lambda x: x.replace(day=1, month=1)
                ),
            )
            .reindex(index=self.net_load_profile.index, method="ffill")
            .divide(
                self.net_load_profile.groupby(pd.Grouper(freq="YS")).transform("count"),
                axis=0,
            )
        )
        return {"fuel": fuel_cost, "vom": vom_cost, "startup": start_cost, "fom": fom}

    def grouper(
        self,
        df: pd.DataFrame | dict[str, pd.DataFrame],
        by: str | None = "technology_description",
        freq: str = "YS",
        col_name: str | None = None,
    ) -> pd.DataFrame:
        """Aggregate a df of generator profiles.

        Columns are grouped using `by` column from
        :attr:`.DispatchModel.fossil_plant_specs` and `freq` determines
        the output time resolution.

        Args:
            df: dataframe to apply grouping to, if a dict of dataframes, does the
                grouping on each and then concatenates them together with keys as
                column name suffix
            by: column from :attr:`.DispatchModel.fossil_plant_specs` to use for
                grouping df columns, if None, no column grouping
            freq: output time resolution
            col_name: if specified, stack the output and use this as the column name,
                if `df` is a dict, each df is stacked and `col_name` if any is
                prepended to the key to form the column name.

        """
        if isinstance(df, dict):
            pref = "" if col_name is None else col_name + "_"
            return pd.concat(
                [
                    self._grouper(df=df_, by=by, freq=freq, col_name=pref + col_name_)
                    for col_name_, df_ in df.items()
                ],
                axis=1,
            )
        return self._grouper(df=df, by=by, freq=freq, col_name=col_name)

    def _grouper(
        self,
        df: pd.DataFrame,
        by: str | None,
        freq: str,
        col_name: str | None = None,
        freq_agg: str = "sum",
    ) -> pd.DataFrame:
        """Aggregate a df of generator profiles.

        Columns are grouped using `by` column from
        :attr:`.DispatchModel.fossil_plant_specs` and `freq` determines
        the output time resolution.

        Args:
            df: dataframe to apply grouping to
            by: column from :attr:`.DispatchModel.fossil_plant_specs` to use for
                grouping df columns, if None, no column grouping
            freq: output time resolution
            col_name: if specified, stack the output and use this as the column name
            freq_agg: aggregation func to use in frequency groupby

        """
        if by is None:
            out = df.groupby([pd.Grouper(freq=freq)]).agg(freq_agg)
            dropna = True
        else:
            df = df.copy()
            col_grouper = self.fossil_plant_specs[by].to_dict()
            df.columns = list(df.columns)
            out = (
                df.rename(columns=col_grouper)
                .groupby(level=0, axis=1)
                .sum()
                .groupby([pd.Grouper(freq=freq)])
                .agg(freq_agg)
            )
            out.columns.name = by
            dropna = False
        if col_name is None:
            return out
        return (
            out.stack(level=out.columns.names, dropna=dropna)
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

    def system_level_summary(self, freq="YS", **kwargs):
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
        **kwargs,
    ):
        """Create granular summary of fossil plant metrics.

        Args:
            by: column from :attr:`.DispatchModel.fossil_plant_specs` to use for
                grouping fossil plants, if None, no column grouping
            freq: output time resolution
        """
        return (
            pd.concat(
                [
                    self._grouper(
                        apply_op_ret_date(
                            pd.DataFrame(
                                1,
                                index=self.net_load_profile.index,
                                columns=self.fossil_plant_specs.index,
                            ),
                            self.fossil_plant_specs.operating_date,
                            self.fossil_plant_specs.retirement_date,
                            self.fossil_plant_specs.capacity_mw,
                        ),
                        by=by,
                        freq=freq,
                        col_name="capacity_mw",
                        freq_agg="max",
                    ),
                    self.grouper(
                        self.historical_dispatch,
                        by=by,
                        freq=freq,
                        col_name="historical_mwh",
                    ),
                    self.grouper(
                        self.historical_cost,
                        by=by,
                        freq=freq,
                        col_name="historical_cost",
                    ),
                    self.grouper(
                        self.fossil_redispatch,
                        by=by,
                        freq=freq,
                        col_name="redispatch_mwh",
                    ),
                    self.grouper(
                        self.redispatch_cost,
                        by=by,
                        freq=freq,
                        col_name="redispatch_cost",
                    ),
                ],
                axis=1,
            )
            .assign(
                avoided_mwh=lambda x: np.maximum(
                    x.historical_mwh - x.redispatch_mwh, 0.0
                ),
                avoided_cost_fuel=lambda x: np.maximum(
                    x.historical_cost_fuel - x.redispatch_cost_fuel, 0.0
                ),
                avoided_cost_vom=lambda x: np.maximum(
                    x.historical_cost_vom - x.redispatch_cost_vom, 0.0
                ),
                avoided_cost_startup=lambda x: np.maximum(
                    x.historical_cost_startup - x.redispatch_cost_startup, 0.0
                ),
                pct_replaced=lambda x: np.maximum(
                    x.avoided_mwh / x.historical_mwh, 0.0
                ),
            )
            .sort_index()
        )

    def to_file(
        self,
        path: Path | str,
        include_output: bool = False,
        compression=ZIP_DEFLATED,
        clobber=False,
        **kwargs,
    ):
        """Save :class:`.DispatchModel` to disk.

        A very ugly process at the moment because of our goal not to use pickle
        and to try to keep the file small-ish. Also need to manage the fact that
        the parquet requirement for string column names causes some problems.
        """
        if not isinstance(path, Path):
            path = Path(path)
        path = path.with_suffix(".zip")
        if path.exists() and not clobber:
            raise FileExistsError(f"{path} exists, to overwrite set `clobber=True`")

        metadata = {
            **self.__meta__,
            "__qualname__": self.__class__.__qualname__,
            "plant_index": list(self.fossil_plant_specs.index),
        }

        with ZipFile(path, "w", compression=compression) as z:
            for df_name, func in self._parquet_out.items():
                try:
                    df_out = getattr(self, df_name)
                    if df_out is not None and not df_out.empty:
                        z.writestr(
                            f"{df_name}.parquet", func(df_out, df_name).to_parquet()
                        )
                except Exception as exc:
                    raise RuntimeError(f"{df_name} {exc!r}") from exc
            if include_output and not self.fossil_redispatch.empty:
                for df_name in ("system_level_summary", "operations_summary"):
                    z.writestr(
                        f"{df_name}.parquet",
                        getattr(self, df_name)(**kwargs).to_parquet(),
                    )
            z.writestr(
                "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=4)
            )

    def __repr__(self) -> str:
        return (
            self.__class__.__qualname__
            + f"({', '.join(f'{k}={v}' for k, v in self.__meta__.items())}, "
            f"n_plants={len(self.fossil_plant_specs)})".replace("self.", "")
        )
