"""Simple dispatch model interface."""
from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_STORED

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    from plotly.graph_objects import Figure

    PLOTLY_INSTALLED = True
except ModuleNotFoundError:
    from typing import Any

    Figure = Any
    PLOTLY_INSTALLED = False

__all__ = ["DispatchModel"]

from etoolbox.datazip import DataZip

from dispatch import __version__
from dispatch.constants import COLOR_MAP, MTDF, PLOT_MAP
from dispatch.engine import dispatch_engine, dispatch_engine_py
from dispatch.helpers import apply_op_ret_date, dispatch_key
from dispatch.metadata import LOAD_PROFILE_SCHEMA, Validator

LOGGER = logging.getLogger(__name__)


class DispatchModel:
    """Class to contain the core dispatch model functionality.

    - allow the core dispatch model to accept data set up for different uses
    - provide a nicer API on top of :func:`.dispatch_engine_py` that accepts
      :mod:`pandas` objects
    - methods for common analysis of dispatch results
    """

    __slots__ = (
        "load_profile",
        "net_load_profile",
        "dispatchable_specs",
        "dispatchable_cost",
        "dispatchable_profiles",
        "storage_specs",
        "re_profiles_ac",
        "re_excess",
        "re_plant_specs",
        "dt_idx",
        "yrs_idx",
        "redispatch",
        "storage_dispatch",
        "system_data",
        "starts",
        "_metadata",
        "_cached",
    )
    _parquet_out = (
        "re_profiles_ac",
        "re_excess",
        "re_plant_specs",
        "storage_dispatch",
        "system_data",
        "storage_specs",
        "dispatchable_specs",
        "dispatchable_cost",
        "dispatchable_profiles",
        "redispatch",
        "load_profile",
        "net_load_profile",
    )

    def __init__(
        self,
        load_profile: pd.Series[float],
        dispatchable_specs: pd.DataFrame,
        dispatchable_profiles: pd.DataFrame,
        dispatchable_cost: pd.DataFrame,
        storage_specs: pd.DataFrame | None = None,
        re_profiles: pd.DataFrame | None = None,
        re_plant_specs: pd.DataFrame | None = None,
        jit: bool = True,
        name: str = "",
    ):
        """Initialize DispatchModel.

        Args:
            load_profile: load profile that resources will be dispatched against. If
                ``re_profiles`` and ``re_plant_specs`` are not provided, this should
                be a net load profile. If they are provided, this *must* be a gross
                profile, or at least, gross of those RE resources.
            dispatchable_specs: rows are dispatchable generators, columns must include:

                -   capacity_mw: generator nameplate/operating capacity
                -   ramp_rate: max 1-hr increase in output, in MW
                -   operating_date: the date the plant entered or will enter service
                -   retirement_date: the date the plant will retire
                -   exclude: (Optional) True means the generator will NOT be
                    redispatched. Its historical data will be preserved and redispatch
                    data will be zero.

                The index must be a :class:`pandas.MultiIndex` of
                ``['plant_id_eia', 'generator_id']``.

            dispatchable_profiles: set the maximum output of each generator in
                each hour.
            dispatchable_cost: cost metrics for each dispatchable generator in
                each year must be tidy with :class:`pandas.MultiIndex` of
                ``['plant_id_eia', 'generator_id', 'datetime']``, columns must
                include:

                -   vom_per_mwh: variable O&M (USD/MWh)
                -   fuel_per_mwh: fuel cost (USD/MWh)
                -   fom_per_kw: fixed O&M (USD/kW)
                -   start_per_kw: generator startup cost (USD/kW)

            storage_specs: rows are storage facilities, for RE+Storage facilities,
                the ``plant_id_eia`` for the storage component must match the
                ``plant_id_eia`` for the RE component in ``re_profiles`` and
                ``re_plant_specs``. Columns must include:

                -   capacity_mw: max charge/discharge capacity in MW
                -   duration_hrs: storage duration in hours
                -   roundtrip_eff: roundtrip efficiency
                -   operating_date: datetime unit starts operating

                The index must be a :class:`pandas.MultiIndex` of
                ``['plant_id_eia', 'generator_id']``.

            re_profiles: normalized renewable profiles, these should be DC profiles,
                especially when they are part of RE+Storage resources, if they are
                AC profiles, make sure the ilr in ``re_plant_specs`` is 1.0.
            re_plant_specs: rows are renewable facilities, for RE+Storage facilities,
                the ``plant_id_eia`` for the RE component must match the
                ``plant_id_eia`` for the storage component in ``storage_specs``.
                Columns must include:

                -   capacity_mw: AC capacity of the generator
                -   ilr: inverter loading ratio, if ilr != 1, the corresponding
                    profile must be a DC profile.
                -   operating_date: datetime unit starts operating

                The index must be a :class:`pandas.MultiIndex` of
                ``['plant_id_eia', 'generator_id']``.

            jit: if ``True``, use numba to compile the dispatch engine, ``False`` is
                mostly for debugging
            name: a name, only used in the ``repr``
        """
        if not name and "balancing_authority_code_eia" in dispatchable_specs:
            name = dispatchable_specs.balancing_authority_code_eia.mode().iloc[0]
        self._metadata: dict[str, str] = {
            "name": name,
            "version": __version__,
            "created": datetime.now().strftime("%c"),
            "jit": jit,
        }

        self.load_profile: pd.Series = LOAD_PROFILE_SCHEMA.validate(load_profile)

        self.dt_idx = self.load_profile.index
        self.yrs_idx = self.dt_idx.to_series().groupby([pd.Grouper(freq="YS")]).first()

        validator = Validator(self, gen_set=dispatchable_specs.index)
        self.dispatchable_specs: pd.DataFrame = validator.dispatchable_specs(
            dispatchable_specs
        ).pipe(self._add_exclude_col)
        self.dispatchable_cost: pd.DataFrame = validator.dispatchable_cost(
            dispatchable_cost
        ).pipe(self.add_total_costs)
        self.storage_specs: pd.DataFrame = validator.storage_specs(storage_specs)
        self.dispatchable_profiles: pd.DataFrame = apply_op_ret_date(
            validator.dispatchable_profiles(dispatchable_profiles),
            self.dispatchable_specs.operating_date,
            self.dispatchable_specs.retirement_date,
        )
        self.re_plant_specs, re_profiles = validator.renewables(
            re_plant_specs, re_profiles
        )
        (
            self.net_load_profile,
            self.re_profiles_ac,
            self.re_excess,
        ) = self.re_and_net_load(re_profiles)

        # create vars with correct column names that will be replaced after dispatch
        self.redispatch = MTDF.reindex(columns=self.dispatchable_specs.index)
        self.storage_dispatch = MTDF.reindex(
            columns=[
                col
                for i in self.storage_specs.index.get_level_values("plant_id_eia")
                for col in (
                    f"charge_{i}",
                    f"discharge_{i}",
                    f"soc_{i}",
                    f"gridcharge_{i}",
                )
            ]
        )
        self.system_data = MTDF.reindex(
            columns=["deficit", "dirty_charge", "curtailment"]
        )
        self.starts = MTDF.reindex(columns=self.dispatchable_specs.index)
        self._cached = {}

    def re_and_net_load(self, re_profiles):
        """Create net_load_profile based on what RE data was provided."""
        if self.re_plant_specs is None or re_profiles is None:
            return (
                self.load_profile,
                None,
                None,
            )
        # ILR adjusted normalized profiles
        temp = re_profiles * self.re_plant_specs.ilr.to_numpy()
        ac_out = np.minimum(temp, 1) * self.re_plant_specs.capacity_mw.to_numpy()
        excess = temp * self.re_plant_specs.capacity_mw.to_numpy() - ac_out
        return self.load_profile - ac_out.sum(axis=1), ac_out, excess

    def dc_charge(self):
        """Align excess_re to match the storage facilities it could charge."""
        dc_charge = pd.DataFrame(
            np.nan, index=self.load_profile.index, columns=self.storage_specs.index
        )
        if self.re_excess is None:
            return dc_charge.fillna(0.0)
        dc_charge = dc_charge.droplevel("generator_id", axis=1)
        return (
            dc_charge.combine_first(
                self.re_excess.groupby(level=0, axis=1)
                .sum()
                .sort_index(axis=1, ascending=False)
            )[dc_charge.columns]
            .set_axis(self.storage_specs.index, axis=1)
            .fillna(0.0)
        )

    def add_total_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add columns for total FOM and total startup from respective unit costs."""
        df = (
            df.reset_index()
            .merge(
                self.dispatchable_specs.capacity_mw.reset_index(),
                on=["plant_id_eia", "generator_id"],
                validate="m:1",
            )
            .set_index(["plant_id_eia", "generator_id", "datetime"])
            .assign(fom=lambda x: x.capacity_mw * x.fom_per_kw * 1000)
        )
        if "startup_cost" not in df:
            df = df.assign(startup_cost=lambda x: x.capacity_mw * x.start_per_kw * 1000)
        return df.drop(columns=["capacity_mw"])

    @staticmethod
    def _add_exclude_col(df: pd.DataFrame) -> pd.DataFrame:
        """Add ``exclude`` column if not already present."""
        if "exclude" in df:
            return df
        return df.assign(exclude=False)

    @classmethod
    def from_file(cls, path: Path | str | BytesIO) -> DispatchModel:
        """Recreate an instance of :class:`.DispatchModel` from disk."""
        if isinstance(path, str):
            path = Path(path)

        def _type_check(meta):
            if meta["__qualname__"] != cls.__qualname__:
                raise TypeError(
                    f"{path.name} represents a `{meta['__qualname__']}` which "
                    f"is not compatible with `{cls.__qualname__}.from_file()`"
                )
            del meta["__qualname__"]

        with DataZip(path, "r") as z:
            metadata = z.read("metadata")
            _type_check(metadata)
            data_dict = dict(z.read_dfs())

        sig = inspect.signature(cls).parameters
        self = cls(
            **{k: v for k, v in (data_dict | metadata).items() if k in sig},
        )
        for k, v in data_dict.items():
            if k not in sig:
                setattr(self, k, v)
        self._metadata.update({k: v for k, v in metadata.items() if k not in sig})
        return self

    @classmethod
    def from_patio(
        cls,
        net_load: pd.Series[float],
        dispatchable_profiles: pd.DataFrame,
        plant_data: pd.DataFrame,
        cost_data: pd.DataFrame,
        storage_specs: pd.DataFrame,
        jit: bool = True,
    ) -> DispatchModel:
        """Create :class:`.DispatchModel` with data from patio.BAScenario."""
        if "operating_date" not in storage_specs:
            storage_specs = storage_specs.assign(
                operating_date=net_load.index.min(),
                plant_id_eia=lambda x: x.index.to_series() * -1,
                generator_id=lambda x: (
                    x.groupby(["plant_id_eia"]).transform("cumcount") + 1
                ).astype(str),
            ).set_index(["plant_id_eia", "generator_id"])
        return cls(
            load_profile=net_load,
            dispatchable_specs=plant_data,
            dispatchable_cost=cost_data,
            dispatchable_profiles=dispatchable_profiles,
            storage_specs=storage_specs,
            jit=jit,
        )

    @classmethod
    def from_fresh(
        cls,
        net_load_profile: pd.Series[float],
        dispatchable_specs: pd.DataFrame,
        dispatchable_cost: pd.DataFrame,
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
        if "operating_date" not in dispatchable_specs:
            dispatchable_specs = dispatchable_specs.assign(
                operating_date=net_load_profile.index.min()
            )

        # insert a `retirement_date` column if it doesn't exist and fill missing values
        # with a date after the dispatch period
        if "retirement_date" not in dispatchable_specs:
            if "planned_retirement_date" not in dispatchable_specs:
                dispatchable_specs = dispatchable_specs.assign(
                    retirement_date=net_load_profile.index.max()
                )
            else:
                dispatchable_specs = dispatchable_specs.rename(
                    columns={"planned_retirement_date": "retirement_date"}
                )
        dispatchable_specs = dispatchable_specs.fillna(
            {"retirement_date": net_load_profile.index.max()}
        )

        # make a boolean array for whether a particular hour comes between
        # a generator's `operating_date` and `retirement_date` or not
        dispatchable_profiles = pd.DataFrame(
            np.expand_dims(dispatchable_specs.capacity_mw.to_numpy(), axis=0).repeat(
                len(net_load_profile.index), axis=0
            ),
            index=net_load_profile.index,
            columns=dispatchable_specs.index,
        )

        return cls(
            load_profile=net_load_profile,
            dispatchable_specs=dispatchable_specs,
            dispatchable_cost=dispatchable_cost,
            dispatchable_profiles=dispatchable_profiles,
            storage_specs=storage_specs,
            jit=jit,
        )

    @property
    def dispatch_func(self) -> Callable:
        """Appropriate dispatch engine depending on ``jit`` setting."""
        return dispatch_engine if self._metadata["jit"] else dispatch_engine_py

    @property
    def is_redispatch(self) -> bool:
        """True if this is redispatch, i.e. has meaningful historical dispatch."""
        # more than 2 unique values are required because any plant that begins
        # operation during the period will have both 0 and its capacity
        return self.dispatchable_profiles.nunique().max() > 2

    @property
    def historical_cost(self) -> dict[str, pd.DataFrame]:
        """Total hourly historical cost by generator."""
        if self.is_redispatch:
            return self._cost(self.dispatchable_profiles)
        else:
            out = self.dispatchable_profiles.copy()
            out.loc[:, :] = np.nan
            return {"fuel": out, "vom": out, "startup": out}

    @property
    def historical_dispatch(self) -> pd.DataFrame:
        """Total hourly historical cost by generator."""
        if self.is_redispatch:
            return self.dispatchable_profiles
        else:
            out = self.dispatchable_profiles.copy()
            out.loc[:, :] = np.nan
            return out

    @property
    def redispatch_cost(self) -> dict[str, pd.DataFrame]:
        """Total hourly redispatch cost by generator."""
        out = self._cost(self.redispatch)
        # zero out FOM of excluded resources
        out["fom"] = out["fom"] * (~self.dispatchable_specs.exclude).to_numpy(
            dtype=float
        )
        return out

    # TODO probably a bad idea to use __call__, but nice to not have to think of a name
    def __call__(self) -> DispatchModel:
        """Run dispatch model."""
        # determine any dispatchable resources that should be excluded from dispatch and
        # zero out their profile so they do not run
        to_exclude = (~self.dispatchable_specs.exclude).to_numpy(dtype=float)
        d_prof = self.dispatchable_profiles.to_numpy(dtype=np.float_, copy=True)
        if np.any(to_exclude == 0.0):
            d_prof = d_prof * to_exclude

        fos_prof, storage, deficits, starts = self.dispatch_func(
            net_load=self.net_load_profile.to_numpy(dtype=np.float_),  # type: ignore
            hr_to_cost_idx=(
                self.net_load_profile.index.year  # type: ignore
                - self.net_load_profile.index.year.min()  # type: ignore
            ).to_numpy(dtype=np.int64),
            historical_dispatch=d_prof,
            dispatchable_ramp_mw=self.dispatchable_specs.ramp_rate.to_numpy(
                dtype=np.float_
            ),
            dispatchable_startup_cost=self.dispatchable_cost.startup_cost.unstack().to_numpy(
                dtype=np.float_
            ),
            dispatchable_marginal_cost=self.dispatchable_cost.total_var_mwh.unstack().to_numpy(
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
            storage_dc_charge=self.dc_charge().to_numpy(dtype=np.float_),
        )
        self.redispatch = pd.DataFrame(
            fos_prof,
            index=self.dt_idx,
            columns=self.dispatchable_profiles.columns,
        )
        self.storage_dispatch = pd.DataFrame(
            np.hstack([storage[:, :, x] for x in range(storage.shape[2])]),
            index=self.dt_idx,
            columns=self.storage_dispatch.columns,
        )
        self.system_data = pd.DataFrame(
            deficits,
            index=self.dt_idx,
            columns=self.system_data.columns,
        )
        self.starts = (
            pd.DataFrame(
                starts.T,
                columns=self.dispatchable_specs.index,
                index=self.yrs_idx,
            )
            .stack([0, 1])  # type: ignore
            .reorder_levels([1, 2, 0])
            .sort_index()
        )
        return self

    def _cost(self, profiles: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Determine total cost based on hourly production and starts."""
        profs = profiles.to_numpy()
        fuel_cost = profs * self.dispatchable_cost.fuel_per_mwh.unstack(
            level=("plant_id_eia", "generator_id")
        ).reindex(index=self.load_profile.index, method="ffill")
        vom_cost = profs * self.dispatchable_cost.vom_per_mwh.unstack(
            level=("plant_id_eia", "generator_id")
        ).reindex(index=self.load_profile.index, method="ffill")
        start_cost = np.where(
            (profs == 0) & (np.roll(profs, -1, axis=0) > 0), 1, 0
        ) * self.dispatchable_cost.startup_cost.unstack(
            level=("plant_id_eia", "generator_id")
        ).reindex(
            index=self.load_profile.index, method="ffill"
        )
        fom = (
            apply_op_ret_date(
                self.dispatchable_cost.fom.unstack(
                    level=("plant_id_eia", "generator_id")
                ),
                self.dispatchable_specs.operating_date.apply(
                    lambda x: x.replace(day=1, month=1)
                ),
                self.dispatchable_specs.retirement_date.apply(
                    lambda x: x.replace(day=1, month=1)
                ),
            )
            .reindex(index=self.load_profile.index, method="ffill")
            .divide(
                self.load_profile.groupby(pd.Grouper(freq="YS")).transform("count"),
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
        :attr:`.DispatchModel.dispatchable_specs` and `freq` determines
        the output time resolution.

        Args:
            df: dataframe to apply grouping to, if a dict of dataframes, does the
                grouping on each and then concatenates them together with keys as
                column name suffix
            by: column from :attr:`.DispatchModel.dispatchable_specs` to use for
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
                    self.strict_grouper(
                        df=df_, by=by, freq=freq, col_name=pref + col_name_
                    )
                    for col_name_, df_ in df.items()
                ],
                axis=1,
            )
        return self.strict_grouper(df=df, by=by, freq=freq, col_name=col_name)

    def strict_grouper(
        self,
        df: pd.DataFrame,
        by: str | None,
        freq: str,
        col_name: str | None = None,
        freq_agg: str = "sum",
    ) -> pd.DataFrame:
        """Aggregate a df of generator profiles.

        Columns are grouped using `by` column from
        :attr:`.DispatchModel.dispatchable_specs` and `freq` determines
        the output time resolution.

        Args:
            df: dataframe to apply grouping to
            by: column from :attr:`.DispatchModel.dispatchable_specs` to use for
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
            col_grouper = self.dispatchable_specs[by].to_dict()
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
            durs = self.system_data.deficit / self.load_profile
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
            comparison = self.load_profile.groupby([pd.Grouper(freq="YS")]).transform(
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
                if hr in self.load_profile.index
            }
        )

    def hourly_data_check(self, cutoff: float = 0.01):
        """Aggregate data for :meth:`.DispatchModel.hrs_to_checl`."""
        max_disp = apply_op_ret_date(
            pd.DataFrame(
                1.0,
                index=self.load_profile.index,
                columns=self.dispatchable_profiles.columns,
            ),
            self.dispatchable_specs.operating_date,
            self.dispatchable_specs.retirement_date,
            self.dispatchable_specs.capacity_mw,
        )
        out = pd.concat(
            {
                "gross_load": self.load_profile,
                "net_load": self.net_load_profile,
                "deficit": self.system_data.deficit,
                "max_dispatch": max_disp.sum(axis=1),
                "redispatch": self.redispatch.sum(axis=1),
                "historical_dispatch": self.dispatchable_profiles.sum(axis=1),
                "net_storage": (
                    self.storage_dispatch.filter(regex="^discharge").sum(axis=1)
                    - self.storage_dispatch.filter(regex="^gridcharge").sum(axis=1)
                ),
                "state_of_charge": self.storage_dispatch.filter(regex="^soc").sum(
                    axis=1
                ),
                "re": self.re_profiles_ac.sum(axis=1),
                "re_excess": self.re_excess.sum(axis=1),
            },
            axis=1,
        ).loc[self.hrs_to_check(cutoff=cutoff), :]
        return out

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

    def system_level_summary(self, freq: str = "YS", **kwargs) -> pd.DataFrame:
        """Create system and storage summary metrics."""
        out = pd.concat(
            [
                # mwh deficit, curtailment, dirty charge
                self.system_data.groupby(pd.Grouper(freq=freq))
                .sum()
                .rename(columns={c: f"{c}_mwh" for c in self.system_data}),
                # max deficit pct of load
                self.system_data[["deficit"]]
                .groupby(pd.Grouper(freq=freq))
                .max()
                .rename(columns={"deficit": "deficit_max_pct_net_load"})
                / self.load_profile.max(),
                # count of deficit greater than 2%
                pd.Series(
                    self.system_data[self.system_data / self.load_profile.max() > 0.02]
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
                        for i in self.storage_specs.index.get_level_values(
                            "plant_id_eia"
                        )
                    },
                    **{
                        f"storage_{i}_max_hrs": self.storage_dispatch[f"soc_{i}"]
                        / self.storage_specs.loc[i, "capacity_mw"]
                        for i in self.storage_specs.index.get_level_values(
                            "plant_id_eia"
                        )
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
                for i in self.storage_specs.index.get_level_values("plant_id_eia")
            },
            **{
                f"storage_{i}_hrs_utilization": out[f"storage_{i}_max_hrs"]
                / self.storage_specs.loc[i, "duration_hrs"]
                for i in self.storage_specs.index.get_level_values("plant_id_eia")
            },
        )

    def re_summary(
        self,
        by: str | None = "technology_description",
        freq: str = "YS",
        **kwargs,
    ) -> pd.DataFrame:
        """Create granular summary of renewable plant metrics."""
        if self.re_profiles_ac is None or self.re_plant_specs is None:
            raise AssertionError(
                "at least one of `re_profiles` and `re_plant_specs` is `None`"
            )
        out = (
            self.re_profiles_ac.groupby([pd.Grouper(freq=freq)])
            .sum()
            .stack(["plant_id_eia", "generator_id"])
            .to_frame(name="redispatch_mwh")
            # .assign(redispatch_mwh=lambda x: x.historical_mwh)
            .reset_index()
            .merge(
                self.re_plant_specs,
                on=["plant_id_eia", "generator_id"],
                validate="m:1",
            )
            .assign(
                capacity_mw=lambda x: x.capacity_mw.where(
                    x.operating_date <= x.datetime, 0
                )
            )
        )
        if by is None:
            return out.set_index(["plant_id_eia", "generator_id", "datetime"])
        return out.groupby([by, "datetime"]).sum()

    def storage_summary(
        self,
        by: str | None = "technology_description",
        freq: str = "YS",
        **kwargs,
    ) -> pd.DataFrame:
        """Create granular summary of storage plant metrics."""
        out = (
            self.storage_dispatch.filter(regex="^dis|^grid")
            .groupby([pd.Grouper(freq=freq)])
            .sum()
            .stack()
            .reset_index()
            .rename(columns={0: "redispatch_mwh"})
        )

        out[["kind", "plant_id_eia"]] = out.level_1.str.split("_", expand=True)
        out = (
            out.astype({"plant_id_eia": int})
            .assign(
                redispatch_mwh=lambda x: x.redispatch_mwh.mask(
                    x.kind == "gridcharge", x.redispatch_mwh * -1
                ),
                generator_id=lambda x: x.plant_id_eia.replace(
                    self.storage_specs.reset_index(
                        "generator_id"
                    ).generator_id.to_dict()
                ),
            )
            .groupby(["plant_id_eia", "generator_id", "datetime"])
            .redispatch_mwh.sum()
            .reset_index()
            .merge(
                self.storage_specs.reset_index(),
                on=["plant_id_eia", "generator_id"],
                validate="m:1",
            )
            .assign(
                capacity_mw=lambda x: x.capacity_mw.where(
                    x.operating_date <= x.datetime, 0
                )
            )
        )
        if by is None:
            return out.set_index(["plant_id_eia", "generator_id", "datetime"])
        return out.groupby([by, "datetime"]).sum()

    def full_output(self, freq: str = "YS") -> pd.DataFrame:
        """Create full operations output."""
        # setup deficit/curtailment as if they were resources for full output, the idea
        # here is that you could rename them purchase/sales.
        def_cur = self.grouper(self.system_data, by=None)[["deficit", "curtailment"]]
        def_cur.columns = pd.MultiIndex.from_tuples(
            [(0, "deficit"), (0, "curtailment")], names=["plant_id_eia", "generator_id"]
        )
        def_cur = (
            def_cur.stack([0, 1])
            .reorder_levels([1, 2, 0])
            .sort_index()
            .to_frame(name="redispatch_mwh")
            .assign(
                technology_description=lambda x: x.index.get_level_values(
                    "generator_id"
                )
            )
        )

        return pd.concat(
            [
                self.dispatchable_summary(by=None, freq=freq, augment=True),
                self.re_summary(by=None, freq=freq),
                self.storage_summary(by=None, freq=freq)
                .reset_index()
                .set_index(["plant_id_eia", "generator_id", "datetime"]),
                def_cur,
            ]
        ).sort_index()

    def load_summary(self, **kwargs):
        """Create summary of load data."""
        return pd.concat(
            [
                self.strict_grouper(
                    self.net_load_profile.to_frame("net_load"), by=None, freq="YS"
                ),
                self.strict_grouper(
                    self.net_load_profile.to_frame("net_load_peak"),
                    by=None,
                    freq="YS",
                    freq_agg="max",
                ),
                self.strict_grouper(
                    self.load_profile.to_frame("gross_load"),
                    by=None,
                    freq="YS",
                ),
                self.strict_grouper(
                    self.load_profile.to_frame("gross_load_peak"),
                    by=None,
                    freq="YS",
                    freq_agg="max",
                ),
            ],
            axis=1,
        )

    def dispatchable_summary(
        self,
        by: str | None = "technology_description",
        freq: str = "YS",
        augment: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Create granular summary of dispatchable plant metrics.

        Args:
            by: column from :attr:`.DispatchModel.dispatchable_specs` to use for
                grouping dispatchable plants, if None, no column grouping
            freq: output time resolution
            augment: include columns from plant_specs columns
        """
        out = (
            pd.concat(
                [
                    self.strict_grouper(
                        apply_op_ret_date(
                            pd.DataFrame(
                                1,
                                index=self.load_profile.index,
                                columns=self.dispatchable_specs.index,
                            ),
                            self.dispatchable_specs.operating_date,
                            self.dispatchable_specs.retirement_date,
                            self.dispatchable_specs.capacity_mw,
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
                        self.redispatch,
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
        if not augment:
            return out
        cols = [
            "plant_name_eia",
            "technology_description",
            "utility_id_eia",
            "final_ba_code",
            "final_respondent_id",
            "respondent_name",
            "balancing_authority_code_eia",
            "prime_mover_code",
            "operating_date",
            "retirement_date",
            "status",
            "owned_pct",
        ]
        return (
            out.reset_index()
            .merge(
                self.dispatchable_specs,
                on=["plant_id_eia", "generator_id"],
                validate="m:1",
                suffixes=(None, "_l"),
            )
            .set_index(out.index.names)[
                list(out.columns)
                + [col for col in cols if col in self.dispatchable_specs]
            ]
        )

    def to_file(
        self,
        path: Path | str | BytesIO,
        include_output: bool = False,
        compression=ZIP_STORED,
        clobber=False,
        **kwargs,
    ) -> None:
        """Save :class:`.DispatchModel` to disk."""
        if isinstance(path, (str, Path)):
            if Path(path).with_suffix(".zip").exists() and not clobber:
                raise FileExistsError(f"{path} exists, to overwrite set `clobber=True`")
            if clobber:
                Path(path).with_suffix(".zip").unlink(missing_ok=True)

        with DataZip(path, "w", compression=compression) as z:
            for df_name in self._parquet_out:
                z.writed(df_name, getattr(self, df_name))
            if include_output and not self.redispatch.empty:
                for df_name in ("full_output", "load_summary"):
                    z.writed(
                        df_name,
                        getattr(self, df_name)(**kwargs),
                    )
            z.writed(
                "metadata",
                {
                    **self._metadata,
                    "__qualname__": self.__class__.__qualname__,
                },
            )

    def _plot_prep(self):
        if "plot_prep" not in self._cached:
            storage = self.storage_dispatch.assign(
                charge=lambda x: -1 * x.filter(regex="^gridcharge").sum(axis=1),
                discharge=lambda x: x.filter(regex="^discharge").sum(axis=1),
            )
            try:
                re = self.re_summary(freq="H").redispatch_mwh.unstack(level=0)
            except AssertionError:
                re = MTDF.reindex(index=self.load_profile.index)

            def _grp(df):
                return df.rename(columns=PLOT_MAP).groupby(level=0, axis=1).sum()

            self._cached["plot_prep"] = (
                pd.concat(
                    [
                        self.grouper(self.redispatch, freq="H").pipe(_grp),
                        re.pipe(_grp),
                        storage[["charge", "discharge"]],
                    ],
                    axis=1,
                )
                .assign(
                    Curtailment=self.system_data.curtailment * -1,
                    Deficit=self.system_data.deficit,
                )
                .rename_axis("resource", axis=1)
                .stack()
                .to_frame(name="net_generation_mwh")
            )
        return self._cached["plot_prep"]

    def _plot_prep_detail(self, begin, end):
        to_cat = [
            self.redispatch.set_axis(
                pd.MultiIndex.from_frame(
                    self.dispatchable_specs.technology_description.reset_index()
                ),
                axis=1,
            ),
            self.re_profiles_ac.set_axis(
                pd.MultiIndex.from_frame(
                    self.re_plant_specs.technology_description.reset_index()
                ),
                axis=1,
            ),
            self.storage_dispatch.filter(like="discharge").set_axis(
                pd.MultiIndex.from_frame(
                    self.storage_specs.assign(
                        technology_description="discharge"
                    ).technology_description.reset_index()
                ),
                axis=1,
            ),
            -1
            * self.storage_dispatch.filter(like="gridcharge").set_axis(
                pd.MultiIndex.from_frame(
                    self.storage_specs.assign(
                        technology_description="charge"
                    ).technology_description.reset_index()
                ),
                axis=1,
            ),
            -1
            * self.system_data.curtailment.to_frame(
                name=(999, "curtailment", "Curtailment")
            ),
            self.system_data.deficit.to_frame(name=(999, "deficit", "Deficit")),
        ]
        return (
            pd.concat(
                to_cat,
                axis=1,
            )
            .loc[begin:end, :]
            .stack([0, 1, 2])
            .to_frame(name="net_generation_mwh")
            .reset_index()
            .assign(resource=lambda x: x.technology_description.replace(PLOT_MAP))
            .query("net_generation_mwh != 0.0")
        )

    def plot_period(self, begin, end=None, by_gen=True) -> Figure:
        """Plot hourly dispatch by generator."""
        begin = pd.Timestamp(begin)
        if end is None:
            end = begin + pd.Timedelta(days=7)
        else:
            end = pd.Timestamp(end)
        net_load = self.net_load_profile.loc[begin:end]
        data = self._plot_prep_detail(begin, end)
        hover_name = "plant_id_eia"
        if not by_gen:
            data = data.groupby(["datetime", "resource"]).sum().reset_index()
            hover_name = "resource"
        out = (
            px.bar(
                data.replace(
                    {"resource": {"charge": "Storage", "discharge": "Storage"}}
                ).sort_values(["resource"], key=dispatch_key),
                x="datetime",
                y="net_generation_mwh",
                color="resource",
                hover_name=hover_name,
                color_discrete_map=COLOR_MAP,
            )
            .add_scatter(
                x=net_load.index,
                y=net_load,
                name="Net Load",
                mode="lines",
                line_color=COLOR_MAP["Net Load"],
                line_dash="dot",
            )
            .update_layout(xaxis_title=None, yaxis_title="MW", yaxis_tickformat=",.0r")
        )
        if self.re_profiles_ac is None or self.re_plant_specs is None:
            return out
        return out.add_scatter(
            x=self.load_profile.loc[begin:end].index,
            y=self.load_profile.loc[begin:end],
            name="Gross Load",
            mode="lines",
            line_color=COLOR_MAP["Gross Load"],
        )

    def plot_year(self, year: int, freq="D") -> Figure:
        """Monthly facet plot of daily dispatch for a year."""
        assert freq in ("H", "D"), "`freq` must be 'D' for day or 'H' for hour"
        out = (
            self._plot_prep()
            .loc[str(year), :]
            .reset_index()
            .groupby([pd.Grouper(freq=freq, key="datetime"), "resource"])
            .sum()
            .reset_index()
            .assign(
                day=lambda z: z.datetime.dt.day,
                hour=lambda z: z.datetime.dt.day * 24 + z.datetime.dt.hour,
                month=lambda z: z.datetime.dt.strftime("%B"),
                resource=lambda z: z.resource.replace(
                    {"charge": "Storage", "discharge": "Storage"}
                ),
            )
            .sort_values(["resource", "month"], key=dispatch_key)
        )
        x, yt, ht = {"D": ("day", "MWh", "resource"), "H": ("hour", "MW", "datetime")}[
            freq
        ]
        return (
            px.bar(
                out,
                x=x,
                y="net_generation_mwh",
                color="resource",
                facet_col="month",
                facet_col_wrap=4,
                height=800,
                width=1000,
                hover_name=ht,
                color_discrete_map=COLOR_MAP,
            )
            .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            .update_xaxes(title=None)
            .for_each_yaxis(
                lambda yaxis: (yaxis.update(title=yt) if yaxis.title.text else None)
                # lambda yaxis: print(yaxis)
            )
            .update_layout(bargap=0)
        )

    def plot_output(self, y: str, color="resource") -> Figure:
        """Plot a columns from :meth:`.DispatchModel.full_output`."""
        return px.bar(
            self.full_output()
            .reset_index()
            .assign(
                year=lambda x: x.datetime.dt.year,
                resource=lambda x: x.technology_description.replace(PLOT_MAP),
                redispatch_cost=lambda x: x.filter(like="redispatch_cost").sum(axis=1),
                historical_cost=lambda x: x.filter(like="historical_cost").sum(axis=1),
            )
            .sort_values(["resource"], key=dispatch_key),
            x="year",
            y=y,
            color=color,
            hover_name="plant_id_eia",
            color_discrete_map=COLOR_MAP,
            width=1000,
        ).update_layout(xaxis_title=None)

    def __repr__(self) -> str:
        if self.re_plant_specs is None:
            re_len = 0
        else:
            re_len = len(self.re_plant_specs)
        return (
            self.__class__.__qualname__
            + f"({', '.join(f'{k}={v}' for k, v in self._metadata.items())}, "
            f"n_dispatchable={len(self.dispatchable_specs)}, n_re={re_len}, "
            f"n_storage={len(self.storage_specs)})"
        )
