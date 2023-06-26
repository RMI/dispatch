"""Simple dispatch model interface."""
from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta
from typing import  ClassVar, Literal

import numpy as np
import pandas as pd
import polars as pl

try:
    import plotly.express as px
    from plotly.graph_objects import Figure

    PLOTLY_INSTALLED = True
except ModuleNotFoundError:
    from typing import Any

    Figure = Any
    PLOTLY_INSTALLED = False

__all__ = ["DispatchModel"]

from etoolbox.datazip import IOMixin

from dispatch import __version__
from dispatch.constants import COLOR_MAP, MTDF, PLOT_MAP
from dispatch.engine import dispatch_engine, dispatch_engine_auto
from dispatch.helpers import dispatch_key, zero_profiles_outside_operating_dates
from dispatch.metadata import LOAD_PROFILE_SCHEMA, IDConverter, Validator

LOGGER = logging.getLogger(__name__)


class DispatchModel(IOMixin):
    """Class to contain the core dispatch model functionality.

    - allow the core dispatch model to accept data set up for different uses
    - provide a nicer API on top of :func:`.dispatch_engine` that accepts
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
        "config",
        "_metadata",
        "_cached",
        "_polars",
        "pl_dispatchable_profiles",
        "pl_dispatchable_cost",
        "pl_dispatchable_specs",
        "pl_re_profiles_ac",
        "pl_re_plant_specs",
        "pl_storage_specs",
    )
    id_schema = {
        "plant_id_eia": pl.Int32,
        "generator_id": pl.Utf8,
        "datetime": pl.Datetime("us"),
    }
    es_schema = {
        "charge": pl.Float32,
        "discharge": pl.Float32,
        "soc": pl.Float32,
        "gridcharge": pl.Float32,
    }
    sys_schema = {
        "deficit": pl.Float32,
        "dirty_charge": pl.Float32,
        "curtailment": pl.Float32,
        "load_adjustment": pl.Float32,
    }
    pl_freq = {"YS": "1y", "AS": "1y", "MS": "1mo", "D": "1d", "H": "1h"}
    default_config: ClassVar[dict[str, str]] = {"dynamic_reserve_coeff": "auto"}

    def __init__(
        self,
        load_profile: pd.Series[float],
        dispatchable_specs: pd.DataFrame,
        dispatchable_profiles: pd.DataFrame,
        dispatchable_cost: pd.DataFrame,
        storage_specs: pd.DataFrame | None = None,
        re_profiles: pd.DataFrame | None = None,
        re_plant_specs: pd.DataFrame | None = None,
        *,
        jit: bool = True,
        name: str = "",
        config: dict | None = None,
        to_pandas: bool = True,
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
                -   min_uptime: (Optional) the minimum number of hours the generator
                    must be on before its output can be reduced.
                -   exclude: (Optional) True means the generator will NOT be
                    redispatched. Its historical data will be preserved and redispatch
                    data will be zero.
                -   no_limit: (Optional) True means the generator will not have the
                    hourly maximum from ``dispatchable_profiles`` enforced, instead the
                    max of historical and capacity_mw will be used. This allows the
                    removal of the cap without affecting historical metrics.

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
                -   total_var_mwh: (Optional) total variable cost (USD/MWh), if not
                    provided it will be calculates as  vom_per_mwh + fuel_per_mwh.
                    If it is provided, this value will be used to determine marginal
                    cost rank but will not be used to calculate operating costs.
                    This allows for a kludgy kind of must-run or uneconomic dispatch.
                -   heat_rate: (Optional) (mmbtu/MWh)
                -   co2_factor: (Optional) (X/mmbtu) X should be tonnes or short
                    tonnes.

            storage_specs: rows are storage facilities, for RE+Storage facilities,
                the ``plant_id_eia`` for the storage component must match the
                ``plant_id_eia`` for the RE component in ``re_profiles`` and
                ``re_plant_specs``. Columns must include:

                -   capacity_mw: max charge/discharge capacity in MW.
                -   duration_hrs: storage duration in hours.
                -   roundtrip_eff: roundtrip efficiency.
                -   operating_date: datetime unit starts operating.
                -   reserve: [Optional] % of state of charge to hold in reserve until
                    after dispatchable startup. If this is not provided or the reserve
                    is 0.0, the reserve will be set dynamically each hour looking
                    out 24 hours.

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
                -   interconnect_mw: (Optional) the interconnect capacity of the
                    renewable facility. By default, this is the same as ``capacity_mw``
                    but can be reduced to reflect facility-specific transmission /
                    interconnection constraints. If the facility has storage, storage
                    can be charged by the constrained excess.
                -   fom_per_kw: (Optional) fixed O&M (USD/kW)

                The index must be a :class:`pandas.MultiIndex` of
                ``['plant_id_eia', 'generator_id']``.

            jit: if ``True``, use numba to compile the dispatch engine, ``False`` is
                mostly for debugging
            name: a name, only used in the ``repr``
            config: a dict of changes to :attr:`.DispatchModel.default_config`, can
                include:

                - dynamic_reserve_coeff: passed to
                  :func:`dispatch.engine.dynamic_reserve` the default value is 'auto'
                  which then tries a number of values and selects the best using
                  :func:`dispatch.engine.choose_best_coefficient`.
              to_pandas: default to always providing outputs as pd.DataFrame.

        >>> pd.options.display.width = 1000
        >>> pd.options.display.max_columns = 6
        >>> pd.options.display.max_colwidth = 30

        Examples
        --------
        **Input Tables**

        The load profile that resources will be dispatched against.

        >>> load_profile = pd.Series(
        ...     550
        ...     + 40 * (np.sin(np.pi * np.arange(8784) / 24 - np.pi / 6)) ** 2
        ...     + 20 * (np.sin(np.pi * np.arange(8784) / 12)) ** 2
        ...     + 250 * (np.cos(np.pi * np.arange(8784) / 4392) ** 2)
        ...     + 200 * (np.sin(np.pi * np.arange(8784) / 8784) ** 2),
        ...     index=pd.date_range(
        ...         "2020-01-01", freq="H", periods=8784, name="datetime"
        ...     ),
        ... )
        >>> load_profile.head()
        datetime
        2020-01-01 00:00:00    810.000000
        2020-01-01 01:00:00    807.197508
        2020-01-01 02:00:00    807.679083
        2020-01-01 03:00:00    810.680563
        2020-01-01 04:00:00    814.998363
        Freq: H, dtype: float64

        Core specification of dispatchable generators.

        >>> dispatchable_specs = pd.DataFrame(
        ...     {
        ...         "capacity_mw": [350, 500, 600],
        ...         "ramp_rate": [350, 250, 100],
        ...         "technology_description": [
        ...             "Natural Gas Fired Combustion Turbine",
        ...             "Natural Gas Fired Combined Cycle",
        ...             "Conventional Steam Coal",
        ...         ],
        ...     },
        ...     index=pd.MultiIndex.from_tuples(
        ...         [(1, "1"), (1, "2"), (2, "1")],
        ...         names=["plant_id_eia", "generator_id"],
        ...     ),
        ... ).assign(
        ...     operating_date=pd.Timestamp("2000-01-01"),
        ...     retirement_date=pd.Timestamp("2050-01-01"),
        ... )
        >>> dispatchable_specs  # doctest: +NORMALIZE_WHITESPACE
                                   capacity_mw  ramp_rate         technology_description operating_date retirement_date
        plant_id_eia generator_id
        1            1                     350        350  Natural Gas Fired Combusti...     2000-01-01      2050-01-01
                     2                     500        250  Natural Gas Fired Combined...     2000-01-01      2050-01-01
        2            1                     600        100        Conventional Steam Coal     2000-01-01      2050-01-01

        Set the maximum output of each generator in each hour. In this case we set the
        maximum in every hour as the generator's nameplate capacity.

        >>> dispatchable_profiles = pd.DataFrame(
        ...     np.tile(dispatchable_specs.capacity_mw.to_numpy(), (8784, 1)),
        ...     index=pd.date_range(
        ...         "2020-01-01", freq="H", periods=8784, name="datetime"
        ...     ),
        ...     columns=dispatchable_specs.index,
        ... )
        >>> dispatchable_profiles.head()  # doctest: +NORMALIZE_WHITESPACE
        plant_id_eia           1         2
        generator_id           1    2    1
        datetime
        2020-01-01 00:00:00  350  500  600
        2020-01-01 01:00:00  350  500  600
        2020-01-01 02:00:00  350  500  600
        2020-01-01 03:00:00  350  500  600
        2020-01-01 04:00:00  350  500  600

        Cost metrics for each dispatchable generator in each year.

        >>> dispatchable_cost = pd.DataFrame(
        ...     {
        ...         "vom_per_mwh": [15.0, 5.0, 2.0],
        ...         "fuel_per_mwh": [45.0, 35.0, 20.0],
        ...         "fom_per_kw": [2, 15, 25],
        ...         "start_per_kw": [0.005, 0.01, 0.03],
        ...         "heat_rate": [10.0, 6.5, 9.5],
        ...         "co2_factor": [0.05291, 0.05291, 0.09713],
        ...     },
        ...     index=pd.MultiIndex.from_tuples(
        ...         [
        ...             (1, "1", pd.Timestamp("2020-01-01")),
        ...             (1, "2", pd.Timestamp("2020-01-01")),
        ...             (2, "1", pd.Timestamp("2020-01-01")),
        ...         ],
        ...         names=["plant_id_eia", "generator_id", "datetime"],
        ...     ),
        ... )
        >>> dispatchable_cost.index.levels[2].freq = "AS-JAN"
        >>> dispatchable_cost  # doctest: +NORMALIZE_WHITESPACE
                                              vom_per_mwh  fuel_per_mwh  fom_per_kw  start_per_kw  heat_rate  co2_factor
        plant_id_eia generator_id datetime
        1            1            2020-01-01         15.0          45.0           2         0.005       10.0     0.05291
                     2            2020-01-01          5.0          35.0          15         0.010        6.5     0.05291
        2            1            2020-01-01          2.0          20.0          25         0.030        9.5     0.09713

        Specifications for storage facilities. For RE+Storage facilities, the
        ``plant_id_eia`` for the storage component must match the ``plant_id_eia`` for
        the RE component in ``re_profiles`` and ``re_plant_specs``.

        >>> storage_specs = pd.DataFrame(
        ...     {
        ...         "capacity_mw": [250, 200],
        ...         "duration_hrs": [4, 12],
        ...         "roundtrip_eff": [0.9, 0.5],
        ...         "technology_description": [
        ...             "Solar Photovoltaic with Energy Storage",
        ...             "Batteries",
        ...         ],
        ...     },
        ...     index=pd.MultiIndex.from_tuples(
        ...         [(5, "es"), (7, "1")], names=["plant_id_eia", "generator_id"]
        ...     ),
        ... ).assign(operating_date=pd.Timestamp("2000-01-01 00:00:00"))
        >>> storage_specs  # doctest: +NORMALIZE_WHITESPACE
                                   capacity_mw  duration_hrs  roundtrip_eff         technology_description operating_date
        plant_id_eia generator_id
        5            es                    250             4            0.9  Solar Photovoltaic with En...     2000-01-01
        7            1                     200            12            0.5                      Batteries     2000-01-01

        Specifications for renewable facilities. Becasue ``plant_id_eia`` 5 shows up in both
        ``storage_specs`` and ``re_plant_specs``, those resources will be DC-coupled.

        >>> re_plant_specs = pd.DataFrame(
        ...     {
        ...         "capacity_mw": [500, 500],
        ...         "ilr": [1.3, 1.0],
        ...         "technology_description": [
        ...             "Solar Photovoltaic with Energy Storage",
        ...             "Onshore Wind",
        ...         ],
        ...     },
        ...     index=pd.MultiIndex.from_tuples(
        ...         [(5, "1"), (6, "1")], names=["plant_id_eia", "generator_id"]
        ...     ),
        ... ).assign(
        ...     operating_date=pd.Timestamp("2000-01-01"),
        ...     retirement_date=pd.Timestamp("2050-01-01"),
        ... )
        >>> re_plant_specs  # doctest: +NORMALIZE_WHITESPACE
                                   capacity_mw  ilr         technology_description operating_date retirement_date
        plant_id_eia generator_id
        5            1                     500  1.3  Solar Photovoltaic with En...     2000-01-01      2050-01-01
        6            1                     500  1.0                   Onshore Wind     2000-01-01      2050-01-01

        Normalized renewable DC profiles.

        >>> re_profiles = pd.DataFrame(
        ...     np.vstack(
        ...         (
        ...             np.sin(np.pi * np.arange(8784) / 24) ** 8,
        ...             np.cos(np.pi * np.arange(8784) / 24) ** 2,
        ...         )
        ...     ).T,
        ...     columns=re_plant_specs.index,
        ...     index=load_profile.index,
        ... )
        >>> re_profiles.round(2).head()  # doctest: +NORMALIZE_WHITESPACE
        plant_id_eia           5     6
        generator_id           1     1
        datetime
        2020-01-01 00:00:00  0.0  1.00
        2020-01-01 01:00:00  0.0  0.98
        2020-01-01 02:00:00  0.0  0.93
        2020-01-01 03:00:00  0.0  0.85
        2020-01-01 04:00:00  0.0  0.75

        **Setting up the model**

        Create the :class:`.DispatchModel` object:

        >>> dm = DispatchModel(
        ...     load_profile=load_profile,
        ...     dispatchable_specs=dispatchable_specs,
        ...     dispatchable_profiles=dispatchable_profiles,
        ...     dispatchable_cost=dispatchable_cost,
        ...     storage_specs=storage_specs,
        ...     re_profiles=re_profiles,
        ...     re_plant_specs=re_plant_specs,
        ...     name="test",
        ... )

        Run the dispatch model, the model runs inplace but also returns itself for
        convenience.

        >>> dm = dm()

        Explore the results, starting with how much load could not be met. The results
        are now returned as :class:`polars.LazyFrame` so `collect()` must be called on
        them to see the results. We convert to :class:`pandas.DataFrame` to show how
        that would be done.

        >>> dm.lost_load().collect().to_pandas()  # doctest: +NORMALIZE_WHITESPACE
              category  count
        0  (-inf, 0.0]   8784

        Generate a full, combined output of all resources at specified frequency.

        >>> dm.full_output(freq="YS").collect().to_pandas()  # doctest: +NORMALIZE_WHITESPACE
           plant_id_eia generator_id  capacity_mw  ... duration_hrs  roundtrip_eff  reserve
        0             0  curtailment          NaN  ...          NaN            NaN      NaN
        1             0      deficit          NaN  ...          NaN            NaN      NaN
        2             1            1        350.0  ...          NaN            NaN      NaN
        3             1            2        500.0  ...          NaN            NaN      NaN
        4             2            1        600.0  ...          NaN            NaN      NaN
        5             5            1        500.0  ...          NaN            NaN      NaN
        6             5           es        250.0  ...          4.0            0.9      0.0
        7             6            1        500.0  ...          NaN            NaN      NaN
        8             7            1        200.0  ...         12.0            0.5      0.0
        <BLANKLINE>
        [9 rows x 28 columns]
        """
        if not name and "balancing_authority_code_eia" in dispatchable_specs:
            name = dispatchable_specs.balancing_authority_code_eia.mode().iloc[0]
        self._metadata: dict[str, str] = {
            "name": name,
            "version": __version__,
            "created": datetime.now().strftime("%c"),
            "jit": jit,
        }
        if config is None:
            self.config = self.default_config
        else:
            self.config = self.default_config | config

        self.load_profile: pd.Series = LOAD_PROFILE_SCHEMA.validate(load_profile)

        self.dt_idx = self.load_profile.index
        self.yrs_idx = self.dt_idx.to_series().groupby([pd.Grouper(freq="YS")]).first()

        validator = Validator(
            self,
            gen_set=dispatchable_specs.index,
            re_set=pd.MultiIndex.from_product(
                [[], []], names=["plant_id_eia", "generator_id"]
            )
            if re_plant_specs is None
            else re_plant_specs.index,
        )
        self.dispatchable_specs: pd.DataFrame = validator.dispatchable_specs(
            dispatchable_specs
        ).pipe(self._add_optional_cols, df_name="dispatchable_specs")
        self.dispatchable_cost: pd.DataFrame = validator.dispatchable_cost(
            dispatchable_cost
        ).pipe(self._add_total_and_missing_cols)
        self.storage_specs: pd.DataFrame = validator.storage_specs(storage_specs).pipe(
            self._add_optional_cols, df_name="storage_specs"
        )
        self.dispatchable_profiles: pd.DataFrame = (
            zero_profiles_outside_operating_dates(
                validator.dispatchable_profiles(dispatchable_profiles),
                self.dispatchable_specs.operating_date,
                self.dispatchable_specs.retirement_date,
            )
        )
        self.re_plant_specs, re_profiles = validator.renewables(
            re_plant_specs, re_profiles
        )
        (
            self.net_load_profile,
            self.re_profiles_ac,
            self.re_excess,
        ) = self.re_and_net_load(re_profiles)

        # set up some translation dicts to assist in transforming to and from polars
        self._polars = IDConverter(
            self.dispatchable_specs,
            self.re_plant_specs,
            self.storage_specs,
            self.dt_idx,
        )
        # create vars with correct column names that will be replaced after dispatch
        # self.redispatch = MTDF.reindex(columns=self.dispatchable_specs.index)
        self.pl_dispatchable_profiles = pl.concat(
            [
                self._polars.disp_big_idx,
                pl.from_numpy(
                    self.dispatchable_profiles.to_numpy().reshape(
                        self.dispatchable_profiles.size, order="F"
                    ),
                    {"historical_mwh": pl.Float32},
                ),
            ],
            how="horizontal",
        ).lazy()
        self.pl_dispatchable_cost = (
            pl.from_pandas(
                self.dispatchable_cost.reset_index(), schema_overrides=self.id_schema
            )
            .fill_nan(None)
            .lazy()
        )
        self.pl_dispatchable_specs = self._polars.from_pandas(self.dispatchable_specs)
        if self.re_plant_specs is not None:
            self.pl_re_profiles_ac = pl.concat(
                [
                    self._polars.re_big_idx,
                    pl.from_numpy(
                        self.re_profiles_ac.to_numpy().reshape(
                            self.re_profiles_ac.size, order="F"
                        ),
                        {"redispatch_mwh": pl.Float32},
                    ),
                ],
                how="horizontal",
            ).lazy()
            self.pl_re_plant_specs = self._polars.from_pandas(
                self.re_plant_specs.reset_index()
            )
        self.pl_storage_specs = self._polars.from_pandas(self.storage_specs)
        self.redispatch = pl.LazyFrame(
            schema=self.id_schema | {"redispatch_mwh": pl.Float32}
        )
        self.storage_dispatch = pl.LazyFrame(schema=self.id_schema | self.es_schema)
        self.system_data = pl.LazyFrame(
            schema={"datetime": pl.Datetime("us")} | self.sys_schema
        )
        self.starts = MTDF.reindex(columns=self.dispatchable_specs.index)
        self._cached = {}

    def re_and_net_load(self, re_profiles):
        """Create net_load_profile based on what RE data was provided."""
        if self.re_plant_specs is None or re_profiles is None:
            warnings.warn(
                "`re_plant_specs` and `re_profiles` will be required in the future.",
                FutureWarning,
                stacklevel=2,
            )
            return (
                self.load_profile,
                None,
                None,
            )
        if "interconnect_mw" not in self.re_plant_specs:
            self.re_plant_specs = self.re_plant_specs.assign(
                interconnect_mw=lambda x: x.capacity_mw
            )
        if "fom_per_kw" not in self.re_plant_specs:
            self.re_plant_specs = self.re_plant_specs.assign(fom_per_kw=np.nan)
        # ILR adjusted normalized profiles
        full_prod = (
            re_profiles
            * self.re_plant_specs.ilr.to_numpy()
            * self.re_plant_specs.capacity_mw.to_numpy()
        )
        ac_out = np.minimum(full_prod, self.re_plant_specs.interconnect_mw.to_numpy())
        excess = full_prod - ac_out
        return self.load_profile - ac_out.sum(axis=1), ac_out, excess

    def dc_charge(self):
        """Align excess_re to match the storage facilities it could charge."""
        dc_charge = pd.DataFrame(
            np.nan, index=self.load_profile.index, columns=self.storage_specs.index
        )
        if self.re_excess is None:
            return dc_charge.fillna(0.0)
        # we need to be able to handle non-unique plant_id_eias for storage and re
        # so we can't just drop storage generator_ids, this requires some gymnastics
        # to select the correct columns from self.re_excess and name them in such a way
        # that combine_first works properly since it combines by index/column
        re_excess = (
            self.re_excess.groupby(level=0, axis=1)
            .sum()
            .sort_index(axis=1, ascending=False)
        )
        re_excess = re_excess.loc[:, [pid for pid, _ in dc_charge if pid in re_excess]]
        # put the correct storage index column names on the re_excess data
        re_excess.columns = dc_charge.loc[:, list(re_excess)].columns
        return (
            dc_charge.combine_first(re_excess)[dc_charge.columns]
            .set_axis(self.storage_specs.index, axis=1)
            .fillna(0.0)
        )

    def _add_total_and_missing_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add optional columns if they are missing.

        Add columns for total FOM and total startup from respective unit costs.
        """
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
        if "total_var_mwh" not in df:
            df = df.assign(
                total_var_mwh=lambda x: x[["vom_per_mwh", "fuel_per_mwh"]].sum(axis=1)
            )
        if "heat_rate" not in df:
            df = df.assign(heat_rate=np.nan)
        if "co2_factor" not in df:
            df = df.assign(co2_factor=np.nan)
        return df.drop(columns=["capacity_mw"])

    @staticmethod
    def _add_optional_cols(df: pd.DataFrame, df_name) -> pd.DataFrame:
        """Add optional column if not already present."""
        default_values = {
            "dispatchable_specs": (
                ("min_uptime", 0),
                ("exclude", False),
                ("no_limit", False),
            ),
            "storage_specs": (("reserve", 0.0),),
        }
        return df.assign(
            **{col: value for col, value in default_values[df_name] if col not in df}
        )

    def __getstate__(self):
        state = {}
        for name in self.__slots__:
            if all((hasattr(self, name), name not in ("_cached", "dt_idx", "_polars"))):
                state[name] = getattr(self, name)
        state["polars_state"] = self._polars.__getstate__()
        if not self.redispatch.collect().is_empty():
            for df_name in ("full_output", "load_summary"):
                try:
                    state[df_name] = getattr(self, df_name)()
                except Exception as exc:
                    LOGGER.warning("unable to write %s, %r", df_name, exc)
        return None, state

    def __setstate__(self, state: tuple[Any, dict]):
        _, state = state
        for k, v in state.items():
            if k in self.__slots__:
                setattr(self, k, v)
        self.dt_idx = self.load_profile.index
        self._cached = {}
        self._polars = IDConverter.__new__(IDConverter)
        self._polars.__setstate__(state["polars_state"])

    @classmethod
    def from_patio(cls, *args, **kwargs) -> DispatchModel:
        """Create :class:`.DispatchModel` with data from patio.BAScenario."""
        raise DeprecationWarning("`from_patio` is no longer supported.")

    @classmethod
    def from_fresh(
        cls,
        net_load_profile: pd.Series[float],
        dispatchable_specs: pd.DataFrame,
        dispatchable_cost: pd.DataFrame,
        storage_specs: pd.DataFrame,
        *,
        jit: bool = True,
    ) -> DispatchModel:
        """Run dispatch without historical hourly operating constraints."""
        warnings.warn(
            "`from_fresh` will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
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
    def is_redispatch(self) -> bool:
        """Determine if this is a redispatch.

        True if this is redispatch, i.e. has meaningful historical dispatch.
        """
        # more than 2 unique values are required because any plant that begins
        # operation during the period will have both 0 and its capacity
        return self.dispatchable_profiles.nunique().max() > 2

    @property
    def historical_dispatch(self) -> pd.DataFrame:
        """Total hourly historical cost by generator."""
        if self.is_redispatch:
            return self.dispatchable_profiles
        else:
            out = self.dispatchable_profiles.copy()
            out.loc[:, :] = np.nan
            return out

    # TODO probably a bad idea to use __call__, but nice to not have to think of a name
    def __call__(self, **kwargs) -> DispatchModel:
        """Run dispatch model."""
        # determine any dispatchable resources that should not be limited by historical
        # dispatch and set their max output to the greater of their capacity and
        # historical dispatch, then apply their operating and retirement dates
        no_limit = self.dispatchable_specs.no_limit.to_numpy()
        if np.any(no_limit):
            d_prof = self.dispatchable_profiles.copy()
            d_prof.loc[:, no_limit] = zero_profiles_outside_operating_dates(
                np.maximum(
                    d_prof.loc[:, no_limit],
                    self.dispatchable_specs.loc[no_limit, "capacity_mw"].to_numpy(),
                ),
                self.dispatchable_specs.loc[no_limit, "operating_date"],
                self.dispatchable_specs.loc[no_limit, "retirement_date"],
            )
            d_prof = d_prof.to_numpy(dtype=np.float_)
        else:
            d_prof = self.dispatchable_profiles.to_numpy(dtype=np.float_, copy=True)
        # determine any dispatchable resources that should be excluded from dispatch and
        # zero out their profile so they do not run
        to_exclude = (~self.dispatchable_specs.exclude).to_numpy(dtype=float)
        if np.any(to_exclude == 0.0):
            d_prof = d_prof * to_exclude

        coeff = (self.config | kwargs)["dynamic_reserve_coeff"]

        if self._metadata["jit"]:
            func = dispatch_engine_auto
            coeff = -10.0 if coeff == "auto" else coeff
        else:
            func = dispatch_engine.py_func
            if coeff == "auto":
                LOGGER.warning(
                    "when `jit == False` we cannot automatically determine "
                    "`dynamic_reserve_coeff` so it will be set to 1."
                )
                coeff = 1

        fos_prof, storage, system, starts = func(
            net_load=self.net_load_profile.to_numpy(dtype=np.float_),
            hr_to_cost_idx=(
                self.net_load_profile.index.year
                - self.net_load_profile.index.year.min()
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
            dispatchable_min_uptime=self.dispatchable_specs.min_uptime.to_numpy(
                dtype=np.int_
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
            storage_reserve=self.storage_specs.reserve.to_numpy(),
            dynamic_reserve_coeff=coeff,
        )
        self.redispatch = pl.concat(
            [
                self._polars.disp_big_idx,
                pl.from_numpy(
                    fos_prof.reshape(fos_prof.size, order="F"),
                    {"redispatch_mwh": pl.Float32},
                ),
            ],
            how="horizontal",
        ).lazy()
        self.storage_dispatch = pl.concat(
            [
                self._polars.storage_big_idx,
                pl.from_numpy(
                    np.vstack([storage[:, :, x] for x in range(storage.shape[2])]),
                    self.es_schema,
                ),
            ],
            how="horizontal",
        ).lazy()
        self.system_data = pl.concat(
            [
                self._polars.dt.collect(),
                pl.from_numpy(system, self.sys_schema),
            ],
            how="horizontal",
        ).lazy()
        # self.starts = (
        #     pd.DataFrame(
        #         starts.T,
        #         columns=self.dispatchable_specs.index,
        #         index=self.yrs_idx,
        #     )
        #     .stack([0, 1])
        #     .reorder_levels([1, 2, 0])
        #     .sort_index()
        # )
        return self

    def lost_load(
        self, comparison: pd.Series[float] | np.ndarray | float | None = None
    ) -> pl.LazyFrame:
        """Value counts of deficit.

        Number of hours during which deficit was in various duration bins.
        """
        if comparison is None:
            comparison = self.load_profile.max()
        bins = map(
            float,
            "0.0, 0.0001, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0".split(
                ", "
            ),
        )
        return (
            (self.system_data.select("deficit").collect().to_series() / comparison)
            .cut(bins)
            .groupby(by="category")
            .count()
            .sort("category")
            .lazy()
        )

    def hrs_to_check(
        self,
        kind: Literal["deficit", "curtailment"] = "deficit",
        cutoff: float = 0.01,
        comparison: pd.Series[float] | float | None = None,
    ) -> pl.Series:
        """Hours from dispatch to look at more closely.

        Hours with positive deficits are ones where not all of net load was served we
        want to be able to easily check the two hours immediately before these positive
        deficit hours.

        Args:
            kind: 'curtailment' or 'deficit'
            cutoff: if deficit / curtailment exceeds this proportion of
                ``comparison``, include the hour
            comparison: default is annual peak load

        Returns: list of hours
        """
        if comparison is None:
            comparison = (
                pl.from_pandas(
                    self.load_profile.groupby([pd.Grouper(freq="YS")])
                    .transform("max")
                    .reset_index()
                )
                .lazy()
                .with_columns(pl.col("datetime").cast(pl.Datetime("us")))
            )

        hrs = (
            self.system_data.join(comparison, on="datetime")
            .filter(pl.col(kind) / pl.col("load_profile") > cutoff)
            .select("datetime")
        )

        return (
            pl.concat(
                [
                    hrs.select(pl.col("datetime") - timedelta(hours=2)),
                    hrs.select(pl.col("datetime") - timedelta(hours=1)),
                    hrs,
                    hrs.select(pl.col("datetime") + timedelta(hours=1)),
                ],
                how="vertical",
            )
            .unique()
            .sort("datetime")
            .collect()
            .to_series()
        )

    def hourly_data_check(
        self, kind: Literal["deficit", "curtailment"] = "deficit", cutoff: float = 0.01
    ) -> pl.LazyFrame:
        """Aggregate data for :meth:`.DispatchModel.hrs_to_check`.

        Args:
            kind: 'curtailment' or 'deficit'
            cutoff: if deficit / curtailment exceeds this proportion of
                ``comparison``, include the hour

        Returns: context for hours preceding deficit or curtailment hours
        """
        disp_block = (
            self.pl_dispatchable_profiles.join(
                self.pl_dispatchable_specs, on=["plant_id_eia", "generator_id"]
            )
            .join(self.redispatch, on=["datetime", "combined_id"])
            .pipe(self._add_capacity)
            .with_columns(
                available=pl.when(pl.col("no_limit"))
                .then(pl.max(pl.col("capacity_mw"), pl.col("historical_mwh")))
                .when(pl.col("exclude"))
                .then(0.0)
                .otherwise(pl.col("historical_mwh"))
            )
            .with_columns(
                headroom=pl.col("available")
                - pl.col("historical_mwh").shift_and_fill(0.0, periods=1)
            )
            .with_columns(
                pl.when(pl.col("headroom") > 0)
                .then(pl.min(pl.col("headroom"), pl.col("ramp_rate")))
                .otherwise(pl.col("headroom"))
                .alias("headroom_hr-1")
            )
            .groupby("datetime")
            .agg(
                pl.sum("capacity_mw").alias("max_dispatch"),
                pl.sum("redispatch_mwh").alias("redispatch"),
                pl.sum("historical_mwh").alias("historical_dispatch"),
                pl.sum("available"),
                pl.sum("headroom_hr-1"),
            )
        )

        return (
            pl.from_pandas(self.load_profile.reset_index())
            .lazy()
            .join(
                pl.from_pandas(self.net_load_profile.reset_index()).lazy(),
                on="datetime",
            )
            .with_columns(pl.col("datetime").cast(pl.Datetime("us")))
            .join(self.system_data, on="datetime")
            .select(
                pl.col("datetime"),
                pl.col("load_profile").alias("gross_load"),
                pl.col("0").alias("net_load"),
                pl.col("load_adjustment"),
                pl.col(kind),
            )
            .join(disp_block, on="datetime")
            .join(
                self.storage_dispatch.groupby("datetime")
                .agg(
                    pl.sum("discharge"),
                    pl.sum("charge"),
                    pl.sum("soc"),
                )
                .select(
                    pl.col("datetime"),
                    (pl.col("discharge") - pl.col("charge")).alias("net_storage"),
                    pl.col("soc").alias("state_of_charge"),
                ),
                on="datetime",
            )
            .join(
                self.pl_re_profiles_ac.groupby("datetime").agg(
                    pl.sum("redispatch_mwh").alias("re")
                ),
                on="datetime",
            )
            .join(
                pl.from_pandas(self.re_excess.sum(axis=1).reset_index())
                .lazy()
                .select(
                    pl.col("datetime").cast(pl.Datetime("us")),
                    pl.col("0").alias("re_excess"),
                ),
                on="datetime",
            )
        ).filter(pl.col("datetime").is_in(self.hrs_to_check(kind=kind, cutoff=cutoff)))

    def storage_capacity(self) -> pl.LazyFrame:
        """Value counts of charge and discharge.

        Number of hours when max of storage charge or discharge was in various bins.
        """
        rates = self.storage_dispatch.with_columns(
            rate=pl.max(pl.col("charge"), pl.col("discharge"))
        )
        # a mediocre way to define the bins...
        d_max = int(np.ceil(rates.select(pl.max("rate")).collect().item()))
        g_bins = [
            y
            for x in range(1, 6)
            for y in (1.0 * 10**x, 2.5 * 10.0**x, 5.0 * 10**x)
        ]
        bins = [0, 0.01] + [x for x in g_bins if x < d_max] + [d_max]

        with pl.StringCache():
            out = (
                pl.concat(
                    [
                        rates.filter(pl.col("combined_id") == x)
                        .select("rate")
                        .collect()
                        .to_series()
                        .cut(bins)
                        .groupby("category")
                        .agg(pl.col("rate").count())
                        .rename({"rate": x})
                        for x in rates.select("combined_id")
                        .unique()
                        .collect()
                        .to_series()
                    ],
                    how="align",
                )
                .sort("category")
                .fill_null(0.0)
            )
        return out.lazy()

    def storage_durations(self) -> pl.LazyFrame:
        """Value counts of state of charge hours.

        Number of hours during which state of charge was in various duration bins.
        """
        durs = (
            self.storage_dispatch.join(
                self.pl_storage_specs, on=["plant_id_eia", "generator_id"]
            )
            .with_columns(dur=pl.col("soc") / pl.col("capacity_mw"))
            .select(["combined_id", "dur"])
        )
        # a mediocre way to define the bins...
        d_max = int(np.ceil(durs.select(pl.max("dur")).collect().item()))
        g_bins = [0.0, 0.01, 2.0, 4.0] + [
            y
            for x in range(1, 6)
            for y in (1.0 * 10**x, 2.5 * 10.0**x, 5.0 * 10**x)
        ]
        bins = [x for x in g_bins if x < d_max] + [d_max]

        with pl.StringCache():
            out = (
                pl.concat(
                    [
                        durs.filter(pl.col("combined_id") == x)
                        .select("dur")
                        .collect()
                        .to_series()
                        .cut(bins)
                        .groupby("category")
                        .agg(pl.col("dur").count())
                        .rename({"dur": x})
                        for x in durs.select("combined_id")
                        .unique()
                        .collect()
                        .to_series()
                    ],
                    how="align",
                )
                .sort("category")
                .fill_null(0.0)
            )

        return out.lazy()

    def system_level_summary(
        self, freq: str = "YS", storage_rollup: dict | None = None, **kwargs
    ) -> pl.LazyFrame:
        """Create system and storage summary metrics.

        Args:
            freq: temporal frequency to aggregate hourly data to.
            storage_rollup: as {group name: [id1, id2, ...]} ids included here will
                be aggregated together according to the group names and will not be
                included in their own columns. Utilization metrics are not available.
            **kwargs:

        Returns: summary of curtailment, deficit, storage and select metrics
        """
        freq = self.pl_freq.get(freq, freq)
        es_ids = self.storage_specs.index.get_level_values("plant_id_eia").unique()
        if storage_rollup is not None:
            mapper = {i: str(i) for i in es_ids} | {
                n: k for k, v in storage_rollup.items() for n in v
            }
        else:
            mapper = {i: str(i) for i in es_ids}
        d_cols = ["max_mw", "max_hrs", "mw_utilization", "hrs_utilization"]
        es_roll_up = (
            self.storage_dispatch.join(
                self.pl_storage_specs, on=["plant_id_eia", "generator_id"]
            )
            .with_columns(es_grp=pl.col("plant_id_eia").map_dict(mapper))
            .groupby(["es_grp", "datetime"])
            .agg(
                pl.sum("charge"),
                pl.sum("discharge"),
                pl.sum("soc"),
                pl.sum("capacity_mw"),
                pl.max("duration_hrs"),
            )
            .with_columns(
                max_mw=pl.max(pl.col("charge"), pl.col("discharge")),
                max_hrs=pl.col("soc") / pl.col("capacity_mw"),
            )
            .sort(["datetime", "es_grp"])
            .groupby_dynamic("datetime", every=freq, period=freq, by="es_grp")
            .agg(
                pl.max("max_mw"),
                pl.max("max_hrs"),
                pl.max("capacity_mw"),
                pl.max("duration_hrs"),
            )
            .with_columns(
                mw_utilization=pl.col("max_mw") / pl.col("capacity_mw"),
                hrs_utilization=pl.col("max_hrs") / pl.col("duration_hrs"),
            )
            .select(["es_grp", "datetime", *d_cols])
        )
        to_join = [
            es_roll_up.filter(pl.col("es_grp") == i)
            .select(["datetime", *d_cols])
            .rename({d: f"storage_{i}_{d}" for d in d_cols})
            for i in sorted(set(mapper.values()), reverse=True)
        ]
        es = to_join.pop(0)
        for df in to_join:
            es = es.join(df, on="datetime")

        return (
            self.system_data.join(
                pl.from_pandas(
                    self.load_profile.reset_index(),
                    schema_overrides={
                        "datetime": pl.Datetime("us"),
                        "load_profile": pl.Float32,
                    },
                )
                .rename({"load_profile": "load_mwh"})
                .lazy(),
                on="datetime",
            )
            .join(
                self.pl_re_profiles_ac.sort("datetime")
                .rename({"redispatch_mwh": "re_mwh"})
                .groupby("datetime")
                .agg(pl.sum("re_mwh")),
                on="datetime",
            )
            .with_columns(
                re_curtailment_mwh=pl.min(pl.col("curtailment"), pl.col("re_mwh")),
                deficit_gt_2pct_count=pl.when(
                    pl.col("deficit") / pl.col("load_mwh").max() > 0.02
                )
                .then(1)
                .otherwise(0),
            )
            .sort("datetime")
            .groupby_dynamic("datetime", every=freq, period=freq)
            .agg(
                pl.sum("deficit"),
                pl.sum("dirty_charge"),
                pl.sum("curtailment"),
                pl.sum("load_mwh"),
                pl.sum("re_mwh"),
                pl.sum("re_curtailment_mwh"),
                (pl.max("deficit") / pl.lit(self.load_profile.max())).alias(
                    "deficit_max_pct_net_load"
                ),
                pl.sum("deficit_gt_2pct_count"),
            )
            .with_columns(
                deficit_pct=pl.col("deficit") / pl.col("load_mwh"),
                curtailment_pct=pl.col("curtailment") / pl.col("load_mwh"),
                re_curtailment_pct=pl.col("re_curtailment_mwh") / pl.col("re_mwh"),
            )
            .rename({c: f"{c}_mwh" for c in ("deficit", "dirty_charge", "curtailment")})
            .join(es, on="datetime")
        )

    def re_summary(
        self,
        by: str | None = "technology_description",
        freq: str = "YS",
        **kwargs,
    ) -> pl.LazyFrame:
        """Create granular summary of renewable plant metrics."""
        if self.re_profiles_ac is None or self.re_plant_specs is None:
            raise AssertionError(
                "at least one of `re_profiles` and `re_plant_specs` is `None`"
            )

        freq = self.pl_freq.get(freq, freq)
        pl_by = ["plant_id_eia", "generator_id"] if by is None else [by]
        id_cols = ["plant_id_eia", "generator_id"]

        tech_col = (
            [pl.first("technology_description")]
            if by != "technology_description"
            else []
        )
        return (
            self.pl_re_profiles_ac.join(self.pl_re_plant_specs, on=id_cols)
            .pipe(self._add_capacity)
            .with_columns(
                redispatch_cost_fom=pl.when(
                    (pl.col("datetime") >= pl.col("operating_date"))
                    & (
                        pl.col("datetime")
                        <= pl.col("retirement_date").fill_null(
                            pl.col("datetime").max() + timedelta(30)
                        )
                    )
                )
                .then(
                    pl.col("capacity_mw")
                    * pl.col("fom_per_kw")
                    * pl.lit(1000)
                    / pl.col("datetime")
                    .count()
                    .over(*id_cols, pl.col("datetime").dt.year())
                )
                .otherwise(pl.lit(0.0)),
            )
            .sort(["datetime", *pl_by])
            .groupby_dynamic("datetime", every=freq, period=freq, by=pl_by)
            .agg(
                pl.max("capacity_mw"),
                pl.sum("redispatch_mwh"),
                pl.sum("redispatch_cost_fom"),
                pl.first("ilr"),
                pl.first("interconnect_mw"),
                pl.first("fom_per_kw"),
                pl.first("operating_date"),
                *tech_col,
            )
        )

    @staticmethod
    def _add_capacity(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.with_columns(
            capacity_mw=pl.when(
                (pl.col("datetime") >= pl.col("operating_date"))
                & (
                    pl.col("datetime")
                    <= pl.col("retirement_date").fill_null(
                        pl.col("datetime").max() + timedelta(30)
                    )
                )
            )
            .then(pl.col("capacity_mw"))
            .otherwise(pl.lit(0.0))
        )

    def storage_summary(
        self,
        by: str | None = "technology_description",
        freq: str = "YS",
        **kwargs,
    ) -> pl.LazyFrame:
        """Create granular summary of storage plant metrics."""
        freq = self.pl_freq.get(freq, freq)
        pl_by = ["plant_id_eia", "generator_id"] if by is None else by
        id_cols = ["plant_id_eia", "generator_id"]

        return (
            self.storage_dispatch.join(self.pl_storage_specs, on=id_cols)
            .with_columns(
                capacity_mw=pl.when(pl.col("datetime") >= pl.col("operating_date"))
                .then(pl.col("capacity_mw"))
                .otherwise(pl.lit(0.0)),
                redispatch_mwh=(pl.col("discharge") - pl.col("gridcharge")),
            )
            .groupby_dynamic("datetime", every=freq, period=freq, by=pl_by)
            .agg(
                pl.first("capacity_mw"),
                pl.sum("redispatch_mwh"),
                pl.sum("discharge"),
                pl.sum("gridcharge"),
                pl.first("duration_hrs"),
                pl.first("roundtrip_eff"),
                pl.first("operating_date"),
                pl.first("technology_description"),
                pl.first("reserve"),
            )
        )

    def full_output(self, freq: str = "YS", *, augment=False) -> pl.LazyFrame:
        """Create full operations output."""
        # setup deficit/curtailment as if they were resources for full output, the idea
        # here is that you could rename them purchase/sales.
        freq = self.pl_freq.get(freq, freq)
        id_cols = ["plant_id_eia", "generator_id"]
        def_cur = (
            self.system_data.sort("datetime")
            .groupby_dynamic("datetime", every=freq, period=freq)
            .agg(pl.sum("deficit"), pl.sum("curtailment") * -1)
            .melt(
                id_vars="datetime",
                value_vars=["deficit", "curtailment"],
                variable_name="generator_id",
                value_name="redispatch_mwh",
            )
            .with_columns(
                plant_id_eia=pl.lit(0).cast(pl.Int32),
                technology_description=pl.col("generator_id"),
            )
            .select([*id_cols, "datetime", "redispatch_mwh", "technology_description"])
        )
        return pl.concat(
            [
                self.dispatchable_summary(by=None, freq=freq, augment=augment),
                self.re_summary(by=None, freq=freq),
                self.storage_summary(by=None, freq=freq),
                def_cur,
            ],
            how="diagonal",
        ).sort(["plant_id_eia", "generator_id", "datetime"])

    def load_summary(self, freq="YS", **kwargs) -> pl.LazyFrame:
        """Create summary of load data."""
        freq = self.pl_freq.get(freq, freq)
        return (
            pl.from_pandas(
                self.net_load_profile.to_frame("net_load")
                .join(self.load_profile.to_frame("gross_load"))
                .reset_index(),
                schema_overrides={
                    "datetime": pl.Datetime("us"),
                    "net_load": pl.Float32,
                    "gross_load": pl.Float32,
                },
            )
            .lazy()
            .sort("datetime")
            .groupby_dynamic("datetime", every=freq, period=freq)
            .agg(
                pl.sum("net_load"),
                pl.max("net_load").alias("net_load_peak"),
                pl.sum("gross_load"),
                pl.max("gross_load").alias("gross_load_peak"),
            )
        )

    def dispatchable_summary(
        self,
        by: str | None = None,
        freq: str = "YS",
        *,
        augment: bool = False,
        **kwargs,
    ) -> pl.LazyFrame:
        """Create granular summary of dispatchable plant metrics.

        Args:
            by: column from :attr:`.DispatchModel.dispatchable_specs` to use for
                grouping dispatchable plants, if None, no column grouping
            freq: output time resolution
            augment: include columns from plant_specs columns
        """
        freq = self.pl_freq.get(freq, freq)
        pl_by = ["plant_id_eia", "generator_id"] if by is None else by
        id_cols = ["plant_id_eia", "generator_id"]

        if self.is_redispatch:
            hist_prof = self.pl_dispatchable_profiles
        else:
            # polars has different nan and null poisoning so to get previous behavior
            # set all values to zero
            hist_prof = self.pl_dispatchable_profiles.with_columns(
                historical_mwh=pl.lit(0.0)
            )

        out = (
            self.pl_dispatchable_profiles.join(self.pl_dispatchable_specs, on=id_cols)
            .pipe(self._add_capacity)
            .groupby_dynamic("datetime", every=freq, period=freq, by=pl_by)
            .agg(pl.max("capacity_mw"))
            .join(
                self._disp_summary_helper(
                    hist_prof,
                    t="historical",
                    by=pl_by,
                    freq=freq,
                    id_cols=id_cols,
                ),
                on=[*id_cols, "datetime"],
            )
            .join(
                self._disp_summary_helper(
                    self.redispatch,
                    t="redispatch",
                    by=pl_by,
                    freq=freq,
                    id_cols=id_cols,
                ),
                on=[*id_cols, "datetime"],
            )
        )
        if not self.is_redispatch:
            out = out.with_columns(historical_cost_fom=pl.lit(0.0))
        if not augment:
            return out.join(
                self.pl_dispatchable_specs.select(
                    "plant_id_eia",
                    "generator_id",
                    "technology_description",
                    "operating_date",
                ),
                on=["plant_id_eia", "generator_id"],
            )
        return out.join(
            self.pl_dispatchable_specs.drop("capacity_mw"), on=id_cols
        ).select(
            list(
                dict.fromkeys(out.columns)
                | dict.fromkeys(self.pl_dispatchable_specs.columns)
            )
        )

    def _disp_summary_helper(
        self, df: pl.LazyFrame, t, by, freq, id_cols
    ) -> pl.LazyFrame:
        out = (
            df.join(self.pl_dispatchable_specs, on=id_cols)
            .pipe(self._add_capacity)
            .join_asof(self.pl_dispatchable_cost, on="datetime", by=id_cols)
            .with_columns(
                (pl.col(f"{t}_mwh") * pl.col("heat_rate"))
                .fill_null(0.0)
                .alias(f"{t}_mmbtu"),
                (pl.col(f"{t}_mwh") * pl.col("heat_rate") * pl.col("co2_factor"))
                .fill_null(0.0)
                .alias(f"{t}_co2"),
                (pl.col(f"{t}_mwh") * pl.col("fuel_per_mwh")).alias(f"{t}_cost_fuel"),
                (pl.col(f"{t}_mwh") * pl.col("vom_per_mwh")).alias(f"{t}_cost_vom"),
                (
                    (
                        (pl.col(f"{t}_mwh") != 0.0)
                        & (pl.col(f"{t}_mwh").shift(1) == 0.0)
                        & (pl.col("plant_id_eia") == pl.col("plant_id_eia").shift(1))
                        & (pl.col("generator_id") == pl.col("generator_id").shift(1))
                    ).cast(pl.Int32)
                    * pl.col("startup_cost")
                ).alias(f"{t}_cost_startup"),
                pl.when(pl.col("capacity_mw").fill_null(0.0) > 0.0)
                .then(
                    pl.col("fom")
                    / pl.col("datetime")
                    .count()
                    .over(*id_cols, pl.col("datetime").dt.year())
                )
                .otherwise(pl.lit(0.0))
                .alias(f"{t}_cost_fom"),
            )
        )
        if t == "redispatch":
            out = out.with_columns(
                redispatch_cost_fom=pl.when(pl.col("exclude"))
                .then(pl.lit(0.0))
                .otherwise(pl.col(f"{t}_cost_fom"))
            )

        return out.groupby_dynamic("datetime", every=freq, period=freq, by=by).agg(
            pl.col(f"{t}_mwh").sum(),
            pl.col(f"{t}_mmbtu").sum(),
            pl.col(f"{t}_co2").sum(),
            pl.col(f"{t}_cost_fuel").sum(),
            pl.col(f"{t}_cost_vom").sum(),
            pl.col(f"{t}_cost_startup").sum(),
            pl.col(f"{t}_cost_fom").sum(),
        )

    def _plot_prep(self):
        storage = (
            self.storage_dispatch.groupby("datetime")
            .agg(pl.sum("discharge"), pl.sum("gridcharge").alias("charge") * -1)
            .melt(
                id_vars="datetime",
                value_vars=["discharge", "charge"],
                value_name="redispatch_mwh",
                variable_name="technology_description",
            )
        )
        try:
            re = (
                self.re_summary(freq="1h")
                .with_columns(pl.col("technology_description").map_dict(PLOT_MAP))
                .groupby("technology_description", "datetime")
                .agg(pl.sum("redispatch_mwh"))
            )
        except AssertionError:
            re = pl.LazyFrame(schema=storage.schema)

        return pl.concat(
            [
                self.redispatch.join(
                    self.pl_dispatchable_specs,
                    on=["plant_id_eia", "generator_id"],
                )
                .with_columns(pl.col("technology_description").map_dict(PLOT_MAP))
                .groupby("technology_description", "datetime")
                .agg(pl.sum("redispatch_mwh")),
                re,
                storage,
                self.system_data.with_columns(pl.col("curtailment") * -1).melt(
                    id_vars="datetime",
                    value_vars=["curtailment", "deficit"],
                    variable_name="technology_description",
                    value_name="redispatch_mwh",
                ),
            ],
            how="diagonal",
        ).rename(
            {
                "technology_description": "resource",
                "redispatch_mwh": "net_generation_mwh",
            }
        )

    def _plot_prep_detail(self, begin, end):
        id_cols = ["plant_id_eia", "generator_id"]
        to_cat = [
            self.redispatch.join(
                self.pl_dispatchable_specs.select(*id_cols, "technology_description"),
                on=id_cols,
            ),
            self.pl_re_profiles_ac.join(
                self.pl_re_plant_specs.select(*id_cols, "technology_description"),
                on=id_cols,
            ),
            self.storage_dispatch.with_columns(charge=pl.col("gridcharge") * -1).melt(
                id_vars=[*id_cols, "datetime"],
                value_vars=["discharge", "charge"],
                variable_name="technology_description",
                value_name="redispatch_mwh",
            ),
            self.system_data.with_columns(curtailment=pl.col("curtailment") * -1)
            .melt(
                id_vars=["datetime"],
                value_vars=["curtailment", "deficit"],
                variable_name="generator_id",
                value_name="redispatch_mwh",
            )
            .with_columns(
                technology_description=pl.col("generator_id"),
                plant_id_eia=pl.lit(999),
            ),
        ]
        redispatch = (
            pl.concat(to_cat, how="diagonal")
            .drop("combined_id")
            .rename({"redispatch_mwh": "net_generation_mwh"})
            .with_columns(
                series=pl.lit("redispatch"),
                resource=pl.col("technology_description")
                .map_dict(PLOT_MAP)
                .fill_null(pl.col("technology_description"))
                .cast(pl.Utf8),
            )
        )
        if self.is_redispatch:
            hist = (
                self.pl_dispatchable_profiles.join(
                    self.pl_dispatchable_specs.select(
                        "plant_id_eia", "generator_id", "technology_description"
                    ),
                    on=["plant_id_eia", "generator_id"],
                )
                .rename({"historical_mwh": "net_generation_mwh"})
                .drop("combined_id")
                .with_columns(
                    series=pl.lit("historical"),
                    resource=pl.col("technology_description").map_dict(PLOT_MAP),
                )
            )
        else:
            hist = pl.LazyFrame(schema=redispatch.schema)

        return pl.concat([redispatch, hist], how="diagonal").filter(
            (pl.col("datetime") >= begin) & (pl.col("datetime") <= end)
        )

    def plot_period(
        self, begin, end=None, *, by_gen=True, compare_hist=False
    ) -> Figure:
        """Plot hourly dispatch by generator."""
        begin = pd.Timestamp(begin)
        end = begin + pd.Timedelta(days=7) if end is None else pd.Timestamp(end)
        net_load = self.net_load_profile.loc[begin:end]
        data = self._plot_prep_detail(begin, end)
        if data.filter(pl.col("series") == "historical").collect().is_empty():
            if compare_hist:
                LOGGER.warning("disabling `compare_hist` because no historical data")
            compare_hist = False
        hover_name = "plant_id_eia"
        if not by_gen:
            data = data.groupby(["datetime", "resource", "series"]).agg(
                pl.sum("net_generation_mwh")
            )
            hover_name = "resource"
        if not compare_hist:
            data = data.filter(pl.col("series") == "redispatch")
            kwargs = {}
        else:
            kwargs = {"facet_row": "series"}
        nl_args = {
            "x": net_load.index,
            "y": net_load,
            "name": "Net Load",
            "mode": "lines",
            "line_color": COLOR_MAP["Net Load"],
            "line_dash": "dot",
        }
        gl_args = {
            "x": self.load_profile.loc[begin:end].index,
            "y": self.load_profile.loc[begin:end],
            "name": "Gross Load",
            "mode": "lines",
            "line_color": COLOR_MAP["Gross Load"],
        }
        # see https://plotly.com/python/facet-plots/#adding-the-same-trace-to-all-facets
        out = (
            px.bar(
                data.collect()
                .to_pandas()
                .replace({"resource": {"charge": "Storage", "discharge": "Storage"}})
                .sort_values(["resource"], key=dispatch_key),
                x="datetime",
                y="net_generation_mwh",
                color="resource",
                hover_name=hover_name,
                color_discrete_map=COLOR_MAP,
                **kwargs,
            )
            .add_scatter(**nl_args)
            .update_layout(xaxis_title=None)
            .update_yaxes(title="MW", tickformat=",.0r")
            .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        )
        if compare_hist:
            out = out.add_scatter(
                **nl_args,
                row=2,
                col=1,
                showlegend=False,
            )
        if self.re_profiles_ac is None or self.re_plant_specs is None:
            return out

        out = out.add_scatter(**gl_args)
        if compare_hist:
            return out.add_scatter(
                **gl_args,
                row=2,
                col=1,
                showlegend=False,
            )
        return out

    def plot_year(self, year: int, freq="D") -> Figure:
        """Monthly facet plot of daily dispatch for a year."""
        if freq not in ("H", "1h", "D", "1d"):
            raise AssertionError(
                "`freq` must be 'D' or '1d' for day, or 'H' or '1h' for hour"
            )
        freq = self.pl_freq.get(freq, freq)
        out = (
            self._plot_prep()
            .filter(pl.col("datetime").dt.year() == year)
            .sort(["resource", "datetime"])
            .groupby_dynamic("datetime", every=freq, period=freq, by="resource")
            .agg(pl.sum("net_generation_mwh"))
            .with_columns(
                day=pl.col("datetime").dt.day(),
                hour=pl.col("datetime").dt.day() * 24 + pl.col("datetime").dt.hour(),
                year=pl.col("datetime").dt.strftime("%Y"),
                month=pl.col("datetime").dt.strftime("%B"),
            )
            .collect()
            .to_pandas()
            .sort_values(["resource", "month"], key=dispatch_key)
        )
        x, yt, ht = {
            "1d": ("day", "MWh", "resource"),
            "1h": ("hour", "MW", "datetime"),
        }[freq]
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
                # facet_row_spacing=0.02,
            )
            .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            .update_xaxes(title=None)
            .for_each_yaxis(
                lambda yaxis: (yaxis.update(title=yt) if yaxis.title.text else None)
            )
            .update_traces(
                marker_line_width=0.1,
            )
            .update_layout(bargap=0)
        )

    def plot_all_years(self) -> Figure:
        """Facet plot of daily dispatch for all years."""
        out = (
            (
                self._plot_prep()
                .sort(["resource", "datetime"])
                .groupby_dynamic("datetime", every="1d", period="1d", by="resource")
                .agg(pl.sum("net_generation_mwh"))
                .with_columns(
                    day=pl.col("datetime").dt.day(),
                    hour=pl.col("datetime").dt.day() * 24
                    + pl.col("datetime").dt.hour(),
                    year=pl.col("datetime").dt.strftime("%Y"),
                    month=pl.col("datetime").dt.strftime("%B"),
                )
            )
            .collect()
            .to_pandas()
            .replace({"resource": {"charge": "Storage", "discharge": "Storage"}})
            .sort_values(["resource", "year", "month"], key=dispatch_key)
        )
        return (
            px.bar(
                out,
                x="day",
                y="net_generation_mwh",
                color="resource",
                facet_col="month",
                facet_row="year",
                height=1750,
                width=1750,
                # hover_name=ht,
                color_discrete_map=COLOR_MAP,
                facet_row_spacing=0.002,
                facet_col_spacing=0.004,
            )
            .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            .update_xaxes(title=None)
            .for_each_yaxis(
                lambda yaxis: (yaxis.update(title="MWh") if yaxis.title.text else None)
            )
            .update_traces(
                marker_line_width=0.1,
            )
            .update_layout(bargap=0)
        )

    def plot_output(self, y: str, color="resource", freq="YS") -> Figure:
        """Plot a columns from :meth:`.DispatchModel.full_output`."""
        to_plot = self.full_output(freq=freq, augment=True).with_columns(
            year=pl.col("datetime").dt.year(),
            month=pl.col("datetime").dt.month(),
            resource=pl.col("technology_description").map_dict(PLOT_MAP),
            redispatch_cost=pl.sum(pl.col("^redispatch_cost.*$")),
            historical_cost=pl.sum(pl.col("^historical_cost.*$")),
            redispatch_mwh=pl.when(pl.col("technology_description") == "curtailment")
            .then(pl.col("redispatch_mwh") * -1)
            .otherwise(pl.col("redispatch_mwh")),
        )
        y_cat = y.removeprefix("redispatch_").removeprefix("historical_")
        b_kwargs = {
            "x": "datetime",
            "y": y,
            "color": color,
            "hover_name": "plant_id_eia",
            "color_discrete_map": COLOR_MAP,
        }
        if series_facet := all(
            ("redispatch_" + y_cat in to_plot, "historical_" + y_cat in to_plot)
        ):
            to_plot1 = to_plot.melt(
                id_vars=[
                    "plant_id_eia",
                    "generator_id",
                    "datetime",
                    "capacity_mw",
                    "plant_name_eia",
                    "technology_description",
                    "year",
                    "month",
                    "resource",
                ],
                value_vars=["redispatch_" + y_cat, "historical_" + y_cat],
                variable_name="series",
                value_name=y_cat,
            ).with_columns(pl.col("series").str.replace("_" + y_cat, ""))
            if (
                series_facet := to_plot1.groupby("series")
                .agg(pl.sum(y_cat))
                .filter(pl.col("series") == "historical")
                .select(y_cat)
                .collect()
                .item()
                > 0.0
            ):
                b_kwargs.update(facet_row="series", y=y_cat)
                to_plot = to_plot1
        if freq == "MS":
            b_kwargs.update(
                facet_col="year", facet_col_wrap=2, x="month", facet_row_spacing=0.003
            )
            if series_facet:
                b_kwargs.update(facet_row="year", facet_col="series", facet_col_wrap=0)
        return (
            px.bar(
                to_plot.filter(
                    (pl.col(b_kwargs["y"]) != 0.0) & pl.col(b_kwargs["y"]).is_not_nan()
                )
                .collect()
                .to_pandas()
                .sort_values(
                    (
                        ["series", "resource", "year"]
                        if series_facet
                        else ["resource", "year"]
                    ),
                    key=dispatch_key,
                ),
                width=800,
                **b_kwargs,
            )
            .update_traces(
                marker_line_width=0.1,
            )
            .update_layout(xaxis_title=None, bargap=0.025)
            .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        )

    def __repr__(self) -> str:
        re_len = 0 if self.re_plant_specs is None else len(self.re_plant_specs)
        return (
            self.__class__.__qualname__
            + f"({', '.join(f'{k}={v}' for k, v in self._metadata.items())}, "
            f"n_dispatchable={len(self.dispatchable_specs)}, n_re={re_len}, "
            f"n_storage={len(self.storage_specs)})"
        )
