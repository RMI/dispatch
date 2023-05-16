"""Simple dispatch model interface."""
from __future__ import annotations

import logging
import warnings
from datetime import datetime

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

from etoolbox.datazip import IOMixin

from dispatch import __version__
from dispatch.constants import COLOR_MAP, MTDF, PLOT_MAP
from dispatch.engine import dispatch_engine
from dispatch.helpers import dispatch_key, zero_profiles_outside_operating_dates
from dispatch.metadata import LOAD_PROFILE_SCHEMA, Validator

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
        "_metadata",
        "_cached",
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
        jit: bool = True,  # noqa: FBT001, FBT002
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

        Explore the results, starting with how much load could not be met.

        >>> dm.lost_load()  # doctest: +NORMALIZE_WHITESPACE
        (-0.001, 0.0001]    8784
        (0.0001, 0.02]         0
        (0.02, 0.05]           0
        (0.05, 0.1]            0
        (0.1, 0.15]            0
        (0.15, 0.2]            0
        (0.2, 0.3]             0
        (0.3, 0.4]             0
        (0.4, 0.5]             0
        (0.5, 0.75]            0
        (0.75, 1.0]            0
        Name: count, dtype: int64

        Generate a full, combined output of all resources at specified frequency.

        >>> dm.full_output(freq="YS").round(1)  # doctest: +NORMALIZE_WHITESPACE
                                              capacity_mw  historical_mwh  historical_mmbtu  ...  duration_hrs  roundtrip_eff  reserve
        plant_id_eia generator_id datetime                                                   ...
        0            curtailment  2020-01-01          NaN             NaN               NaN  ...           NaN            NaN      NaN
                     deficit      2020-01-01          NaN             NaN               NaN  ...           NaN            NaN      NaN
        1            1            2020-01-01        350.0             0.0               0.0  ...           NaN            NaN      NaN
                     2            2020-01-01        500.0             0.0               0.0  ...           NaN            NaN      NaN
        2            1            2020-01-01        600.0             0.0               0.0  ...           NaN            NaN      NaN
        5            1            2020-01-01        500.0             NaN               NaN  ...           NaN            NaN      NaN
                     es           2020-01-01        250.0             NaN               NaN  ...           4.0            0.9      0.0
        6            1            2020-01-01        500.0             NaN               NaN  ...           NaN            NaN      NaN
        7            1            2020-01-01        200.0             NaN               NaN  ...          12.0            0.5      0.0
        <BLANKLINE>
        [9 rows x 30 columns]
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

    def _add_total_and_missing_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add optional columns if they are missing.

        Add columns for total FOM and total startup from respective unit
        costs.
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

    def __setstate__(self, state: tuple[Any, dict]):
        _, state = state
        for k, v in state.items():
            if k in self.__slots__:
                setattr(self, k, v)
        self.dt_idx = self.load_profile.index
        self._cached = {}

    def __getstate__(self):
        state = {}
        for name in self.__slots__:
            if all((hasattr(self, name), name not in ("_cached", "dt_idx"))):
                state[name] = getattr(self, name)
        if not self.redispatch.empty:
            for df_name in ("full_output", "load_summary"):
                try:
                    state[df_name] = getattr(self, df_name)()
                except Exception as exc:
                    LOGGER.warning("unable to write %s, %r", df_name, exc)
        return None, state

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

        True if this is redispatch, i.e. has meaningful historical
        dispatch.
        """
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

        no_limit = self.dispatchable_specs.no_limit.to_numpy()
        if np.any(no_limit):
            d_prof[:, no_limit] = np.maximum(
                d_prof[:, no_limit],
                self.dispatchable_specs.loc[no_limit, "capacity_mw"].to_numpy(),
            )

        func = dispatch_engine if self._metadata["jit"] else dispatch_engine.py_func

        fos_prof, storage, deficits, starts = func(
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
            .stack([0, 1])
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
            zero_profiles_outside_operating_dates(
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
        """Value counts of deficit.

        Number of hours during which deficit was in various duration
        bins.
        """
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

        Hours with positive deficits are ones where not all of net load
        was served we want to be able to easily check the two hours
        immediately before these positive deficit hours.
        """
        if comparison is None:
            comparison = self.load_profile.groupby([pd.Grouper(freq="YS")]).transform(
                "max"
            )
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
        """Aggregate data for :meth:`.DispatchModel.hrs_to_check`."""
        max_disp = zero_profiles_outside_operating_dates(
            pd.DataFrame(
                1.0,
                index=self.load_profile.index,
                columns=self.dispatchable_profiles.columns,
            ),
            self.dispatchable_specs.operating_date,
            self.dispatchable_specs.retirement_date,
            self.dispatchable_specs.capacity_mw.mask(
                self.dispatchable_specs.exclude, 0.0
            ),
        )

        available = np.where(
            self.dispatchable_specs.no_limit.to_numpy(),
            np.maximum(
                self.dispatchable_profiles,
                self.dispatchable_specs.loc[:, "capacity_mw"].to_numpy(),
            ),
            np.where(
                ~self.dispatchable_specs.exclude.to_numpy(),
                self.dispatchable_profiles,
                0.0,
            ),
        )

        headroom = available - np.roll(self.redispatch, 1, axis=0)
        max_ramp_from_previous = np.where(
            headroom > 0,
            np.minimum(headroom, self.dispatchable_specs.ramp_rate.to_numpy()),
            headroom,
        )

        out = pd.concat(
            {
                "gross_load": self.load_profile,
                "net_load": self.net_load_profile,
                "deficit": self.system_data.deficit,
                "max_dispatch": max_disp.sum(axis=1),
                "redispatch": self.redispatch.sum(axis=1),
                "historical_dispatch": self.dispatchable_profiles.sum(axis=1),
                "available": pd.Series(
                    available.sum(axis=1), index=self.load_profile.index
                ),
                "headroom_hr-1": pd.Series(
                    max_ramp_from_previous.sum(axis=1), index=self.load_profile.index
                ),
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
        """Value counts of charge and discharge.

        Number of hours when storage charge or discharge was in various
        bins.
        """
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
        """Value counts of state of charge hours.

        Number of hours during which state of charge was in various
        duration bins.
        """
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
                        / (
                            self.storage_specs.loc[i, "capacity_mw"].item()
                            if np.all(self.storage_specs.loc[i, "capacity_mw"] > 0)
                            else 1.0
                        )
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
                / (
                    self.storage_specs.loc[i, "capacity_mw"].item()
                    if np.all(self.storage_specs.loc[i, "capacity_mw"] > 0)
                    else 1.0
                )
                for i in self.storage_specs.index.get_level_values("plant_id_eia")
            },
            **{
                f"storage_{i}_hrs_utilization": out[f"storage_{i}_max_hrs"]
                / (
                    self.storage_specs.loc[i, "duration_hrs"].item()
                    if np.all(self.storage_specs.loc[i, "duration_hrs"] > 0)
                    else 1.0
                )
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
        fom = (
            zero_profiles_outside_operating_dates(
                (
                    self.re_plant_specs.capacity_mw
                    * 1000
                    * self.re_plant_specs.fom_per_kw
                )
                .to_frame(name=self.yrs_idx[0])
                .reindex(self.yrs_idx, axis=1, method="ffill")
                .T,
                self.re_plant_specs.operating_date.apply(
                    lambda x: x.replace(day=1, month=1)
                ),
                self.re_plant_specs.retirement_date.apply(
                    lambda x: x.replace(day=1, month=1)
                ),
            )
            .reindex(index=self.load_profile.index, method="ffill")
            .divide(
                self.load_profile.groupby(pd.Grouper(freq="YS")).transform("count"),
                axis=0,
            )
        )
        out = (
            self.re_profiles_ac.groupby([pd.Grouper(freq=freq)])
            .sum()
            .stack(["plant_id_eia", "generator_id"])
            .to_frame(name="redispatch_mwh")
            .merge(
                fom.groupby([pd.Grouper(freq=freq)])
                .sum()
                .stack(["plant_id_eia", "generator_id"])
                .to_frame(name="redispatch_cost_fom"),
                left_index=True,
                right_index=True,
                validate="1:1",
            )
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
        return out.groupby([by, "datetime"]).sum(numeric_only=True)

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
        def_cur = self.grouper(self.system_data, by=None, freq=freq)[
            ["deficit", "curtailment"]
        ]
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
        *,
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
        hr = self.dispatchable_cost.heat_rate.unstack([0, 1]).reindex(
            index=self.load_profile.index, method="ffill"
        )
        co2 = self.dispatchable_cost.co2_factor.unstack([0, 1]).reindex(
            index=self.load_profile.index, method="ffill"
        )

        out = (
            pd.concat(
                [
                    self.strict_grouper(
                        zero_profiles_outside_operating_dates(
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
                        self.historical_dispatch * hr,
                        by=by,
                        freq=freq,
                        col_name="historical_mmbtu",
                    ),
                    self.grouper(
                        self.historical_dispatch * hr * co2,
                        by=by,
                        freq=freq,
                        col_name="historical_co2",
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
                        self.redispatch * hr,
                        by=by,
                        freq=freq,
                        col_name="redispatch_mmbtu",
                    ),
                    self.grouper(
                        self.redispatch * hr * co2,
                        by=by,
                        freq=freq,
                        col_name="redispatch_co2",
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
                avoided_mmbtu=lambda x: np.maximum(
                    x.historical_mmbtu - x.redispatch_mmbtu, 0.0
                ),
                avoided_co2=lambda x: np.maximum(
                    x.historical_co2 - x.redispatch_co2, 0.0
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
                pct_replaced=lambda x: np.nan_to_num(
                    np.maximum(x.avoided_mwh / x.historical_mwh, 0.0)
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
            "state",
            "latitude",
            "longitude",
        ]
        return (
            out.reset_index()
            .merge(
                self.dispatchable_specs.reset_index(),
                on=["plant_id_eia", "generator_id"],
                validate="m:1",
                suffixes=(None, "_l"),
            )
            .set_index(out.index.names)[
                list(out.columns)
                + [col for col in cols if col in self.dispatchable_specs]
            ]
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

        def arrange(df):
            return (
                df.loc[begin:end, :]
                .rename_axis(
                    columns=["plant_id_eia", "generator_id", "technology_description"]
                )
                .stack([0, 1, 2])
                .to_frame(name="net_generation_mwh")
                .reset_index()
                .assign(resource=lambda x: x.technology_description.replace(PLOT_MAP))
                .query("net_generation_mwh != 0.0 & net_generation_mwh.notna()")
            )

        return pd.concat(
            [
                arrange(
                    pd.concat(
                        to_cat,
                        axis=1,
                    )
                ).assign(series="redispatch"),
                arrange(
                    self.historical_dispatch.set_axis(
                        pd.MultiIndex.from_frame(
                            self.dispatchable_specs.technology_description.reset_index()
                        ),
                        axis=1,
                    )
                ).assign(series="historical"),
            ],
            axis=0,
        )

    def plot_period(
        self, begin, end=None, *, by_gen=True, compare_hist=False
    ) -> Figure:
        """Plot hourly dispatch by generator."""
        begin = pd.Timestamp(begin)
        end = begin + pd.Timedelta(days=7) if end is None else pd.Timestamp(end)
        net_load = self.net_load_profile.loc[begin:end]
        data = self._plot_prep_detail(begin, end)
        if data.query("series == 'historical'").empty:
            if compare_hist:
                LOGGER.warning("disabling `compare_hist` because no historical data")
            compare_hist = False
        hover_name = "plant_id_eia"
        if not by_gen:
            data = data.groupby(["datetime", "resource"]).sum().reset_index()
            hover_name = "resource"
        if not compare_hist:
            data = data.query("series == 'redispatch'")
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
                data.replace(
                    {"resource": {"charge": "Storage", "discharge": "Storage"}}
                ).sort_values(["resource"], key=dispatch_key),
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
        if freq not in ("H", "D"):
            raise AssertionError("`freq` must be 'D' for day or 'H' for hour")
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
            self._plot_prep()
            .reset_index()
            .groupby([pd.Grouper(freq="D", key="datetime"), "resource"])
            .sum()
            .reset_index()
            .assign(
                day=lambda z: z.datetime.dt.day,
                hour=lambda z: z.datetime.dt.day * 24 + z.datetime.dt.hour,
                year=lambda z: z.datetime.dt.strftime("%Y"),
                month=lambda z: z.datetime.dt.strftime("%B"),
                resource=lambda z: z.resource.replace(
                    {"charge": "Storage", "discharge": "Storage"}
                ),
            )
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
        to_plot = (
            self.full_output(freq=freq)
            .reset_index()
            .assign(
                year=lambda x: x.datetime.dt.year,
                month=lambda x: x.datetime.dt.month,
                resource=lambda x: x.technology_description.replace(PLOT_MAP),
                redispatch_cost=lambda x: x.filter(like="redispatch_cost").sum(axis=1),
                historical_cost=lambda x: x.filter(like="historical_cost").sum(axis=1),
                redispatch_mwh=lambda x: x.redispatch_mwh.mask(
                    x.technology_description == "curtailment", x.redispatch_mwh * -1
                ),
            )
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
                var_name="series",
                value_name=y_cat,
            ).assign(series=lambda x: x.series.str.split("_" + y_cat, expand=True)[0])
            if (
                series_facet := to_plot1.groupby("series")[y_cat]  # noqa: PD008
                .sum()
                .at["historical"]
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
                to_plot[to_plot[b_kwargs["y"]] != 0.0]
                .dropna(subset=b_kwargs["y"])
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
