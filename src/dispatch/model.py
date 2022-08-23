"""Simple dispatch model, engine and interface."""


from __future__ import annotations

import logging

import numpy as np
import pandas as pd

__all__ = ["DispatchModel"]

from dispatch.engine import dispatch_engine, dispatch_engine_compiled

LOGGER = logging.getLogger(__name__)
MTDF = pd.DataFrame()
"""An empty :py:class:`pd.DataFrame`."""


class DispatchModel:
    """Class to contain the core dispatch model functionality.

    - allow the core dispatch model to accept data set up for different uses
    - provide a nicer API that accepts pandas objects, rather than the core numba/numpy engine
    - methods for common analysis of dispatch results
    """

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
                startup_cost: cost to start up the generator
                datetime(freq='YS'): a column for each year with marginal cost data
            fossil_profiles: set the maximum output of each generator in each hour
            storage_specs: rows are types of storage, columns must contain:
                capacity_mw: max charge/discharge capacity in MW
                duration_hrs: storage duration in hours
                roundtrip_eff: roundtrip efficiency
            re_profiles: ??
            re_plant_specs: ??
            jit: if True, use numba to compile the dispatch engine, False is mostly for debugging
            name: a name, only used in the repr
        """
        self.net_load_profile = net_load_profile
        self.jit = jit

        self.dt_idx = self.net_load_profile.index
        self.yrs_idx = self.dt_idx.to_series().groupby([pd.Grouper(freq="YS")]).first()

        # make sure we have all the `fossil_plant_specs` columns we need
        for col in ("capacity_mw", "ramp_rate", "startup_cost"):
            assert col in fossil_plant_specs, f"`storage_specs` requires `{col}` column"
        assert all(
            x in fossil_plant_specs for x in self.yrs_idx
        ), "`fossil_plant_specs` requires columns for plant cost with 'YS' datetime names"
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
                [0.0, 0, 1.0], columns=["capacity_mw", "duration_hrs", "roundtrip_eff"]
            )
        else:
            for col in ("capacity_mw", "duration_hrs", "roundtrip_eff"):
                assert col in storage_specs, f"`storage_specs` requires `{col}` column"
            self.storage_specs = storage_specs

        assert len(fossil_profiles) == len(self.net_load_profile)
        assert fossil_profiles.shape[1] == len(self.fossil_plant_specs)
        self.fossil_profiles = fossil_profiles

        self.re_plant_specs = re_plant_specs
        self.re_profiles = re_profiles

        self.disp_func = dispatch_engine_compiled if self.jit else dispatch_engine

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

    def __repr__(self) -> str:
        return (
            self.__class__.__qualname__
            + f"({self.name=}, {self.jit=}, n_plants={len(self.fossil_plant_specs)}, ...)".replace(
                "self.", ""
            )
        )

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
        return cls(
            net_load_profile=net_load_profile,
            fossil_plant_specs=fossil_plant_specs,
            fossil_profiles=pd.DataFrame(
                fossil_plant_specs.capacity_mw
                * np.ones((len(net_load_profile), fossil_plant_specs.shape[0])),
                columns=fossil_plant_specs.index,
                index=net_load_profile.index,
            ),
            storage_specs=storage_specs,
            jit=jit,
        )

    # TODO probably a bad idea to use __call__, but nice to not have to think of a name
    def __call__(self) -> None:
        """Run dispatch model."""
        fos_prof, storage, deficits, starts = self.disp_func(
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
        """Number of hours where storage charge or discharge was in various bins."""
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
