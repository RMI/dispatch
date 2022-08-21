"""Simple dispatch model, engine and interface."""


from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numba import njit

__all__ = ["ba_dispatch", "DispatchModel"]


LOGGER = logging.getLogger(__name__)
MTDF = pd.DataFrame()


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
            jit: if True, use numba to compile the dispatch function
            name: a name, only used in the repr
        """
        self.net_load_profile = net_load_profile
        self.jit = jit
        self.name = name

        self.dt_idx = self.net_load_profile.index
        self.yrs_idx = self.dt_idx.to_series().groupby([pd.Grouper(freq="YS")]).first()

        # make sure we have all the `fossil_plant_specs` columns we need
        for col in ("capacity_mw", "ramp_rate", "startup_cost"):
            assert col in fossil_plant_specs, f"`storage_specs` requires `{col}` column"
        assert all(
            x in fossil_plant_specs for x in self.yrs_idx
        ), "`fossil_plant_specs` requires columns for plant cost with 'YS' datetime names"
        self.fossil_plant_specs: pd.DataFrame = fossil_plant_specs

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

        self.disp_func = ba_dispatch if self.jit else _ba_dispatch

        self.fossil_redispatch = MTDF.copy()
        self.storage_dispatch = MTDF.copy()
        self.system_data = pd.DataFrame(
            columns=["deficit", "dirty_charge", "curtailment"]
        )
        self.starts: pd.DataFrame | pd.Series[int] = pd.DataFrame(
            columns=self.fossil_plant_specs.index
        )

    def __repr__(self) -> str:
        return (
            self.__class__.__qualname__
            + f"({self.name=}, n_plants={len(self.fossil_plant_specs)}, ...)".replace(
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

    def __call__(self) -> None:
        """Run dispatch model."""
        fos_prof, storage, deficits, starts = self.disp_func(
            net_load=self.net_load_profile.to_numpy(dtype=np.float64),
            hr_to_cost_idx=(
                self.net_load_profile.index.year  # type: ignore
                - self.net_load_profile.index.year.min()  # type: ignore
            ).to_numpy(dtype=np.int64),
            fossil_profiles=self.fossil_profiles.to_numpy(dtype=np.float64),
            fossil_ramp_mw=self.fossil_plant_specs.ramp_rate.to_numpy(dtype=np.float64),
            fossil_startup_cost=self.fossil_plant_specs.startup_cost.to_numpy(
                dtype=np.float64
            ),
            fossil_marginal_cost=self.fossil_plant_specs[self.yrs_idx].to_numpy(
                dtype=np.float64
            ),
            storage_mw=self.storage_specs.capacity_mw.to_numpy(dtype=np.float64),
            storage_hrs=self.storage_specs.duration_hrs.to_numpy(dtype=np.int64),
            storage_eff=self.storage_specs.roundtrip_eff.to_numpy(dtype=np.float64),
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

    def storage_durations(self) -> pd.DataFrame:
        """Number of hours during which state of charge was in various duration bins."""
        df = self.storage_dispatch.filter(like="soc")
        durs = df / self.storage_specs.capacity_mw.to_numpy()
        d_min = int(np.floor(durs.min().min()))
        d_max = int(np.ceil(durs.max().max()))
        g_bins = [d_min, 0.01, 2, 4] + [
            y
            for x in range(1, 6)
            for y in (1.0 * 10**x, 2.5 * 10.0**x, 5.0 * 10**x)
        ]
        bins = [x for x in g_bins if x < d_max] + [d_max]
        return pd.concat(
            [pd.value_counts(pd.cut(durs[col], bins)).sort_index() for col in durs],
            axis=1,
        )

    def storage_capacity(self) -> pd.DataFrame:
        """Number of hours where storage charge or discharge was in various bins."""
        rates = self.storage_dispatch.filter(like="charge")
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

    def lost_load(
        self, comparison: pd.Series[float] | np.ndarray | float | None = None
    ) -> pd.Series[int]:
        """Number of hours during which deficit was in various duration bins."""
        if comparison is None:
            durs = self.system_data.deficit / self.net_load_profile
        else:
            durs = self.system_data.deficit / comparison
        bins = [
            0.0,
            0.0001,
            0.02,
            0.04,
            0.06,
            0.08,
            0.1,
            0.15,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]
        return pd.value_counts(pd.cut(durs, bins, include_lowest=True)).sort_index()

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


def _ba_dispatch(
    net_load: np.ndarray,
    hr_to_cost_idx: np.ndarray,
    fossil_profiles: np.ndarray,
    fossil_ramp_mw: np.ndarray,
    fossil_startup_cost: np.ndarray,
    fossil_marginal_cost: np.ndarray,
    storage_mw: np.ndarray,
    storage_hrs: np.ndarray,
    storage_eff: np.ndarray = np.array((0.9, 0.5)),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-ready dispatch engine.

    1.  first iterate through operating plants
    2.  then charge/discharge storage
    3.  if there is still a deficit, iterate through non-operating plants
        and turn them on if required

    Args:
        net_load: net load, as in net of RE generation, negative net load means
            excess renewables
        hr_to_cost_idx: an array that contains for each hour, the index of the correct
            column in fossil_marginal_cost that contains cost data for that hour
        fossil_profiles: historic plant dispatch, acts as an hourly upper constraint
            on this dispatch
        fossil_startup_cost: startup cost in $ for each fossil generator
        fossil_marginal_cost: annual marginal cost for each fossil generator in $/MWh
            rows are generators and columns are years
        storage_mw: max charge/discharge rate for storage in MW
        storage_hrs: duration of storage
        storage_eff: storage round-trip efficiency


    Returns:
        fossil_redispatch: new hourly fossil dispatch
        storage: hourly charge, discharge, and state of charge data
        system_level: hourly deficit, dirty charge, and total curtailment data
        starts: count of starts for each plant in each year


    """
    assert (
        len(storage_mw) == len(storage_hrs) == len(storage_eff)
    ), "storage data does not match"
    storage_soc_max: np.ndarray = storage_mw * storage_hrs

    assert (
        fossil_ramp_mw.shape[0]
        == fossil_startup_cost.shape[0]
        == fossil_marginal_cost.shape[0]
        == fossil_profiles.shape[1]
    ), "fossil plant data does not match"

    assert (
        len(net_load) == len(hr_to_cost_idx) == len(fossil_profiles)
    ), "profile lengths do not match"

    # internal fossil data we need to track; (0) current run op_hours
    # (1) whether we touched the plant in the first round of dispatch
    fossil_op_data: np.ndarray = np.zeros((fossil_ramp_mw.shape[0], 2), dtype=np.int64)
    # need to set op_hours to 1 for plants that we are starting off as operating
    fossil_op_data[:, 0] = np.where(fossil_profiles[0, :] > 0.0, 1, 0)

    # create an array to keep track of re-dispatch
    fossil_redispatch: np.ndarray = np.zeros_like(fossil_profiles)
    # to avoid having to do the first hour differently, we just assume original
    # dispatch in that hour and then skip it
    fossil_redispatch[0, :] = fossil_profiles[0, :]

    # create an array to determine the marginal cost dispatch order for each year
    # the values in each column represent the canonical indexes for each resource
    # and they are in the order of increasing marginal cost for that year (column)
    marginal_ranks: np.ndarray = np.hstack(
        (np.arange(fossil_marginal_cost.shape[0]).reshape(-1, 1), fossil_marginal_cost)
    )
    for i in range(1, 1 + fossil_marginal_cost[0, :].shape[0]):
        marginal_ranks[:, i] = marginal_ranks[marginal_ranks[:, i].argsort()][:, 0]
    marginal_ranks = marginal_ranks[:, 1:].astype(np.int64)

    # create an array in the order of startup rank whose elements are the canonical
    # resource index for that startup rank
    start_ranks: np.ndarray = np.vstack(
        (np.arange(fossil_startup_cost.shape[0]), fossil_startup_cost)
    ).T
    start_ranks = start_ranks[start_ranks[:, 1].argsort()][:, 0].astype(np.int64)

    # array to keep track of starts by year
    starts: np.ndarray = np.zeros_like(marginal_ranks)

    # array to keep track of hourly storage data. rows are hours, columns
    # (0) charge, (1) discharge, (2) state of charge
    storage: np.ndarray = np.zeros((len(net_load), 3, len(storage_mw)))

    # array to keep track of system level data (0) deficits, (1) dirty charge,
    # (2) curtailment
    system_level: np.ndarray = np.zeros_like(storage[:, :, 0])

    # the big loop where we iterate through all the hours
    for hr, (deficit, yr) in enumerate(zip(net_load, hr_to_cost_idx)):
        # because of the look-backs, the first hour has to be done differently
        # here we just skip it because its only one hour and we assume
        # historical fossil dispatch
        if hr == 0:
            # if there is excess RE we charge the battery in the first hour
            if deficit < 0.0:
                for es_i in range(storage.shape[2]):
                    # charge storage
                    storage[hr, 0, es_i] = min(
                        -deficit,
                        storage_mw[es_i],
                        storage_soc_max[es_i] / storage_eff[es_i],
                    )
                    storage[hr, 2, es_i] = storage[hr, 0, es_i] * storage_eff[es_i]
                    deficit += storage[hr, 0, es_i]
                    if deficit == 0.0:
                        break
            continue

        # want to figure out how much we'd like to dispatch fossil given
        # that we'd like to use storage before fossil
        max_discharge = 0.0
        for es_i in range(storage.shape[2]):
            max_discharge += min(storage[hr - 1, 2, es_i], deficit, storage_mw[es_i])
        prov_deficit = max(0, deficit - max_discharge)

        # new hour so reset where we keep track if we've touched a plant for this hour
        fossil_op_data[:, 1] = 0

        # dispatch plants in the order of their marginal cost for year yr
        for r in marginal_ranks[:, yr]:
            op_hours = fossil_op_data[r, 0]
            # we are only dealing with plants already operating here
            if op_hours == 0:
                continue
            ramp = fossil_ramp_mw[r]
            previous = fossil_redispatch[hr - 1, r]
            # a plant's output is the lesser of historical, max hour output based on
            # ramping constraints and then the greater of actual need and the min hour
            # output based on ramping constraints
            r_out = min(
                fossil_profiles[hr, r],
                previous + ramp,
                max(prov_deficit, previous - ramp),
            )

            # if we ran this hour, update op_hours col, if not set to 0
            fossil_op_data[r, 0] = op_hours + 1 if r_out > 0.0 else 0
            # we took care of this plant for this hour so don't want to touch
            # it again in start-up loop
            fossil_op_data[r, 1] = 1
            fossil_redispatch[hr, r] = r_out
            # keep a running total of remaining deficit, having this value be negative
            # just makes the loop code more complicated, if it actually should be
            # negative we capture that below
            prov_deficit = max(0, prov_deficit - r_out)

        # calculate the true deficit as the hour's net load less actual dispatch
        # of fossil plants in hr that were also operating in hr - 1
        deficit -= np.sum(fossil_redispatch[hr, :])

        # # negative deficit means excess generation, so we charge the battery
        # # and move on to the next hour
        if deficit < 0.0:
            for es_i in range(storage.shape[2]):
                # calculate the amount of charging, to account for battery capacity, we
                # make sure that `charge` would not put `soc` over `storage_soc_max`
                soc = storage[hr - 1, 2, es_i]
                charge = min(
                    -deficit,
                    storage_mw[es_i],
                    (storage_soc_max[es_i] - soc) / storage_eff[es_i],
                )
                # calculate new `soc` and check that it makes sense
                soc = soc + charge * storage_eff[es_i]
                assert soc <= storage_soc_max[es_i]
                # store charge and new soc
                storage[hr, 0, es_i], storage[hr, 2, es_i] = charge, soc
                # calculate the amount of charging that was dirty
                # TODO check that this calculation is actually correct
                system_level[hr, 1] += min(max(0, charge - net_load[hr] * -1), charge)
                # calculate the amount of total curtailment
                # TODO check that this calculation is actually correct
                system_level[hr, 2] += -deficit - charge
                deficit += charge
            continue

        # discharge batteries, the amount is the lesser of state of charge,
        # deficit, and the max MW of the battery
        for es_i in range(storage.shape[2]):
            discharge = min(storage[hr - 1, 2, es_i], deficit, storage_mw[es_i])
            storage[hr, 1, es_i] = discharge
            storage[hr, 2, es_i] = storage[hr - 1, 2, es_i] - discharge
            deficit -= discharge

        assert (
            deficit >= 0.0
        ), "negative deficit after, discharge, this shouldn't happen"
        # once we've dealt with operating plants and storage, if there is no positive
        # deficit we can skip startups and go on to the next hour
        if deficit == 0.0:
            continue

        # TODO check that this start_ranks ordering system is working properly
        for r in start_ranks:
            # we are only dealing with plants not already operating here
            if fossil_op_data[r, 1]:
                continue
            ramp = fossil_ramp_mw[r]

            # a fossil plant's output during an hour is the lesser of the deficit,
            # the plant's historical output, and the plant's re-dispatch output
            # in the previous hour + the plant's one hour max ramp
            r_out = min(
                deficit, fossil_profiles[hr, r], fossil_redispatch[hr - 1, r] + ramp
            )
            if r_out > 0.0:
                fossil_op_data[r, 0] = 1
                starts[r, yr] = starts[r, yr] + 1
                fossil_redispatch[hr, r] = r_out
                deficit -= r_out

                if deficit == 0.0:
                    break

        if deficit == 0.0:
            continue

        # if we end up here that means we never got the deficit to zero, we want
        # to keep track of that
        system_level[hr, 0] = deficit

    assert not np.any(
        storage[storage[:, 1, 0] > np.roll(storage[:, 2, 0], 1)]
    ), "discharge exceeded previous state of charge in at least 1 hour for es0"
    assert not np.any(
        storage[storage[:, 1, 1] > np.roll(storage[:, 2, 1], 1)]
    ), "discharge exceeded previous state of charge in at least 1 hour for es1"
    assert np.all(
        fossil_redispatch <= fossil_profiles * (1 + 1e-4)
    ), "redispatch exceeded historical dispatch in at least 1 hour"

    return fossil_redispatch, storage, system_level, starts


ba_dispatch = njit(_ba_dispatch, error_model="numpy")
