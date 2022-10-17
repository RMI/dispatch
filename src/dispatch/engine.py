"""Dispatch engine."""
from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["dispatch_engine_compiled", "dispatch_engine"]


def dispatch_engine(
    net_load: np.ndarray,
    hr_to_cost_idx: np.ndarray,
    historical_dispatch: np.ndarray,
    dispatchable_ramp_mw: np.ndarray,
    dispatchable_startup_cost: np.ndarray,
    dispatchable_marginal_cost: np.ndarray,
    storage_mw: np.ndarray,
    storage_hrs: np.ndarray,
    storage_eff: np.ndarray,
    storage_op_hour: np.ndarray,
    storage_dc_charge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch engine that can be compiled with :func:`numba.jit`.

    For each hour...

    1.  first iterate through operating plants
    2.  then charge/discharge storage
    3.  if there is still a deficit, iterate through non-operating plants
        and turn them on if required

    Args:
        net_load: net load, as in net of RE generation, negative net load means
            excess renewables
        hr_to_cost_idx: an array that contains for each hour, the index of the correct
            column in ``dispatchable_marginal_cost`` that contains cost data for that
            hour
        historical_dispatch: historic plant dispatch, acts as an hourly upper
            constraint on this dispatch
        dispatchable_ramp_mw: max one hour ramp in MW
        dispatchable_startup_cost: startup cost in $ for each dispatchable generator
        dispatchable_marginal_cost: annual marginal cost for each dispatchable
            generator in $/MWh rows are generators and columns are years
        storage_mw: max charge/discharge rate for storage in MW
        storage_hrs: duration of storage
        storage_eff: storage round-trip efficiency
        storage_op_hour: first hour in which storage is available, i.e. the index of
            the operating date
        storage_dc_charge: an array whose columns match each storage facility, and a
            row for each hour representing how much energy would be curtailed at an
            attached renewable facility because it exceeds the system's inverter. This
            then represents how much the storage in an RE+Storage facility could be
            charged by otherwise curtailed energy when ilr>1.


    Returns:
        redispatch: new hourly dispatch
        storage: hourly charge, discharge, and state of charge data
        system_level: hourly deficit, dirty charge, and total curtailment data
        starts: count of starts for each plant in each year
    """
    _validate_inputs(
        net_load,
        hr_to_cost_idx,
        historical_dispatch,
        dispatchable_ramp_mw,
        dispatchable_startup_cost,
        dispatchable_marginal_cost,
        storage_mw,
        storage_hrs,
        storage_eff,
        storage_dc_charge,
    )

    storage_soc_max: np.ndarray = storage_mw * storage_hrs

    marginal_ranks, start_ranks = make_rank_arrays(
        dispatchable_marginal_cost, dispatchable_startup_cost
    )

    # create an array to keep track of re-dispatch
    redispatch: np.ndarray = np.zeros_like(historical_dispatch)
    # array to keep track of starts by year
    starts: np.ndarray = np.zeros_like(marginal_ranks)
    # array to keep track of hourly storage data. rows are hours, columns
    # (0) charge, (1) discharge, (2) state of charge, (3) grid charge
    storage: np.ndarray = np.zeros((len(net_load), 4, len(storage_mw)))
    # array to keep track of system level data (0) deficits, (1) dirty charge,
    # (2) curtailment
    system_level: np.ndarray = np.zeros((len(net_load), 3))
    # internal dispatch data we need to track; (0) current run op_hours
    # (1) whether we touched the plant in the first round of dispatch
    op_data: np.ndarray = np.zeros(
        (dispatchable_marginal_cost.shape[0], 2), dtype=np.int64
    )

    # to avoid having to do the first hour differently, we just assume original
    # dispatch in that hour and then skip it
    redispatch[0, :] = historical_dispatch[0, :]
    # need to set op_hours to 1 for plants that we are starting off as operating
    op_data[:, 0] = np.where(historical_dispatch[0, :] > 0.0, 1, 0)

    # the big loop where we iterate through all the hours
    for hr, (deficit, yr) in enumerate(zip(net_load, hr_to_cost_idx)):
        # because of the look-backs, the first hour has to be done differently
        # here we just skip it because its only one hour and we assume
        # historical fossil dispatch
        if hr == 0:
            # if there is excess RE we charge the battery in the first hour
            if deficit < 0.0 or np.any(storage_dc_charge[hr, :] > 0.0):
                for es_i in range(storage.shape[2]):
                    # skip the `es_i` storage resource if it is not yet in operation
                    if storage_op_hour[es_i] > hr:
                        continue

                    storage[hr, :, es_i] = charge_storage(
                        deficit=deficit,
                        soc=0.0,  # previous soc
                        dc_charge=storage_dc_charge[hr, es_i],
                        mw=storage_mw[es_i],
                        soc_max=storage_soc_max[es_i],
                        eff=storage_eff[es_i],
                    )

                    # update the deficit if we are grid charging
                    if deficit < 0.0:
                        deficit += storage[hr, 3, es_i]
            continue

        # want to figure out how much we'd like to dispatch fossil given
        # that we'd like to use storage before fossil
        max_discharge = 0.0
        for es_i in range(storage.shape[2]):
            max_discharge += min(storage[hr - 1, 2, es_i], deficit, storage_mw[es_i])
        prov_deficit = max(0.0, deficit - max_discharge)

        # new hour so reset where we keep track if we've touched a plant for this hour
        op_data[:, 1] = 0

        # dispatch plants in the order of their marginal cost for year yr
        for r in marginal_ranks[:, yr]:
            op_hours = op_data[r, 0]
            # we are only dealing with plants already operating here
            if op_hours == 0:
                continue
            ramp = dispatchable_ramp_mw[r]
            previous = redispatch[hr - 1, r]
            # a plant's output is the lesser of historical, max hour output based on
            # ramping constraints and then the greater of actual need and the min hour
            # output based on ramping constraints
            r_out = min(
                historical_dispatch[hr, r],
                previous + ramp,
                max(prov_deficit, previous - ramp),
            )

            # if we ran this hour, update op_hours col, if not set to 0
            op_data[r, 0] = op_hours + 1 if r_out > 0.0 else 0
            # we took care of this plant for this hour so don't want to touch
            # it again in start-up loop
            op_data[r, 1] = 1
            redispatch[hr, r] = r_out
            # keep a running total of remaining deficit, having this value be negative
            # just makes the loop code more complicated, if it actually should be
            # negative we capture that when we calculate the actual deficit based
            # on redispatch below
            prov_deficit = max(0, prov_deficit - r_out)

        # calculate the true deficit as the hour's net load less actual dispatch
        # of fossil plants in hr that were also operating in hr - 1
        deficit -= np.sum(redispatch[hr, :])

        # negative deficit means excess generation, so we charge the battery
        # if there is DC-coupled storage, we also charge that storage if there is
        # RE generation that would otherwise be curtailed
        if deficit < 0.0 or np.any(storage_dc_charge[hr, :] > 0.0):
            for es_i in range(storage.shape[2]):
                # skip the `es_i` storage resource if it is not yet in operation
                if storage_op_hour[es_i] > hr:
                    continue
                storage[hr, :, es_i] = charge_storage(
                    deficit=deficit,
                    soc=storage[hr - 1, 2, es_i],  # previous soc
                    dc_charge=storage_dc_charge[hr, es_i],
                    mw=storage_mw[es_i],
                    soc_max=storage_soc_max[es_i],
                    eff=storage_eff[es_i],
                )
                # alias grid_charge
                grid_charge = storage[hr, 3, es_i]
                # calculate the amount of charging that was dirty
                # TODO check that this calculation is actually correct
                system_level[hr, 1] += min(
                    max(0, grid_charge - net_load[hr] * -1), grid_charge
                )
                # if we are charging from the grid, need to update the deficit
                if deficit < 0.0:
                    deficit += grid_charge
            # store the amount of total curtailment
            # TODO check that this calculation is actually correct

        # this 'continue' and storing of the deficit needs an extra check because
        # sometimes we charge storage direct from DC-coupled RE even when there
        # is a positive deficit
        if deficit <= 0.0:
            system_level[hr, 2] = -deficit
            continue

        # discharge batteries, the amount is the lesser of state of charge,
        # deficit, and the max MW of the battery, we have to either charge
        # or discharge every hour whether there is a deficit or not to propagate
        # state of charge forward
        for es_i in range(storage.shape[2]):
            # skip the `es_i` storage resource if it is not yet in operation or
            # there was excess generation from a DC-coupled RE facility
            if storage_op_hour[es_i] > hr or storage_dc_charge[hr, es_i] > 0.0:
                continue
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
        for r in start_ranks[:, yr]:
            # we are only dealing with plants not already operating here
            if op_data[r, 1]:
                continue
            ramp = dispatchable_ramp_mw[r]

            # a fossil plant's output during an hour is the lesser of the deficit,
            # the plant's historical output, and the plant's re-dispatch output
            # in the previous hour + the plant's one hour max ramp
            r_out = min(
                deficit, historical_dispatch[hr, r], redispatch[hr - 1, r] + ramp
            )
            if r_out > 0.0:
                op_data[r, 0] = 1
                starts[r, yr] = starts[r, yr] + 1
                redispatch[hr, r] = r_out
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
        redispatch <= historical_dispatch * (1 + 1e-4)
    ), "redispatch exceeded historical dispatch in at least 1 hour"

    if np.all(storage_dc_charge == 0.0):
        assert np.all(
            storage[:, 0, :] == storage[:, 3, :]
        ), "charge != gridcharge when storage_dc_charge is all 0.0"
    else:
        assert np.all(
            storage[:, 0, :] >= storage[:, 3, :]
        ), "gridcharge exceeded charge for at least one storage facility/hour"

    # for es_i in range(storage.shape[2]):
    #     if np.any(
    #         storage[storage[:, 1, es_i] > np.roll(storage[:, 2, es_i], 1)]
    #     ):
    #         raise AssertionError(f"discharge exceeded previous state of charge in at least 1 hour for {es_i}")

    return redispatch, storage, system_level, starts


@njit
def make_rank_arrays(
    marginal_cost: np.ndarray, startup_cost: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Turn cost arrays into rank arrays.

    Args:
        marginal_cost: array of marginal costs
        startup_cost: array of startup cost

    Returns:
        marginal_ranks
        start_ranks
    """
    # create an array to determine the marginal cost dispatch order for each year
    # the values in each column represent the canonical indexes for each resource
    # and they are in the order of increasing marginal cost for that year (column)
    marginal_ranks: np.ndarray = np.hstack(
        (
            np.arange(marginal_cost.shape[0]).reshape(-1, 1),
            marginal_cost,
        )
    )
    for i in range(1, 1 + marginal_cost[0, :].shape[0]):
        marginal_ranks[:, i] = marginal_ranks[marginal_ranks[:, i].argsort()][:, 0]
    marginal_ranks = marginal_ranks[:, 1:].astype(np.int64)
    # create an array to determine the startup cost order for each year
    # the values in each column represent the canonical indexes for each resource
    # and they are in the order of increasing startup cost for that year (column)
    start_ranks: np.ndarray = np.hstack(
        (
            np.arange(startup_cost.shape[0]).reshape(-1, 1),
            startup_cost,
        )
    )
    for i in range(1, 1 + startup_cost[0, :].shape[0]):
        start_ranks[:, i] = start_ranks[start_ranks[:, i].argsort()][:, 0]
    start_ranks = start_ranks[:, 1:].astype(np.int64)
    return marginal_ranks, start_ranks


@njit
def charge_storage(
    deficit: float,
    soc: float,
    dc_charge: float,
    mw: float,
    soc_max: float,
    eff: float,
) -> tuple[float, float, float, float]:
    """Calculations for charging storage.

    Args:
        deficit: amount of charging possible from the grid
        soc: state of charge before charging
        dc_charge: power available from DC-coupled RE
        mw: storage power capacity
        soc_max: storage energy capacity
        eff: round-trip efficiency of storage

    Returns:
        charge: total charge in the hour
        discharge: always 0.0, a placeholder
        soc: state of charge after charging
        grid_charge: portion of ``charge`` that came from the grid

    """
    # because we can now end up in this loop when deficit is positive,
    # we need to prevent that positive deficit from mucking up our
    # calculations
    _grid_charge = -deficit if deficit < 0.0 else 0.0
    charge = min(
        # _grid_charge represents grid charging,
        # dc_charge represents charging from a DC-coupled  RE facility
        _grid_charge + dc_charge,
        mw,
        # calculate the amount of charging, to account for battery capacity, we
        # make sure that `charge` would not put `soc` over `soc_max`
        (soc_max - soc) / eff,
    )
    # we charge from DC-coupled RE before charging from the grid
    grid_charge = max(0.0, charge - dc_charge)
    # calculate new `soc` and check that it makes sense
    soc = soc + charge * eff
    assert soc <= soc_max
    return charge, 0.0, soc, grid_charge


@njit
def _validate_inputs(
    net_load,
    hr_to_cost_idx,
    historical_dispatch,
    ramp_mw,
    startup_cost,
    marginal_cost,
    storage_mw,
    storage_hrs,
    storage_eff,
    storage_dc_charge,
):
    if not (
        len(storage_mw)
        == len(storage_hrs)
        == len(storage_eff)
        == storage_dc_charge.shape[1]
    ):
        raise AssertionError("storage data does not match")
    if not (
        ramp_mw.shape[0]
        == startup_cost.shape[0]
        == marginal_cost.shape[0]
        == historical_dispatch.shape[1]
    ):
        raise AssertionError("shapes of dispatchable plant data do not match")
    if not (
        len(net_load)
        == len(hr_to_cost_idx)
        == len(historical_dispatch)
        == len(storage_dc_charge)
    ):
        raise AssertionError("profile lengths do not match")
    if not (
        len(np.unique(hr_to_cost_idx))
        == marginal_cost.shape[1]
        == startup_cost.shape[1]
    ):
        raise AssertionError(
            "# of unique values in `hr_to_cost_idx` does not match # of columns "
            "in `dispatchable_marginal_cost` and `dispatchable_startup_cost`"
        )


dispatch_engine_compiled = njit(dispatch_engine, error_model="numpy")
