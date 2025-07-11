"""Dispatch engine."""

from __future__ import annotations

import numpy as np
from numba import njit

CHARGE_IDX = 0
DISCHARGE_IDX = 1
SOC_IDX = 2
GRIDCHARGE_IDX = 3

DEFICIT_IDX = 0
DIRTY_IDX = 1
CURTAILMENT_IDX = 2
ADJUSTMENT_IDX = 3


@njit(error_model="numpy", cache=True)
def dispatch_engine_auto(
    net_load: np.ndarray,
    hr_to_cost_idx: np.ndarray,
    historical_dispatch: np.ndarray,
    dispatchable_ramp_mw: np.ndarray,
    dispatchable_startup_cost: np.ndarray,
    dispatchable_marginal_cost: np.ndarray,
    dispatchable_min_uptime: np.ndarray,
    storage_mw: np.ndarray,
    storage_charge_mw: np.ndarray,
    storage_hrs: np.ndarray,
    storage_charge_eff: np.ndarray,
    storage_discharge_eff: np.ndarray,
    storage_op_hour: np.ndarray,
    storage_dc_charge: np.ndarray,
    storage_reserve: np.ndarray,
    dynamic_reserve_coeff: float,
    marginal_for_startup_rank: bool = False,  # noqa: FBT001, FBT002
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Autotune ``dynamic_reserve_coeff`` in :func:`.dispatch_engine`.

    Args:
        net_load: net load, as in net of RE generation, negative net load means
            excess renewables
        hr_to_cost_idx: an array that contains for each hour, the index of the correct
            column in ``dispatchable_marginal_cost`` that contains cost data for that
            hour
        historical_dispatch: historic generator dispatch, acts as an hourly upper
            constraint on this dispatch
        dispatchable_ramp_mw: max one hour ramp in MW
        dispatchable_startup_cost: startup cost in $ for each dispatchable generator
        dispatchable_marginal_cost: annual marginal cost for each dispatchable
            generator in $/MWh rows are generators and columns are years
        dispatchable_min_uptime: minimum hrs a generator must operate before its
            output can be reduced
        storage_mw: max discharge rate for storage in MW
        storage_charge_mw: max charge rate for storage in MW
        storage_hrs: duration of storage
        storage_charge_eff: storage charge efficiency
        storage_discharge_eff: storage discharge efficiency
        storage_op_hour: first hour in which storage is available, i.e. the index of
            the operating date
        storage_dc_charge: an array whose columns match each storage facility, and a
            row for each hour representing how much energy would be curtailed at an
            attached renewable facility because it exceeds the system's inverter. This
            then represents how much the storage in an RE+Storage facility could be
            charged by otherwise curtailed energy when ilr>1.
        storage_reserve: portion of a storage facility's SOC that will be held in
            reserve until after dispatchable resource startup.
        dynamic_reserve_coeff: coefficient passed to
            :func:`dispatch.engine.dynamic_reserve` to use in exponential function.
            ``-10.0`` means autotune this parameter.
        marginal_for_startup_rank: if True, rather than using
            ``dispatchable_startup_cost`` to order generators for startup, use marginal
            cost rank


    Returns:
        -   **redispatch** (:class:`numpy.ndarray`) - new hourly dispatch
        -   **storage** (:class:`numpy.ndarray`) - hourly charge, discharge, and state
            of charge data
        -   **system_level** (:class:`numpy.ndarray`) - hourly deficit, dirty charge,
            and total curtailment dat
        -   **starts** (:class:`numpy.ndarray`) - count of starts for each generator in
            each year
    """
    validate_inputs(
        net_load,
        hr_to_cost_idx,
        historical_dispatch,
        dispatchable_ramp_mw,
        dispatchable_startup_cost,
        dispatchable_marginal_cost,
        storage_mw,
        storage_charge_mw,
        storage_hrs,
        storage_charge_eff,
        storage_discharge_eff,
        storage_dc_charge,
        storage_reserve,
    )
    reserve_is_dynamic = dynamic_reserve_coeff == -10.0
    have_dynamic_reserve_storage = np.sum(storage_mw[storage_reserve == 0.0]) > 0.0
    if reserve_is_dynamic and have_dynamic_reserve_storage:
        coeffs = np.arange(0.0, 1.5, 0.25)
        results = []
        comparison = np.zeros((len(coeffs), 4))
        for ix, coeff in enumerate(coeffs):
            redispatch, storage, system_level, starts = dispatch_engine(
                net_load=net_load,
                hr_to_cost_idx=hr_to_cost_idx,
                historical_dispatch=historical_dispatch,
                dispatchable_ramp_mw=dispatchable_ramp_mw,
                dispatchable_startup_cost=dispatchable_startup_cost,
                dispatchable_marginal_cost=dispatchable_marginal_cost,
                dispatchable_min_uptime=dispatchable_min_uptime,
                storage_mw=storage_mw,
                storage_charge_mw=storage_charge_mw,
                storage_hrs=storage_hrs,
                storage_charge_eff=storage_charge_eff,
                storage_discharge_eff=storage_discharge_eff,
                storage_op_hour=storage_op_hour,
                storage_dc_charge=storage_dc_charge,
                storage_reserve=storage_reserve,
                dynamic_reserve_coeff=coeff,
                marginal_for_startup_rank=marginal_for_startup_rank,
            )
            comparison[ix, :] = np.sum(system_level, axis=0) / np.sum(net_load)
            results.append((redispatch, storage, system_level, starts))
        return results[choose_best_coefficient(comparison)]

    return dispatch_engine(
        net_load=net_load,
        hr_to_cost_idx=hr_to_cost_idx,
        historical_dispatch=historical_dispatch,
        dispatchable_ramp_mw=dispatchable_ramp_mw,
        dispatchable_startup_cost=dispatchable_startup_cost,
        dispatchable_marginal_cost=dispatchable_marginal_cost,
        dispatchable_min_uptime=dispatchable_min_uptime,
        storage_mw=storage_mw,
        storage_charge_mw=storage_charge_mw,
        storage_hrs=storage_hrs,
        storage_charge_eff=storage_charge_eff,
        storage_discharge_eff=storage_discharge_eff,
        storage_op_hour=storage_op_hour,
        storage_dc_charge=storage_dc_charge,
        storage_reserve=storage_reserve,
        dynamic_reserve_coeff=dynamic_reserve_coeff,
        marginal_for_startup_rank=marginal_for_startup_rank,
    )


@njit(error_model="numpy", cache=True)
def choose_best_coefficient(comparison: np.ndarray) -> int:
    """Select the optimal combination of deficit and curtailment.

    Args:
        comparison:

    Returns: index of the optimal case
    """
    mins = np.argmin(comparison, axis=0)
    if mins[DEFICIT_IDX] == mins[CURTAILMENT_IDX]:
        return mins[DEFICIT_IDX]
    comparison = np.column_stack((comparison, np.arange(0, len(comparison))))
    ok_deficit = comparison[comparison[:, DEFICIT_IDX] <= 1e-4]
    if len(ok_deficit) > 0:
        return int(ok_deficit[np.argmin(ok_deficit[:, CURTAILMENT_IDX], axis=0), 4])
    return int(np.argmin(comparison[:, DEFICIT_IDX]))


@njit(error_model="numpy", cache=True)
def dispatch_engine(  # noqa: C901
    net_load: np.ndarray,
    hr_to_cost_idx: np.ndarray,
    historical_dispatch: np.ndarray,
    dispatchable_ramp_mw: np.ndarray,
    dispatchable_startup_cost: np.ndarray,
    dispatchable_marginal_cost: np.ndarray,
    dispatchable_min_uptime: np.ndarray,
    storage_mw: np.ndarray,
    storage_charge_mw: np.ndarray,
    storage_hrs: np.ndarray,
    storage_charge_eff: np.ndarray,
    storage_discharge_eff: np.ndarray,
    storage_op_hour: np.ndarray,
    storage_dc_charge: np.ndarray,
    storage_reserve: np.ndarray,
    dynamic_reserve_coeff: float,
    marginal_for_startup_rank: bool,  # noqa: FBT001
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch engine that can be compiled with :func:`numba.jit`.

    For each hour...

    1.  first iterate through operating generators
    2.  then charge/discharge storage
    3.  if there is still a deficit, iterate through non-operating generators
        and turn them on if required

    Args:
        net_load: net load, as in net of RE generation, negative net load means
            excess renewables
        hr_to_cost_idx: an array that contains for each hour, the index of the correct
            column in ``dispatchable_marginal_cost`` that contains cost data for that
            hour
        historical_dispatch: historic generator dispatch, acts as an hourly upper
            constraint on this dispatch
        dispatchable_ramp_mw: max one hour ramp in MW
        dispatchable_startup_cost: startup cost in $ for each dispatchable generator
        dispatchable_marginal_cost: annual marginal cost for each dispatchable
            generator in $/MWh rows are generators and columns are years
        dispatchable_min_uptime: minimum hrs a generator must operate before its
            output can be reduced
        storage_mw: max discharge rate for storage in MW
        storage_charge_mw: max charge rate for storage in MW
        storage_hrs: duration of storage
        storage_charge_eff: storage charge efficiency
        storage_discharge_eff: storage discharge efficiency
        storage_op_hour: first hour in which storage is available, i.e. the index of
            the operating date
        storage_dc_charge: an array whose columns match each storage facility, and a
            row for each hour representing how much energy would be curtailed at an
            attached renewable facility because it exceeds the system's inverter. This
            then represents how much the storage in an RE+Storage facility could be
            charged by otherwise curtailed energy when ilr>1.
        storage_reserve: portion of a storage facility's SOC that will be held in
            reserve until after dispatchable resource startup.
        dynamic_reserve_coeff: coefficient passed to
            :func:`dispatch.engine.dynamic_reserve` to use in exponential function.
        marginal_for_startup_rank: if True, rather than using
            ``dispatchable_startup_cost`` to order generators for startup, use marginal
            cost rank


    Returns:
        -   **redispatch** (:class:`numpy.ndarray`) - new hourly dispatch
        -   **storage** (:class:`numpy.ndarray`) - hourly charge, discharge, and state
            of charge data
        -   **system_level** (:class:`numpy.ndarray`) - hourly deficit, dirty charge,
            and total curtailment data
        -   **starts** (:class:`numpy.ndarray`) - count of starts for each generator in
            each year
    """
    storage_soc_max: np.ndarray = storage_mw * storage_hrs

    marginal_ranks, start_ranks = make_rank_arrays(
        dispatchable_marginal_cost, dispatchable_startup_cost
    )
    if marginal_for_startup_rank:
        start_ranks = marginal_ranks

    # create an array to keep track of re-dispatch
    redispatch: np.ndarray = np.zeros_like(historical_dispatch)
    # array to keep track of starts by year
    starts: np.ndarray = np.zeros_like(marginal_ranks)
    # array to keep track of hourly storage data. rows are hours, columns
    # (0) charge, (1) discharge, (2) state of charge, (3) grid charge
    storage: np.ndarray = np.zeros((len(net_load), 4, len(storage_mw)))
    # array to keep track of system level data (0) deficits, (1) dirty charge,
    # (2) curtailment, (3) adjustment to deficit
    system_level: np.ndarray = np.zeros((len(net_load), 4))
    # internal dispatch data we need to track; (0) current run operating_hours
    # (1) whether we touched the generator in the first round of dispatch
    operating_data: np.ndarray = np.zeros(
        (dispatchable_marginal_cost.shape[0], 2), dtype=np.int64
    )

    # to avoid having to do the first hour differently, we just assume original
    # dispatch in that hour and then skip it
    redispatch[0, :] = historical_dispatch[0, :]
    # need to set operating_hours to 1 for generators that we are starting
    # off as operating
    operating_data[:, 0] = np.where(historical_dispatch[0, :] > 0.0, 1, 0)

    # the big loop where we iterate through all the hours
    for hr, (deficit, yr) in enumerate(zip(net_load, hr_to_cost_idx)):
        # storage reserves are dynamic so we need to determine the current hour's
        # reserve to use throughout
        storage_reserve_ = dynamic_reserve(
            hr=hr,
            reserve=storage_reserve,
            net_load=net_load,
            coeff=dynamic_reserve_coeff,
        )

        # because of the look-backs, the first hour has to be done differently
        # here we just skip it because its only one hour and we assume
        # historical fossil dispatch
        if hr == 0:
            # if there is excess RE we charge the battery in the first hour
            if deficit < 0.0 or np.any(storage_dc_charge[hr, :] > 0.0):
                for storage_idx in range(storage.shape[2]):
                    # skip the `es_i` storage resource if it is not yet in operation
                    if storage_op_hour[storage_idx] > hr:
                        continue

                    storage[hr, :, storage_idx] = charge_storage(
                        deficit=deficit,
                        # previous state_of_charge
                        state_of_charge=storage_soc_max[storage_idx]
                        * storage_reserve_[storage_idx],
                        dc_charge=storage_dc_charge[hr, storage_idx],
                        mw=storage_charge_mw[storage_idx],
                        max_state_of_charge=storage_soc_max[storage_idx],
                        eff=storage_charge_eff[storage_idx],
                    )

                    # update the deficit if we are grid charging
                    if deficit < 0.0:
                        deficit += storage[hr, GRIDCHARGE_IDX, storage_idx]
            continue

        deficit_adjustment = adjust_for_storage_reserve(
            state_of_charge=storage[hr - 1, SOC_IDX, :],
            mw=storage_mw,
            reserve=storage_reserve_,
            # only want to adjust the provisional deficit for storage that is
            # operating
            max_state_of_charge=storage_soc_max * (hr >= storage_op_hour),
        )
        system_level[hr, ADJUSTMENT_IDX] = deficit_adjustment
        provisional_deficit = max(
            0.0,
            deficit + deficit_adjustment,
        )

        # new hour so reset where we keep track if we've touched a generator for
        # this hour
        operating_data[:, 1] = 0

        # read whole numpy rows because numpy uses row-major ordering, this is
        # potentially bad or useless intuition-based optimization
        historical_hr_dispatch: np.ndarray = historical_dispatch[hr, :]
        previous_hr_redispatch: np.ndarray = redispatch[hr - 1, :]

        # dispatch generators in the order of their marginal cost for year yr
        for generator_idx in marginal_ranks[:, yr]:
            operating_hours = operating_data[generator_idx, 0]
            # we are only dealing with generators already operating here
            if operating_hours == 0:
                continue
            generator_output = calculate_generator_output(
                desired_mw=provisional_deficit,
                max_mw=historical_hr_dispatch[generator_idx],
                previous_mw=previous_hr_redispatch[generator_idx],
                ramp_mw=dispatchable_ramp_mw[generator_idx],
                current_uptime=operating_hours,
                min_uptime=dispatchable_min_uptime[generator_idx],
            )
            redispatch[hr, generator_idx] = generator_output
            # update operating hours and mark that we took care of this generator
            operating_data[generator_idx, :] = (
                (operating_hours + 1 if generator_output > 0.0 else 0),
                1,
            )
            # keep a running total of remaining deficit, having this value be negative
            # just makes the loop code more complicated, if it actually should be
            # negative we capture that when we calculate the true deficit based
            # on redispatch below
            provisional_deficit = max(0, provisional_deficit - generator_output)

        # calculate the true deficit as the hour's net load less actual dispatch
        # of fossil generators in hr that were also operating in hr - 1
        deficit -= np.sum(redispatch[hr, :])

        # negative deficit means excess generation, so we charge the battery
        # if there is DC-coupled storage, we also charge that storage if there is
        # RE generation that would otherwise be curtailed
        if deficit < 0.0 or np.any(storage_dc_charge[hr, :] > 0.0):
            for storage_idx in range(storage.shape[2]):
                # skip the `es_i` storage resource if it is not yet in operation
                if storage_op_hour[storage_idx] > hr:
                    continue
                storage[hr, :, storage_idx] = charge_storage(
                    deficit=deficit,
                    state_of_charge=storage[
                        hr - 1, SOC_IDX, storage_idx
                    ],  # previous state_of_charge
                    dc_charge=storage_dc_charge[hr, storage_idx],
                    mw=storage_charge_mw[storage_idx],
                    max_state_of_charge=storage_soc_max[storage_idx],
                    eff=storage_charge_eff[storage_idx],
                )
                # alias grid_charge
                grid_charge = storage[hr, GRIDCHARGE_IDX, storage_idx]
                # calculate the amount of charging that was dirty
                # TODO check that this calculation is actually correct
                system_level[hr, DIRTY_IDX] += min(
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
            # store excess generation (-deficit) as curtailment
            system_level[hr, CURTAILMENT_IDX] = -deficit
            continue

        # discharge batteries, the amount is the lesser of state of charge,
        # deficit, and the max MW of the battery, we have to either charge
        # or discharge every hour whether there is a deficit or not to propagate
        # state of charge forward
        for storage_idx in range(storage.shape[2]):
            # skip the `storage_idx` storage resource if it is not yet in operation or
            # there was excess generation from a DC-coupled RE facility
            if (
                storage_op_hour[storage_idx] > hr
                or storage_dc_charge[hr, storage_idx] > 0.0
            ):
                # if we got here because of DC-coupled charging, we still need to
                # propagate forward state_of_charge
                storage[hr, SOC_IDX, storage_idx] = storage[
                    hr - 1, SOC_IDX, storage_idx
                ]
                continue
            discharge = discharge_storage(
                desired_mw=deficit,
                state_of_charge=storage[hr - 1, SOC_IDX, storage_idx],
                mw=storage_mw[storage_idx],
                max_state_of_charge=storage_soc_max[storage_idx],
                reserve=storage_reserve_[storage_idx],
                eff=storage_discharge_eff[storage_idx],
            )
            storage[hr, DISCHARGE_IDX, storage_idx] = discharge
            # with discharge efficiency taken into account, we can encounter floating
            # point errors that cause discharge to be a tiny bit larger than soc
            storage[hr, SOC_IDX, storage_idx] = max(
                0.0,
                storage[hr - 1, SOC_IDX, storage_idx]
                - discharge / storage_discharge_eff[storage_idx],
            )
            deficit -= discharge

        # once we've dealt with operating generators and storage, if there is no
        # positive deficit we can skip startups and go on to the next hour
        if deficit == 0.0:
            continue

        # we also need to make sure that the deficit is not negative, if it is,
        # something has gone wrong
        assert deficit >= 0.0, "negative deficit after discharge, this shouldn't happen"

        # TODO check that this start_ranks ordering system is working properly
        for generator_idx in start_ranks[:, yr]:
            # we are only dealing with generators not already operating here
            if operating_data[generator_idx, 1]:
                continue
            ramp_mw = dispatchable_ramp_mw[generator_idx]

            # a fossil generator's output during an hour is the lesser of the deficit,
            # the generator's historical output, and the generator's re-dispatch output
            # in the previous hour + the generator's one hour max ramp
            assert previous_hr_redispatch[generator_idx] == 0, ""
            generator_output = min(
                deficit, historical_hr_dispatch[generator_idx], ramp_mw
            )
            if generator_output > 0.0:
                operating_data[generator_idx, 0] = 1
                starts[generator_idx, yr] = starts[generator_idx, yr] + 1
                redispatch[hr, generator_idx] = generator_output
                deficit -= generator_output

                if deficit == 0.0:
                    break

        if deficit == 0.0:
            continue

        # discharge batteries again but with no reserve
        for storage_idx in range(storage.shape[2]):
            # skip the `storage_idx` storage resource if it is not yet in operation
            if storage_op_hour[storage_idx] > hr:
                continue
            discharge = discharge_storage(
                desired_mw=deficit,
                # may have already used the battery this hour so use the current hour
                # state of charge
                state_of_charge=storage[hr, SOC_IDX, storage_idx],
                # already used some of the battery's MW this hour
                mw=storage_mw[storage_idx] - storage[hr, DISCHARGE_IDX, storage_idx],
                max_state_of_charge=storage_soc_max[storage_idx],
                reserve=0.0,
                eff=storage_discharge_eff[storage_idx],
            )
            storage[hr, DISCHARGE_IDX, storage_idx] += discharge
            # with discharge efficiency taken into account, we can encounter floating
            # point errors that cause discharge to be a tiny bit larger than soc
            storage[hr, SOC_IDX, storage_idx] = max(
                0.0,
                storage[hr, SOC_IDX, storage_idx]
                - discharge / storage_discharge_eff[storage_idx],
            )
            deficit -= discharge

            if deficit == 0.0:
                break

        if deficit == 0.0:
            continue

        # if we end up here that means we never got the deficit to zero, we want
        # to keep track of that
        system_level[hr, DEFICIT_IDX] = deficit

    # assert not np.any(
    #     storage[storage[:, 1, 0] > np.roll(storage[:, 2, 0], 1)]
    # ), "discharge exceeded previous state of charge in at least 1 hour for es0"
    # assert not np.any(
    #     storage[storage[:, 1, 1] > np.roll(storage[:, 2, 1], 1)]
    # ), "discharge exceeded previous state of charge in at least 1 hour for es1"
    assert np.all(redispatch <= historical_dispatch * (1 + 1e-4)), (
        "redispatch exceeded historical dispatch in at least 1 hour"
    )

    if np.all(storage_dc_charge == 0.0):
        assert np.all(storage[:, 0, :] == storage[:, 3, :]), (
            "charge != gridcharge when storage_dc_charge is all 0.0"
        )
    else:
        # if not np.all(storage[:, 0, :] >= storage[:, 3, :]):
        #     raise AssertionError(
        #     "gridcharge exceeded charge for at least one storage facility/hour"
        # )
        compare(
            storage[:, 0, :],
            storage[:, 3, :],
            "gridcharge exceeded charge for at least one storage facility/hour",
        )

    for es_i in range(storage.shape[2]):
        if np.any(
            storage[
                storage[:, DISCHARGE_IDX, es_i] > np.roll(storage[:, SOC_IDX, es_i], 1)
            ]
        ):
            print(es_i)
            raise AssertionError(
                "discharge exceeded previous state of charge in at least 1 hour for above storage number"
            )

    return redispatch, storage, system_level, starts


@njit(cache=True)
def compare(a, b, msg):
    """Check that all entries in a are >= to the corresponding value in b."""
    assert a.shape == b.shape
    for a_, b_ in zip(a, b):
        if np.all(np.isclose(a_, b_) | (a_ >= b_)):
            continue
        else:
            raise AssertionError(msg)


@njit(cache=True)
def dynamic_reserve(
    hr: int,
    reserve: np.ndarray,
    net_load: np.ndarray,
    coeff: float,
) -> np.ndarray:
    r"""Adjust storage reserve based on 24hr net load.

    Args:
        hr: hour index
        reserve: storage reserve defaults
        net_load: net load profile
        coeff: coefficient in exponential

    .. math::
       :label: reserve

          ramp &= \frac{max(load_{h+1}, ..., load_{h+24})}{load_h} - 1

          reserve &= 1 - e^{-coeff \times ramp}

    Returns: adjusted reserves
    """
    projected = net_load[hr + 1 : hr + 24]
    if len(projected) == 0:
        return reserve
    if net_load[hr] == 0:
        # equivalent to shifting up the whole profile by the max, muting the %
        # change relative to shifting by the average, median, etc.
        projected_increase = 1.0 if np.max(projected) > 0 else 0
    else:
        projected_increase = max(0.0, np.max(projected) / net_load[hr] - 1)
    return np.where(
        reserve == 0.0, np.around(1 - np.exp(-coeff * projected_increase), 2), reserve
    )


@njit(cache=True)
def adjust_for_storage_reserve(
    state_of_charge: np.ndarray,
    mw: np.ndarray,
    reserve: np.ndarray,
    max_state_of_charge: np.ndarray,
) -> float:
    """Adjustment to deficit to restore or use storage reserve.

    Args:
        state_of_charge: state of charge before charging/discharging
        mw: storage power capacity
        reserve: target filling to this reserve level or allow discharging to
            2x this level
        max_state_of_charge: storage energy capacity

    Returns: amount by which provisional deficit must be adjusted to use or replace
        storage reserve
    """
    # if we are below reserve we actually want to increase the provisional
    # deficit to restore the reserve
    under_reserve = np.minimum(
        state_of_charge - reserve * max_state_of_charge,
        0.0,
    )
    augment_deficit = sum(
        np.where(under_reserve < 0.0, np.maximum(under_reserve, -mw), 0.0)
    )
    if augment_deficit < 0.0:
        return -augment_deficit

    # reserve cannot exceed one
    _reserve = np.minimum(1.0, 2.0 * reserve)
    # but if we don't need to restore the reserve, we want to check if we have
    # excess reserve and adjust down the provisional deficit
    return -sum(
        np.minimum(
            mw,
            # keep SOC above 2x typical reserve
            np.maximum(
                state_of_charge - _reserve * max_state_of_charge,
                0.0,
            ),
        )
    )


@njit(cache=True)
def discharge_storage(
    desired_mw: float,
    state_of_charge: float,
    mw: float,
    max_state_of_charge: float,
    reserve: float = 0.0,
    eff: float = 1.0,
) -> float:
    """Calculations for discharging storage.

    Args:
        desired_mw: amount of power we want from storage
        state_of_charge: state of charge before charging
        mw: storage power capacity
        max_state_of_charge: storage energy capacity
        reserve: prevent discharge below this portion of ``max_state_of_charge``
        eff: discharge efficiency of storage

    Returns: amount of storage discharge
    """
    out = min(
        desired_mw,
        mw,
        # prevent discharge below reserve (or full SOC if reserve is 0.0)
        max(0.0, state_of_charge - max_state_of_charge * reserve) * eff,
    )
    if out / eff > state_of_charge and not np.isclose(out / eff, state_of_charge):
        raise ValueError("discharge / eff > state of charge")
    return out


@njit(cache=True)
def calculate_generator_output(
    desired_mw: float,
    max_mw: float,
    previous_mw: float,
    ramp_mw: float,
    current_uptime: int = 0,
    min_uptime: int = 0,
) -> float:
    """Determine period output for a generator.

    Args:
        desired_mw: desired generator output
        max_mw: maximum output of generator this period
        previous_mw: generator output in the previous period
        ramp_mw: maximum one-period change in generator output in MW, up or down
        current_uptime: as of the end of the previous period, for how many periods
            has the generator been operating
        min_uptime: the minimum duration the generator must operate before its
            output is reduced

    Returns:
        output of the generator for given period
    """
    if current_uptime >= min_uptime:
        return min(
            # we limit output to historical as a transmission
            # and operating constraint proxy
            max_mw,
            # maximum output subject to ramping constraints
            previous_mw + ramp_mw,
            # max of desired output and minimum output subject to ramping constraints
            max(desired_mw, previous_mw - ramp_mw),
        )
    return min(
        # we limit output to historical as a transmission
        # and operating constraint proxy
        max_mw,
        # maximum output subject to ramping constraints
        previous_mw + ramp_mw,
        # max of desired output and previous because we cannot ramp down yet
        max(desired_mw, previous_mw),
    )


@njit(cache=True)
def make_rank_arrays(
    marginal_cost: np.ndarray, startup_cost: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Turn cost arrays into rank arrays.

    Args:
        marginal_cost: array of marginal costs
        startup_cost: array of startup cost

    Returns:
        -   **marginal_ranks** (:class:`numpy.ndarray`) - marginal rank array
        -   **start_ranks** (:class:`numpy.ndarray`) - start rank array
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


@njit(cache=True)
def charge_storage(
    deficit: float,
    state_of_charge: float,
    dc_charge: float,
    mw: float,
    max_state_of_charge: float,
    eff: float,
) -> tuple[float, float, float, float]:
    """Calculations for charging storage.

    Args:
        deficit: amount of charging possible from the grid
        state_of_charge: state of charge before charging
        dc_charge: power available from DC-coupled RE
        mw: storage power capacity
        max_state_of_charge: storage energy capacity
        eff: round-trip efficiency of storage

    Returns:
        A tuple with the same organization of columns of internal ``storage`` in
        :func:`.dispatch_engine`.

        -   **charge** (:class:`float`) - total charge in the hour
        -   **discharge** (:class:`float`) - always 0.0, a placeholder
        -   **soc** (:class:`float`) - tate of charge after charging
        -   **grid_charge** (:class:`float`) -  portion of ``charge`` that came from
            the grid
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
        # make sure that `charge` would not put `state_of_charge` over
        # `max_state_of_charge`
        (max_state_of_charge - state_of_charge) / eff,
    )
    # we charge from DC-coupled RE before charging from the grid
    grid_charge = max(0.0, charge - dc_charge)
    # calculate new `state_of_charge` and check that it makes sense
    state_of_charge = state_of_charge + charge * eff
    if (
        not np.isclose(state_of_charge, max_state_of_charge)
        and state_of_charge > max_state_of_charge
    ):
        raise AssertionError("state_of_charge > max_state_of_charge")
    return charge, 0.0, state_of_charge, grid_charge


@njit(cache=True)
def validate_inputs(
    net_load,
    hr_to_cost_idx,
    historical_dispatch,
    ramp_mw,
    startup_cost,
    marginal_cost,
    storage_mw,
    storage_charge_mw,
    storage_hrs,
    storage_charge_eff,
    storage_discharge_eff,
    storage_dc_charge,
    storage_reserve,
) -> None:
    """Validate shape of inputs."""
    if not (
        len(storage_mw)
        == len(storage_charge_mw)
        == len(storage_hrs)
        == len(storage_charge_eff)
        == len(storage_discharge_eff)
        == len(storage_reserve)
        == storage_dc_charge.shape[1]
    ):
        raise AssertionError("storage data does not match")
    if not (
        ramp_mw.shape[0]
        == startup_cost.shape[0]
        == marginal_cost.shape[0]
        == historical_dispatch.shape[1]
    ):
        raise AssertionError("shapes of dispatchable generator data do not match")
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
