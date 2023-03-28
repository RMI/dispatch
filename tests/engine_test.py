"""Dispatch engine tests."""


import numpy as np
import pytest
from dispatch.engine import (
    calculate_generator_output,
    charge_storage,
    dispatch_engine,
    make_rank_arrays,
    validate_inputs,
)
from etoolbox.utils.testing import idfn

NL = [
    -500.0,
    -250.0,
    100.0,
    400.0,
    800.0,
    1000.0,
    1200.0,
    800.0,
    600.0,
    -500.0,
    -250.0,
    100.0,
    400.0,
    800.0,
    1000.0,
    1200.0,
    800.0,
    600.0,
]
CAP = [500, 400, 300]


def test_engine():
    """Trivial test for the dispatch engine."""
    redispatch, es, sl, st = dispatch_engine.py_func(
        net_load=np.array(NL),
        hr_to_cost_idx=np.zeros(len(NL), dtype=int),
        historical_dispatch=np.array([CAP] * len(NL)),
        dispatchable_ramp_mw=np.array(CAP),
        dispatchable_startup_cost=np.array([[1000.0], [1000.0], [100.0]]),
        dispatchable_marginal_cost=np.array([[10.0], [20.0], [50.0]]),
        storage_mw=np.array([400, 200]),
        storage_hrs=np.array([4, 12]),
        storage_eff=np.array((0.9, 0.9)),
        storage_op_hour=np.array((0, 0)),
        storage_dc_charge=np.zeros((len(NL), 2)),
    )
    assert np.all(redispatch.sum(axis=1) + es[:, 1, :].sum(axis=1) >= NL)


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"storage_mw": np.array([400, 200, 100])}, AssertionError),
        ({"storage_dc_charge": np.zeros((len(NL), 5))}, AssertionError),
        ({"startup_cost": np.array([[1000.0], [1000.0]])}, AssertionError),
        ({"net_load": np.array(NL)[1:]}, AssertionError),
        (
            {"hr_to_cost_idx": np.append(np.zeros(len(NL) - 1, dtype=int), 1)},
            AssertionError,
        ),
    ],
    ids=idfn,
)
def test_validate_inputs(override, expected):
    """Test input validator."""
    base = dict(  # noqa: C408
        net_load=np.array(NL),
        hr_to_cost_idx=np.zeros(len(NL), dtype=int),
        historical_dispatch=np.array([CAP] * len(NL)),
        ramp_mw=np.array(CAP),
        startup_cost=np.array([[1000.0], [1000.0], [100.0]]),
        marginal_cost=np.array([[10.0], [20.0], [50.0]]),
        storage_mw=np.array([400, 200]),
        storage_hrs=np.array([4, 12]),
        storage_eff=np.array((0.9, 0.9)),
        storage_dc_charge=np.zeros((len(NL), 2)),
    )
    validate_inputs.py_func(**base)
    over = base | override
    with pytest.raises(expected):
        validate_inputs.py_func(**over)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {
                "deficit": 0,
                "state_of_charge": 0,
                "dc_charge": 0,
                "mw": 10,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (0, 0, 0, 0),
        ),
        (
            {
                "deficit": -5,
                "state_of_charge": 0,
                "dc_charge": 0,
                "mw": 10,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (5, 0, 5, 5),
        ),
        (
            {
                "deficit": 0,
                "state_of_charge": 0,
                "dc_charge": 5,
                "mw": 10,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (5, 0, 5, 0),
        ),
        (
            {
                "deficit": 500,
                "state_of_charge": 0,
                "dc_charge": 5,
                "mw": 10,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (5, 0, 5, 0),
        ),
        (
            {
                "deficit": -5,
                "state_of_charge": 0,
                "dc_charge": 5,
                "mw": 5,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (5, 0, 5, 0),
        ),
        (
            {
                "deficit": -5,
                "state_of_charge": 0,
                "dc_charge": 3,
                "mw": 5,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (5, 0, 5, 2),
        ),
        (
            {
                "deficit": -5,
                "state_of_charge": 0,
                "dc_charge": 5,
                "mw": 10,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (10, 0, 10, 5),
        ),
        (
            {
                "deficit": -5,
                "state_of_charge": 5,
                "dc_charge": 5,
                "mw": 10,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (10, 0, 15, 5),
        ),
        (
            {
                "deficit": -5,
                "state_of_charge": 5,
                "dc_charge": 5,
                "mw": 10,
                "max_state_of_charge": 20,
                "eff": 0.5,
            },
            (10, 0, 10, 5),
        ),
        (
            {
                "deficit": -5,
                "state_of_charge": 15,
                "dc_charge": 5,
                "mw": 10,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (5, 0, 20, 0),
        ),
        (
            {
                "deficit": -2,
                "state_of_charge": 15,
                "dc_charge": 5,
                "mw": 10,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (5, 0, 20, 0),
        ),
        (
            {
                "deficit": -5,
                "state_of_charge": 0,
                "dc_charge": 0,
                "mw": 1,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (1, 0, 1, 1),
        ),
        (
            {
                "deficit": 0,
                "state_of_charge": 0,
                "dc_charge": 5,
                "mw": 1,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (1, 0, 1, 0),
        ),
        (
            {
                "deficit": 500,
                "state_of_charge": 0,
                "dc_charge": 5,
                "mw": 1,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (1, 0, 1, 0),
        ),
        (
            {
                "deficit": -5,
                "state_of_charge": 0,
                "dc_charge": 5,
                "mw": 1,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (1, 0, 1, 0),
        ),
        (
            {
                "deficit": -5,
                "state_of_charge": 5,
                "dc_charge": 5,
                "mw": 1,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (1, 0, 6, 0),
        ),
        (
            {
                "deficit": -5,
                "state_of_charge": 5,
                "dc_charge": 5,
                "mw": 1,
                "max_state_of_charge": 20,
                "eff": 0.9,
            },
            (1, 0, 5.9, 0),
        ),
        (
            {
                "deficit": -2,
                "state_of_charge": 15,
                "dc_charge": 5,
                "mw": 1,
                "max_state_of_charge": 20,
                "eff": 1.0,
            },
            (1, 0, 16, 0),
        ),
    ],
    ids=idfn,
)
def test_charge_storage(kwargs, expected):
    """Test storage charging calculations."""
    assert charge_storage.py_func(**kwargs) == expected


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"desired_mw": 5, "max_mw": 2, "previous_mw": 0, "ramp_mw": 1}, 1),
        ({"desired_mw": 5, "max_mw": 10, "previous_mw": 10, "ramp_mw": 1}, 9),
        ({"desired_mw": 5, "max_mw": 10, "previous_mw": 5, "ramp_mw": 1}, 5),
        ({"desired_mw": 5, "max_mw": 4, "previous_mw": 5, "ramp_mw": 1}, 4),
        ({"desired_mw": 5, "max_mw": 4, "previous_mw": 2, "ramp_mw": 1}, 3),
        ({"desired_mw": 5, "max_mw": 0, "previous_mw": 5, "ramp_mw": 1}, 0),
    ],
    ids=idfn,
)
def test_dispatch_generator(kwargs, expected):
    """Test the logic of the calculate_generator_output function."""
    assert calculate_generator_output.py_func(**kwargs) == expected


def test_make_rank_arrays():
    """Test cost rank setup."""
    m_cost = np.array([[50.0, 50.0], [25.0, 75.0]])
    s_cost = np.array([[250.0, 250.0], [500.0, 500.0]])
    m, s = make_rank_arrays.py_func(m_cost, s_cost)
    assert np.all(m == np.array([[1, 0], [0, 1]]))
    assert np.all(s == np.array([[0, 0], [1, 1]]))
