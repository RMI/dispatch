"""Dispatch engine tests."""


import numpy as np
import pytest
from dispatch.engine import (
    adjust_for_storage_reserve,
    calculate_generator_output,
    charge_storage,
    choose_best_coefficient,
    discharge_storage,
    dispatch_engine,
    dispatch_engine_auto,
    dynamic_reserve,
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


@pytest.mark.parametrize("py_func", [True, False], ids=idfn)
@pytest.mark.parametrize("marginal_for_startup_rank", [True, False], ids=idfn)
def test_engine(py_func, marginal_for_startup_rank):
    """Trivial test for the dispatch engine."""
    in_dict = dict(  # noqa: C408
        net_load=np.array(NL),
        hr_to_cost_idx=np.zeros(len(NL), dtype=int),
        historical_dispatch=np.array([CAP] * len(NL)),
        dispatchable_ramp_mw=np.array(CAP),
        dispatchable_startup_cost=np.array([[1000.0], [1000.0], [100.0]]),
        dispatchable_marginal_cost=np.array([[10.0], [20.0], [50.0]]),
        dispatchable_min_uptime=np.zeros_like(CAP, dtype=np.int_),
        storage_mw=np.array([400, 200]),
        storage_hrs=np.array([4, 12]),
        storage_eff=np.array((0.9, 0.9)),
        storage_op_hour=np.array((0, 0)),
        storage_dc_charge=np.zeros((len(NL), 2)),
        storage_reserve=np.array([0.1, 0.1]),
        dynamic_reserve_coeff=1.5,
        marginal_for_startup_rank=marginal_for_startup_rank,
    )
    if py_func:
        redispatch, es, sl, st = dispatch_engine.py_func(**in_dict)
    else:
        redispatch, es, sl, st = dispatch_engine(**in_dict)
    assert np.all(redispatch.sum(axis=1) + es[:, 1, :].sum(axis=1) >= NL)


@pytest.mark.parametrize("marginal_for_startup_rank", [True, False], ids=idfn)
@pytest.mark.parametrize("py_func", [True, False], ids=idfn)
@pytest.mark.parametrize(
    ("coeff", "reserve"),
    [
        ("auto", 0.0),
        ("auto", 0.1),
        (0.5, 0.0),
        (0.5, 0.1),
        (0.0, 0.0),
        (0.0, 0.1),
    ],
    ids=idfn,
)
def test_dispatch_engine_auto(py_func, coeff, reserve, marginal_for_startup_rank):
    """Trivial test for the dispatch engine."""
    in_dict = dict(  # noqa: C408
        net_load=np.array(NL),
        hr_to_cost_idx=np.zeros(len(NL), dtype=int),
        historical_dispatch=np.array([CAP] * len(NL), dtype=float),
        dispatchable_ramp_mw=np.array(CAP, dtype=float),
        dispatchable_startup_cost=np.array([[1000.0], [1000.0], [100.0]], dtype=float),
        dispatchable_marginal_cost=np.array([[10.0], [20.0], [50.0]], dtype=float),
        dispatchable_min_uptime=np.zeros_like(CAP, dtype=np.int_),
        storage_mw=np.array([400, 200], dtype=float),
        storage_hrs=np.array([4, 12], dtype=float),
        storage_eff=np.array((0.9, 0.9), dtype=float),
        storage_op_hour=np.array((0, 0)),
        storage_dc_charge=np.zeros((len(NL), 2), dtype=float),
        storage_reserve=np.array([reserve, reserve], dtype=float),
        dynamic_reserve_coeff=-10.0 if coeff == "auto" else coeff,
        marginal_for_startup_rank=marginal_for_startup_rank,
    )
    if py_func:
        redispatch, es, sl, st = dispatch_engine_auto.py_func(**in_dict)
    else:
        redispatch, es, sl, st = dispatch_engine_auto(**in_dict)
    assert np.all(redispatch.sum(axis=1) + es[:, 1, :].sum(axis=1) >= NL)


def test_engine_pyfunc_numba():
    """Trivial numba compilation does not change results."""
    redispatch_py, es_py, sl_py, st_py = dispatch_engine.py_func(
        net_load=np.array(NL),
        hr_to_cost_idx=np.zeros(len(NL), dtype=int),
        historical_dispatch=np.array([CAP] * len(NL)),
        dispatchable_ramp_mw=np.array(CAP),
        dispatchable_startup_cost=np.array([[1000.0], [1000.0], [100.0]]),
        dispatchable_marginal_cost=np.array([[10.0], [20.0], [50.0]]),
        dispatchable_min_uptime=np.zeros_like(CAP, dtype=np.int_),
        storage_mw=np.array([400, 200]),
        storage_hrs=np.array([4, 12]),
        storage_eff=np.array((0.9, 0.9)),
        storage_op_hour=np.array((0, 0)),
        storage_dc_charge=np.zeros((len(NL), 2)),
        storage_reserve=np.array([0.1, 0.1]),
        dynamic_reserve_coeff=1.5,
        marginal_for_startup_rank=False,
    )
    redispatch, es, sl, st = dispatch_engine(
        net_load=np.array(NL),
        hr_to_cost_idx=np.zeros(len(NL), dtype=int),
        historical_dispatch=np.array([CAP] * len(NL)),
        dispatchable_ramp_mw=np.array(CAP),
        dispatchable_startup_cost=np.array([[1000.0], [1000.0], [100.0]]),
        dispatchable_marginal_cost=np.array([[10.0], [20.0], [50.0]]),
        dispatchable_min_uptime=np.zeros_like(CAP, dtype=np.int_),
        storage_mw=np.array([400, 200]),
        storage_hrs=np.array([4, 12]),
        storage_eff=np.array((0.9, 0.9)),
        storage_op_hour=np.array((0, 0)),
        storage_dc_charge=np.zeros((len(NL), 2)),
        storage_reserve=np.array([0.1, 0.1]),
        dynamic_reserve_coeff=1.5,
        marginal_for_startup_rank=False,
    )
    assert np.all(redispatch == redispatch_py)
    assert np.all(es == es_py)
    assert np.all(sl == sl_py)
    assert np.all(st == st_py)


def test_engine_marginal_for_startup_rank():
    """Test that ``marginal_for_startup_rank`` uses lower marginal cost generators."""
    in_dict = dict(  # noqa: C408
        net_load=np.array(NL),
        hr_to_cost_idx=np.zeros(len(NL), dtype=int),
        historical_dispatch=np.array([CAP] * len(NL)),
        dispatchable_ramp_mw=np.array(CAP),
        dispatchable_startup_cost=np.array([[1000.0], [1000.0], [100.0]]),
        dispatchable_marginal_cost=np.array([[10.0], [20.0], [50.0]]),
        dispatchable_min_uptime=np.zeros_like(CAP, dtype=np.int_),
        storage_mw=np.array([400, 200]),
        storage_hrs=np.array([4, 12]),
        storage_eff=np.array((0.9, 0.9)),
        storage_op_hour=np.array((0, 0)),
        storage_dc_charge=np.zeros((len(NL), 2)),
        storage_reserve=np.array([0.1, 0.1]),
        dynamic_reserve_coeff=1.5,
    )
    redis_m, *_ = dispatch_engine.py_func(**in_dict, marginal_for_startup_rank=True)
    redis, *_ = dispatch_engine.py_func(**in_dict, marginal_for_startup_rank=False)
    diff = redis_m - redis
    assert diff[:, 0].sum() > 0
    assert diff[:, 2].sum() < 0


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"storage_mw": np.array([400, 200, 100])}, AssertionError),
        ({"storage_dc_charge": np.zeros((len(NL), 5))}, AssertionError),
        ({"startup_cost": np.array([[1000.0], [1000.0]])}, AssertionError),
        ({"net_load": np.array(NL)[1:]}, AssertionError),
        (
            {"hr_to_cost_idx": np.append(np.zeros(len(NL) - 1, dtype=int), 1)},
            AssertionError,
        ),
        (
            {
                "storage_reserve": np.array(
                    [
                        400,
                    ]
                )
            },
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
        storage_reserve=np.array((0.0, 0.1)),
    )
    validate_inputs.py_func(**base)
    over = base | override
    with pytest.raises(expected):
        validate_inputs.py_func(**over)


# fmt: off
@pytest.mark.parametrize("py_func", [True, False])
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"deficit": 0, "state_of_charge": 0, "dc_charge": 0, "mw": 10, "max_state_of_charge": 20, "eff": 1.0},
            (0, 0, 0, 0),
        ),
        (
            {"deficit": -5, "state_of_charge": 0, "dc_charge": 0, "mw": 10, "max_state_of_charge": 20, "eff": 1.0},
            (5, 0, 5, 5),
        ),
        (
            {"deficit": 0, "state_of_charge": 0, "dc_charge": 5, "mw": 10, "max_state_of_charge": 20, "eff": 1.0},
            (5, 0, 5, 0),
        ),
        (
            {"deficit": 500, "state_of_charge": 0, "dc_charge": 5, "mw": 10, "max_state_of_charge": 20, "eff": 1.0},
            (5, 0, 5, 0),
        ),
        (
            {"deficit": -5, "state_of_charge": 0, "dc_charge": 5, "mw": 5, "max_state_of_charge": 20, "eff": 1.0},
            (5, 0, 5, 0),
        ),
        (
            {"deficit": -5, "state_of_charge": 0, "dc_charge": 3, "mw": 5, "max_state_of_charge": 20, "eff": 1.0},
            (5, 0, 5, 2),
        ),
        (
            {"deficit": -5, "state_of_charge": 0, "dc_charge": 5, "mw": 10, "max_state_of_charge": 20, "eff": 1.0},
            (10, 0, 10, 5),
        ),
        (
            {"deficit": -5, "state_of_charge": 5, "dc_charge": 5, "mw": 10, "max_state_of_charge": 20, "eff": 1.0},
            (10, 0, 15, 5),
        ),
        (
            {"deficit": -5, "state_of_charge": 5, "dc_charge": 5, "mw": 10, "max_state_of_charge": 20, "eff": 0.5},
            (10, 0, 10, 5),
        ),
        (
            {"deficit": -5, "state_of_charge": 15, "dc_charge": 5, "mw": 10, "max_state_of_charge": 20, "eff": 1.0},
            (5, 0, 20, 0),
        ),
        (
            {"deficit": -2, "state_of_charge": 15, "dc_charge": 5, "mw": 10, "max_state_of_charge": 20, "eff": 1.0},
            (5, 0, 20, 0),
        ),
        (
            {"deficit": -5, "state_of_charge": 0, "dc_charge": 0, "mw": 1, "max_state_of_charge": 20, "eff": 1.0},
            (1, 0, 1, 1),
        ),
        (
            {"deficit": 0, "state_of_charge": 0, "dc_charge": 5, "mw": 1, "max_state_of_charge": 20, "eff": 1.0},
            (1, 0, 1, 0),
        ),
        (
            {"deficit": 500, "state_of_charge": 0, "dc_charge": 5, "mw": 1, "max_state_of_charge": 20, "eff": 1.0},
            (1, 0, 1, 0),
        ),
        (
            {"deficit": -5, "state_of_charge": 0, "dc_charge": 5, "mw": 1, "max_state_of_charge": 20, "eff": 1.0},
            (1, 0, 1, 0),
        ),
        (
            {"deficit": -5, "state_of_charge": 5, "dc_charge": 5, "mw": 1, "max_state_of_charge": 20, "eff": 1.0},
            (1, 0, 6, 0),
        ),
        (
            {"deficit": -5, "state_of_charge": 5, "dc_charge": 5, "mw": 1, "max_state_of_charge": 20, "eff": 0.9},
            (1, 0, 5.9, 0),
        ),
        (
            {"deficit": -2, "state_of_charge": 15, "dc_charge": 5, "mw": 1, "max_state_of_charge": 20, "eff": 1.0},
            (1, 0, 16, 0),
        ),
    ],
    ids=idfn,
)
def test_charge_storage(py_func, kwargs, expected):
    """Test storage charging calculations."""
    if py_func:
        assert charge_storage.py_func(**kwargs) == expected
    else:
        assert charge_storage(**kwargs) == expected


@pytest.mark.parametrize("py_func", [True, False])
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"desired_mw": 1, "state_of_charge": 2, "mw": 3, "max_state_of_charge": 6}, 1),
        ({"desired_mw": 1, "state_of_charge": 5, "mw": 3, "max_state_of_charge": 6}, 1),
        ({"desired_mw": 5, "state_of_charge": 2, "mw": 3, "max_state_of_charge": 6}, 2),
        ({"desired_mw": 9, "state_of_charge": 5, "mw": 3, "max_state_of_charge": 6}, 3),
        ({"desired_mw": 1, "state_of_charge": 2, "mw": 3, "max_state_of_charge": 6, "reserve": 0.5}, 0),
        ({"desired_mw": 1, "state_of_charge": 5, "mw": 3, "max_state_of_charge": 6, "reserve": 0.5}, 1),
        ({"desired_mw": 5, "state_of_charge": 2, "mw": 3, "max_state_of_charge": 6, "reserve": 0.5}, 0),
        ({"desired_mw": 9, "state_of_charge": 5, "mw": 3, "max_state_of_charge": 6, "reserve": 0.5}, 2),
    ],
    ids=idfn,
)
def test_discharge_storage(py_func, kwargs, expected):
    """Test storage discharging calculations."""
    if py_func:
        assert discharge_storage.py_func(**kwargs) == expected
    else:
        assert discharge_storage(**kwargs) == expected


@pytest.mark.parametrize("py_func", [True, False])
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"state_of_charge": [8], "mw": [2], "reserve": [0.25], "max_state_of_charge": [8]}, -2.0),
        ({"state_of_charge": [7], "mw": [2], "reserve": [0.25], "max_state_of_charge": [8]}, -2.0),
        ({"state_of_charge": [6], "mw": [2], "reserve": [0.25], "max_state_of_charge": [8]}, -2.0),
        ({"state_of_charge": [5], "mw": [2], "reserve": [0.25], "max_state_of_charge": [8]}, -1.0),
        ({"state_of_charge": [4], "mw": [2], "reserve": [0.25], "max_state_of_charge": [8]}, 0.0),
        ({"state_of_charge": [3], "mw": [2], "reserve": [0.25], "max_state_of_charge": [8]}, 0.0),
        ({"state_of_charge": [2], "mw": [2], "reserve": [0.25], "max_state_of_charge": [8]}, 0.0),
        ({"state_of_charge": [1], "mw": [2], "reserve": [0.25], "max_state_of_charge": [8]}, 1.0),
        ({"state_of_charge": [0], "mw": [2], "reserve": [0.25], "max_state_of_charge": [8]}, 2.0),
        ({"state_of_charge": [0, 0], "mw": [2, 2], "reserve": 2 * [0.25], "max_state_of_charge": [8, 8]}, 4.0),
        ({"state_of_charge": [1, 0], "mw": [2, 2], "reserve": 2 * [0.25], "max_state_of_charge": [8, 8]}, 3.0),
        ({"state_of_charge": [8, 0], "mw": [2, 2], "reserve": 2 * [0.25], "max_state_of_charge": [8, 8]}, 2.0),
        ({"state_of_charge": [8, 1], "mw": [2, 2], "reserve": 2 * [0.25], "max_state_of_charge": [8, 8]}, 1.0),
        ({"state_of_charge": [8, 2], "mw": [2, 2], "reserve": 2 * [0.25], "max_state_of_charge": [8, 8]}, -2.0),
        ({"state_of_charge": [8, 3], "mw": [2, 2], "reserve": 2 * [0.25], "max_state_of_charge": [8, 8]}, -2.0),
        ({"state_of_charge": [8, 4], "mw": [2, 2], "reserve": 2 * [0.25], "max_state_of_charge": [8, 8]}, -2.0),
        ({"state_of_charge": [8, 5], "mw": [2, 2], "reserve": 2 * [0.25], "max_state_of_charge": [8, 8]}, -3.0),
        ({"state_of_charge": [8, 6], "mw": [2, 2], "reserve": 2 * [0.25], "max_state_of_charge": [8, 8]}, -4.0),
        ({"state_of_charge": [8, 7], "mw": [2, 2], "reserve": 2 * [0.25], "max_state_of_charge": [8, 8]}, -4.0),
        ({"state_of_charge": [8, 8], "mw": [2, 2], "reserve": 2 * [0.25], "max_state_of_charge": [8, 8]}, -4.0),
    ],
    ids=idfn,
)
def test_adjust_for_storage_reserve(py_func, kwargs, expected):
    """Test adjustments for storage reserve calculations."""
    if py_func:
        assert adjust_for_storage_reserve.py_func(
            **{k: np.array(v, dtype=float) for k, v in kwargs.items()}
        ) == expected
    else:
        assert adjust_for_storage_reserve(
            **{k: np.array(v, dtype=float) for k, v in kwargs.items()}
        ) == expected


@pytest.mark.parametrize("py_func", [True, False])
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"desired_mw": 5, "max_mw": 2, "previous_mw": 0, "ramp_mw": 1}, 1),
        ({"desired_mw": 5, "max_mw": 10, "previous_mw": 10, "ramp_mw": 1}, 9),
        ({"desired_mw": 5, "max_mw": 10, "previous_mw": 5, "ramp_mw": 1}, 5),
        ({"desired_mw": 5, "max_mw": 4, "previous_mw": 5, "ramp_mw": 1}, 4),
        ({"desired_mw": 5, "max_mw": 4, "previous_mw": 2, "ramp_mw": 1}, 3),
        ({"desired_mw": 5, "max_mw": 0, "previous_mw": 5, "ramp_mw": 1}, 0),
        ({"desired_mw": 5, "max_mw": 2, "previous_mw": 0, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 0}, 1),
        ({"desired_mw": 5, "max_mw": 9, "previous_mw": 9, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 0}, 8),
        ({"desired_mw": 5, "max_mw": 9, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 0}, 5),
        ({"desired_mw": 5, "max_mw": 4, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 0}, 4),
        ({"desired_mw": 5, "max_mw": 4, "previous_mw": 2, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 0}, 3),
        ({"desired_mw": 5, "max_mw": 0, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 0}, 0),
        ({"desired_mw": 5, "max_mw": 9, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 2}, 5),
        ({"desired_mw": 5, "max_mw": 4, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 2}, 4),
        ({"desired_mw": 5, "max_mw": 4, "previous_mw": 2, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 2}, 3),
        ({"desired_mw": 5, "max_mw": 0, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 2}, 0),
        ({"desired_mw": 0, "max_mw": 9, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 2}, 5),
        ({"desired_mw": 4, "max_mw": 9, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 2}, 5),
        ({"desired_mw": 7, "max_mw": 9, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 2}, 6),
        ({"desired_mw": 5, "max_mw": 9, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 3}, 5),
        ({"desired_mw": 5, "max_mw": 4, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 3}, 4),
        ({"desired_mw": 5, "max_mw": 4, "previous_mw": 2, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 3}, 3),
        ({"desired_mw": 5, "max_mw": 0, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 3}, 0),
        ({"desired_mw": 0, "max_mw": 9, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 3}, 5),
        ({"desired_mw": 4, "max_mw": 9, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 3}, 5),
        ({"desired_mw": 7, "max_mw": 9, "previous_mw": 5, "ramp_mw": 1, "current_uptime": 1, "min_uptime": 3}, 6),
    ],
    ids=idfn,
)
def test_dispatch_generator(py_func, kwargs, expected):
    """Test the logic of the calculate_generator_output function."""
    if py_func:
        assert calculate_generator_output.py_func(**kwargs) == expected
    else:
        assert calculate_generator_output(**kwargs) == expected
# fmt: on


@pytest.mark.parametrize("py_func", [True, False], ids=idfn)
def test_make_rank_arrays(py_func):
    """Test cost rank setup."""
    m_cost = np.array([[50.0, 50.0], [25.0, 75.0]])
    s_cost = np.array([[250.0, 250.0], [500.0, 500.0]])
    if py_func:
        m, s = make_rank_arrays.py_func(m_cost, s_cost)
    else:
        m, s = make_rank_arrays(m_cost, s_cost)
    assert np.all(m == np.array([[1, 0], [0, 1]]))
    assert np.all(s == np.array([[0, 0], [1, 1]]))


@pytest.mark.parametrize("py_func", [True, False])
@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (12, 24, 0.95),
        (12, 18, 0.89),
        (12, 16, 0.86),
        (12, 14, 0.83),
        (12, 10, 0.71),
        (12, 8, 0.63),
        (12, 6, 0.53),
        (12, 4, 0.39),
        (12, 3, 0.31),
        (12, 2, 0.22),
        (12, 1, 0.12),
        (12, 0, 0.0),
    ],
    ids=idfn,
)
def test_dynamic_reserve(py_func, a, b, expected):
    """Test dynamic reserve."""
    net_load = a + b * np.sin(np.arange(0, 4, np.pi / 24))
    if py_func:
        result = dynamic_reserve.py_func(
            hr=0, reserve=np.array([0.0, 0.1]), net_load=net_load, coeff=1.5
        )
    else:
        result = dynamic_reserve(
            hr=0, reserve=np.array([0.0, 0.1]), net_load=net_load, coeff=1.5
        )
    assert np.all(result == np.array([expected, 0.1]))


@pytest.mark.parametrize("py_func", [True, False], ids=idfn)
@pytest.mark.parametrize(
    ("comp", "expected"),
    [
        (np.array([[0, 1e7], [1e-6, 1e6], [1e-5, 1e5], [1e-4, 1e4], [1e-3, 1e3]]), 3),
        (np.array([[100, 1e7], [7e-2, 1e6], [2e-4, 1e5], [6e-4, 1e4], [1e-3, 1e3]]), 2),
        (np.array([[1e-3, 1e3], [2e-3, 1e5], [6e-3, 1e4], [1e-2, 1e5]]), 0),
    ],
    ids=idfn,
)
def test_optimal_idx(py_func, comp, expected):
    """Test selection of optimal index."""
    comp = np.column_stack(
        (comp[:, 0], np.zeros(len(comp)), comp[:, 1], np.zeros(len(comp)))
    )
    if py_func:
        assert choose_best_coefficient.py_func(comp) == expected
    else:
        assert choose_best_coefficient(comp) == expected
