"""Dispatch engine tests."""


import numpy as np
import pytest

# noinspection PyProtectedMember
from dispatch.engine import charge_storage, dispatch_engine, make_rank_arrays
from dispatch.helpers import idfn

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
    redispatch, es, sl, st = dispatch_engine(
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
    "deficit, soc, dc_charge, mw, soc_max, eff, expected",
    [
        (0, 0, 0, 10, 20, 1, (0, 0, 0, 0)),
        (-5, 0, 0, 10, 20, 1, (5, 0, 5, 5)),
        (0, 0, 5, 10, 20, 1, (5, 0, 5, 0)),
        (500, 0, 5, 10, 20, 1, (5, 0, 5, 0)),
        (-5, 0, 5, 5, 20, 1, (5, 0, 5, 0)),
        (-5, 0, 3, 5, 20, 1, (5, 0, 5, 2)),
        (-5, 0, 5, 10, 20, 1, (10, 0, 10, 5)),
        (-5, 5, 5, 10, 20, 1, (10, 0, 15, 5)),
        (-5, 5, 5, 10, 20, 0.5, (10, 0, 10, 5)),
        (-5, 15, 5, 10, 20, 1, (5, 0, 20, 0)),
        (-2, 15, 5, 10, 20, 1, (5, 0, 20, 0)),
        (-5, 0, 0, 1, 20, 1, (1, 0, 1, 1)),
        (0, 0, 5, 1, 20, 1, (1, 0, 1, 0)),
        (500, 0, 5, 1, 20, 1, (1, 0, 1, 0)),
        (-5, 0, 5, 1, 20, 1, (1, 0, 1, 0)),
        (-5, 5, 5, 1, 20, 1, (1, 0, 6, 0)),
        (-5, 5, 5, 1, 20, 0.9, (1, 0, 5.9, 0)),
        (-2, 15, 5, 1, 20, 1, (1, 0, 16, 0)),
    ],
    ids=idfn,
)
def test_charge_storage(deficit, soc, dc_charge, mw, soc_max, eff, expected):
    """Test storage charging calculations."""
    assert charge_storage(deficit, soc, dc_charge, mw, soc_max, eff) == expected


def test_make_rank_arrays():
    """Test cost rank setup."""
    m_cost = np.array([[50.0, 50.0], [25.0, 75.0]])
    s_cost = np.array([[250.0, 250.0], [500.0, 500.0]])
    m, s = make_rank_arrays(m_cost, s_cost)
    assert np.all(m == np.array([[1, 0], [0, 1]]))
    assert np.all(s == np.array([[0, 0], [1, 1]]))
