"""Dispatch engine tests."""


import numpy as np

from dispatch.engine import dispatch_engine

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
    re, es, sl, st = dispatch_engine(
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
    assert True
