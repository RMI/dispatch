"""Where dispatch tests will go."""
import numpy as np
import pandas as pd

from dispatch import DispatchModel


def setup_dm(fossil_profiles, fossil_specs, re_profiles, re, storage):
    """Setup `DispatchModel`."""
    fossil_profiles.columns = fossil_specs.index
    dm = DispatchModel.from_patio(
        fossil_profiles.sum(axis=1) - re_profiles @ re,
        fossil_profiles=fossil_profiles,
        plant_data=fossil_specs,
        storage_specs=storage,
    )
    return dm


def test_from_patio(fossil_profiles, re_profiles, fossil_specs):
    """Dummy test to quiet pytest."""
    dm = setup_dm(
        fossil_profiles,
        fossil_specs,
        re_profiles,
        np.array([5000.0, 5000.0, 0.0, 0.0]),
        pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
    )
    assert dm


def test_low_lost_load(fossil_profiles, re_profiles, fossil_specs):
    """Dummy test to quiet pytest."""
    fossil_profiles.columns = fossil_specs.index
    dm = setup_dm(
        fossil_profiles,
        fossil_specs,
        re_profiles,
        np.array([5000.0, 5000.0, 0.0, 0.0]),
        pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
    )
    dm()
    assert (dm.lost_load() / dm.lost_load().sum()).iloc[0] > 0.9999
