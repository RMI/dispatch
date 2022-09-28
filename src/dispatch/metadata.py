"""Metadata and :mod:`pandera` stuff."""
import logging
from typing import Any

import numpy as np
import pandas as pd
import pandera as pa

LOGGER = logging.getLogger(__name__)


DT_SCHEMA = pa.Index(pa.Timestamp, name="datetime")
PID_SCHEMA = pa.Index(int, name="plant_id_eia")
GID_SCHEMA = pa.Index(str, name="generator_id")

DISPATCHABLE_SPECS_SCHEMA = pa.DataFrameSchema(
    index=pa.MultiIndex(
        [PID_SCHEMA, GID_SCHEMA],
        unique=["plant_id_eia", "generator_id"],
        strict=True,
        coerce=True,
    ),
    columns={
        "capacity_mw": pa.Column(float),
        "ramp_rate": pa.Column(float),
        "startup_cost": pa.Column(float),
        "operating_date": pa.Column(pa.Timestamp),
        "retirement_date": pa.Column(pa.Timestamp, nullable=True),
    },
    coerce=True,
)
DISPATCHABLE_COST_SCHEMA = pa.DataFrameSchema(
    index=pa.MultiIndex(
        [PID_SCHEMA, GID_SCHEMA, DT_SCHEMA],
        unique=["plant_id_eia", "generator_id", "datetime"],
        strict=True,
        coerce=True,
    ),
    columns={
        "vom_per_mwh": pa.Column(float),
        "fuel_per_mwh": pa.Column(float),
        "total_var_mwh": pa.Column(float, required=False),
        "fom_per_kw": pa.Column(float),
        "start_per_kw": pa.Column(float),
    },
    coerce=True,
)
NET_LOAD_SCHEMA = pa.SeriesSchema(pa.Float, index=DT_SCHEMA, coerce=True)
STORAGE_SPECS_SCHEMA = pa.DataFrameSchema(
    columns={
        "capacity_mw": pa.Column(float),
        "duration_hrs": pa.Column(int),
        "roundtrip_eff": pa.Column(float),
        "operating_date": pa.Column(pa.Timestamp),
    },
    index=pa.Index(int, unique=True),
    strict=True,
    coerce=True,
)


class Validator:
    """Validator for :class:`.DispatchModel` inputs."""

    def __init__(self, obj: Any):
        """Init Validator."""
        self.obj = obj
        self.gen_set = obj.dispatchable_specs.index
        self.net_load_profile = obj.net_load_profile

    def dispatchable_profiles(
        self, dispatchable_profiles: pd.DataFrame
    ) -> pd.DataFrame:
        """Validate dispatchable_profiles."""
        dispatchable_profiles = pa.DataFrameSchema(
            index=DT_SCHEMA,
            columns={x: pa.Column(float) for x in self.gen_set},
            coerce=True,
            strict=True,
        ).validate(dispatchable_profiles)
        if not np.all(dispatchable_profiles.index == self.net_load_profile.index):
            raise AssertionError(
                "`dispatchable_profiles` and `net_load_profile` must be the same length"
            )

        return dispatchable_profiles

    def dispatchable_cost(self, dispatchable_cost: pd.DataFrame) -> pd.DataFrame:
        """Validate dispatchable_cost."""
        dispatchable_cost = DISPATCHABLE_COST_SCHEMA.validate(dispatchable_cost)
        # make sure al
        if not np.all(
            dispatchable_cost.reset_index(
                level="datetime", drop=True
            ).index.drop_duplicates()
            == self.gen_set
        ):
            raise AssertionError(
                "generators in `dispatchable_cost` do not match generators in `dispatchable_specs`"
            )
        marg_freq = pd.infer_freq(dispatchable_cost.index.get_level_values(2).unique())
        self.obj.__meta__["marginal_cost_freq"] = marg_freq
        if "YS" not in marg_freq and "AS" not in marg_freq:
            raise AssertionError("Cost data must be `YS`")
        marg_dts = dispatchable_cost.index.get_level_values("datetime")
        missing_prds = [
            d
            for d in self.net_load_profile.resample(marg_freq).first().index
            if d not in marg_dts
        ]
        if missing_prds:
            raise AssertionError(f"{missing_prds} not in `dispatchable_cost`")
        if "total_var_mwh" not in dispatchable_cost:
            dispatchable_cost = dispatchable_cost.assign(
                total_var_mwh=lambda x: x[["vom_per_mwh", "fuel_per_mwh"]].sum(axis=1)
            )
        return dispatchable_cost

    def storage_specs(self, storage_specs: pd.DataFrame) -> pd.DataFrame:
        """Validate storage_specs."""
        if storage_specs is None:
            LOGGER.warning("Careful, dispatch without storage is untested")
            storage_specs = pd.DataFrame(
                [0.0, 0, 1.0, self.net_load_profile.index.max()],
                columns=[
                    "capacity_mw",
                    "duration_hrs",
                    "roundtrip_eff",
                    "operating_date",
                ],
            )
        return STORAGE_SPECS_SCHEMA.validate(storage_specs)
