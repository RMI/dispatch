"""Metadata and :mod:`pandera` stuff."""
import logging
from typing import Any

import pandas as pd
import pandera as pa

LOGGER = logging.getLogger(__name__)


DT_SCHEMA = pa.Index(pa.Timestamp, name="datetime")
PID_SCHEMA = pa.Index(int, name="plant_id_eia")
GID_SCHEMA = pa.Index(str, name="generator_id")
LOAD_PROFILE_SCHEMA = pa.SeriesSchema(pa.Float, index=DT_SCHEMA, coerce=True)


class Validator:
    """Validator for :class:`.DispatchModel` inputs."""

    dispatchable_cost_schema = pa.DataFrameSchema(
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
            "startup_cost": pa.Column(float, required=False),
            "heat_rate": pa.Column(
                float,
                pa.Check.in_range(0.0, 30.0),
                required=False,
                nullable=True,
            ),
            "co2_factor": pa.Column(
                float,
                pa.Check.in_range(0.0, 0.2),
                required=False,
                nullable=True,
            ),
        },
        coerce=True,
    )

    def __init__(self, obj: Any, gen_set: pd.Index):
        """Init Validator."""
        self.obj = obj
        self.gen_set = gen_set
        self.load_profile = obj.load_profile
        self.storage_specs_schema = pa.DataFrameSchema(
            index=pa.MultiIndex(
                [PID_SCHEMA, GID_SCHEMA],
                unique=["plant_id_eia"],
                strict=True,
                coerce=True,
            ),
            columns={
                "capacity_mw": pa.Column(float, pa.Check.greater_than_or_equal_to(0)),
                "duration_hrs": pa.Column(int, pa.Check.greater_than_or_equal_to(0)),
                "roundtrip_eff": pa.Column(float, pa.Check.in_range(0, 1)),
                "operating_date": pa.Column(
                    pa.Timestamp,
                    pa.Check.less_than(self.load_profile.index.max()),
                    description="operating_date in storage_specs",
                ),
            },
            # strict=True,
            coerce=True,
            title="storage_specs",
        )
        self.renewable_specs_schema = pa.DataFrameSchema(
            index=pa.MultiIndex(
                [PID_SCHEMA, GID_SCHEMA],
                unique=["plant_id_eia", "generator_id"],
                strict=True,
                coerce=True,
            ),
            columns={
                "capacity_mw": pa.Column(float, pa.Check.greater_than_or_equal_to(0)),
                "ilr": pa.Column(float, pa.Check.in_range(1.0, 10.0)),
                "operating_date": pa.Column(
                    pa.Timestamp,
                    pa.Check.less_than(self.load_profile.index.max()),
                    description="operating_date in renewable_specs",
                ),
                "retirement_date": pa.Column(pa.Timestamp, nullable=True),
                "interconnect_mw": pa.Column(pa.Float, nullable=False, required=False),
                "fom_per_kw": pa.Column(pa.Float, nullable=True, required=False),
            },
            coerce=True,
        )
        self.dispatchable_specs_schema = pa.DataFrameSchema(
            index=pa.MultiIndex(
                [PID_SCHEMA, GID_SCHEMA],
                unique=["plant_id_eia", "generator_id"],
                strict=True,
                coerce=True,
            ),
            columns={
                "capacity_mw": pa.Column(float, pa.Check.greater_than_or_equal_to(0)),
                "ramp_rate": pa.Column(float, pa.Check.greater_than(0)),
                "operating_date": pa.Column(
                    pa.Timestamp,
                    pa.Check.less_than(self.load_profile.index.max()),
                    description="operating_date in dispatchable_specs",
                ),
                "retirement_date": pa.Column(pa.Timestamp, nullable=True),
                "exclude": pa.Column(pa.Bool, nullable=False, required=False),
            },
            coerce=True,
        )

    def dispatchable_specs(self, dispatchable_specs: pd.DataFrame) -> pd.DataFrame:
        """Validate dispatchable_specs."""
        return self.dispatchable_specs_schema.validate(dispatchable_specs)

    def dispatchable_profiles(
        self, dispatchable_profiles: pd.DataFrame
    ) -> pd.DataFrame:
        """Validate dispatchable_profiles.

        changed the limit to 2e4 but these look like errors...

        E           pandera.errors.SchemaError: <Schema Column(name=(55178, 'CT-1'), type=DataType(float64))> failed element-wise validator 0:
        E           <Check in_range: in_range(0.0, 10000.0)>
        E           failure cases:
        E                           index  failure_case
        E           0 2011-03-08 18:00:00  10842.857422
        E           1 2011-03-10 09:00:00  12690.857422
        """
        dispatchable_profiles = pa.DataFrameSchema(
            index=DT_SCHEMA,
            columns={
                x: pa.Column(float, pa.Check.in_range(0.0, 2e4)) for x in self.gen_set
            },
            ordered=True,
            coerce=True,
            strict=True,
        ).validate(dispatchable_profiles)
        try:
            pd.testing.assert_index_equal(
                dispatchable_profiles.index, self.load_profile.index
            )
        except AssertionError as exc:
            raise AssertionError(
                "`dispatchable_profiles` and `load_profile` indexes must match"
            ) from exc

        return dispatchable_profiles

    def dispatchable_cost(self, dispatchable_cost: pd.DataFrame) -> pd.DataFrame:
        """Validate dispatchable_cost."""
        try:
            marg_freq = dispatchable_cost.index.levels[2].freqstr
            if marg_freq is None:
                raise RuntimeError
        except (AttributeError, RuntimeError):
            marg_freq = pd.infer_freq(
                dispatchable_cost.index.get_level_values(2).unique()
            )

        dispatchable_cost = self.dispatchable_cost_schema.validate(dispatchable_cost)
        # make sure al
        try:
            pd.testing.assert_index_equal(
                dispatchable_cost.reset_index(
                    level="datetime", drop=True
                ).index.drop_duplicates(),
                self.gen_set,
            )
        except AssertionError as exc:
            raise AssertionError(
                "generators in `dispatchable_cost` do not match generators in `dispatchable_specs`"
            ) from exc
        self.obj._metadata["marginal_cost_freq"] = marg_freq
        if "YS" not in marg_freq and "AS" not in marg_freq:
            raise AssertionError("Cost data must be `YS`")
        marg_dts = dispatchable_cost.index.get_level_values("datetime")
        missing_prds = [
            d
            for d in self.load_profile.resample(marg_freq).first().index
            if d not in marg_dts
        ]
        if missing_prds:
            raise AssertionError(f"{missing_prds} not in `dispatchable_cost`")
        return dispatchable_cost

    def storage_specs(self, storage_specs: pd.DataFrame) -> pd.DataFrame:
        """Validate storage_specs."""
        return self.storage_specs_schema.validate(storage_specs)

    def renewables(
        self, re_plant_specs: pd.DataFrame | None, re_profiles: pd.DataFrame | None
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Validate renewable specs and profiles."""
        if re_plant_specs is None or re_profiles is None:
            return re_plant_specs, re_profiles
        re_plant_specs = self.renewable_specs_schema.validate(re_plant_specs)
        re_profiles = pa.DataFrameSchema(
            index=DT_SCHEMA,
            columns={
                x: pa.Column(float, pa.Check.in_range(0.0, 1.0))
                for x in re_plant_specs.index
            },
            ordered=True,
            coerce=True,
            strict=True,
        ).validate(re_profiles)
        try:
            pd.testing.assert_index_equal(re_profiles.index, self.load_profile.index)
        except AssertionError as exc:
            raise AssertionError(
                "`re_profiles` and `load_profile` indexes must match"
            ) from exc
        return re_plant_specs, re_profiles
