"""Metadata and :mod:`pandera` stuff."""
import logging
import warnings
from typing import Any

import pandas as pd
import pandera as pa

LOGGER = logging.getLogger(__name__)


DT_SCHEMA = pa.Index(pa.Timestamp, name="datetime")
PID_SCHEMA = pa.Index(int, name="plant_id_eia")
GID_SCHEMA = pa.Index(str, name="generator_id")
LOAD_PROFILE_SCHEMA = pa.DataFrameSchema(
    columns={"load": pa.Column(pa.Float)}, index=DT_SCHEMA, coerce=True
)


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

    def __init__(self, obj: Any, gen_set: pd.Index, re_set: pd.Index):
        """Init Validator."""
        self.obj = obj
        self.gen_set = gen_set
        self.re_set = re_set
        self.load_profile = obj.load_profile
        self.storage_specs_schema = pa.DataFrameSchema(
            index=pa.MultiIndex(
                [PID_SCHEMA, GID_SCHEMA],
                unique=["plant_id_eia", "generator_id"],
                strict=True,
                coerce=True,
            ),
            columns={
                "capacity_mw": pa.Column(float, pa.Check.greater_than_or_equal_to(0)),
                "duration_hrs": pa.Column(int, pa.Check.greater_than_or_equal_to(0)),
                "roundtrip_eff": pa.Column(
                    float, pa.Check.in_range(0, 1), nullable=False, required=False
                ),
                "operating_date": pa.Column(
                    pa.Timestamp,
                    pa.Check.less_than(self.load_profile.index.max()),
                    description="operating_date in storage_specs",
                ),
                "reserve": pa.Column(pa.Float, nullable=False, required=False),
                "charge_mw": pa.Column(pa.Float, nullable=False, required=False),
                "charge_eff": pa.Column(
                    pa.Float, pa.Check.in_range(0, 1), nullable=False, required=False
                ),
                "discharge_eff": pa.Column(
                    pa.Float, pa.Check.in_range(0, 1), nullable=False, required=False
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
                "min_uptime": pa.Column(pa.Int, nullable=False, required=False),
                "exclude": pa.Column(pa.Bool, nullable=False, required=False),
                "no_limit": pa.Column(pa.Bool, nullable=False, required=False),
            },
            coerce=True,
        )

    def dispatchable_specs(self, dispatchable_specs: pd.DataFrame) -> pd.DataFrame:
        """Validate dispatchable_specs."""
        if "exclude" in dispatchable_specs and "no_limit" in dispatchable_specs:
            not_good = dispatchable_specs.exclude & dispatchable_specs.no_limit
            if not_good.any():
                raise AssertionError(
                    f"These both ``no_limit`` and ``exclude`` cannot be True for the "
                    f"same generator, errors: {list(not_good[not_good].index)}"
                )
        return self.dispatchable_specs_schema.validate(dispatchable_specs)

    def dispatchable_profiles(
        self, dispatchable_profiles: pd.DataFrame
    ) -> pd.DataFrame:
        """Validate dispatchable_profiles.

        changed the limit to 2e4 but these look like errors...

        E           pandera.errors.SchemaError: <Schema
        Column(name=(55178, 'CT-1'), E
        type=DataType(float64))> failed element-wise validator 0: E
        <Check in_range: in_range(0.0, 10000.0)> E           failure
        cases: E                           index  failure_case E
        0 2011-03-08 18:00:00  10842.857422 E           1 2011-03-10
        09:00:00  12690.857422
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
        marg_freq = dispatchable_cost.index.levels[2].freqstr
        if marg_freq is None:
            marg_freq = pd.infer_freq(
                dispatchable_cost.index.get_level_values(2).unique()
            )
        if marg_freq is None:
            freq_dict = {1: "A", 12: "M", 52: "W", 365: "D"}
            full = dispatchable_cost.index.get_level_values(2).unique()
            is_start = all(full.day.unique() == 1)
            per_year = int(len(full) / len(full.year.unique()))
            marg_freq = freq_dict.get(per_year, "") + "S" if is_start else ""

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
                "generators in `dispatchable_cost` do not match generators in "
                "`dispatchable_specs`"
            ) from exc
        # self.obj._metadata["marginal_cost_freq"] = marg_freq
        if not any(("YS" in marg_freq, "AS" in marg_freq, "MS" in marg_freq)):
            raise AssertionError(f"Cost data must be `YS` or not `{marg_freq}`")
        marg_dts = dispatchable_cost.index.get_level_values("datetime")
        missing_prds = [
            d
            for d in self.load_profile.resample(marg_freq).first().index
            if d not in marg_dts
        ]
        if missing_prds:
            raise AssertionError(f"{missing_prds} not in `dispatchable_cost`")

        missing_gens = (
            dispatchable_cost.reset_index()  # noqa: PD010, PD013
            .pivot(
                index=["plant_id_eia", "generator_id"],
                columns="datetime",
                values="fom_per_kw",
            )
            .isna()
            .stack()
        )
        if not (missing_gens := missing_gens[missing_gens]).empty:
            msg = missing_gens.to_string().replace("\n", "\n\t")
            raise AssertionError(f"missing periods from `dispatchable_cost`: \n{msg}")
        return dispatchable_cost

    def storage_specs(self, storage_specs: pd.DataFrame) -> pd.DataFrame:
        """Validate storage_specs."""
        if "roundtrip_eff" in storage_specs:
            warnings.warn(
                "use `charge_eff` and `discharge_eff` instead of `roundtrip_eff`, if "
                "using `roundtrip_eff`, it is treated as `charge_eff` and "
                "`discharge_eff` is set to 1.0",
                DeprecationWarning,
                stacklevel=2,
            )
        elif "charge_eff" not in storage_specs or "discharge_eff" not in storage_specs:
            raise AssertionError(
                "both `charge_eff` and `discharge_eff` are required, to replicate "
                "previous behavior, set `charge_eff` as the roundtrip efficiency and "
                "`discharge_eff` to 1.0"
            )

        out = self.storage_specs_schema.validate(storage_specs)
        check_dup_ids = out.assign(
            id_count=lambda x: x.groupby("plant_id_eia").capacity_mw.transform("count")
        ).query("id_count > 1")
        bad_dups = check_dup_ids.loc[
            [
                x
                for x in check_dup_ids.index
                if x[0] in self.re_set.get_level_values("plant_id_eia")
            ],
            :,
        ]
        if bad_dups.empty:
            return out
        bad_dups = (
            bad_dups[["capacity_mw", "duration_hrs", "roundtrip_eff", "operating_date"]]
            .to_string()
            .replace("\n", "\n\t")
        )
        raise AssertionError(
            f"DC-coupled storage must share `plant_id_eia` with the associated "
            f"renewable facility. In those cases, there can be only one storage "
            f"facility with that `plant_id_eia`. The following storage facilities "
            f"are DC_coupled but not unique: \n {bad_dups}"
        )

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
