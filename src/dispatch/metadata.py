"""Metadata and :mod:`pandera` stuff."""
import pandera as pa

DT_SCHEMA = pa.Index(pa.Timestamp, name="datetime")
PID_SCHEMA = pa.Index(int, name="plant_id_eia")
GID_SCHEMA = pa.Index(str, name="generator_id")

FOSSIL_PROFILE_SCHEMA = pa.DataFrameSchema(
    index=DT_SCHEMA,
)
FOSSIL_SPECS_SCHEMA = pa.DataFrameSchema(
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
MARGINAL_COST_SCHEMA = pa.SeriesSchema(
    float,
    index=pa.MultiIndex(
        [PID_SCHEMA, GID_SCHEMA, DT_SCHEMA],
        unique=["plant_id_eia", "generator_id", "datetime"],
        strict=True,
        coerce=True,
    ),
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
