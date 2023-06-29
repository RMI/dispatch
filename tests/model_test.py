"""Where dispatch tests will go."""
import logging
import sys
from io import BytesIO

import numpy as np
import pandas as pd
import pandera as pa
import polars as pl
import pytest
from dispatch import DispatchModel
from dispatch.helpers import zero_profiles_outside_operating_dates
from etoolbox.datazip import DataZip
from etoolbox.utils.testing import idfn

logger = logging.getLogger(__name__)


def test_new_no_dates(fossil_profiles, re_profiles, fossil_specs, fossil_cost):
    """Test that :meth:`.DispatchModel.from_fresh` fills in missing dates."""
    fossil_specs.iloc[
        0, fossil_specs.columns.get_loc("retirement_date")
    ] = fossil_profiles.index.max() - pd.Timedelta(weeks=15)
    self = DispatchModel.from_fresh(
        net_load_profile=fossil_profiles.sum(axis=1),
        dispatchable_specs=fossil_specs.drop(
            columns=["retirement_date", "operating_date"]
        ),
        dispatchable_cost=fossil_cost,
        storage_specs=pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
            index=pd.MultiIndex.from_tuples(
                [(-99, "es"), (-98, "es")], names=["plant_id_eia", "generator_id"]
            ),
        ),
        jit=True,
    )
    dates = self.dispatchable_specs[
        ["operating_date", "retirement_date"]
    ].drop_duplicates()
    assert fossil_profiles.index.min() == dates.operating_date.item()
    assert fossil_profiles.index.max() == dates.retirement_date.item()


@pytest.mark.parametrize(
    ("attr", "cols", "expected"),
    [
        ("redispatch", ["redispatch_mwh"], {"f": 399_672_109, "r": 370_228_932}),
        (
            "storage_dispatch",
            ["charge", "discharge", "soc", "gridcharge"],
            {"f": 266_660_550, "r": 235_078_425},
        ),
        (
            "system_data",
            ["deficit", "dirty_charge", "curtailment", "load_adjustment"],
            {"f": 49_509_985, "r": 75_636_546},
        ),
    ],
    ids=idfn,
)
def test_redispatch_total(ent_dm, attr, cols, expected):
    """High-level test that results have not changed."""
    ind, ent_dm = ent_dm
    assert sum(
        getattr(ent_dm, attr).select(cols).sum().collect()
    ).item() == pytest.approx(expected[ind], rel=2.5e-4)


@pytest.mark.parametrize("comparison", [None, "load_max"], ids=idfn)
def test_low_lost_load(mini_dm, comparison):
    """Dummy test that there isn't much lost load."""
    if comparison is not None:
        comparison = mini_dm.load_profile.max()
    assert (
        mini_dm.lost_load(comparison)
        .select(pl.col("count") / pl.col("count").sum())
        .collect()[0, "count"]
        > 0.998
    )


class TestIO:
    """Tests for IO functionality."""

    def test_write_and_read_full(self, temp_dir, ent_fresh):
        """Test that DispatchModel can be written and read."""
        dm = DispatchModel(**ent_fresh)
        file = temp_dir / "test_write_and_read_full.zip"
        try:
            dm.to_file(file)
            x = DispatchModel.from_file(file)
            x()
            x.to_file(file, clobber=True, include_output=True)
        except Exception as exc:
            raise AssertionError(f"{exc!r}") from exc
        else:
            assert True

    def test_write_and_read_bytes(self, ent_fresh):
        """Test that DispatchModel can be saved to and read from BytesIO."""
        dm = DispatchModel(**ent_fresh)
        file = BytesIO()
        try:
            dm.to_file(file)
            x = DispatchModel.from_file(file)
            x()
            x.to_file(file, clobber=True, include_output=True)
        except Exception as exc:
            raise AssertionError(f"{exc!r}") from exc
        else:
            assert True


class TestOutputs:
    """Tests for summaries and outputs."""

    def test_operations_summary(self, ent_dm):
        """Setup for testing cost and grouper methods."""
        _, ent_dm = ent_dm
        x = ent_dm.dispatchable_summary(by=None)
        assert (
            x.drop(
                [
                    "historical_mmbtu",
                    "historical_co2",
                    "redispatch_mmbtu",
                    "redispatch_co2",
                ]
            )
            .collect()
            .to_pandas()
            .notna()
            .all()
            .all()
        )

    def test_storage_summary(self, ent_dm):
        """Setup for testing cost and grouper methods."""
        _, ent_dm = ent_dm
        x = ent_dm.storage_summary(by=None)
        assert not x.collect().is_empty()

    def test_full_output(self, ent_dm):
        """Setup for testing cost and grouper methods."""
        _, ent_dm = ent_dm
        x = ent_dm.full_output()
        assert not x.collect().is_empty()

    def test_load_summary(self, ent_dm):
        """Setup for testing cost and grouper methods."""
        _, ent_dm = ent_dm
        x = ent_dm.load_summary()
        assert not x.collect().is_empty()

    @pytest.mark.parametrize(
        ("func", "args", "drop_cols", "expected"),
        [
            ("dispatchable_summary", {"by": None}, (), "notna"),
            (
                "re_summary",
                {"by": None},
                ("owned_pct", "retirement_date", "fom_per_kw", "redispatch_cost_fom"),
                "notna",
            ),
            ("system_level_summary", {}, (), "notna"),
            ("load_summary", {}, (), "notna"),
            ("storage_durations", {}, (), "notna"),
            ("storage_capacity", {}, (), "notna"),
            ("hourly_data_check", {}, (), "notna"),
            ("hrs_to_check", {}, (), "notna"),
            # ("dc_charge", {}, (), "notna"),
            pytest.param("full_output", {}, (), "notna", marks=pytest.mark.xfail),
            ("dispatchable_summary", {"by": None}, (), "notempty"),
            ("re_summary", {"by": None}, ("owned_pct", "retirement_date"), "notempty"),
            ("system_level_summary", {}, (), "notempty"),
            ("load_summary", {}, (), "notempty"),
            ("storage_durations", {}, (), "notempty"),
            ("storage_capacity", {}, (), "notempty"),
            ("hourly_data_check", {}, (), "notempty"),
            # ("dc_charge", {}, (), "notempty"),
            ("full_output", {}, (), "notempty"),
        ],
        ids=idfn,
    )
    def test_outputs_parametric(self, ent_dm, func, args, drop_cols, expected):
        """Test that outputs are not empty or do not have unexpected nans."""
        ind, ent_dm = ent_dm
        df = getattr(ent_dm, func)(**args)
        if isinstance(df, pl.LazyFrame | pl.DataFrame):
            df = df.drop([c for c in drop_cols if c in df.columns])
            df = df.collect()
        if expected == "notna":
            assert df.to_pandas().notna().all().all()
        elif expected == "notempty":
            assert not df.is_empty()
        else:
            raise AssertionError


@pytest.mark.skipif(
    (sys.platform == "win32") & (sys.version_info > (3, 10)),
    reason="plotly or kaleido intermittently hangs on windows in python 3.11",
)
class TestPlotting:
    """Tests for plotting methods."""

    def test_plot_all_years(self, ent_dm, temp_dir):
        """Test that outputs are not empty or do not have unexpected nans."""
        ind, ent_dm = ent_dm
        y = ent_dm.plot_all_years()
        img_path = temp_dir / "test_plot_all_years.pdf"
        try:
            y.write_image(str(img_path))
        except Exception as exc:
            raise AssertionError("unable to write image") from exc

    @pytest.mark.parametrize("freq", ["D", "H"], ids=idfn)
    def test_plotting(self, mini_dm, temp_dir, freq):
        """Testing plotting function."""
        y = mini_dm.plot_year(2015, freq=freq)
        img_path = temp_dir / f"test_plotting_{freq}.pdf"
        try:
            y.write_image(str(img_path))
        except Exception as exc:
            raise AssertionError("unable to write image") from exc

    def test_plot_detail_ent(self, ent_fresh, temp_dir):
        """Testing plotting function."""
        img_path = temp_dir / "test_plot_detail_ent.pdf"
        self = DispatchModel(**ent_fresh)()
        x = self.plot_period("2034-01-01", "2034-01-05", compare_hist=True)
        with DataZip(temp_dir / "test_img", "w") as dz:
            dz["img"] = x
        try:
            x.write_image(img_path)
        except Exception as exc:
            raise AssertionError("unable to write image") from exc
        else:
            assert True

    def test_plot_period_comp(self, ent_redispatch, temp_dir):
        """Testing plotting function."""
        img_path = temp_dir / "test_plot_period_comp.pdf"
        self = DispatchModel(**ent_redispatch)()
        x = self.plot_period("2034-01-01", "2034-01-05", compare_hist=False)
        try:
            x.write_image(img_path)
        except Exception as exc:
            raise AssertionError("unable to write image") from exc
        else:
            assert True

    def test_plot_year_ent(self, ent_fresh, temp_dir):
        """Testing plotting function."""
        img_path = temp_dir / "test_plot_year_ent.pdf"
        self = DispatchModel(**ent_fresh)()
        x = self.plot_year(2034)
        try:
            x.write_image(img_path)
        except Exception as exc:
            raise AssertionError("unable to write image") from exc
        else:
            assert True
        finally:
            img_path.unlink(missing_ok=True)

    @pytest.mark.parametrize(
        ("col", "freq"),
        [("capacity_mw", "YS"), ("redispatch_mwh", "YS"), ("redispatch_mwh", "MS")],
        ids=idfn,
    )
    def test_plot_output(self, ent_dm, temp_dir, col, freq):
        """Testing plotting function."""
        ind, ent_dm = ent_dm
        img_path = temp_dir / f"test_plot_output_{ind}_{col}_{freq}.pdf"
        x = ent_dm.plot_output(col, freq=freq)
        try:
            x.write_image(img_path)
        except Exception as exc:
            raise AssertionError("unable to write image") from exc
        else:
            assert True


def test_alt_total_var_mwh(
    mini_dm, fossil_specs, fossil_profiles, re_profiles, fossil_cost
):
    """Test impact of total_var_mwh.

    Test that changing total_var_mwh changes dispatch but not cost calculations.
    """
    fossil_cost = fossil_cost.copy()
    fossil_cost.loc[(3648, "4", "2018-01-01"), "total_var_mwh"] = 0.0
    re = np.array([5000.0, 5000.0, 0.0, 0.0])
    fossil_profiles.columns = fossil_specs.index
    fossil_profiles = zero_profiles_outside_operating_dates(
        fossil_profiles, fossil_specs.operating_date, fossil_specs.retirement_date
    )

    alt = DispatchModel(
        load_profile=fossil_profiles.sum(axis=1) - re_profiles @ re,
        dispatchable_profiles=fossil_profiles,
        dispatchable_cost=fossil_cost,
        dispatchable_specs=fossil_specs,
        storage_specs=pd.DataFrame(
            [
                (-1, "es", 5000, 4, 0.9, fossil_profiles.index.min()),
                (-2, "es", 2000, 8760, 0.5, fossil_profiles.index.min()),
            ],
            columns=[
                "plant_id_eia",
                "generator_id",
                "capacity_mw",
                "duration_hrs",
                "roundtrip_eff",
                "operating_date",
            ],
        ).set_index(["plant_id_eia", "generator_id"]),
    )()
    filter0 = pl.col("datetime").dt.year() == 2017
    assert (
        alt.redispatch.filter(filter0)
        .collect()
        .frame_equal(mini_dm.redispatch.filter(filter0).collect())
    )
    filter1 = (
        (pl.col("datetime").dt.year() == 2018)
        & (pl.col("plant_id_eia") == 3648)
        & (pl.col("generator_id") == "4")
    )
    assert np.all(
        alt.redispatch.filter(filter1).collect()
        >= mini_dm.redispatch.filter(filter1).collect()
    )
    assert (
        not alt.dispatchable_summary(by=None)
        .collect()
        .frame_equal(mini_dm.dispatchable_summary(by=None).collect())
    )
    alt.redispatch = mini_dm.redispatch
    assert (
        alt.dispatchable_summary(by=None)
        .collect()
        .frame_equal(mini_dm.dispatchable_summary(by=None).collect())
    )


def assert_dispatch_is_different(data, existing):
    """Assert that dispatch and given profiles are not the same."""
    self = DispatchModel(**data)
    self()
    if existing == "existing":
        cols = self.pl_dispatchable_specs.filter(pl.col("plant_id_eia") > 0).select(
            ["plant_id_eia", "generator_id"]
        )
    else:
        cols = self.pl_dispatchable_specs.filter(pl.col("plant_id_eia") < 0).select(
            ["plant_id_eia", "generator_id"]
        )
    comp = (
        self.redispatch.join(cols, on=["plant_id_eia", "generator_id"], how="inner")
        .select(pl.col("redispatch_mwh").round(0).alias("comp"))
        .collect()
        .frame_equal(
            self.pl_dispatchable_profiles.join(
                cols, on=["plant_id_eia", "generator_id"], how="inner"
            )
            .select(pl.col("historical_mwh").round(0).alias("comp"))
            .collect()
        )
    )
    assert not comp, f"dispatch of {existing} failed"


@pytest.mark.parametrize("existing", ["existing", "additions"], ids=idfn)
def test_redispatch_different(ent_redispatch, existing):
    """Test that redispatch and historical are not the same."""
    assert_dispatch_is_different(ent_redispatch, existing)


@pytest.mark.parametrize("existing", ["existing", "additions"], ids=idfn)
def test_fresh_different(ent_fresh, existing):
    """Test that dispatch and full capacity profiles are not the same."""
    assert_dispatch_is_different(ent_fresh, existing)


@pytest.mark.parametrize(
    ("gen", "col_set", "col", "expected"),
    [
        ((55380, "CTG1"), "redispatch_", "mwh", 0.0),
        ((55380, "CTG1"), "redispatch_", "cost_fuel", 0.0),
        ((55380, "CTG1"), "redispatch_", "cost_vom", 0.0),
        ((55380, "CTG1"), "redispatch_", "cost_startup", 0.0),
        ((55380, "CTG1"), "redispatch_", "cost_fom", 0.0),
        ((55380, "CTG1"), "historical_", "mwh", 1.0),
        ((55380, "CTG1"), "historical_", "cost_fuel", 1.0),
        ((55380, "CTG1"), "historical_", "cost_vom", 1.0),
        ((55380, "CTG1"), "historical_", "cost_startup", 1.0),
        ((55380, "CTG1"), "historical_", "cost_fom", 1.0),
        ((55380, "CTG2"), "redispatch_", "mwh", 1.0),
        ((55380, "CTG2"), "redispatch_", "cost_fuel", 1.0),
        ((55380, "CTG2"), "redispatch_", "cost_vom", 1.0),
        ((55380, "CTG2"), "redispatch_", "cost_startup", 1.0),
        ((55380, "CTG2"), "redispatch_", "cost_fom", 1.0),
        ((55380, "CTG2"), "historical_", "mwh", 1.0),
        ((55380, "CTG2"), "historical_", "cost_fuel", 1.0),
        ((55380, "CTG2"), "historical_", "cost_vom", 1.0),
        ((55380, "CTG2"), "historical_", "cost_startup", 1.0),
        ((55380, "CTG2"), "historical_", "cost_fom", 1.0),
    ],
    ids=idfn,
)
def test_dispatchable_exclude(
    ent_out_for_excl_test, ent_out_for_test, gen, col_set, col, expected
):
    """Test the effect of excluding a generator from dispatch."""
    filt = (pl.col("plant_id_eia") == gen[0]) & (pl.col("generator_id") == gen[1])
    col = col_set + col
    excl_test_item = ent_out_for_excl_test.filter(filt).select(col).collect().item()
    ent_out_for_test_item = ent_out_for_test.filter(filt).select(col).collect().item()
    if expected == 0.0:
        assert excl_test_item == expected
        assert excl_test_item != ent_out_for_test_item
    else:
        rel = None
        if gen == (55380, "CTG2") and col_set == "redispatch_":
            # we do not expect this generator's output in redispatch to be the same
            # whether CTG1 is excluded or not because excluding CTG1 affects other
            # generator dispatch
            rel = 1e-1
        assert excl_test_item > 1e4
        assert excl_test_item == pytest.approx(ent_out_for_test_item, rel=rel)


@pytest.mark.parametrize(
    ("gen", "col_set", "col", "expected"),
    [
        ((55380, "CTG1"), "redispatch_", "mwh", 25033526),
        ((55380, "CTG1"), "redispatch_", "cost_fuel", 359337060),
        ((55380, "CTG1"), "redispatch_", "cost_vom", 10919931),
        ((55380, "CTG1"), "redispatch_", "cost_startup", 79979045),
        ((55380, "CTG1"), "redispatch_", "cost_fom", 1689013.875),
        ((55380, "CTG1"), "historical_", "mwh", 1.0),
        ((55380, "CTG1"), "historical_", "cost_fuel", 1.0),
        ((55380, "CTG1"), "historical_", "cost_vom", 1.0),
        ((55380, "CTG1"), "historical_", "cost_startup", 1.0),
        ((55380, "CTG1"), "historical_", "cost_fom", 1.0),
        ((55380, "CTG2"), "redispatch_", "mwh", 1.0),
        ((55380, "CTG2"), "redispatch_", "cost_fuel", 1.0),
        ((55380, "CTG2"), "redispatch_", "cost_vom", 1.0),
        ((55380, "CTG2"), "redispatch_", "cost_startup", 1.0),
        ((55380, "CTG2"), "redispatch_", "cost_fom", 1.0),
        ((55380, "CTG2"), "historical_", "mwh", 1.0),
        ((55380, "CTG2"), "historical_", "cost_fuel", 1.0),
        ((55380, "CTG2"), "historical_", "cost_vom", 1.0),
        ((55380, "CTG2"), "historical_", "cost_startup", 1.0),
        ((55380, "CTG2"), "historical_", "cost_fom", 1.0),
    ],
    ids=idfn,
)
def test_dispatchable_no_limit(
    ent_out_for_no_limit_test, ent_out_for_test, gen, col_set, col, expected
):
    """Test the effect of excluding a generator from dispatch."""
    filt = (pl.col("plant_id_eia") == gen[0]) & (pl.col("generator_id") == gen[1])
    col = col_set + col
    no_limit_test_item = (
        ent_out_for_no_limit_test.filter(filt).select(col).collect().item()
    )
    test_item = ent_out_for_test.filter(filt).select(col).collect().item()
    if expected == 1.0:
        rel = None
        if gen == (55380, "CTG2") and col_set == "redispatch_":
            # we do not expect this generator's output in redispatch to be the same
            # whether CTG1 is excluded or not because excluding CTG1 affects other
            # generator dispatch
            rel = 1e-1
        assert no_limit_test_item > 1e4
        assert no_limit_test_item == pytest.approx(test_item, rel=rel)
    else:
        assert no_limit_test_item >= test_item, "no limit was not greater"
        assert no_limit_test_item == pytest.approx(expected)


@pytest.mark.parametrize(
    ("idx_to_change", "exception"),
    [
        ("dispatchable_specs", AssertionError),
        ("dispatchable_cost", AssertionError),
        ("re_plant_specs", pa.errors.SchemaError),
        ("dispatchable_profiles", AssertionError),
        ("re_profiles", AssertionError),
    ],
    ids=idfn,
)
def test_bad_index(ent_fresh, idx_to_change, exception):
    """Test that removing data from inputs raises errors."""
    if idx_to_change == "dispatchable_cost":
        to_drop = len(
            ent_fresh["dispatchable_cost"].index.get_level_values("datetime").unique()
        )
    else:
        to_drop = 2
    ent_fresh[idx_to_change] = ent_fresh[idx_to_change].iloc[:-to_drop, :]
    with pytest.raises(exception):
        DispatchModel(**ent_fresh)


def test_no_limit_late_operating_date(ent_redispatch):
    """Test that no_limit resources still respect their operating dates."""
    ent_redispatch["dispatchable_specs"] = ent_redispatch["dispatchable_specs"].assign(
        exclude=False, no_limit=False
    )
    ent_redispatch["dispatchable_specs"].loc[
        (55380, "CTG1"), ["operating_date", "retirement_date", "no_limit"]
    ] = (pd.to_datetime(2025, format="%Y"), pd.to_datetime(2030, format="%Y"), True)
    dm = DispatchModel(**ent_redispatch)()
    df = (
        dm.dispatchable_summary(by=None)
        .filter((pl.col("plant_id_eia") == 55380) & (pl.col("generator_id") == "CTG1"))
        .select(["datetime", "redispatch_mwh"])
        .collect()
    )
    assert np.all(
        df.filter(pl.col("datetime").dt.year() <= 2024).select("redispatch_mwh") == 0.0
    )
    assert np.all(
        df.filter(
            (pl.col("datetime").dt.year() >= 2025)
            & (pl.col("datetime").dt.year() <= 2029)
        ).select("redispatch_mwh")
        > 0
    )
    assert np.all(
        df.filter(pl.col("datetime").dt.year() >= 2032).select("redispatch_mwh") == 0.0
    )


@pytest.mark.parametrize(
    ("re_ids", "expected"),
    [
        ((-38,), "notempty"),
        ((-38, -43), AssertionError),
    ],
    ids=idfn,
)
def test_non_unique_storage_ids(ent_redispatch, re_ids, expected):
    """Test that non-unique storage IDs are allowed if not connected to RE."""
    ent_redispatch["storage_specs"] = pd.concat(
        [ent_redispatch["storage_specs"]]
        + [
            ent_redispatch["storage_specs"]
            .reset_index()
            .query("plant_id_eia == @x")
            .assign(generator_id="2")
            .set_index(["plant_id_eia", "generator_id"])
            for x in re_ids
        ]
    ).sort_index()
    if isinstance(expected, str):
        dm = DispatchModel(**ent_redispatch)()
        assert not dm.system_level_summary(freq="YS").collect().is_empty()
    else:
        with pytest.raises(expected):
            DispatchModel(**ent_redispatch)()


def test_system_level_summary_rollup(ent_redispatch):
    """Test that storage_rollup in system_level_summary works."""
    dm = DispatchModel(**ent_redispatch)()
    assert (
        not dm.system_level_summary(freq="YS", storage_rollup={"other": [-40, -39]})
        .collect()
        .is_empty()
    )


def test_bad_exclude_no_limit(ent_redispatch):
    """Test that both ``exclude`` and ``no_limit`` fails."""
    ent_redispatch["dispatchable_specs"] = ent_redispatch["dispatchable_specs"].assign(
        exclude=False, no_limit=False
    )
    ent_redispatch["dispatchable_specs"].loc[
        (55380, "CTG1"), ["exclude", "no_limit"]
    ] = True
    with pytest.raises(AssertionError):
        DispatchModel(**ent_redispatch)


def test_interconnect_mw(ent_fresh):
    """Test that interconnect_mw affects re_profiles and excess_re."""
    raw = DispatchModel(**ent_fresh)
    re_specs = (
        ent_fresh["re_plant_specs"]
        .copy()
        .assign(interconnect_mw=lambda x: x.capacity_mw)
    )
    new_max = re_specs.iloc[0, re_specs.columns.get_loc("interconnect_mw")] * 0.5
    re_specs.iloc[0, re_specs.columns.get_loc("interconnect_mw")] = new_max
    changed = DispatchModel(**(ent_fresh | {"re_plant_specs": re_specs}))
    assert changed.re_profiles_ac.iloc[:, 0].max() == new_max
    assert (
        raw.re_profiles_ac.iloc[:, 0].max()
        == re_specs.iloc[0, re_specs.columns.get_loc("capacity_mw")]
        * ent_fresh["re_profiles"].iloc[:, 0].max()
    )
    assert changed.re_excess.iloc[:, 0].sum() > 0.0
    assert raw.re_excess.iloc[:, 0].sum() == 0.0


def test_repr(ent_fresh):
    """Test repr."""
    self = DispatchModel(**ent_fresh)
    assert "n_dispatchable=24" in repr(self)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (
            "deficit",
            {
                "57": 0.0,
                "aps": 4.631323e-7,
                "epe": 3.2663479e-20,
                "fpc": 2.8729943e-6,
                "fpl": 1.512245e-4,
                "ldwp": 1.640462e-4,
                "miso": 2.684052e-6,
                "nyis": 3.62941e-6,
                "pac": 4.23462e-6,
                "pjm": 2.26282e-6,
                "psco": 5.255995e-5,
                "tva": 7.56522347e-6,
            },
        ),
        (
            "curtailment",
            {
                "57": 1.087665e-5,
                "aps": 9.35579e-6,
                "epe": 4.1907477e-7,
                "fpc": 9.07320749e-6,
                "fpl": 8.60302e-5,
                "ldwp": 4.794772e-2,
                "miso": 7.311725e-5,
                "nyis": 4.40259e-10,
                "pac": 6.05654e-6,
                "pjm": 3.74322e-5,
                "psco": 2.117926e-4,
                "tva": 5.7364898e-7,
            },
        ),
    ],
    ids=idfn,
)
def test_deficit_curtailment(ba_dm, kind, expected):
    """Test that deficit and curtailment have not regressed.

    The goal is to have a test that can be run as we make changes to the engine to check
    that key performance metrics are not getting worse.
    """
    name, dm = ba_dm
    actual = dm.system_data.select(kind).sum().collect().item() / dm.load_profile.sum()
    expected = expected[name]
    assert actual <= expected
    if not np.isclose(actual, expected, rtol=1e-6, atol=0):
        logger.warning("%s %s: expected=%e, actual=%e", name, kind, expected, actual)


def sweep(test_dir):
    """Run dispatch on many BAs with different coefficient settings."""
    bas = [
        "57",
        "193",
        "aps",
        "caiso",
        "duke",
        "epe",
        "erco",
        "fpc",
        "fpl",
        "ldwp",
        "miso",
        "nyis",
        "pac",
        "pjm",
        "psco",
        "tva",
    ]
    coeffs = np.arange(0.0, 2.0, 0.1)
    df = pd.DataFrame(
        index=pd.MultiIndex.from_product([bas, coeffs], names=["ba", "coeff"]),
        columns=["deficit", "curtailment"],
    )
    for ba in bas:
        try:
            data = dict(DataZip(test_dir / f"data/ba/{ba}.zip").items())
            dm = DispatchModel(**data)
            for coeff in coeffs:
                dm(dynamic_reserve_coeff=coeff)
                df.loc[(ba, coeff), "deficit"] = (
                    dm.system_data.deficit.sum() / dm.load_profile.sum()
                )
                df.loc[(ba, coeff), "curtailment"] = (
                    dm.system_data.curtailment.sum() / dm.load_profile.sum()
                )
        except Exception as exc:
            logger.warning("%s, %r", ba, exc)
    return df


@pytest.mark.skip(reason="exploration not testing")
def test_sweep(test_dir):
    """Explore impact of different coefficients on deficit and curtailment."""
    import plotly.express as px

    df = sweep(test_dir)
    f2 = (
        px.scatter(
            df.reset_index()
            .dropna()
            .melt(id_vars=["ba", "coeff"], value_vars=["deficit", "curtailment"]),
            x="coeff",
            y="value",
            color="variable",
            facet_col="ba",
            facet_col_wrap=3,
            facet_col_spacing=0.1,
            height=900,
        )
        .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        .update_yaxes(
            matches=None, showticklabels=True, tickformat=".1e", rangemode="tozero"
        )
    )
    f2.write_image(test_dir / "data/def.pdf")
    raise AssertionError()


@pytest.mark.skip(reason="for debugging only")
def test_ent(ent_fresh):
    """Harness for testing dispatch."""
    self = DispatchModel(**ent_fresh, jit=False)
    self()
    raise AssertionError


@pytest.mark.skip(reason="for debugging only")
def test_file(ent_fresh):
    """Harness for testing dispatch."""
    from pathlib import Path

    self = DispatchModel.from_file(Path.home() / "Documents/gp_dm.zip")
    self.plot_year(2008)

    raise AssertionError
