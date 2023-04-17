"""Where dispatch tests will go."""
from io import BytesIO

import numpy as np
import pandas as pd
import pandera as pa
import pytest
from dispatch import DispatchModel
from dispatch.helpers import zero_profiles_outside_operating_dates
from etoolbox.datazip import DataZip
from etoolbox.utils.testing import idfn


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
    "attr, expected",
    [
        ("redispatch", {"f": 449456776, "r": 383683946}),
        ("storage_dispatch", {"f": 269892073, "r": 221453923}),
        ("system_data", {"f": 112821177, "r": 95078855}),
        ("starts", {"f": 77109, "r": 54628}),
    ],
    ids=idfn,
)
def test_redispatch_total(ent_dm, attr, expected):
    """High-level test that results have not changed."""
    ind, ent_dm = ent_dm
    assert getattr(ent_dm, attr).sum().sum() == pytest.approx(expected[ind])


@pytest.mark.parametrize("comparison", [None, "load_max"], ids=idfn)
def test_low_lost_load(mini_dm, comparison):
    """Dummy test that there isn't much lost load."""
    if comparison is None:
        assert (mini_dm.lost_load() / mini_dm.lost_load().sum()).iloc[0] > 0.998
    else:
        assert (
            mini_dm.lost_load(mini_dm.load_profile.max())
            / mini_dm.lost_load(mini_dm.load_profile.max()).sum()
        ).iloc[0] > 0.998


def test_marginal_cost(mini_dm):
    """Setup for testing cost and grouper methods."""
    x = mini_dm.grouper(mini_dm.historical_cost, "technology_description")
    assert not x.empty


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

    def test_operations_summary(self, mini_dm):
        """Setup for testing cost and grouper methods."""
        x = mini_dm.dispatchable_summary(by=None)
        assert x.notna().all().all()

    def test_storage_summary(self, mini_dm):
        """Setup for testing cost and grouper methods."""
        x = mini_dm.storage_summary(by=None)
        assert not x.empty

    @pytest.mark.parametrize(
        "func, args, drop_cols, expected",
        [
            (
                "dispatchable_summary",
                {"by": None, "augment": True},
                (
                    "utility_id_eia",
                    "final_respondent_id",
                    "retirement_date",
                    "final_ba_code",
                    "respondent_name",
                    "balancing_authority_code_eia",
                    "plant_name_eia",
                    "prime_mover_code",
                    "state",
                    "latitude",
                    "longitude",
                ),
                "notna",
            ),
            ("dispatchable_summary", {"by": None}, (), "notna"),
            (
                "re_summary",
                {"by": None},
                ("owned_pct", "retirement_date", "fom_per_kw"),
                "notna",
            ),
            ("system_level_summary", {}, (), "notna"),
            ("load_summary", {}, (), "notna"),
            ("storage_durations", {}, (), "notna"),
            ("storage_capacity", {}, (), "notna"),
            ("hourly_data_check", {}, (), "notna"),
            ("dc_charge", {}, (), "notna"),
            pytest.param("full_output", {}, (), "notna", marks=pytest.mark.xfail),
            (
                "dispatchable_summary",
                {"by": None, "augment": True},
                (
                    "utility_id_eia",
                    "final_respondent_id",
                    "retirement_date",
                    "final_ba_code",
                    "respondent_name",
                    "balancing_authority_code_eia",
                    "plant_name_eia",
                    "prime_mover_code",
                ),
                "notempty",
            ),
            ("dispatchable_summary", {"by": None}, (), "notempty"),
            ("re_summary", {"by": None}, ("owned_pct", "retirement_date"), "notempty"),
            ("system_level_summary", {}, (), "notempty"),
            ("load_summary", {}, (), "notempty"),
            ("storage_durations", {}, (), "notempty"),
            ("storage_capacity", {}, (), "notempty"),
            ("hourly_data_check", {}, (), "notempty"),
            ("dc_charge", {}, (), "notempty"),
            ("full_output", {}, (), "notempty"),
        ],
        ids=idfn,
    )
    def test_outputs_parametric(self, ent_dm, func, args, drop_cols, expected):
        """Test that outputs are not empty or do not have unexpected nans."""
        ind, ent_dm = ent_dm
        df = getattr(ent_dm, func)(**args)
        df = df[[c for c in df if c not in drop_cols]]
        if expected == "notna":
            assert df.notna().all().all()
        elif expected == "notempty":
            assert not df.empty
        else:
            raise AssertionError


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
        "col, freq",
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

    Test that changing total_var_mwh changes dispatch but not cost
    calculations.
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

    assert (
        alt.redispatch.loc["2017", :].compare(mini_dm.redispatch.loc["2017", :]).empty
    )
    assert np.all(
        alt.redispatch.loc["2018", (3648, "4")]
        >= mini_dm.redispatch.loc["2018", (3648, "4")]
    )
    assert (
        not alt.dispatchable_summary(by=None)
        .compare(mini_dm.dispatchable_summary(by=None))
        .empty
    )
    alt.redispatch = mini_dm.redispatch
    assert (
        alt.dispatchable_summary(by=None)
        .compare(mini_dm.dispatchable_summary(by=None))
        .empty
    )


def assert_dispatch_is_different(data, existing):
    """Assert that dispatch and given profiles are not the same."""
    self = DispatchModel(**data)
    self()
    if existing == "existing":
        cols = [tup for tup in self.dispatchable_profiles.columns if tup[0] > 0]
    else:
        cols = [tup for tup in self.dispatchable_profiles.columns if tup[0] < 0]
    comp = (
        self.redispatch.loc[:, cols]
        .round(0)
        .compare(self.dispatchable_profiles.loc[:, cols].round(0))
    )
    assert not comp.empty, f"dispatch of {existing} failed"


@pytest.mark.parametrize("existing", ["existing", "additions"], ids=idfn)
def test_redispatch_different(ent_redispatch, existing):
    """Test that redispatch and historical are not the same."""
    assert_dispatch_is_different(ent_redispatch, existing)


@pytest.mark.parametrize("existing", ["existing", "additions"], ids=idfn)
def test_fresh_different(ent_fresh, existing):
    """Test that dispatch and full capacity profiles are not the same."""
    assert_dispatch_is_different(ent_fresh, existing)


@pytest.mark.parametrize(
    "gen, col_set, col, expected",
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
    if expected == 0.0:
        assert ent_out_for_excl_test.loc[gen, col_set + col] == expected
        assert (
            ent_out_for_excl_test.loc[gen, col_set + col]
            != ent_out_for_test.loc[gen, col_set + col]
        )
    else:
        rel = None
        if gen == (55380, "CTG2") and col_set == "redispatch_":
            # we do not expect this generator's output in redispatch to be the same
            # whether CTG1 is excluded or not because excluding CTG1 affects other
            # generator dispatch
            rel = 1e-1
        assert ent_out_for_excl_test.loc[gen, col_set + col] > 1e4
        assert ent_out_for_excl_test.loc[gen, col_set + col] == pytest.approx(
            ent_out_for_test.loc[gen, col_set + col],
            rel=rel,
        )


@pytest.mark.parametrize(
    "gen, col_set, col, expected",
    [
        ((55380, "CTG1"), "redispatch_", "mwh", 25367968),
        ((55380, "CTG1"), "redispatch_", "cost_fuel", 364137734),
        ((55380, "CTG1"), "redispatch_", "cost_vom", 11065819),
        ((55380, "CTG1"), "redispatch_", "cost_startup", 64935667),
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
    if expected == 1.0:
        rel = None
        if gen == (55380, "CTG2") and col_set == "redispatch_":
            # we do not expect this generator's output in redispatch to be the same
            # whether CTG1 is excluded or not because excluding CTG1 affects other
            # generator dispatch
            rel = 1e-1
        assert ent_out_for_no_limit_test.loc[gen, col_set + col] > 1e4
        assert ent_out_for_no_limit_test.loc[gen, col_set + col] == pytest.approx(
            ent_out_for_test.loc[gen, col_set + col],
            rel=rel,
        )
    else:
        assert (
            ent_out_for_no_limit_test.loc[gen, col_set + col]
            >= ent_out_for_test.loc[gen, col_set + col]
        ), "no limit was not greater"
        assert ent_out_for_no_limit_test.loc[gen, col_set + col] == pytest.approx(
            expected
        )


@pytest.mark.parametrize(
    "idx_to_change, exception",
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
