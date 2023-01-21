"""Where dispatch tests will go."""
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from dispatch import DispatchModel
from dispatch.helpers import apply_op_ret_date, idfn


def test_new(fossil_profiles, re_profiles, fossil_specs, fossil_cost):
    """Dummy test to quiet pytest."""
    fossil_specs.iloc[
        0, fossil_specs.columns.get_loc("retirement_date")
    ] = fossil_profiles.index.max() - pd.Timedelta(weeks=15)
    self = DispatchModel.from_fresh(
        net_load_profile=fossil_profiles.sum(axis=1),
        dispatchable_specs=fossil_specs,
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
    self()
    assert self


def test_new_no_dates(fossil_profiles, re_profiles, fossil_specs, fossil_cost):
    """Dummy test to quiet pytest."""
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


def test_new_with_dates(fossil_profiles, re_profiles, fossil_specs, fossil_cost):
    """Test operating and retirement dates for fossil and storage."""
    fossil_specs.iloc[
        0, fossil_specs.columns.get_loc("retirement_date")
    ] = fossil_profiles.index.max() - pd.Timedelta(weeks=15)
    fossil_specs.loc[8066, "retirement_date"] = pd.Timestamp(
        year=2018, month=12, day=31
    )
    self = DispatchModel.from_fresh(
        net_load_profile=fossil_profiles.sum(axis=1),
        dispatchable_specs=fossil_specs,
        dispatchable_cost=fossil_cost,
        storage_specs=pd.DataFrame(
            [
                (5000, 4, 0.9, pd.Timestamp(year=2016, month=1, day=1)),
                (2000, 8760, 0.5, pd.Timestamp(year=2019, month=1, day=1)),
            ],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff", "operating_date"],
            index=pd.MultiIndex.from_tuples(
                [(-99, "es"), (-98, "es")], names=["plant_id_eia", "generator_id"]
            ),
        ),
        jit=True,
    )
    self()
    assert self


def test_low_lost_load(mini_dm):
    """Dummy test to quiet pytest."""
    assert (mini_dm.lost_load() / mini_dm.lost_load().sum()).iloc[0] > 0.998


def test_write_and_read(
    fossil_profiles, re_profiles, fossil_specs, temp_dir, fossil_cost
):
    """Test that DispatchModel can be written and read."""
    fossil_profiles.columns = fossil_specs.index
    dm = DispatchModel.from_patio(
        fossil_profiles.sum(axis=1)
        - re_profiles @ np.array([5000.0, 5000.0, 0.0, 0.0]),
        dispatchable_profiles=fossil_profiles,
        cost_data=fossil_cost,
        plant_data=fossil_specs,
        storage_specs=pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
    )
    file = temp_dir / "test_write_and_read.zip"
    try:
        dm.to_file(file)
        x = DispatchModel.from_file(file)
        x()
        x.to_file(file, clobber=True, include_output=False)
    except Exception as exc:
        raise AssertionError(f"{exc!r}") from exc
    else:
        assert True


def test_write_and_read_full(temp_dir, ent_fresh):
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


def test_write_and_read_bytes(ent_fresh):
    """Test that DispatchModel can be written and read."""
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


def test_marginal_cost(mini_dm):
    """Setup for testing cost and grouper methods."""
    x = mini_dm.grouper(mini_dm.historical_cost, "technology_description")
    assert not x.empty


def test_alt_total_var_mwh(
    mini_dm, fossil_specs, fossil_profiles, re_profiles, fossil_cost
):
    """Test that changing total_var_mwh changes dispatch but not cost calculations."""
    fossil_cost = fossil_cost.copy()
    fossil_cost.loc[(3648, "4", "2018-01-01"), "total_var_mwh"] = 0.0
    re = np.array([5000.0, 5000.0, 0.0, 0.0])
    fossil_profiles.columns = fossil_specs.index
    fossil_profiles = apply_op_ret_date(
        fossil_profiles, fossil_specs.operating_date, fossil_specs.retirement_date
    )
    alt = DispatchModel.from_patio(
        fossil_profiles.sum(axis=1) - re_profiles @ re,
        dispatchable_profiles=fossil_profiles,
        cost_data=fossil_cost,
        plant_data=fossil_specs,
        storage_specs=pd.DataFrame(
            [(5000, 4, 0.9), (2000, 8760, 0.5)],
            columns=["capacity_mw", "duration_hrs", "roundtrip_eff"],
        ),
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


def test_operations_summary(mini_dm):
    """Setup for testing cost and grouper methods."""
    x = mini_dm.dispatchable_summary(by=None)
    assert x.notna().all().all()


def test_storage_summary(mini_dm):
    """Setup for testing cost and grouper methods."""
    x = mini_dm.storage_summary(by=None)
    assert not x.empty


out_params = [
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
]


@pytest.mark.parametrize(
    "func, args, drop_cols, expected",
    out_params,
    ids=idfn,
)
def test_outputs_parametric(ent_dm, func, args, drop_cols, expected):
    """Test that outputs are not empty or do not have unexpected nans."""
    ind, ent_dm = ent_dm
    df = getattr(ent_dm, func)(**args)
    df = df[[c for c in df if c not in drop_cols]]
    if expected == "notna":
        assert df.notna().all().all()
    elif expected == "notempty":
        assert not df.empty
    else:
        assert False


@pytest.mark.parametrize(
    "attr, expected",
    [
        ("redispatch", {"f": 421292628, "r": 389777545}),
        ("storage_dispatch", {"f": 152460988, "r": 144263839}),
        ("system_data", {"f": 81154038, "r": 99539021}),
        ("starts", {"f": 113795, "r": 68253}),
    ],
    ids=idfn,
)
def test_redispatch_total(ent_dm, attr, expected):
    """High-level test that results have not changed."""
    ind, ent_dm = ent_dm
    assert getattr(ent_dm, attr).sum().sum() == pytest.approx(expected[ind])


def test_plot_all_years(ent_dm, temp_dir):
    """Test that outputs are not empty or do not have unexpected nans."""
    ind, ent_dm = ent_dm
    y = ent_dm.plot_all_years()
    img_path = temp_dir / "test_plot_all_years.pdf"
    try:
        y.write_image(str(img_path))
    except Exception as exc:
        raise AssertionError("unable to write image") from exc


@pytest.mark.parametrize("freq", ["D", "H"], ids=idfn)
def test_plotting(mini_dm, temp_dir, freq):
    """Testing plotting function."""
    y = mini_dm.plot_year(2015, freq=freq)
    img_path = temp_dir / f"test_plotting_{freq}.pdf"
    try:
        y.write_image(str(img_path))
    except Exception as exc:
        raise AssertionError("unable to write image") from exc


def test_plot_detail_ent(ent_fresh, temp_dir):
    """Testing plotting function."""
    img_path = temp_dir / "test_plot_detail_ent.pdf"
    self = DispatchModel(**ent_fresh)()
    x = self.plot_period("2034-01-01", "2034-01-05", compare_hist=True)
    try:
        x.write_image(img_path)
    except Exception as exc:
        raise AssertionError("unable to write image") from exc
    else:
        assert True


def test_plot_period_comp(ent_redispatch, temp_dir):
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


def test_plot_year_ent(ent_fresh, temp_dir):
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
def test_plot_output(ent_dm, temp_dir, col, freq):
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


def test_repr(ent_fresh):
    """Test repr."""
    self = DispatchModel(**ent_fresh)
    assert "n_dispatchable=24" in repr(self)


@pytest.mark.parametrize("existing", ["existing", "additions"], ids=idfn)
def test_redispatch_different(ent_redispatch, existing):
    """Test that redispatch and historical are not the same."""
    self = DispatchModel(**ent_redispatch)
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
    assert not comp.empty


@pytest.mark.parametrize("existing", ["existing", "additions"], ids=idfn)
def test_fresh_different(ent_fresh, existing):
    """Test that dispatch and full capacity profiles (fresh) are not the same."""
    self = DispatchModel(**ent_fresh)
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


def test_interconnect_mw(ent_fresh):
    """Test that interconnect_mw produces expected re_profiles and excess_re."""
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


@pytest.mark.skip(reason="for debugging only")
def test_ent(ent_fresh):
    """Harness for testing dispatch."""
    self = DispatchModel(**ent_fresh, jit=False)
    self()
    assert False
