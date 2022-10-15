"""Tests for :mod:`dispatch.helpers`."""

import pandas as pd
import pytest

from dispatch.helpers import DataZip, copy_profile


def test_copy_profile(ent_fresh):
    """Test copy_profile."""
    load = ent_fresh["load_profile"]
    yr_range = range(2050, 2055)
    out = copy_profile(load.loc["2025"], yr_range)
    assert tuple(out.index.year.unique()) == tuple(yr_range)


def test_dfs_to_from_zip(test_dir):
    """Dfs are same after being written and read back."""
    df_dict = {
        "a": pd.DataFrame(
            [[0, 1], [2, 3]],
            columns=pd.MultiIndex.from_tuples([(0, "a"), (1, "b")]),
        ),
        "b": pd.Series([1, 2, 3, 4]),
    }
    try:
        DataZip.dfs_to_zip(
            test_dir / "df_test",
            df_dict,
        )
        df_load = DataZip.dfs_from_zip(test_dir / "df_test")
        for a, b in zip(df_dict.values(), df_load.values()):
            assert a.compare(b).empty
    except Exception as exc:
        raise AssertionError("Something broke") from exc
    finally:
        (test_dir / "df_test.zip").unlink(missing_ok=True)


def test_datazip(test_dir):
    """Test :class:`.DataZip`."""
    df_dict = {
        "a": pd.DataFrame(
            [[0, 1], [2, 3]],
            columns=pd.MultiIndex.from_tuples([(0, "a"), (1, "b")]),
        ),
        "b": pd.Series([1, 2, 3, 4]),
    }
    try:
        with DataZip(test_dir / "obj.zip", "w") as z:
            z.writed("a", df_dict["a"])
            z.writed("b", df_dict["b"])
            z.writed("c", {1: 3, "3": "fifteen", 5: (0, 1)})
            with pytest.raises(FileExistsError):
                z.writed("c", {1: 3, "3": "fifteen", 5: (0, 1)})
            with pytest.raises(FileExistsError):
                z.writed("b", df_dict["b"])

        with DataZip(test_dir / "obj.zip", "r") as z:
            z.namelist()
    finally:
        (test_dir / "obj.zip").unlink(missing_ok=True)
