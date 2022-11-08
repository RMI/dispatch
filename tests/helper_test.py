"""Tests for :mod:`dispatch.helpers`."""

import pandas as pd

from dispatch.helpers import copy_profile, dispatch_key


def test_copy_profile(ent_fresh):
    """Test copy_profile."""
    load = ent_fresh["load_profile"]
    yr_range = range(2050, 2055)
    out = copy_profile(load.loc["2025"], yr_range)
    assert tuple(out.index.year.unique()) == tuple(yr_range)


def test_dispatch_key_series():
    """Test dispatch key func on :class:`pandas.Series`."""
    idx = (
        pd.Series(["May", "April", "Solar", "Gas CC", "March"])
        .sort_values(key=dispatch_key)
        .reset_index(drop=True)
    )
    pd.testing.assert_series_equal(
        idx, pd.Series(["Gas CC", "Solar", "March", "April", "May"])
    )


def test_dispatch_key_idx():
    """Test dispatch key func on :class:`pandas.Index`."""
    idx = pd.Index(["May", "April", "Solar", "Gas CC", "March"]).sort_values(
        key=dispatch_key
    )
    pd.testing.assert_index_equal(
        idx, pd.Index(["Gas CC", "Solar", "March", "April", "May"])
    )


def test_dispatch_key_list():
    """Test dispatch key func on :class:`list`."""
    idx = sorted(["May", "April", "Solar", "Gas CC", "March"], key=dispatch_key)
    assert idx == ["Gas CC", "Solar", "March", "April", "May"]


# def test_dfs_to_from_zip(test_dir):
#     """Dfs are same after being written and read back."""
#     df_dict = {
#         "a": pd.DataFrame(
#             [[0, 1], [2, 3]],
#             columns=pd.MultiIndex.from_tuples([(0, "a"), (1, "b")]),
#         ),
#         "b": pd.Series([1, 2, 3, 4]),
#     }
#     try:
#         DataZip.dfs_to_zip(
#             test_dir / "df_test",
#             df_dict,
#         )
#         df_load = DataZip.dfs_from_zip(test_dir / "df_test")
#         for a, b in zip(df_dict.values(), df_load.values()):
#             assert a.compare(b).empty
#     except Exception as exc:
#         raise AssertionError("Something broke") from exc
#     finally:
#         (test_dir / "df_test.zip").unlink(missing_ok=True)
#
#
# def test_datazip(test_dir):
#     """Test :class:`.DataZip`."""
#     df_dict = {
#         "a": pd.DataFrame(
#             [[0, 1], [2, 3]],
#             columns=pd.MultiIndex.from_tuples([(0, "a"), (1, "b")]),
#         ),
#         "b": pd.Series([1, 2, 3, 4]),
#     }
#     try:
#         with DataZip(test_dir / "obj.zip", "w") as z:
#             z.writed("a", df_dict["a"])
#             z.writed("b", df_dict["b"])
#             z.writed("c", {1: 3, "3": "fifteen", 5: (0, 1)})
#             z.writed("d", "hello world")
#             with pytest.raises(FileExistsError):
#                 z.writed("c", {1: 3, "3": "fifteen", 5: (0, 1)})
#             with pytest.raises(FileExistsError):
#                 z.writed("b", df_dict["b"])
#     except Exception as exc:
#         raise AssertionError("Something broke") from exc
#     else:
#         with DataZip(test_dir / "obj.zip", "r") as z1:
#             for n, t in (("a", pd.DataFrame), ("b", pd.Series), ("c", dict), ('d', str)):
#                 assert isinstance(z1.read(n), t)
#             assert "a" in z1.bad_cols
#     finally:
#         (test_dir / "obj.zip").unlink(missing_ok=True)
#
#
# def test_datazip_w(test_dir):
#     """Test writing to existing :class:`.DataZip`."""
#     df_dict = {
#         "a": pd.DataFrame(
#             [[0, 1], [2, 3]],
#             columns=pd.MultiIndex.from_tuples([(0, "a"), (1, "b")]),
#         ),
#     }
#     try:
#         with DataZip(test_dir / "obj.zip", "w") as z0:
#             z0.writed("a", df_dict["a"])
#     except Exception as exc:
#         raise AssertionError("Something broke") from exc
#     else:
#         with DataZip(test_dir / "obj.zip", "r") as z1:
#             assert "a" in z1.bad_cols
#         with pytest.raises(ValueError):
#             with DataZip(test_dir / "obj.zip", "a") as z2a:
#                 z2a.namelist()
#         with pytest.raises(ValueError):
#             with DataZip(test_dir / "obj.zip", "x") as z2x:
#                 z2x.namelist()
#         with pytest.raises(FileExistsError):
#             with DataZip(test_dir / "obj.zip", "w") as z3:
#                 z3.namelist()
#     finally:
#         (test_dir / "obj.zip").unlink(missing_ok=True)
#
#
# def test_dm_in_datazip(test_dir, ent_fresh):
#     """Test creating a datazip with a :class:`.DispatchModel` in it."""
#     dm = DispatchModel(**ent_fresh)
#     df = pd.DataFrame(
#         [[0, 1], [2, 3]],
#         columns=pd.MultiIndex.from_tuples([(0, "a"), (1, "b")]),
#     )
#
#     try:
#         with DataZip(test_dir / "obj.zip", "w") as z0:
#             z0.writed("df", df)
#             z0.writed("dm", dm)
#         with DataZip(test_dir / "obj.zip", "r") as self:
#             dm1 = self.read("dm")
#             assert isinstance(dm1, DispatchModel)
#     finally:
#         (test_dir / "obj.zip").unlink(missing_ok=True)
#
#
# def test_namedtuple_in_datazip(test_dir, ent_fresh):
#     """Test creating a datazip with a :class:`.DispatchModel` in it."""
#     dm = DispatchModel(**ent_fresh)
#     df = pd.DataFrame(
#         [[0, 1], [2, 3]],
#         columns=pd.MultiIndex.from_tuples([(0, "a"), (1, "b")]),
#     )
#     om = ObjMeta("this", "that")
#
#     try:
#         with DataZip(test_dir / "obj.zip", "w") as z0:
#             z0.writed("df", df)
#             z0.writed("dm", dm)
#             z0.writed("om", om)
#             z0.writed("tup", (1, 2, (1, 2, 3)))
#             z0.writed("l", [om, om])
#             z0.writed("d", {"a": (1, 2, 3), "b": [32, 45], 3: 4})
#         with DataZip(test_dir / "obj.zip", "r") as self:
#             om1 = self.read("om")
#             self.read("tup")
#             assert isinstance(om1, ObjMeta)
#     finally:
#         (test_dir / "obj.zip").unlink(missing_ok=True)
