"""Test the PUDL console scripts from within PyTest."""

import pkg_resources
import pytest

# Obtain a list of all deployed entry point scripts to test:
ENTRY_POINTS = [
    ep.name
    for ep in pkg_resources.iter_entry_points("console_scripts")
    if ep.module_name.startswith("cheshire")
]


@pytest.mark.parametrize("ep", ENTRY_POINTS)
@pytest.mark.script_launch_mode("inprocess")
def test_pudl_scripts(script_runner, ep: str) -> None:  # type: ignore
    """Run each deployed console script with --help as a basic test.

    The script_runner fixture is provided by the pytest-console-scripts plugin.
    """
    ret = script_runner.run(ep, "--help", print_result=False)
    assert ret.success  # nosec: B101


@pytest.mark.parametrize(
    "alpha,beta",
    [
        ("2", "2"),
        pytest.param("a", "2", marks=pytest.mark.xfail),
        pytest.param("2", "b", marks=pytest.mark.xfail),
        pytest.param("a", "b", marks=pytest.mark.xfail),
    ],
)
@pytest.mark.script_launch_mode("inprocess")
def test_winston_args(script_runner, alpha: str, beta: str) -> None:  # type: ignore
    """Try running the script with bad inputs."""
    ret = script_runner.run("winston", "--alpha", alpha, "--beta", beta)
    assert ret.success  # nosec: B101
