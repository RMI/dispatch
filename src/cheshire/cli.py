"""A skeleton of a command line interface to be deployed as an entry point script.

It takes two numbers and does something to them, printing out the result.

"""
from __future__ import annotations

import argparse
import logging
import sys

from cheshire.dummy import do_something

# This is the module-level logger, for any logs
logger = logging.getLogger(__name__)


def parse_command_line(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments. See the -h option for details.

    Args:
        argv (str): Command line arguments, including caller filename.

    Returns:
        dict: Dictionary of command line arguments and their parsed values.

    """

    def formatter(prog) -> argparse.HelpFormatter:  # type: ignore
        """This is a hack to create HelpFormatter with a particular width."""
        return argparse.HelpFormatter(prog, width=88)

    # Use the module-level docstring as the script's description in the help message.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=formatter)

    parser.add_argument(
        "-a",
        "--alpha",
        type=int,
        help="An integer to do something to. Defaults to two (2).",
        default=2,
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=int,
        help="Another integer to do something to. Defaults to two (2).",
        default=2,
    )

    arguments = parser.parse_args(argv[1:])
    return arguments


def main() -> int:
    """Demonstrate a really basic command line interface (CLI) that takes arguments."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s",
        level=logging.INFO,
    )

    args = parse_command_line(sys.argv)
    caligula = do_something(a=args.alpha, b=args.beta)
    print(
        "If you are a man Winston, you are the last man: "
        f"{args.alpha} + {args.beta} = {caligula}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
