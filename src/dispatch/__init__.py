"""RMI's electricity dispatch model."""

import logging

__author__ = "RMI"
__contact__ = "aengel@rmi.org"
__maintainer__ = "Alex Engel"
__license__ = "BSD 3-Clause License"
__maintainer_email__ = "aengel@rmi.org"
__docformat__ = "restructuredtext en"
__description__ = "A simple and efficient dispatch model."

try:
    from dispatch._version import version as __version__
except ImportError:
    __version__ = "unknown"

from dispatch.helpers import (
    apply_op_ret_date,
    copy_profile,
    zero_profiles_outside_operating_dates,
)
from dispatch.model import DispatchModel

__all__ = [
    "DispatchModel",
    "__version__",
    "apply_op_ret_date",
    "copy_profile",
    "zero_profiles_outside_operating_dates",
]

__projecturl__ = "https://github.com/rmi/dispatch"
__downloadurl__ = "https://github.com/rmi/dispatch"

# Create a root logger for use anywhere within the package.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
