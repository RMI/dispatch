"""A template repository for a Python package created by Catalyst Cooperative."""
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

from dispatch.engine import dispatch_engine, dispatch_engine_py
from dispatch.helpers import DataZip, apply_op_ret_date, copy_profile
from dispatch.model import DispatchModel

__all__ = [
    "DispatchModel",
    "dispatch_engine_py",
    "dispatch_engine",
    "copy_profile",
    "apply_op_ret_date",
    "DataZip",
    "__version__",
]

__projecturl__ = "https://github.com/rmi-electricity/dispatch"
__downloadurl__ = "https://github.com/rmi-electricity/dispatch"

# Create a root logger for use anywhere within the package.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
