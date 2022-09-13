"""A template repository for a Python package created by Catalyst Cooperative."""
import logging

import pkg_resources

from dispatch.engine import dispatch_engine, dispatch_engine_compiled
from dispatch.model import DispatchModel

__all__ = ["DispatchModel", "dispatch_engine", "dispatch_engine_compiled"]

__author__ = "RMI"
__contact__ = "aengel@rmi.org"
__maintainer__ = "Alex Engel"
__license__ = "BSD 3-Clause License"
__maintainer_email__ = "aengel@rmi.org"
__version__ = pkg_resources.get_distribution("rmi.dispatch").version
__docformat__ = "restructuredtext en"
__description__ = "A simple and efficient dispatch model."

__projecturl__ = "https://github.com/rmi-electricity/dispatch"
__downloadurl__ = "https://github.com/rmi-electricity/dispatch"

# Create a root logger for use anywhere within the package.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
