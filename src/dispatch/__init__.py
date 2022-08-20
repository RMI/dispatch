"""A template repository for a Python package created by Catalyst Cooperative."""
import logging

import pkg_resources

from patio.dispatch.dispatch import *

__all__ = []
__all__.extend(dispatch.__all__)

__author__ = "RMI"
__contact__ = "aengel@rmi.org"
__maintainer__ = "Alex Engel"
# __license__ = "MIT License"
__maintainer_email__ = "aengel@rmi.org"
__version__ = pkg_resources.get_distribution("dispatch").version
__docformat__ = "restructuredtext en"
__description__ = "A simple and efficient dispatch model."

__projecturl__ = "https://github.com/rmi-electricity/dispatch"
__downloadurl__ = "https://github.com/rmi-electricity/dispatch"

# Create a root logger for use anywhere within the package.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
