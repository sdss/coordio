# encoding: utf-8
# isort:skip_file

import os
import warnings

from sdsstools import get_config, get_logger, get_package_version

from .exceptions import CoordIOError


# PYPI package name
NAME = 'sdss-coordio'

# Loads config. For coordio we don't allow overridding with user config files.
config = get_config('coordio', allow_user=False)

# Inits the logging system as NAME.
log = get_logger(NAME)

# Package name should be PYPI package name
__version__ = get_package_version(path=__file__, package_name=NAME)


from .exceptions import CoordIOUserWarning
from .sofa_bindings import SOFA

try:
    sofa = SOFA()
except OSError:
    warnings.warn('Cannot load libsofa. '
                  'Most functionality will not be available.',
                  CoordIOUserWarning)
    sofa = None


from .defaults import calibration
from .exceptions import *
from .iers import IERS
from .site import Site
from .sky import *
from .telescope import *
from .time import Time
from .wok import Wok
from .tangent import Tangent
from .guide import Guide
from .positioner import PositionerBoss, PositionerApogee, PositionerMetrology
from .utils import *
from .fitData import TransRotScaleModel
