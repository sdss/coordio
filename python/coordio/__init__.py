# encoding: utf-8

import warnings

from sdsstools import get_config, get_logger, get_package_version


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
    warnings.warn('Cannot load libsofa.', CoordIOUserWarning)
    sofa = None


from .iers import IERS
from .time import Time
