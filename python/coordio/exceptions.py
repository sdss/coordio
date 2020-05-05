# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-12-05 12:01:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-12-05 12:19:32

from __future__ import print_function, division, absolute_import


class CoordioError(Exception):
    """A custom core Coordio exception"""

    def __init__(self, message=None):

        message = 'There has been an error' \
            if not message else message

        super(CoordioError, self).__init__(message)


class CoordioNotImplemented(CoordioError):
    """A custom exception for not yet implemented features."""

    def __init__(self, message=None):

        message = 'This feature is not implemented yet.' \
            if not message else message

        super(CoordioNotImplemented, self).__init__(message)


class CoordioAPIError(CoordioError):
    """A custom exception for API errors"""

    def __init__(self, message=None):
        if not message:
            message = 'Error with Http Response from Coordio API'
        else:
            message = 'Http response error from Coordio API. {0}'.format(message)

        super(CoordioAPIError, self).__init__(message)


class CoordioApiAuthError(CoordioAPIError):
    """A custom exception for API authentication errors"""
    pass


class CoordioMissingDependency(CoordioError):
    """A custom exception for missing dependencies."""
    pass


class CoordioWarning(Warning):
    """Base warning for Coordio."""


class CoordioUserWarning(UserWarning, CoordioWarning):
    """The primary warning class."""
    pass


class CoordioSkippedTestWarning(CoordioUserWarning):
    """A warning for when a test is skipped."""
    pass


class CoordioDeprecationWarning(CoordioUserWarning):
    """A warning for deprecated features."""
    pass
