# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-12-05 12:01:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-12-05 12:19:32


class CoordIOError(Exception):
    """A custom core coordIO exception"""

    def __init__(self, message=None):

        message = 'There has been an error' \
            if not message else message

        super(CoordIOError, self).__init__(message)


class CoordinateError(CoordIOError):
    """Coordinate error."""


class CoordIOWarning(Warning):
    """Base warning for coordIO."""


class CoordIOUserWarning(UserWarning, CoordIOWarning):
    """The primary warning class."""
    pass
