#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-16
# @Filename: sofa_bindings.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

# Bindings for SOFA procedures. This does not intent to be a comprehensive
# wrapping of the library, just of those functions that are needed for our
# purposes.

import ctypes
import datetime
import importlib
import os
from ctypes import byref, c_char_p, c_double, c_int, POINTER


# List of argument times for some of the SOFA functions.
ARGTYPES = [

    # const char *scale, int iy, int im, int id,
    #   int ihr, int imn, double sec,
    #   double *d1, double *d2
    ('iauDtf2d', (c_char_p, c_int, c_int, c_int,
                  c_int, c_int, c_double,
                  POINTER(c_double), POINTER(c_double))),

    # double utc1, double utc2, double *tai1, double *tai2
    ('iauUtctai', (c_double, c_double, POINTER(c_double), POINTER(c_double))),

    # double tai1, double tai2, double dta, double *ut11, double *ut12
    ('iauTaiut1', (c_double, c_double, c_double,
                   POINTER(c_double), POINTER(c_double))),

    # double tai1, double tai2, double *utc1, double *utc2
    ('iauTaiutc', (c_double, c_double, POINTER(c_double), POINTER(c_double))),

    # int iy, int im, int id, double fd, double *deltat
    ('iauDat', (c_int, c_int, c_int, c_double, POINTER(c_double)))

]


class SOFA(ctypes.CDLL):
    """Load the SOFA C extension.

    Parameters
    ----------
    name : str
        The path to the shared object. If `None`, will try the standard
        variants of ``libsofa`` for this architecture. If the library is not
        in a standard location the path needs to be added to
        ``LD_LIBRARY_PATHS``.

    """

    def __init__(self, name=None):

        if name is None:
            # Try all the usual suffixes.
            success = False
            mod_path = os.path.join(os.path.dirname(__file__), 'libsofa')
            for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                try:
                    super().__init__(mod_path + suffix)
                    success = True
                    break
                except OSError:
                    pass
            if success is False:
                raise OSError('Could not find a valid libsofa extension.')
        else:
            super().__init__(name)

        # Assign argtypes. This also adds the functions to the class dir().
        for func_name, argtypes in ARGTYPES:
            func = self.__getattr__(func_name)
            func.argtypes = argtypes

    def get_internal_date(self, date=None, scale='UTC'):
        """Returns the internal representation of a date.

        Parameters
        ----------
        date : ~datetime.datetime
            The date to convert. If `None`, `~datetime.datetime.now` will be
            used.
        scale : str
            The scale of the date.

        Returns
        -------
        d1, d2 : double
            The two-part Julian date.

        """

        if date is None:
            date = datetime.datetime.now()

        d1 = c_double()
        d2 = c_double()

        res = self.iauDtf2d(scale.upper().encode(), date.year, date.month,
                            date.day, date.hour, date.minute,
                            date.second + date.microsecond / 1e6,
                            byref(d1), byref(d2))

        if res != 0:
            raise ValueError(f'iauDtf2d return with error code {res}.')

        return d1.value, d2.value
