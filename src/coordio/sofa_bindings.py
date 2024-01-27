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
from ctypes import POINTER, c_char_p, c_double, c_int

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
    ('iauDat', (c_int, c_int, c_int, c_double, POINTER(c_double))),

    # double rc, double dc, double pr, double pd, double px, double rv,
    # double utc1, double utc2, double dut1,
    # double elong, double phi, double hm, double xp, double yp,
    # double phpa, double tc, double rh, double wl,
    # double *aob, double *zob, double *hob,
    # double *dob, double *rob, double *eo
    ('iauAtco13', (c_double, c_double, c_double, c_double, c_double, c_double,
                   c_double, c_double, c_double,
                   c_double, c_double, c_double, c_double, c_double,
                   c_double, c_double, c_double, c_double,
                   POINTER(c_double), POINTER(c_double), POINTER(c_double),
                   POINTER(c_double), POINTER(c_double), POINTER(c_double))),


    ('iauAtoc13', (c_char_p, c_double, c_double, c_double, c_double, c_double,
                   c_double, c_double, c_double, c_double, c_double, c_double,
                   c_double, c_double, c_double, POINTER(c_double), POINTER(c_double))),

    # double date1, double date2, double ut, double elong, double u, double v
    ('iauDtdb', (c_double, c_double, c_double, c_double, c_double, c_double)),

    # double ra1, double dec1, double pmr1, double pmd1,
    # double px1, double rv1,
    # double ep1a, double ep1b, double ep2a, double ep2b,
    # double *ra2, double *dec2, double *pmr2, double *pmd2,
    # double *px2, double *rv2
    ('iauPmsafe', (c_double, c_double, c_double, c_double, c_double, c_double,
                   c_double, c_double, c_double, c_double,
                   POINTER(c_double), POINTER(c_double), POINTER(c_double),
                   POINTER(c_double), POINTER(c_double), POINTER(c_double))),

    # compute position angle http://www.iausofa.org/2019_0722_C/sofa/hd2pa.c
    # args: hour angle, declination, site latitude (radians)
    ('iauHd2pa', (c_double, c_double, c_double)),

    # Horizon to equatorial coordinates: transform azimuth and altitude
    # to hour angle and declination.
    # args: double az, double alt, double latitude, double *ha, double *dec
    ('iauAe2hd', (c_double, c_double, c_double,
                  POINTER(c_double), POINTER(c_double))),

    # Earth rotation angle
    # sum of the arguments should be jd in the UT1 scale
    # args: double jd, double 0
    ('iauEra00', (c_double, c_double)),

    # Determine the constants A and B in the atmospheric refraction model
    # dZ = A tan Z + B tan^3 Z.
    # args: double phpa, double tc, double rh, double wl, double *refa, double *refb
    ('iauRefco', (c_double, c_double, c_double, c_double,
                  POINTER(c_double), POINTER(c_double))),
    # Greenwich mean sidereal time (model consistent with IAU 2000 resolutions).
    # args: double uta, double utb, double tta, double ttb
    ('iauGmst00', (c_double, c_double, c_double, c_double))
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

        # Some special cases of function that do not return
        # an integer error check.
        self.iauDtdb.restype = c_double
        self.iauHd2pa.restype = c_double
        self.iauEra00.restype = c_double
        self.iauGmst00.restype = c_double

    def get_internal_date(self, date=None, scale='UTC'):
        """Returns the internal representation of a date.

        Parameters
        ----------
        date : ~datetime.datetime
            The date to convert. If `None`, `~datetime.datetime.utcnow` will
            be used and ``scale`` will be ignored.
        scale : str
            The scale of the date.

        Returns
        -------
        d1, d2 : `float`
            The two-part Julian date.

        """

        if date is None:
            date = datetime.datetime.now(datetime.timezone.utc)
            scale = 'UTC'

        d1 = c_double()
        d2 = c_double()

        res = self.iauDtf2d(scale.upper().encode(), date.year, date.month,
                            date.day, date.hour, date.minute,
                            date.second + date.microsecond / 1e6,
                            d1, d2)

        if res != 0:
            raise ValueError(f'iauDtf2d return with error code {res}.')

        return d1.value, d2.value
