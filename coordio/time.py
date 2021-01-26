#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-16
# @Filename: time.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import ctypes
import datetime
import time

import numpy

from . import sofa
from .exceptions import CoordIOError
from .iers import IERS


def utc_to_tai(jd):
    """Convert UTC to TAI. Wrapper around iauUtctai."""

    utc1 = int(jd)
    utc2 = jd - utc1

    tai1 = ctypes.c_double()
    tai2 = ctypes.c_double()

    res = sofa.iauUtctai(utc1, utc2, tai1, tai2)

    if res != 0:
        raise ValueError(f'iauUtctai return with error code {res}.')

    return tai1.value + tai2.value


class Time:
    """A time storage and conversion class.

    All times, regardless of the input format and scale are converted and
    stored as TAI Julian dates.

    Parameters
    ----------
    value
        The input time value. If `None`, the current `datetime.datetime.now`
        system time will be used.
    format : str
        The format of the input time.
    scale : str
        The time scale of the input value. Valid values are ``UTC`` and
        ``TAI``.

    Attributes
    ----------
    jd : float
        The Julian date in the TAI scale.

    """

    def __init__(self, value=None, format='jd', scale='UTC'):

        if isinstance(value, datetime.datetime) or value is None:
            jd1, jd2 = sofa.get_internal_date(value, scale)
            self.jd = jd1 + jd2
        elif format == 'jd':
            self.jd = value
        else:
            raise NotImplementedError('Not implemented format.')

        if scale == 'UTC':
            self.jd = utc_to_tai(self.jd)
        elif scale == 'TAI':
            pass
        else:
            raise CoordIOError('Invalid scale.')

        self._iers = None

        # Store the current time as a UNIX time. We'll use this later to know
        # how much time has passed and do a quick update to now.
        self.__internal_time = time.time() if value is None else None

    def __repr__(self):
        return f'<Time (JD={self.jd})>'

    @property
    def iers(self):
        """Gets the IERS table."""

        if self._iers is None:
            self._iers = IERS()

        return self._iers

    @property
    def jd1(self):
        """Returns the integer part of the Julian data."""

        return int(self.jd)

    @property
    def jd2(self):
        """Returns the fractional part of the Julian data."""

        return self.jd - self.jd1

    @property
    def mjd(self):
        """Returns the modified Julian date ``(JD-2400000.5)``."""

        return self.jd - 2400000.5

    def to_now(self, scale='UTC'):
        """Updates the internal value to the current TAI date.

        Parameters
        ----------
        scale : str
            The scale of the system clock. Assumes it's UTC but it can be
            changed if the internal clock is synced to TAI or other timescale.

        Returns
        -------
        now : `float`
            The new TAI JD. Also updates the internal value.

        """

        # If __internal_time is set this means we initialised this
        # time as now(). We can simply add how long it has passed
        # since then.
        if self.__internal_time:
            new_now = time.time()
            delta_jd = (new_now - self.__internal_time) / 86400.
            self.jd += delta_jd
            self.__internal_time = new_now
            return self.jd

        # Otherwise reinitialise the object.
        self.__init__(scale=scale)
        return self.jd

    def get_dut1(self):
        """Returns the delta UT1-UTC."""

        if not self.iers:
            raise CoordIOError('IERS table not loaded.')

        return self.iers.get_delta_ut1_utc(jd=self.jd)

    def to_utc(self):
        """Returns the date converted to JD in the UTC scale."""

        utc1 = ctypes.c_double()
        utc2 = ctypes.c_double()

        res = sofa.iauTaiutc(self.jd1, self.jd2, utc1, utc2)

        if res != 0:
            raise ValueError(f'iauTaiutc return with error code {res}.')

        return utc1.value + utc2.value

    def to_ut1(self):
        """Returns the date converted to JD in the UT1 scale."""

        # DTA is UT1-UTC, which can be expressed as delta_UT1-delta_AT.

        # delta_UT1=UT1-UTC and is given by the IERS tables, in seconds.
        # This will update the table if it's out of date.
        delta_ut1 = self.iers.get_delta_ut1_utc(self.jd)

        # delta_AT=TAI-UTC and can be obtained with the iauDat function but
        # it's easier to calculate it using iauTaiutc and subtracting TAI.
        delta_at = self.jd - self.to_utc()

        # delta_UT1-TAI, in seconds, for iauTaiut1
        dta = delta_ut1 - delta_at * 86400.

        ut1 = ctypes.c_double()
        ut2 = ctypes.c_double()

        res = sofa.iauTaiut1(self.jd1, self.jd2, dta, ut1, ut2)

        if res != 0:
            raise ValueError(f'iauTaiut1 return with error code {res}.')

        return ut1.value + ut2.value

    def to_tt(self):
        """Returns the date converted to JD in the TT scale."""

        return self.jd + 32.184 / 86400.

    def to_tdb(self, longitude=0.0, latitude=0.0, altitude=0.0):
        """Returns the date converted to JD in the TDB scale.

        Parameters
        ----------
        longitude : float
            The East-positive longitude of the site, in degrees.
        latitude : float
            The site latitude in degrees.
        altitude : float
            The altitude (elevation) of the site above see level, in meters.
            Defaults to zero meters.

        """

        # We need to call iauDtdb that models the delta between TT and TDB.
        # It depends on the location of the observer on the Earth, and the
        # fractional part of the UT1 time.

        # Earth radius in km
        RE = 6378.1

        rlong = numpy.radians(longitude)
        rlat = numpy.radians(latitude)

        # Distance from Earth spin axis (km)
        u = (RE + altitude / 1000.) * numpy.cos(rlat)

        # Distance north of equatorial plane (km)
        v = (RE + altitude / 1000.) * numpy.sin(rlat)

        ut1 = self.to_ut1()
        ut1_1 = ut1 - int(ut1)

        tt = self.to_tt()
        tt_1 = int(tt)
        tt_2 = tt - tt_1

        delta_tdb_tt = sofa.iauDtdb(tt_1, tt_2, ut1_1, rlong, u, v)

        return tt + delta_tdb_tt / 86400.
