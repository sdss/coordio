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

from . import sofa
from .exceptions import CoordIOError
from .iers import IERS


def utc_to_tai(jd):
    """Convert UTC to TAI. Wrapper around iauUtctai."""

    utc1 = int(jd)
    utc2 = jd - utc1

    tai1 = ctypes.c_double()
    tai2 = ctypes.c_double()

    res = sofa.iauUtctai(utc1, utc2, ctypes.byref(tai1), ctypes.byref(tai2))

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
    iers : .IERS
        The .IERS object with the TAI to UT1 delta values.

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

        self.iers = IERS()

        # Store the current time as a UNIX time. We'll use this later to know
        # how much time has passed and do a quick update to now.
        self.__internal_time = time.time() if value is None else None

    def __repr__(self):
        return f'<Time (JD={self.jd})>'

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
        now : float
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

    def to_utc(self):
        """Returns the date converted to JD in the UTC scale."""

        utc1 = ctypes.c_double()
        utc2 = ctypes.c_double()

        res = sofa.iauTaiutc(self.jd1, self.jd2,
                             ctypes.byref(utc1),
                             ctypes.byref(utc2))

        if res != 0:
            raise ValueError(f'iauTaiutc return with error code {res}.')

        return utc1.value + utc2.value

    def to_ut1(self):
        """Returns the date converted to JD in the UT1 scale."""

        # DTA is UT1-UTC, which can be expressed as delta_UT1-delta_AT.

        # delta_UT1=UT1-UTC and is given by the IERS tables.
        # This will update the table if it's out of date.
        delta_ut1 = self.iers.get_delta_ut1_utc(self.mjd)

        # delta_AT=TAI-UTC and can be obtained with the iauDat function but
        # it's easier to calculate it using iauTaiutc and subtracting TAI.
        delta_at = self.to_utc() - self.jd

        dta = delta_ut1 - delta_at

        ut1 = ctypes.c_double()
        ut2 = ctypes.c_double()

        res = sofa.iauTaiut1(self.jd1, self.jd2, dta,
                             ctypes.byref(ut1),
                             ctypes.byref(ut2))

        if res != 0:
            raise ValueError(f'iauTaiut1 return with error code {res}.')

        return ut1.value + ut2.value
