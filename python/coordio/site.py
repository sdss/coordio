#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-17
# @Filename: site.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import copy

import numpy

from . import config
from .exceptions import CoordIOError
from .time import Time


g = 9.80665       # m / s^2
R0 = 8.314462618  # J / (mol K)
T0 = 288.16       # K
M = 0.02896968    # kg / mol
p0 = 1013.25      # millibar (hPa)


class Site:
    r"""A representation of an observing site.

    Parameters
    ----------
    name : str
        The name of the site. If the name matches that of a site in the
        configuration file those values will be used, overridden by the
        input keywords.
    latitude : float
        The site latitude in degrees.
    longitude : float
        The East-positive longitude of the site, in degrees.
    altitude : float
        The altitude (elevation) of the site above see level, in meters.
        Defaults to zero meters.
    pressure : float
        The atmospheric pressure at the site, in millibar (same as hPa). If
        not provided the pressure will be calculated using the altitude
        :math:`h` and the approximate expression

        .. math::

            p \sim -p_0 \exp\left( \dfrac{g h M} {T_0 R_0} \right)

        where :math:`p_0` is the pressure at sea level, :math:`M` is the molar
        mass of the air, :math:`R_0` is the universal gas constant, and
        :math:`T_0=288.16\,{\rm K}` is the standard sea-level temperature.
    temperature : float
        The site temperature, in degrees Celsius. Defaults to
        :math:`10^\circ{\rm C}`.
    rh : float
        The relative humidity, in the range :math:`0-1`. Defaults to 0.5.
    system_time_scale : str
        The time scale of the system time. Defaults to UTC.

    Attributes
    ----------
    time : .Time
        A `.Time` instance describing the time for the observation. Use
        `.set_time` to set the time.

    """

    def __init__(self, name, latitude=None, longitude=None,
                 altitude=None, pressure=None, temperature=None,
                 rh=None, system_time_scale=None):

        kwargs = dict(latitude=latitude,
                      longitude=longitude,
                      altitude=altitude,
                      pressure=pressure,
                      temperature=temperature,
                      rh=rh,
                      system_time_scale=system_time_scale)

        if name in config['site']:
            config_kwargs = config['site'][name].copy()
            for kw in kwargs:
                if kwargs[kw] is None and kw in config_kwargs:
                    kwargs[kw] = config_kwargs[kw]

        self.name = name

        for key, value in kwargs.items():
            setattr(self, key, value)

        if not self.longitude or not self.latitude:
            raise CoordIOError('Longitude and latitude need to be defined.')

        # Set defaults
        self.altitude = self.altitude or 0.
        self.temperature = self.temperature or 10.
        self.rh = self.rh or 0.5
        self.system_time_scale = self.system_time_scale or 'UTC'

        if self.pressure is None:
            self.pressure = p0 * numpy.exp(-(self.altitude * g * M) / (T0 * R0))

        self.time = None

    def __repr__(self):
        return (f'<Site {self.name!r} '
                f'(longitude={self.longitude}, latitude={self.latitude})>')

    def set_time(self, value=None, scale='TAI'):
        """Sets the time of the observation.

        Parameters
        ----------
        value : float, .Time, or None
            A `.Time` instance or the Julian date of the time to use.
            If `None`, the current system datetime will be used.
        scale : str
            The time scale of the Julian date. Ignored if ``value`` is a
            `.Time` instance.

        """

        if isinstance(value, Time):
            self.time = value
        else:
            self.time = Time(value, format='jd', scale=scale)

    def copy(self):
        """Returns a copy of the object."""

        return copy.copy(self)
