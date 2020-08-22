#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-17
# @Filename: sky.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

# IAU-defined sky coordinate systems and transformations.

import ctypes

import numpy

from . import sofa
from .coordinate import Coordinate
from .exceptions import CoordinateError
from .time import Time


__all__ = ['ICRS', 'Observed']


class ICRS(Coordinate):
    """A representation of ICRS coordinates.

    Parameters
    ----------
    value : numpy.ndarray
        A Nx2 Numpy array with the RA and Dec coordinates of the targets.
    epoch : numpy.ndarray
        A 1D array with the epoch of the coordinates for each target,
        as a TDB Julian date (although for most applications the small
        differences between scales will not matter). Defaults to J2000.
    pmra : numpy.ndarray
        A 1D array with the proper motion in the RA axis for the N targets,
        in milliarcsec/yr. Must be a true angle, i.e, it must include the
        ``cos(dec)`` term.
    pmdec : numpy.ndarray
        A 1D array with the proper motion in the RA axis for the N targets,
        in milliarcsec/yr.
    parallax : numpy.ndarray
        A 1D array with the parallax for the N targets, in milliarcsec.
    rvel : numpy.ndarray
        A 1D array with the radial velocity in km/s, positive when receding.

    """

    __extra_arrays__ = ['epoch', 'pmra', 'pmdec', 'parallax', 'rvel']

    def __new__(cls, value, **kwargs):

        obj = super().__new__(cls, value, **kwargs)

        if kwargs.get('epoch', None) is None:
            obj.epoch += 2451545.0

        return obj

    def to_epoch(self, jd, site=None):
        """Convert the coordinates to a new epoch.

        Parameters
        ----------
        jd : float
            The Julian date, in TAI scale, of the output epoch.
        site : .Site
            The site of the observation. Used to determine the TDB-TT offset.
            If not provided, it assumes longitude and latitude zero.

        Returns
        -------
        icrs : `.ICRS`
            A new `.ICRS` object with the coordinates, proper motion, etc. in
            the new epoch.

        """

        rra = numpy.radians(self[:, 0])
        rdec = numpy.radians(self[:, 1])
        rpmra = numpy.radians(self.pmra / 1000. / 3600.) / numpy.cos(rdec)
        rpmdec = numpy.radians(self.pmdec / 1000. / 3600.)

        # Using TDB is probably an overkill.

        tai = Time(jd, scale='TAI')

        if site:
            epoch2 = tai.to_tdb(longitude=site.longitude,
                                latitude=site.latitude,
                                altitude=site.altitude)
        else:
            epoch2 = tai.to_tdb()

        epoch2_1 = int(epoch2)
        epoch2_2 = epoch2 - epoch2_1

        ra2 = ctypes.c_double()
        dec2 = ctypes.c_double()
        pmra2 = ctypes.c_double()
        pmdec2 = ctypes.c_double()
        parallax2 = ctypes.c_double()
        rvel2 = ctypes.c_double()

        new_icrs = self.copy()

        for ii in range(self.shape[0]):

            epoch1_1 = float(int(self.epoch[ii]))
            epoch1_2 = self.epoch[ii] - epoch1_1

            res = sofa.iauPmsafe(rra[ii], rdec[ii], rpmra[ii], rpmdec[ii],
                                 self.parallax[ii] / 1000., self.rvel[ii],
                                 epoch1_1, epoch1_2, epoch2_1, epoch2_2,
                                 ra2, dec2, pmra2, pmdec2, parallax2, rvel2)

            if res > 1 or res < 0:
                raise CoordinateError(f'iauPmsafe return with '
                                      f'error code {res}.')

            new_icrs[ii, :] = numpy.rad2deg([ra2.value, dec2.value])
            new_icrs.pmra[ii] = numpy.rad2deg(pmra2.value) * 3600. * 1000.
            new_icrs.pmra[ii] *= numpy.cos(dec2.value)
            new_icrs.pmdec[ii] = numpy.rad2deg(pmdec2.value) * 3600. * 1000.
            new_icrs.parallax[ii] = parallax2.value * 1000.
            new_icrs.rvel[ii] = rvel2.value

        return new_icrs

    def to_observed(self, site):
        """Converts from ICRS to topocentric observed coordinates for a site.

        Parameters
        ----------
        site : .Site
            The site from where observations will occur, along with the time
            of the observation.

        Returns
        -------
        observed : `.Observed`
            The observed coordinates.

        """

        # Prepare to call iauAtco13
        #  Given:
        #     rc,dc  double   ICRS right ascension at J2000.0 (radians)
        #     pr     double   RA proper motion (radians/year)
        #     pd     double   Dec proper motion (radians/year)
        #     px     double   parallax (arcsec)
        #     rv     double   radial velocity (km/s, +ve if receding)
        #     utc1   double   UTC as a 2-part...
        #     utc2   double   ...quasi Julian Date
        #     dut1   double   UT1-UTC (seconds)
        #     elong  double   longitude (radians, east +ve)
        #     phi    double   latitude (geodetic, radians)
        #     hm     double   height above ellipsoid (m, geodetic)
        #     xp,yp  double   polar motion coordinates (radians)
        #     phpa   double   pressure at the observer (hPa = mB)
        #     tc     double   ambient temperature at the observer (deg C)
        #     rh     double   relative humidity at the observer (range 0-1)
        #     wl     double   wavelength (micrometers)
        #
        #  Returned:
        #     aob    double*  observed azimuth (radians: N=0,E=90)
        #     zob    double*  observed zenith distance (radians)
        #     hob    double*  observed hour angle (radians)
        #     dob    double*  observed declination (radians)
        #     rob    double*  observed right ascension (CIO-based, radians)
        #     eo     double*  equation of the origins (ERA-GST)

        # TODO: maybe write this as Cython or C?

        # We need the epoch to be J2000.0 because that's what iauAtco13 likes.
        icrs_2000 = self.to_epoch(2451545.0, site=site)

        rra = numpy.radians(icrs_2000[:, 0])
        rdec = numpy.radians(icrs_2000[:, 1])
        rpmra = numpy.radians(icrs_2000.pmra / 1000. / 3600.) / numpy.cos(rdec)
        rpmdec = numpy.radians(icrs_2000.pmdec / 1000. / 3600.)

        rlong = numpy.radians(site.longitude)
        rlat = numpy.radians(site.latitude)

        if site.time is None:
            time = Time(scale=site.system_time_scale)
        else:
            time = site.time

        utc = time.to_utc()
        utc1 = int(utc)
        utc2 = utc - utc1
        dut1 = time.get_dut1()

        az_obs = ctypes.c_double()
        zen_obs = ctypes.c_double()
        ha_obs = ctypes.c_double()
        dec_obs = ctypes.c_double()
        ra_obs = ctypes.c_double()
        eo_obs = ctypes.c_double()

        observed = Observed(numpy.zeros(icrs_2000.shape, dtype=numpy.float64),
                            radec=numpy.zeros(icrs_2000.shape,
                                              dtype=numpy.float64),
                            site=site)

        for ii in range(len(rra)):

            sofa.iauAtco13(rra[ii], rdec[ii], rpmra[ii], rpmdec[ii],
                           icrs_2000.parallax[ii] / 1000., icrs_2000.rvel[ii],
                           utc1, utc2, dut1,
                           rlong, rlat, site.altitude, 0.0, 0.0,
                           site.pressure, site.temperature, site.rh,
                           site.wavelength / 10000.,
                           az_obs, zen_obs, ha_obs, dec_obs, ra_obs, eo_obs)

            observed[ii, :] = [90 - numpy.rad2deg(zen_obs.value),
                               numpy.rad2deg(az_obs.value)]
            observed.radec[ii, :] = numpy.rad2deg([ra_obs.value,
                                                   dec_obs.value])
            observed.ha[ii] = numpy.rad2deg(ha_obs.value)

        return observed


class Observed(Coordinate):
    """The observed coordinates of a series of targets.

    The array contains the Alt/Az coordinates of the targets. Their RA/Dec
    coordinates can be accessed via the ``radec`` attribute.

    """

    __extra_arrays__ = ['radec', 'ha', 'site']