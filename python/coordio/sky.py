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
from .exceptions import CoordinateError, CoordIOError, CoordIOWarning
from .time import Time
from .site import Site
# from .telescope import Field
from . import defaults


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
    wavelength : numpy.ndarray
        A 1D array with he observing wavelength, in angstrom.
        Defaults to 7500 angstrom.

    """

    __extra_arrays__ = ['epoch', 'pmra', 'pmdec', 'parallax', 'rvel', 'wavelength']

    def __new__(cls, value, **kwargs):

        obj = super().__new__(cls, value, **kwargs)

        if kwargs.get('epoch', None) is None:
            obj.epoch += defaults.epoch

        if kwargs.get('wavelength', None) is None:
            obj.wavelength += defaults.wavelength

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


class Observed(Coordinate):
    """The observed coordinates of a series of targets.

    The array contains the Alt/Az coordinates of the targets. Their RA/Dec
    coordinates can be accessed via the ``ra`` and ``dec`` attributes.
    If `.ICRS` or `.Field` is passed, Alt/Az coordinates are computed.

    Parameters
    ----------
    value : numpy.ndarray
        A Nx2 Numpy array with the Alt and Az coordinates of the targets,
        in degrees.  Or a coordIO.ICRS array.  Or a coordio.Field array
    wavelength : numpy.ndarray
        A 1D array with he observing wavelength, in angstrom.
        If not explicitly passed, it tries to inheret from value.wavelength,
        if that doesn't exist, it is set to default specified in:
        `.defaults.wavelength`
    site : .Site
        The site from where observations will occur, along with the time
        of the observation.  Mandatory argument.

    """

    __extra_arrays__ = ['wavelength']
    __extra_params__ = ['site']
    __computed_arrays__ = ['ra', 'dec', 'ha', 'pa']

    def __new__(cls, value, **kwargs):

        if kwargs.get('site', None) is None:
            raise CoordIOError('Site must be passed to Observed')

        else:
            site = kwargs.get('site')
            if not isinstance(site, Site):
                raise CoordIOError('Must pass Site to Observed')
            if site.time is None:
                raise CoordIOError("Time of observation must be specified")

        # should we prefer wavelength passed, or wavelength
        # existing on value (if it does exist).  Here preferring passed
        if kwargs.get('wavelength', None) is None:
            if hasattr(value, "wavelength"):
                kwargs["wavelength"] = value.wavelength

        obj = super().__new__(cls, value, **kwargs)

        if kwargs.get('wavelength', None) is None:
            obj.wavelength += defaults.wavelength

        # check if a coordinate was passed that we can just
        # 'cast' into Observed
        if isinstance(value, Coordinate):

            if value.coordSysName == 'ICRS':
                obj.fromICRS(value)

            elif value.coordSysName == 'Field':
                obj.fromField(value)

            else:
                raise CoordIOError(
                    'Cannot create Observed from %s'%value.coordSysName
                )

        else:
            # raw numpy array supplied compute values
            obj.fromRaw()

        return obj

    def fromICRS(self, icrsCoords):
        """Converts from ICRS to topocentric observed coordinates for a site.

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
        icrs_2000 = icrsCoords.to_epoch(2451545.0, site=self.site)

        rra = numpy.radians(icrs_2000[:, 0])
        rdec = numpy.radians(icrs_2000[:, 1])
        rpmra = numpy.radians(icrs_2000.pmra / 1000. / 3600.) / numpy.cos(rdec)
        rpmdec = numpy.radians(icrs_2000.pmdec / 1000. / 3600.)

        rlong = numpy.radians(self.site.longitude)
        rlat = numpy.radians(self.site.latitude)

        time = self.site.time

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

        # initialize output arrays
        # obs_arr = numpy.zeros(icrs_2000.shape, dtype=numpy.float64)
        # cls.radec = numpy.zeros(icrs_2000.shape, dtype=numpy.float64)
        # cls.pa = numpy.zeros(icrs_2000.shape[0], dtype=numpy.float64)
        # cls.ha = numpy.zeros(icrs_2000.shape[0], dtype=numpy.float64)

        # pa and ha are initialized to zeros automatically
        # radec needs explicit initialization because it's
        # not a 1D array
        # observed = Observed(numpy.zeros(icrs_2000.shape, dtype=numpy.float64),
        #                     radec=numpy.zeros(icrs_2000.shape,
        #                                       dtype=numpy.float64),
        #                     wavelength=numpy.copy(self.wavelength),
        #                     site=site)

        for ii in range(len(rra)):

            sofa.iauAtco13(rra[ii], rdec[ii], rpmra[ii], rpmdec[ii],
                           icrs_2000.parallax[ii] / 1000., icrs_2000.rvel[ii],
                           utc1, utc2, dut1,
                           rlong, rlat, self.site.altitude, 0.0, 0.0,
                           self.site.pressure, self.site.temperature,
                           self.site.rh, icrs_2000.wavelength[ii] / 10000.,
                           az_obs, zen_obs, ha_obs, dec_obs, ra_obs, eo_obs)

            # self is Alt,Az array
            self[ii, :] = [90 - numpy.rad2deg(zen_obs.value),
                               numpy.rad2deg(az_obs.value)]
            self.ra[ii] = numpy.rad2deg(ra_obs.value)
            self.dec[ii] = numpy.rad2deg(dec_obs.value)
            self.ha[ii] = numpy.rad2deg(ha_obs.value)

            # compute the pa
            self.pa[ii] = numpy.rad2deg(
                sofa.iauHd2pa(ha_obs.value, dec_obs.value, rlat)
            )

    def fromField(self, fieldCoords):
        raise NotImplementedError()

    def fromRaw(self):
        # compute ra, dec, ha, pa here...
        ra_obs = ctypes.c_double()
        dec_obs = ctypes.c_double()
        ha_obs = ctypes.c_double()
        rlat = numpy.radians(self.site.latitude)
        rlong = numpy.radians(self.site.longitude)
        # ut1 = self.site.time.to_ut1()
        ut1 = self.site.time.to_ut1()
        print("ut1", ut1)

        for ii, (alt,az) in enumerate(self):
            raz = numpy.radians(az)
            ralt = numpy.radians(alt)
            sofa.iauAe2hd(raz, ralt, rlat, ha_obs, dec_obs)
            self.ha[ii]  = numpy.degrees(ha_obs.value)
            self.dec[ii] = numpy.degrees(dec_obs.value)
            self.pa[ii]  = numpy.degrees(
                              sofa.iauHd2pa(ha_obs.value, dec_obs.value, rlat)
                           )
            # earth rotation angle (from SOFA docs)
            # https://www.iausofa.org/2017_0420_C/sofa/sofa_ast_c.pdf
            era = sofa.iauEra00(ut1, 0)
            _ra = numpy.degrees(era + rlong - ha_obs.value)
            while _ra < 0:
                _ra += 360
            while _ra > 360:
                _ra -= 360
            self.ra[ii] = _ra


