#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-17
# @Filename: sky.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

# IAU-defined sky coordinate systems and transformations.

from __future__ import annotations

import ctypes
import warnings
from typing import TYPE_CHECKING

import numpy

from . import conv, defaults, sofa
from .coordinate import Coordinate, Coordinate2D, verifySite, verifyWavelength
from .exceptions import CoordIOError, CoordIOUserWarning
from .time import Time

if TYPE_CHECKING:
    from .site import Site


__all__ = ['ICRS', 'Observed']


class ICRS(Coordinate2D):
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
            obj.epoch += defaults.EPOCH

        if isinstance(value, Coordinate):

            if value.coordSysName == 'Observed':
                obj._fromObserved(value)

            else:
                raise CoordIOError(
                    'Cannot convert to ICRS from %s' % value.coordSysName
                )

        return obj

    def _fromObserved(self, obsCoords):
        """Converts from `.Observed` coordinates.  Epoch is the
        time specifified by the site.

        """

        rlong = numpy.radians(obsCoords.site.longitude)
        rlat = numpy.radians(obsCoords.site.latitude)
        rZD = numpy.radians(90 - obsCoords[:, 0])
        rAz = numpy.radians(obsCoords[:, 1])
        wavelength = obsCoords.wavelength / 10000.

        _type = "A".encode()  # coords are azimuth, zenith dist

        time = obsCoords.site.time

        utc = time.to_utc()
        utc1 = int(utc)
        utc2 = utc - utc1
        dut1 = time.get_dut1()

        _ra = ctypes.c_double()
        _dec = ctypes.c_double()

        ra = numpy.zeros(len(obsCoords))
        dec = numpy.zeros(len(obsCoords))

        for ii in range(len(rAz)):

            sofa.iauAtoc13(
                _type, rAz[ii], rZD[ii], utc1, utc2, dut1,
                rlong, rlat, obsCoords.site.altitude, 0.0, 0.0,
                obsCoords.site.pressure, obsCoords.site.temperature,
                obsCoords.site.rh, wavelength[ii], _ra, _dec
            )
            ra[ii] = numpy.degrees(_ra.value)
            dec[ii] = numpy.degrees(_dec.value)

        self[:, 0] = ra
        self[:, 1] = dec

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

        new_icrs = ICRS(numpy.zeros(self.shape, dtype=self.dtype))

        for ii in range(self.shape[0]):

            epoch1_1 = float(int(self.epoch[ii]))
            epoch1_2 = self.epoch[ii] - epoch1_1

            res = sofa.iauPmsafe(rra[ii], rdec[ii], rpmra[ii], rpmdec[ii],
                                 self.parallax[ii] / 1000., self.rvel[ii],
                                 epoch1_1, epoch1_2, epoch2_1, epoch2_2,
                                 ra2, dec2, pmra2, pmdec2, parallax2, rvel2)

            if res > 1 or res < 0:
                warnings.warn(f'iauPmsafe return with error code {res}.',
                              CoordIOUserWarning)

            new_icrs[ii, :] = numpy.rad2deg([ra2.value, dec2.value])
            new_icrs.pmra[ii] = numpy.rad2deg(pmra2.value) * 3600. * 1000.
            new_icrs.pmra[ii] *= numpy.cos(dec2.value)
            new_icrs.pmdec[ii] = numpy.rad2deg(pmdec2.value) * 3600. * 1000.
            new_icrs.parallax[ii] = parallax2.value * 1000.
            new_icrs.rvel[ii] = rvel2.value
            new_icrs.epoch[ii] = jd

        return new_icrs


class Observed(Coordinate2D):
    """The observed coordinates of a series of targets.

    The array contains the Alt/Az coordinates of the targets. Their RA/Dec
    coordinates can be accessed via the ``ra`` and ``dec`` attributes.
    If `.ICRS` or `.Field` is passed, Alt/Az coordinates are computed.

    Parameters
    ----------
    value : numpy.ndarray
        A Nx2 Numpy array with the Alt and Az coordinates of the targets,
        in degrees.  Or `.ICRS` instance.  Or a `.Field` instance.
    wavelength : numpy.ndarray
        A 1D array with he observing wavelength, in angstrom.
        If not explicitly passed, it tries to inheret from value.wavelength,
        if that doesn't exist, it is set to default specified in:
        `defaults.wavelength`
    site : .Site
        The site from where observations will occur, along with the time
        of the observation.  Mandatory argument.

    Attributes
    -----------
    ra : numpy.ndarray
        Nx1 Numpy array, observed RA in degrees
    dec : numpy.ndarray
        Nx1 Numpy array, observed Dec in degrees
    ha : numpy.ndarray
        Nx1 Numpy array, hour angle in degrees
    pa : numpy.ndarray
        Nx1 Numpy array, position angle in degrees.  By SOFA: the angle between
        the direction to the north celestial pole and direction to the zenith.
        range is [-180, 180].  The sign is according to:
        -ha --> -pa, +ha --> +pa

    """

    __extra_arrays__ = ['wavelength']
    __extra_params__ = ['site']  # mandatory
    __computed_arrays__ = ['ra', 'dec', 'ha', 'pa']

    ra: numpy.ndarray
    dec: numpy.ndarray
    ha: numpy.ndarray
    pa: numpy.ndarray

    wavelength: numpy.ndarray
    site: Site

    def __new__(cls, value, **kwargs):
        # should we do range checks (eg alt < 90)? probably.

        verifySite(kwargs)
        verifyWavelength(kwargs, len(value), strict=False)

        obj = super().__new__(cls, value, **kwargs)

        # check if a coordinate was passed that we can just
        # 'cast' into Observed
        if isinstance(value, Coordinate):

            if value.coordSysName == 'ICRS':
                obj._fromICRS(value)

            elif value.coordSysName == 'Field':
                obj._fromField(value)

            else:
                raise CoordIOError(
                    'Cannot convert to Observed from %s' % value.coordSysName
                )

        else:
            # raw numpy array supplied compute values
            obj._fromRaw()

        return obj

    def _fromICRS(self, icrsCoords):
        """Converts from ICRS to topocentric observed coordinates for a site.
        Automatically executed after initialization with `.ICRS`.

        Computes and sets ra, dec, ha, pa arrays.

        Parameters:
        ------------
        icrsCoords : `.ICRS`
            ICRS coordinates from which to convert to observed coordinates

        """

        # eventually move this to coordio.conv?

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

        for ii in range(len(rra)):

            sofa.iauAtco13(
                rra[ii], rdec[ii], rpmra[ii], rpmdec[ii],
                icrs_2000.parallax[ii] / 1000., icrs_2000.rvel[ii],
                utc1, utc2, dut1,
                rlong, rlat, self.site.altitude, 0.0, 0.0,
                self.site.pressure, self.site.temperature,
                self.site.rh, self.wavelength[ii] / 10000.,
                az_obs, zen_obs, ha_obs, dec_obs, ra_obs, eo_obs
            )

            altAz = [
                90 - numpy.rad2deg(zen_obs.value),
                numpy.rad2deg(az_obs.value)
            ]
            self[ii, :] = altAz

            self.ra[ii] = numpy.rad2deg(ra_obs.value)
            self.dec[ii] = numpy.rad2deg(dec_obs.value)
            self.ha[ii] = numpy.rad2deg(ha_obs.value)

            # compute the pa
            self.pa[ii] = numpy.rad2deg(
                sofa.iauHd2pa(ha_obs.value, dec_obs.value, rlat)
            )

    def _fromField(self, fieldCoords):
        """Converts from field coordinates to topocentric observed
        coordinates for a site. Automatically executed after initialization
        with `.Field`.

        Computes and sets ra, dec, ha, pa arrays.

        Parameters:
        ------------
        fieldCoords : `.Field`
            Field coordinates from which to convert to observed coordinates

        """
        # get field center info
        altCenter, azCenter = fieldCoords.field_center.flatten()
        pa = float(fieldCoords.field_center.pa)  # parallactic angle

        alt, az = conv.fieldToObserved(
            fieldCoords.x, fieldCoords.y, fieldCoords.z,
            altCenter, azCenter, pa
        )

        self[:, 0] = alt
        self[:, 1] = az

        self._fromRaw()

    def _fromRaw(self):
        """Automatically executed after initialization with
        an Nx2 `numpy.ndarray` of Alt/Az coords.

        Computes and sets ra, dec, ha, pa arrays.

        """

        self[:, 1] = self[:, 1] % 360

        # compute ra, dec, ha, pa here...
        dec_obs = ctypes.c_double()
        ha_obs = ctypes.c_double()
        rlat = numpy.radians(self.site.latitude)
        rlong = numpy.radians(self.site.longitude)
        ut1 = self.site.time.to_ut1()

        for ii, (alt, az) in enumerate(self):
            raz = numpy.radians(az)
            ralt = numpy.radians(alt)
            sofa.iauAe2hd(raz, ralt, rlat, ha_obs, dec_obs)
            self.ha[ii] = numpy.degrees(ha_obs.value)
            self.dec[ii] = numpy.degrees(dec_obs.value)
            self.pa[ii] = numpy.degrees(
                sofa.iauHd2pa(ha_obs.value, dec_obs.value, rlat)
            )
            # earth rotation angle (from SOFA docs)
            # https://www.iausofa.org/2017_0420_C/sofa/sofa_ast_c.pdf
            era = sofa.iauEra00(ut1, 0)  # time is sum of the 2 args
            _ra = numpy.degrees(era + rlong - ha_obs.value)
            _ra = _ra % 360  # wrap ra

            self.ra[ii] = _ra

    @classmethod
    def fromEquatorial(cls, coords, hadec=False, site=None, **kwargs):
        """Initialises `.Observed` coordinates  from equatorial topocentric.

        Parameters
        ----------
        coords : numpy.ndarray
            Nx2 Numpy array with the RA and Dec coordinates of the targets.
            These must be topocentric equatorial coordinates that will be
            converted to horizontal topocentric used to initialise `.Observed`.
            Alternatively, if ``hadec=True`` then the first column in the array
            must be the hour angle in degrees.
        hadec : bool
            Whether the coordinates are HA and declination.
        site : .Site
            The site from where observations will occur, along with the time
            of the observation.  Mandatory argument.
        kwargs
            Other arguments to pass to `.Observed`.

        """

        coords = numpy.array(coords)

        if len(coords.shape) != 2 or coords.shape[1] != 2:
            raise CoordIOError('coords must be Nx2 array.')

        if site is None:
            raise CoordIOError('A site must be specified.')

        lat_r = numpy.radians(site.latitude)
        dec_d = coords[:, 1]
        dec_r = numpy.radians(dec_d)

        if hadec is False:
            ra_d = coords[:, 0]
            ra_r = numpy.radians(ra_d)

            ut = site.time.to_ut1()
            tt = site.time.to_tt()

            gmst = sofa.iauGmst00(ut, 0.0, tt, 0.0)
            lst_r = gmst + numpy.radians(site.longitude)

            ha_r = lst_r - ra_r

        else:
            ha_r = numpy.radians(coords[:, 0])

        # h = altitude, A = azimuth
        sin_h = (numpy.sin(dec_r) * numpy.sin(lat_r) +
                 numpy.cos(dec_r) * numpy.cos(lat_r) * numpy.cos(ha_r))
        h_r = numpy.arcsin(sin_h)
        h_d = numpy.degrees(h_r)

        sin_A = -numpy.sin(ha_r) * numpy.cos(dec_r) / numpy.cos(h_r)
        cos_A = ((numpy.sin(dec_r) - numpy.sin(lat_r) * numpy.sin(h_r)) /
                 (numpy.cos(lat_r) * numpy.cos(h_r)))
        A_r = numpy.arctan2(sin_A, cos_A)
        A_d = numpy.degrees(A_r) % 360

        return cls(numpy.array([h_d, A_d]).T, site=site, **kwargs)
