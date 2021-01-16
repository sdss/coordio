#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-21
# @Filename: test_sky.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import astropy.time
import numpy
from astropy import units as u
from astropy.coordinates import AltAz, Distance, EarthLocation, SkyCoord

from coordio import ICRS, Observed, Site

wavelength = 7000

icrs = ICRS([[100, 10], [101., 11]],
            pmra=[100, 0.1], pmdec=[-15, -0.5],
            parallax=[10., 10.], rvel=[100, 1000],
            epoch=[2451545, 2451545], wavelength=wavelength)

astropy_icrs = SkyCoord(ra=[100, 101] * u.deg, dec=[10, 11] * u.deg,
                        frame='icrs',
                        pm_ra_cosdec=[100, 0.1] * u.mas / u.yr,
                        pm_dec=[-15., -0.5] * u.mas / u.yr,
                        distance=Distance(parallax=[10., 10.] * u.mas),
                        radial_velocity=[100, 1000] * u.km / u.s,
                        obstime=astropy.time.Time(2451545.0,
                                                  format='jd',
                                                  scale='tai'))

site = Site('APO')
astropy_location = EarthLocation.from_geodetic(lon=site.longitude,
                                               lat=site.latitude,
                                               height=site.altitude)


def test_icrs():

    icrs = ICRS([[100, 10], [101., 11]])

    assert isinstance(icrs, ICRS)
    assert (icrs.epoch == 2451545.0).all()


def test_to_epoch():

    icrs_2020 = icrs.to_epoch(jd=2458863, site=site)

    new_obstime = astropy.time.Time(2458863, format='jd', scale='tdb')
    new_astropy_icrs = astropy_icrs.apply_space_motion(new_obstime=new_obstime)

    numpy.testing.assert_array_almost_equal(new_astropy_icrs.ra.deg,
                                            icrs_2020[:, 0],
                                            decimal=6)
    numpy.testing.assert_array_almost_equal(new_astropy_icrs.dec.deg,
                                            icrs_2020[:, 1],
                                            decimal=6)

    astropy_pm_ra_cosdec = (new_astropy_icrs.pm_ra.value *
                            numpy.cos(numpy.radians(new_astropy_icrs.dec.deg)))
    numpy.testing.assert_array_almost_equal(astropy_pm_ra_cosdec,
                                            icrs_2020.pmra,
                                            decimal=6)
    numpy.testing.assert_array_almost_equal(new_astropy_icrs.pm_dec.value,
                                            icrs_2020.pmdec,
                                            decimal=6)


def test_to_observed():

    site.set_time(2458863, scale='TAI')

    observed = icrs.to_observed(site)

    new_obstime = astropy.time.Time(2458863, format='jd', scale='tai')
    new_astropy_icrs = astropy_icrs.apply_space_motion(new_obstime=new_obstime)
    astropy_observed = new_astropy_icrs.transform_to(
        AltAz(location=astropy_location,
              obstime=new_obstime,
              pressure=site.pressure * u.mbar,
              temperature=site.temperature * u.deg_C,
              relative_humidity=site.rh,
              obswl=wavelength * u.Angstrom))

    assert isinstance(observed, Observed)

    numpy.testing.assert_allclose(observed[:, 0],
                                  astropy_observed.alt.deg,
                                  atol=3e-5, rtol=1e-7)
    numpy.testing.assert_allclose(
        observed[:, 1], astropy_observed.az.deg,
        atol=3e-5 / numpy.cos(numpy.radians(astropy_observed.alt.deg)).max(),
        rtol=1e-7)
