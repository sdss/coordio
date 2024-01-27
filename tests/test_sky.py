#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-21
# @Filename: test_sky.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import sys

import astropy.time
import numpy
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, Distance, EarthLocation, SkyCoord
from coordio import ICRS, CoordIOError, Observed, Site

wavelength = 7000

icrs = ICRS([[100, 10], [101., 11]],
            pmra=[100, 0.1], pmdec=[-15, -0.5],
            parallax=[10., 10.], rvel=[100, 1000],
            epoch=[2451545, 2451545])


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

    if sys.version_info < (3, 8):
        cos_dec = numpy.cos(numpy.radians(new_astropy_icrs.dec.deg))
        astropy_pm_ra_cosdec = new_astropy_icrs.pm_ra.value * cos_dec
    else:
        astropy_pm_ra_cosdec = new_astropy_icrs.pm_ra_cosdec.value

    numpy.testing.assert_array_almost_equal(astropy_pm_ra_cosdec,
                                            icrs_2020.pmra,
                                            decimal=6)
    numpy.testing.assert_array_almost_equal(new_astropy_icrs.pm_dec.value,
                                            icrs_2020.pmdec,
                                            decimal=6)


def test_to_observed():

    site.set_time(2458863, scale='TAI')

    # test for failure if site isn't provided
    with pytest.raises(CoordIOError):
        observed = Observed(icrs)

    observed = Observed(icrs, site=site, wavelength=wavelength)

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

    # test that raw initialization of observed yeilds the same answers
    obsArr = numpy.array(observed)
    observed2 = Observed(obsArr, site=site)

    numpy.testing.assert_allclose(observed[:, 0], observed2[:, 0],
                                  atol=3e-7, rtol=0)

    numpy.testing.assert_allclose(observed[:, 1], observed2[:, 1],
                                  atol=3e-7, rtol=0)

    numpy.testing.assert_allclose(observed.ra, observed2.ra,
                                  atol=3e-7, rtol=0)

    numpy.testing.assert_allclose(observed.dec, observed2.dec,
                                  atol=3e-7, rtol=0)

    numpy.testing.assert_allclose(observed.ha, observed2.ha,
                                  atol=3e-7, rtol=0)

    numpy.testing.assert_allclose(observed.pa, observed2.pa,
                                  atol=3e-7, rtol=0)

    # _icrs = ICRS(observed, wavelength=wavelength)
    # numpy.testing.assert_array_almost_equal(icrs, _icrs, decimal=5)


def test_icrs_obs_cycle():
    time = 2451545
    icrs = ICRS([[100, 10], [101., 11], [180., 30]], epoch=time)

    site.set_time(time, scale='TAI')
    observed = Observed(icrs, site=site)
    _icrs = ICRS(observed)

    a1 = SkyCoord(ra=icrs[:, 0] * u.deg, dec=icrs[:, 1] * u.deg)
    a2 = SkyCoord(ra=_icrs[:, 0] * u.deg, dec=_icrs[:, 1] * u.deg)
    sep = a1.separation(a2)
    assert numpy.max(numpy.array(sep) * 3600) < 0.5


@pytest.mark.skipif(sys.version_info < (3, 8), reason='requires python 3.8+ for HADec')
@pytest.mark.parametrize('hadec', [True, False])
def test_observed_from_equatorial(hadec):
    from astropy.coordinates import HADec

    time = 2451545
    site.set_time(time, scale='TAI')

    ras = numpy.random.uniform(0, 360, 100)
    decs = numpy.random.uniform(-90, 90, 100)

    # Get HA for cases when hadec=True
    astropy_time = astropy.time.Time(time, format='jd', scale='tai',
                                     location=astropy_location)
    astropy_lst = astropy_time.sidereal_time('mean').deg

    ha_d = astropy_lst - ras
    if hadec:
        eqs = numpy.array((ha_d, decs)).T
    else:
        eqs = numpy.array([ras, decs]).T

    observed = Observed.fromEquatorial(eqs, hadec=hadec, site=site,
                                       wavelength=wavelength)

    assert observed.shape == (100, 2)
    assert numpy.all(observed[:, 0] > -90) and numpy.all(observed[:, 0] < 90)
    assert numpy.all(observed[:, 1] > 0) and numpy.all(observed[:, 1] < 360)

    # Now calculate the same using astropy and compare.
    hadec_coords = HADec(ha=ha_d * u.deg, dec=decs * u.deg,
                         obstime=astropy_time, location=astropy_location)

    astropy_altaz = hadec_coords.transform_to(AltAz(location=astropy_location,
                                                    obstime=astropy_time))

    assert numpy.allclose(observed[:, 0], astropy_altaz.alt.deg, atol=1e-6)
    assert numpy.allclose(observed[:, 1], astropy_altaz.az.deg % 360, atol=1e-6)
