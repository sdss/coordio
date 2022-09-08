#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-21
# @Filename: test_site.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import astropy.time
import pytest

from coordio import Site, Time, config


def test_site_apo():

    site = Site('APO')

    apo = config['site']['APO']

    assert site.longitude == apo['longitude']
    assert site.latitude == apo['latitude']

    assert site.pressure is not None
    assert site.pressure < 1000.

    assert site.time is None


def test_set_time():

    site = Site('APO')

    # Jan 15th, 2020
    time = Time(2458863.5, scale='TAI')

    site.set_time(time)
    assert site.time == time

    site.set_time(scale='UTC')
    assert site.time != time
    assert site.time.jd == pytest.approx(astropy.time.Time.now().tai.jd,
                                         abs=1e-8)


def test_copy():

    site = Site('APO')
    site_copy = site.copy()

    assert isinstance(site_copy, Site)
    assert site_copy != site


if __name__ == "__main__":
    test_set_time()
