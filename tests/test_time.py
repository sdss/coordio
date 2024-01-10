#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-21
# @Filename: test_time.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import astropy.time
import pytest

from coordio import IERS, Time


def test_time_now(mock_iers):

    time = Time()

    assert time.jd == pytest.approx(astropy.time.Time.now().tai.jd,
                                    abs=5e-8)

    assert time.jd1 + time.jd2 == time.jd
    assert isinstance(time.iers, IERS)
    assert time.jd - 2400000.5 == time.mjd


def test_to_scales():

    # Jan 15th, 2020
    time = Time(2458863.5, scale='TAI')
    test_time = astropy.time.Time(2458863.5, format='jd', scale='tai')

    assert time.to_utc() == pytest.approx(test_time.utc.jd, abs=1e-9)
    assert time.to_ut1() == pytest.approx(test_time.ut1.jd, abs=1e-9)
    assert time.to_tdb() == pytest.approx(test_time.tdb.jd, abs=1e-9)


def test_get_dut1():

    # Jan 15th, 2020
    time = Time(2458863.5, scale='TAI')

    assert time.get_dut1() == pytest.approx(-0.1799645, abs=1e-9)


def test_to_now():

    # Jan 15th, 2020
    time = Time(2458863.5, scale='TAI')

    time.to_now()
    assert time.jd == pytest.approx(astropy.time.Time.now().tai.jd, abs=1e-8)

    # Now test __internal_time
    time.to_now()
    assert time.jd == pytest.approx(astropy.time.Time.now().tai.jd, abs=1e-8)
