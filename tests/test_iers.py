#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-21
# @Filename: test_iers.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import pytest

from coordio import IERS, CoordIOUserWarning


@pytest.fixture()
def clear_iers_instance():
    # Because of Python mangling for double underscores we need to modify
    # the instance like this: https://bit.ly/3hktKEm.

    # Clear before and after the test, so that this also clears in case that previous
    # tests set the instance.

    IERS._IERS__instance = None
    yield
    IERS._IERS__instance = None


def test_iers(tmpdir, clear_iers_instance):

    iers_file = tmpdir / 'finals2000A.data.csv'

    # raise RuntimeError(iers_file)

    with pytest.warns(CoordIOUserWarning) as ww:
        iers = IERS(path=tmpdir)

    assert 'Downloading IERS table from' in str(ww[0].message)
    assert iers is not None
    assert iers.data is not None

    assert iers_file.exists


def test_is_valid(iers_data_path):

    iers = IERS(iers_data_path)
    assert iers.is_valid(2458863.0)


def test_is_valid_out_of_range():

    iers = IERS()

    with pytest.warns(CoordIOUserWarning):
        assert iers.is_valid(10000000) is False


def test_get_delta_ut1_utc():

    iers = IERS()

    # Jan 15th, 2020
    assert iers.get_delta_ut1_utc(2458863.5) == pytest.approx(-0.1799645)
