#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-16
# @Filename: test_coordio.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import ctypes

import astropy.time
import pytest

from coordio import sofa
from coordio.sofa_bindings import SOFA


def test_load():

    assert sofa._name is not None
    assert 'libsofa' in sofa._name


def test_load_fails(mocker):

    mocker.patch.object(ctypes.CDLL, '__init__', side_effect=OSError)

    with pytest.raises(OSError):
        SOFA()


def test_load_fails_with_name():

    with pytest.raises(OSError):
        SOFA('libblah.so')


def test_get_internal_date():

    assert sofa.get_internal_date() is not None

    j2000 = astropy.time.Time(2000.0, format='jyear')
    assert sofa.get_internal_date(j2000.to_datetime()) == (2451544.5, 0.5)

    now = astropy.time.Time.now()
    assert sum(sofa.get_internal_date()) == pytest.approx(now.jd, abs=1e-8)


def test_get_internal_date_error(mocker):

    mocker.patch.object(sofa, 'iauDtf2d', return_value=-1)

    with pytest.raises(ValueError):
        sofa.get_internal_date()
