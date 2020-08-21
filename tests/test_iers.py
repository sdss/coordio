#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-21
# @Filename: test_iers.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import pathlib
import shutil
import urllib.request

import pytest

from coordio import IERS, CoordIOUserWarning


@pytest.fixture(autouse=True)
def clear_iers_instance():

    yield

    # Because of Python mangling for double underscores we need to modify
    # the instance like this: https://bit.ly/3hktKEm
    IERS._IERS__instance = None


@pytest.fixture
def iers_data_path():
    """Ensure we only download the IERS table once."""

    tmp_path = pathlib.Path(__file__).parent / 'data'

    yield tmp_path


@pytest.fixture
def mock_download(mocker, iers_data_path, tmpdir):

    iers_data_file = iers_data_path / 'finals2000A.data.csv'

    def copy_file(*_):
        shutil.copyfile(iers_data_file, tmpdir / 'finals2000A.data.csv')

    # Mock urllib.request.urlretrieve so that tests work offline and faster.
    mocker.patch.object(urllib.request, 'urlretrieve', side_effect=copy_file)


def test_iers(mock_download, tmpdir):

    iers_file = tmpdir / 'finals2000A.data.csv'

    with pytest.warns(CoordIOUserWarning) as ww:
        iers = IERS(path=tmpdir)

    assert 'Downloading IERS table from' in str(ww[0].message)
    assert iers is not None
    assert iers.data is not None

    assert iers_file.exists


def test_is_valid(iers_data_path):

    iers = IERS(iers_data_path)
    assert iers.is_valid(58906)


def test_is_valid_out_of_range(mock_download, iers_data_path):

    iers = IERS(iers_data_path)

    with pytest.warns(CoordIOUserWarning):
        assert iers.is_valid(10000000) is False


def test_get_delta_ut1_utc(iers_data_path):

    iers = IERS(iers_data_path)

    assert iers.get_delta_ut1_utc(58906.5) == pytest.approx(
        0.5 * (iers.get_delta_ut1_utc(58906) + iers.get_delta_ut1_utc(58907))
    )
