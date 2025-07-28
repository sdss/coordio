#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-21
# @Filename: conftest.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import pathlib
import urllib.request

import pytest

from coordio import config
from coordio.iers import IERS


@pytest.fixture(autouse=True)
def hack_config(tmpdir):

    orig_config = config.copy()

    config['iers']['path'] = str(tmpdir)

    yield

    config.update(orig_config)


@pytest.fixture
def iers_data_path():

    tmp_path = pathlib.Path(__file__).parent / 'data'

    yield tmp_path


@pytest.fixture(autouse=True)
def mock_iers(mocker, iers_data_path):
    # Mock MJD
    mocker.patch.object(IERS, '_get_current_jd', return_value= 2448690)

    iers_data_file = iers_data_path / 'finals2000A.data.csv'

    # Mock urllib.request.urlopen so that tests work offline and faster.
    mocker.patch.object(urllib.request, 'urlopen',
                        return_value=open(iers_data_file, 'rb'))
