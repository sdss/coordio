#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-21
# @Filename: conftest.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from coordio.iers import IERS
import pathlib
import shutil
import urllib.request

import pytest

from coordio import config


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
def mock_iers(mocker, iers_data_path, tmpdir):

    iers_data_file = iers_data_path / 'finals2000A.data.csv'

    def copy_file(*_):
        shutil.copyfile(iers_data_file, tmpdir / 'finals2000A.data.csv')

    # Mock urllib.request.urlretrieve so that tests work offline and faster.
    mocker.patch.object(urllib.request, 'urlretrieve', side_effect=copy_file)


@pytest.fixture(autouse=True)
def clear_iers_instance():
    # Because of Python mangling for double underscores we need to modify
    # the instance like this: https://bit.ly/3hktKEm.

    # Clear before and after the test, so that this also clears in case that previous
    # tests set the instance.

    IERS._IERS__instance = None
    yield
    IERS._IERS__instance = None
