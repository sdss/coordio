#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-16
# @Filename: iers.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import os
import urllib
import warnings

import numpy

from . import config, sofa
from .exceptions import CoordIOError, CoordIOUserWarning


BASE_URL = 'ftp://ftp.iers.org/products/eop/rapid/'


class IERS:

    __instance = None

    def __new__(cls, path=None, channel='finals', download=True):

        if cls.__instance is not None:

            if cls.__instance.is_valid():
                return cls.__instance
            else:
                # We take the default route which takes care of updating the
                # file.
                pass

        obj = super().__new__(cls)
        obj.data = None
        obj.channel = channel or config['iers']['channel']

        path = path or config['iers']['path']

        if channel == 'finals':
            obj.path = os.path.join(os.path.expanduser(path),
                                    'finals2000A.data.csv')
        else:
            raise NotImplementedError('Only finals channels is implemented.')

        if os.path.exists(obj.path):
            cls.load_data(obj)
            if not cls.is_valid(obj):
                warnings.warn('Current IERS file is out of date. '
                              'Redownloading.', CoordIOUserWarning)
                cls.update_data(obj, channel=obj.channel)
        else:
            cls.update_data(obj, channel=obj.channel)

        # Check data one last time.
        if not cls.is_valid(obj):
            raise CoordIOError('IERS table is not valid. '
                               'This should not have happened.')

        cls.__instance = obj

        return cls.__instance

    def update_data(self, path=None, channel=None):

        path = path or self.path
        channel = channel or self.channel

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        if channel == 'finals':
            URL = BASE_URL + 'standard/csv/finals.data.csv'
            warnings.warn(f'Downloading IERS table from {URL}.',
                          CoordIOUserWarning)
            urllib.request.urlretrieve(URL, path)
        else:
            raise NotImplementedError('Only finals channels is implemented.')

        self.load_data(path=path)

    def _get_current_jd(self):

        if sofa is None:
            raise CoordIOError('SOFA library not available.')

        jd1, jd2 = sofa.get_internal_date()
        jd = jd1 + jd2

        return jd

    def is_valid(self, jd=None, offset=5, download=True):

        if self.data is None:
            raise CoordIOError('IERS data has not been loaded')

        if jd is None:
            jd = self._get_current_jd()

        mjd = jd - 2400000.5 + offset

        max_mjd = self.data['MJD'].max()

        if int(mjd) < max_mjd and int(mjd) + 1 < max_mjd:
            return True
        else:
            if download:
                self.update_data()
                return self.is_valid(jd, offset, download=False)
            return False

    def load_data(self, path=None):

        path = path or self.path

        if not os.path.exists(path):
            raise FileNotFoundError(f'File {path!r} not found.')

        self.data = numpy.genfromtxt(path, delimiter=';', names=True,
                                     dtype=None, encoding='UTF-8')

        # Trim rows without UT1-UTC data
        self.data = self.data[~numpy.isnan(self.data['UT1UTC'])]

    def get_delta_ut1_utc(self, jd=None, download=True):

        if jd is None:
            jd = self._get_current_jd()

        if not self.is_valid(jd, offset=0, download=download):
            raise CoordIOError('IERS table is out of date.')

        # Whats the timescale for MJD in the IERS tables?
        # Probably doesn't matter.
        mjd = jd - 2400000.5

        return numpy.interp(mjd, self.data['MJD'], self.data['UT1UTC'])
