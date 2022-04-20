#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-16
# @Filename: iers.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import os
import urllib.request
import warnings

import numpy

from . import config, sofa
from .exceptions import CoordIOError, CoordIOUserWarning


BASE_URL = 'https://datacenter.iers.org/products/eop/rapid/'


class IERS:
    """Wrapper around the IERS bulletins.

    Parameters
    ----------
    path : str
        The path where the IERS table, in CSV format, lives or will be saved
        to. If `None`, defaults to ``config['iers']['path']``.
    channel : str
        The IERS channel to use. Right now only ``finals`` is implemented.
    download : bool
        If the IERS table is not found on disk or it's out of date, download
        an updated copy.

    Attributes
    ----------
    data : ~numpy.ndarray
        A record array with the parsed IERS data, trimmed to remove rows for
        which ``UT1-UTC`` is not defined.

    """

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
            obj.path = os.path.join(os.path.expanduser(path), 'finals2000A.data.csv')
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
        """Update the IERS table, downloading the latest available version."""

        path = path or self.path
        channel = channel or self.channel

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        if channel == 'finals':
            URL = BASE_URL + 'standard/csv/finals.data.csv'
            warnings.warn(f'Downloading IERS table from {URL}.', CoordIOUserWarning)
            url = urllib.request.urlopen(URL)
            data = url.read()

            with open(path, 'wb') as fd:
                fd.write(data)
            url.close()
        else:
            raise NotImplementedError('Only finals channels is implemented.')

        self.load_data(path=path)

    def _get_current_jd(self):
        """Returns the current JD in the scale of the computer clock."""

        jd1, jd2 = sofa.get_internal_date()
        jd = jd1 + jd2

        return jd

    def is_valid(self, jd=None, offset=5, download=True):
        """Determines whether a JD is included in the loaded IERS table.

        Parameters
        ----------
        jd : float
            The Julian date to check. There is a certain ambiguity in the
            format of the MJD in the IERS tables but for most purposes the
            difference between UTC and TAI should be meaningless for these
            purposes.
        offset : int
            Number of days to look ahead. If ``jd+offset`` is not included in
            the IERS table the method will return `False`.
        download : bool
            Whether to automatically download an updated version of the IERS
            table if the requested date is not included in the current one.

        Returns
        -------
        is_valid : `bool`
            Whether the date is valid within the current (or newly downloaded)
            IERS table.

        """

        if self.data is None:
            raise CoordIOError('IERS data has not been loaded')

        if jd is None:
            jd = self._get_current_jd()

        mjd = int(jd - 2400000.5)

        min_mjd = self.data['MJD'].min()
        max_mjd = self.data['MJD'].max()

        if mjd - offset > min_mjd and mjd + offset < max_mjd:
            return True
        else:
            if download:
                self.update_data()
                return self.is_valid(jd, offset, download=False)
            return False

    def load_data(self, path=None):
        """Loads an IERS table in CSV format from ``path``."""

        path = path or self.path

        if not os.path.exists(path):
            raise FileNotFoundError(f'File {path!r} not found.')

        self.data = numpy.genfromtxt(path, delimiter=';', names=True,
                                     dtype=None, encoding='UTF-8')

        # Trim rows without UT1-UTC data
        self.data = self.data[~numpy.isnan(self.data['UT1UTC'])]

    def get_delta_ut1_utc(self, jd=None, download=True):
        """Returns the interpolated ``UT1-UTC`` value, in seconds."""

        if jd is None:
            jd = self._get_current_jd()

        if not self.is_valid(jd, offset=0, download=download):
            raise CoordIOError('IERS table is out of date.')

        # Whats the timescale for MJD in the IERS tables?
        # Probably doesn't matter.
        mjd = jd - 2400000.5

        return numpy.interp(mjd, self.data['MJD'], self.data['UT1UTC'])
