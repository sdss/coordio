#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-11-13
# @Filename: calibration.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import warnings
from typing import Optional

import pandas

from coordio.exceptions import CoordIOUserWarning


class Calibration:
    """Calibration class.

    This class holds several Pandas data frames containing the wok and
    positioner calibrations. On instantiation it loads the calibration
    from ``$WOKCALIB_DIR``, if it exists. Additional calibrations can
    be manually added by calling `.add_calibration`, which are appended
    to the existing calibrations.

    Parameters
    ----------
    paths : str or or list of str or `None`
        The paths to the directory containing the calibration files.
        Defaults to ``$WOKCALIB_DIR``.

    """

    def __init__(self, paths: Optional[str | list[str]] = None):

        if paths is None:
            paths = os.environ.get('WOKCALIB_DIR', '').split(':')
        self.paths = paths

        self.FP_MODEL = pandas.DataFrame()
        self.wokOrient = pandas.DataFrame()
        self.positionerTable = pandas.DataFrame()
        self.wokCoords = pandas.DataFrame()
        self.fiducialCoords = pandas.DataFrame()
        self.fiberAssignments = pandas.DataFrame()
        self.gfaCoords = pandas.DataFrame()

        self.VALID_HOLE_IDS: set[str] = set([])
        self.VALID_GUIDE_IDS: list[str] = []

        self.sites: list[str] = []

        self.fps_calibs_version = 'unknown'

        for path in self.paths:
            if not os.path.exists(path):
                warnings.warn('Calibration path or $WOKCALIB_DIR are not set. '
                              'Coordinate transformations will fail.',
                              CoordIOUserWarning)
                return

            self.add_calibration(path)

        try:
            import fps_calibrations  # type: ignore
            self.fps_calibs_version = fps_calibrations.get_version()
        except ImportError:
            warnings.warn('Cannot retrieve the version of the wok calibrations. '
                          'Consider adding the root of fps_calibrations to PYTHONPATH.',
                          CoordIOUserWarning)

    def add_calibration(self,
                        path: str,
                        focalPlaneFile='focalPlaneModel.csv',
                        wokOrientFile='wokOrientation.csv',
                        positionerTableFile='positionerTable.csv',
                        wokCoordsFile='wokCoords.csv',
                        fiducialCoordsFile='fiducialCoords.csv',
                        fiberAssignmentsFile='fiberAssignments.csv',
                        gfaCoordsFile="gfaCoords.csv"):
        """Add calibration files from a ``path``."""

        path = str(path)
        if not os.path.exists(path):
            raise FileExistsError(f'Path {path} does not exist.')

        # Focal plane model
        self._add_calibration(path, focalPlaneFile, 'FP_MODEL',
                              indices=['site', 'direction', 'waveCat'])

        # Wok orientation
        self._add_calibration(path, wokOrientFile, 'wokOrient', indices='site')

        # Wok coordinates
        self._add_calibration(path, wokCoordsFile, 'wokCoords',
                              indices=['site', 'holeID'], index_col=0)

        self.VALID_HOLE_IDS = set(self.wokCoords.reset_index().holeID.tolist())
        self.VALID_GUIDE_IDS = [ID for ID in self.VALID_HOLE_IDS if ID.startswith('GFA')]

        # Positioner data
        self._add_calibration(path, positionerTableFile, 'positionerTable',
                              indices=['site', 'holeID'], index_col=0)
        self.sites = list(set(self.positionerTable.index.get_level_values(0)))

        # Fiducials
        self._add_calibration(path, fiducialCoordsFile, 'fiducialCoords',
                              indices=['site', 'holeID'], index_col=0)

        # Fibre assignments
        self._add_calibration(path, fiberAssignmentsFile, 'fiberAssignments',
                              indices=['site', 'holeID'], index_col=0)

        # Fibre assignments
        self._add_calibration(path, gfaCoordsFile, 'gfaCoords',
                              indices=['site', 'id'], index_col=0)

    def _add_calibration(self, path: str, file: str, variable: str,
                         indices: str | list | None = None, index_col=None):
        """Loads and concatenates one file."""

        path = os.path.join(path, file)
        if not os.path.exists(path):
            warnings.warn(f'Cannot find file {path}.', CoordIOUserWarning)
            return

        # Sets the file name.
        setattr(self, variable + 'File', path)

        new = pandas.read_csv(path, comment="#", index_col=index_col)
        if len(new) == 0:
            return path

        # Check that we don't have already loaded a calibration for this site.
        if 'site' not in new:
            warnings.warn(f'Column site not found in {path}', CoordIOUserWarning)
            return
        else:
            new_sites = set(new.site.tolist())

        current = getattr(self, variable)
        if len(current) > 0:
            sites = set(current.reset_index().site.tolist())
            if len(sites & new_sites) > 0:
                raise ValueError('Some new sites already exist in calibration.')

        if indices:
            new.set_index(indices, inplace=True)

        setattr(self, variable, pandas.concat([current, new]))

        return path
