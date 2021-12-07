from __future__ import annotations

import numpy

from . import calibration, defaults
from .coordinate import Coordinate, Coordinate2D
from .exceptions import CoordIOError


class Guide(Coordinate2D):
    """Guide coordinates are 2D cartesian xy coordinates.  Units are pixels.
    The origin is the lower left corner of the lower left pixel when looking
    at the CCD through the camera window.  Thus guide images are taken in
    "selfie-mode".  So the coordinate (0.5, 0.5) is the center of the lower
    left pixel. -y points (generally) toward the wok vertex.  True rotations
    and locations of GFAs in the wok are measured/calibrated after
    installation and on-sky characterization.

    Parameters
    -----------
    value : numpy.ndarray
        A Nx2 array where [x,y] are columns.  Or `.Tangent`
    xBin : int
        CCD x bin factor, defaults to 1
    yBin : int
        CCD y bin factor, defaults to 1

    Attributes
    -----------
    guide_warn : numpy.ndarray
        boolean array indicating coordinates outside the CCD FOV
    """

    __extra_params__ = ["xBin", "yBin"]
    __warn_arrays__ = ["guide_warn"]

    xBin: int
    yBin: int
    guide_warn: numpy.ndarray

    def __new__(cls, value, **kwargs):

        xBin = kwargs.get("xBin", None)
        if xBin is None:
            kwargs["xBin"] = 1

        yBin = kwargs.get("yBin", None)
        if yBin is None:
            kwargs["yBin"] = 1

        if isinstance(value, Coordinate):
            if value.coordSysName == "Tangent":
                # going from 3D -> 2D coord sys
                # reduce dimensionality
                arrInit = numpy.zeros((len(value), 2))
                obj = super().__new__(cls, arrInit, **kwargs)
                obj._fromTangent(value)
            else:
                raise CoordIOError(
                    'Cannot convert to Guide from %s'%value.coordSysName
                )
        else:
            obj = super().__new__(cls, value, **kwargs)
            obj._fromRaw()

        return obj

    def _fromTangent(self, tangentCoords):
        """Convert from tangent coordinates to guide coordinates

        """
        if tangentCoords.holeID not in calibration.VALID_GUIDE_IDS:
            raise CoordIOError(
                "Cannot convert from wok hole %s to Guide" %
                tangentCoords.holeID
            )
        # make sure the guide wavelength is specified for all coords
        # this may be a little too extreme of a check
        badWl = numpy.sum(
            tangentCoords.wavelength - defaults.INST_TO_WAVE["GFA"]
        ) != 0
        if badWl:
            raise CoordIOError(
                "Cannont convert to Guide coords from non-guide wavelength"
            )

        # use coords projected to tangent xy plane
        # could almost equivalently use tangentCoords[:,0:2]
        # and we may want to ditch the projection eventually if its
        # unnecessary

        # note, moved this calculation into
        # conv.tangentToGuide, may want to just use that one
        xT = tangentCoords.xProj
        yT = tangentCoords.yProj

        xPix = (1 / self.xBin) * (
            defaults.MICRONS_PER_MM / defaults.GFA_PIXEL_SIZE * xT + \
            defaults.GFA_CHIP_CENTER
        )

        yPix = (1 / self.yBin) * (
            defaults.MICRONS_PER_MM / defaults.GFA_PIXEL_SIZE * yT + \
            defaults.GFA_CHIP_CENTER
        )

        self[:, 0] = xPix
        self[:, 1] = yPix

        self._fromRaw()

    def _fromRaw(self):
        """Find coords that may be out of the FOV and tag them

        """

        # note this might be wrong when xBin is 3
        maxX = defaults.GFA_CHIP_CENTER * 2 / self.xBin
        maxY = defaults.GFA_CHIP_CENTER * 2 / self.yBin
        xPix = self[:, 0]
        yPix = self[:, 1]

        arg = numpy.argwhere((xPix < 0) | (xPix > maxX) | \
                             (yPix < 0) | (yPix > maxY))

        if len(arg) == 0:
            # everything in range...
            return

        arg = arg.squeeze()
        self.guide_warn[arg] = True
