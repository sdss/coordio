from __future__ import annotations

from typing import TYPE_CHECKING

import numpy

from . import calibration, conv, defaults
from .coordinate import Coordinate, Coordinate2D, verifySite
from .exceptions import CoordIOError


if TYPE_CHECKING:
    from .site import Site


class PositionerBase(Coordinate2D):
    """A representation of Positioner coordinates.  Alpha/Beta coordinates
    in degrees.

    When converting from tangent coordinates,
    Robot's are always "right handed" with alpha in [0, 360], beta in [0, 180]

    Parameters
    ------------
    value : numpy.ndarray
        A Nx2 array with columns [alpha, beta] in degrees
    site : `.Site`
        mandatory parameter
    holeID : str
        valid identifier, one of calibration.VALID_HOLE_IDS

    """

    __extra_params__ = ["site", "holeID"]
    __warn_arrays__ = ["positioner_warn"]
    __fiber_type__ = None  # overridden by subclasses

    # For typing.
    positioner_warn: numpy.ndarray
    holeID: str
    site: Site

    def __new__(cls, value, **kwargs):

        verifySite(kwargs, strict=False)

        holeID = kwargs.get("holeID", None)
        if holeID is None:
            raise CoordIOError("Must specify holeID for Positioner Coords")
        if holeID not in calibration.VALID_HOLE_IDS:
            raise CoordIOError("Must be valid holeID for Positioner Coords")

        if isinstance(value, Coordinate):
            if value.coordSysName == "Tangent":
                if holeID.startswith("GFA"):
                    raise CoordIOError(
                        "Guide holeID supplied for Positioner coord"
                    )
                # going from 3D to 2D coordsys
                # initialize array
                initArr = numpy.zeros((len(value), 2))
                obj = super().__new__(cls, initArr, **kwargs)
                obj._loadFiberData()
                obj._fromTangent(value)
            else:
                raise CoordIOError(
                    'Cannot convert to Positioner from %s' % value.coordSysName
                )
        else:
            obj = super().__new__(cls, value, **kwargs)
            obj._loadFiberData()
            obj._fromRaw()

        if obj.__fiber_type__ not in ["Apogee", "Boss", "Metrology"]:
            raise CoordIOError("valid __fiber_type__ must be specified!")

        return obj

    def _loadFiberData(self):
        fiberData = defaults.getPositionerData(self.site.name, self.holeID)

        if self.__fiber_type__ == "Metrology":
            xFiber = fiberData[1]
            yFiber = fiberData[2]

        elif self.__fiber_type__ == "Apogee":
            xFiber = fiberData[3]
            yFiber = fiberData[4]

        elif self.__fiber_type__ == "Boss":
            xFiber = fiberData[5]
            yFiber = fiberData[6]
        else:
            raise CoordIOError("Fiber not specified for positioner coords")

        self.alphaArmLength = fiberData[0]
        self.xFiber = xFiber
        self.yFiber = yFiber

        self.alphaOffset = fiberData[7]
        self.betaOffset = fiberData[8]

    def _fromTangent(self, tangentCoords):
        """Convert from tangent coords to alpha betas
        """
        xTangent = tangentCoords.xProj
        yTangent = tangentCoords.yProj

        # this will always return a right hand orientation
        alphaDeg, betaDeg, isOK = conv.tangentToPositioner(
            xTangent,
            yTangent,
            self.xFiber,
            self.yFiber,
            self.alphaArmLength,
            self.alphaOffset,
            self.betaOffset
        )

        self[:, 0] = alphaDeg
        self[:, 1] = betaDeg
        self.positioner_warn[:] = isOK == False
        self._fromRaw()

    def _fromRaw(self):

        # TODO: maybe not do this?
        self = self % 360


class PositionerBoss(PositionerBase):
    __fiber_type__ = "Boss"


class PositionerApogee(PositionerBase):
    __fiber_type__ = "Apogee"


class PositionerMetrology(PositionerBase):
    __fiber_type__ = "Metrology"
