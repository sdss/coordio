from __future__ import annotations

from typing import TYPE_CHECKING

from . import conv, defaults
from .coordinate import Coordinate, Coordinate3D, verifySite
from .exceptions import CoordIOError


if TYPE_CHECKING:
    from .site import Site


class Wok(Coordinate3D):
    """A representation of a Wok Coordinates.  A 3D Cartesian coordinate
    system.  The orgin is the vertex of the Wok. -x points toward the Boss
    slithead.  +z points from earth to sky.  +y is dictated by right hand
    rule (so it points toward GFA-S2).

    A rotation about focal coords' +z is applied to observe fields at non-zero position
    angles

    With perfect alignment to the rotator at zero position angle, the xyz
    axes of the wok are idenentical to those of the focal plane system.
    The origin of the wok is translated along z with respect to the focal
    plane system by an amount determined by the ZEMAX models of the focal
    plane.

    True translations and tilts of the wok with respect to the focal plane
    are measured on sky (at zero position angle) and stored in the
    ``etc/wokOrientation.csv`` file.

    Parameters
    ------------
    value : numpy.ndarray
        A Nx3 array where [x,y,z] are columns. Or `.FocalPlane`.  Or `.Tangent`.
    site : `.Site`
        site name determines which wok parameters to use.  Mandatory parameter.
    obsAngle : float
        Position angle of observation. Angle measured from (image) North
        through East to wok +y. So obsAngle of 45 deg, wok +y points NE.
        Defaults to zero.
    """

    __extra_params__ = ["site", "obsAngle"]

    site: Site
    obsAngle: float

    def __new__(cls, value, **kwargs):

        verifySite(kwargs, strict=False)

        obsAngle = kwargs.get("obsAngle", None)
        if obsAngle is None:
            kwargs["obsAngle"] = 0.0

        obj = super().__new__(cls, value, **kwargs)

        if isinstance(value, Coordinate):
            if value.coordSysName == "FocalPlane":
                obj._fromFocalPlane(value)
            elif value.coordSysName == "Tangent":
                obj._fromTangent(value)
            else:
                raise CoordIOError(
                    "Cannot convert to Wok from %s" % value.coordSysName
                )

        return obj

    def _fromFocalPlane(self, fpCoords):
        """Converts from focal plane coords to wok coords

        Parameters:
        -------------
        fpCoords : `.FocalPlane`

        """
        xOff, yOff, zOff, tiltX, tiltY = defaults.getWokOrient(self.site.name)
        xWok, yWok, zWok = conv.focalToWok(
            fpCoords[:, 0], fpCoords[:, 1], fpCoords[:, 2],
            self.obsAngle, xOff, yOff, zOff, tiltX, tiltY, fpCoords.fpScale,
            projectFlat=True  # project flat when using a flat wok model
        )

        self[:, 0] = xWok
        self[:, 1] = yWok
        self[:, 2] = zWok

    def _fromTangent(self, tangentCoords):
        """Converts from tangent coords to wok coords

        Parameters:
        -------------
        tangentCoords : `.Tangent`

        """
        tx = tangentCoords[:, 0]
        ty = tangentCoords[:, 1]
        tz = tangentCoords[:, 2]

        b, iHat, jHat, kHat = defaults.getHoleOrient(
            tangentCoords.site.name, tangentCoords.holeID
        )
        positioner_data = defaults.getPositionerData(self.site.name,
                                                     tangentCoords.holeID)

        xWok, yWok, zWok = conv.tangentToWok(
            tx, ty, tz, b, iHat, jHat, kHat,
            scaleFac=tangentCoords.scaleFactor,
            dx=positioner_data[9], dy=positioner_data[10]
        )

        self[:, 0] = xWok
        self[:, 1] = yWok
        self[:, 2] = zWok
