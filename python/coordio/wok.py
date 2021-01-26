import numpy

from .coordinate import Coordinate, verifySite
from .exceptions import CoordIOError
from . import conv
from . import defaults


class Wok(Coordinate):
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

    def __new__(cls, value, **kwargs):

        verifySite(kwargs, strict=False)

        obsAngle = kwargs.get("obsAngle", None)
        if obsAngle is None:
            kwargs["obsAngle"] = 0.0

        if isinstance(value, Coordinate):
            if value.coordSysName == "FocalPlane":
                obj = super().__new__(cls, value, **kwargs)
                obj._fromFocalPlane(value)
            elif value.coordSysName == "Tangent":
                # wok is a 3D coord sys, tangent is 2D Cartesian
                initArray = numpy.zeros((len(value), 3))
                obj = super().__new__(cls, initArray, **kwargs)
                obj._fromTangent(value)
            else:
                raise CoordIOError(
                    "Cannot convert to Field from %s"%value.coordSysName
                )
        else:
            obj = super().__new__(cls, value, **kwargs)

    def _fromFocalPlane(self, fpCoords):
        """Converts from focal plane coords to wok coords

        Parameters:
        -------------
        fpCoords : `.FocalPlane`

        """
        xOff, yOff, zOff, tiltX, tiltY = defaults.getWokOrient(self.site.name)
        xWok, yWok, zWok = conv.focalToWok(
            fpCoords[:,0], fpCoords[:,1], fpCoords[:,2],
            self.obsAngle, xOff, yOff, zOff, tiltX, tiltY
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
        pass





