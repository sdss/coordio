import numpy

from .coordinate import Coordinate, verifySite
from .exceptions import CoordIOError
from . import defaults
from . import conv


class Tangent(Coordinate):
    """A representation of Tangent Coordinates.  A 3D Cartesian coordinate
    system.  The xy plane of this coordinate system is tangent to the wok
    surface at a specific location (one of holeID in etc/wokCoords.csv).
    For positioners, +x axis is a aligned with the alpha=0 direction.  For
    GFA's the -y axis points toward the wok coordinate system origin. +z axis
    (generally) points from the earth to sky.  Origin lies 143 mm above the
    wok surface.  This puts the Tangent xy plane in the plane of the fiber or
    chip.

    Parameters
    ------------
    value : numpy.ndarray
        A Nx3 array where [x,y,z] are columns. Or `.Wok`.  Or `.Positioner`.
        Or `.Guide`.
    site : `.Site`
        site name determines which wok parameters to use.  Mandatory parameter.
    holeID : str
        vaild identifier, one of defaults.VALID_HOLE_IDS
    scaleFactor : float
        multiplicative factor to apply, modeling thermal expansion/contraction
        of wok holes with respect to each other.  Defaults to 1

    """

    __extra_params__ = ["site", "holeID", "scaleFactor"]

    def __new__(cls, value, **kwargs):

        verifySite(kwargs, strict=False)

        holeID = kwargs.get("holeID", None)
        if holeID is None:
            raise CoordIOError("Must specify holeID for Tangent Coords")
        if holeID not in defaults.VALID_HOLE_IDS:
            raise CoordIOError("Must valid holeID for Tangent Coords")
        scaleFactor = kwargs.get("scaleFactor", None)
        if scaleFactor is None:
            # default to scale factor of 1
            kwargs["scaleFactor"] = 1

        if isinstance(value, Coordinate):
            if value.coordSysName == "Positioner":
                # going from 2D to 3D coordsys
                # initialize array
                initArr = numpy.zeros((len(value), 3))
                obj = super().__new__(cls, initArr, **kwargs)
                obj._fromPositioner(value)
            elif value.coordSysName == "Guide":
                # going from 2D to 3D coordsys
                # initialize array
                initArr = numpy.zeros((len(value), 3))
                obj = super().__new__(cls, initArr, **kwargs)
                obj._fromGuide(value)
            elif value.coordSysName == "Wok":
                # going from 3D to 3D coordsys
                obj = super().__new__(cls, value, **kwargs)
                obj._fromWok(value)
            else:
                raise CoordIOError(
                    "Cannot convert to Field from %s"%value.coordSysName
                )

        else:
            obj = super().__new__(cls, value, **kwargs)

        return obj

    def _fromPositioner(self, posCoords):
        """Convert from positioner coords to tangent coords

        Parameters
        ------------
        posCoords : `.Positioner`

        """
        pass

    def _fromGuide(self, guideCoords):
        """Convert from guide coords to tangent coords

        Parameters
        ------------
        guideCoords : `.Guide`
        """
        pass

    def _fromWok(self, wokCoords):
        """Convert from wok coords to tangent coords

        Parameters
        -----------
        wokCoords : `.Wok`
        """
        xWok = wokCoords[:, 0]
        yWok = wokCoords[:, 1]
        zWok = wokCoords[:, 2]

        b, iHat, jHat, kHat = defaults.getHoleOrient(self.site.name, self.holeID)
        tx, ty, tz = conv.wokToTangent(
            xWok, yWok, zWok, b, iHat, jHat, kHat, scaleFac=self.scaleFactor
        )
        self[:, 0] = tx
        self[:, 1] = ty
        self[:, 2] = tz


