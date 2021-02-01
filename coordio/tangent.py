import numpy
import warnings

from .coordinate import Coordinate, Coordinate3D, verifySite, verifyWavelength
from .telescope import FocalPlane
from .wok import Wok
from .exceptions import CoordIOError
from . import defaults
from . import conv


def _getRayOrigins(site, holeID, scaleFactor, obsAngle):
    """Return the location of the spherical focal surface center
    in tangent coordinates for a specific wok location

    site : `.Site
        site with name attribute "APO" or "LCO"
    holeID : str
        A valid hole identifier
    scaleFactor : float
        Scale factor for wok expansion
    obsAngle : float
        observation (position) angle in Deg

    Returns
    --------
    apCen : numpy.ndarray
        3 element array in focal plane coords for center of apogee focal sphere
    bossCen : numpy.ndarray
        3 element array in focal plane coords for center of boss focal sphere
    gfaCen : numpy.ndarray
        3 element array in focal plane coords for center of gfa focal sphere

    """
    outList = []
    direction = "focal"  # irrelevant, just getting sphere param
    for waveCat in ["Apogee", "Boss", "GFA"]:
        R, b, c0, c1, c2, c3, c4 = defaults.getFPModelParams(
            site.name, direction, waveCat
        )
        fpXYZ = [[0, 0, b]] # sphere's center in focal plane coords
        fpCoords = FocalPlane(fpXYZ, site=site)
        wokCoords = Wok(fpCoords, site=site, obsAngle=obsAngle)
        tanCoords = TangentNoProj(
            wokCoords, site=site, holeID=holeID,
            scaleFactor=scaleFactor, obsAngle=obsAngle
        )
        tanCoords = numpy.array(tanCoords).squeeze()
        outList.append(tanCoords)

    return tuple(outList)


class Tangent(Coordinate3D):
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
    wavelength : float or numpy.ndarray
        wavelength used for projecting rays to tangent surfaces (from sphere
        model origin).  Defaults to GFA wavelength

    Attributes
    ---------------
    xProj : numpy.ndarray
        x projection of coordinate to the xy plane
    yProj : numpy.ndarray
        y projection of coordinate to xy plane
    distProj : numpy.ndarray
        distance of projection (mm) proxy for focus offset
        positive when above tangent surface, negative when coord
        is below tangent surface
    obsAngle : float
        Position angle of observation. Angle measured from (image) North
        through East to wok +y. So obsAngle of 45 deg, wok +y points NE.
        Defaults to zero.
    """

    __extra_params__ = ["site", "holeID", "scaleFactor", "obsAngle"]
    __extra_arrays__ = ["wavelength"]
    __computed_arrays__ = ["xProj", "yProj", "distProj"]

    def __new__(cls, value, **kwargs):

        verifySite(kwargs, strict=False)
        verifyWavelength(kwargs, len(value), strict=True)
        # print("gahhh", kwargs["wavelength"])

        holeID = kwargs.get("holeID", None)
        if holeID is None:
            raise CoordIOError("Must specify holeID for Tangent Coords")
        if holeID not in defaults.VALID_HOLE_IDS:
            raise CoordIOError("Must be valid holeID for Tangent Coords")
        scaleFactor = kwargs.get("scaleFactor", None)
        if scaleFactor is None:
            # default to scale factor of 1
            kwargs["scaleFactor"] = 1

        if isinstance(value, Coordinate):
            if "Positioner" in value.coordSysName:
                if holeID.startswith("GFA"):
                    raise CoordIOError(
                        "Guide holeID supplied for Positioner coord"
                    )
                # going from 2D to 3D coordsys
                # initialize array
                initArr = numpy.zeros((len(value), 3))
                obj = super().__new__(cls, initArr, **kwargs)
                obj._fromPositioner(value)
            elif value.coordSysName == "Guide":
                # going from 2D to 3D coordsys
                # initialize array
                if not holeID.startswith("GFA"):
                    raise CoordIOError(
                        "Cannot convert from guide coords to non-GFA location"
                    )
                initArr = numpy.zeros((len(value), 3))
                obj = super().__new__(cls, initArr, **kwargs)
                obj._fromGuide(value)
            elif value.coordSysName == "Wok":
                # going from 3D to 3D coordsys
                obj = super().__new__(cls, value, **kwargs)
                obj._fromWok(value)
            else:
                raise CoordIOError(
                    "Cannot convert to Tangent from %s" % value.coordSysName
                )

        else:
            obj = super().__new__(cls, value, **kwargs)
            obj._fromRaw()

        return obj

    def _fromPositioner(self, posCoords):
        """Convert from positioner coords to tangent coords

        Parameters
        ------------
        posCoords : `.Positioner`

        """
        if not numpy.isfinite(numpy.sum(posCoords)):
            warnings.warn("NaN values propigated from positioner coordinates")
        xTangent, yTangent = conv.positionerToTangent(
            posCoords[:, 0], posCoords[:, 1], posCoords.xFiber,
            posCoords.yFiber, posCoords.alphaArmLength
        )
        self[:, 0] = xTangent
        self[:, 1] = yTangent
        self[:, 2] = 0

        self._fromRaw()

    def _fromGuide(self, guideCoords):
        """Convert from guide coords to tangent coords

        Parameters
        ------------
        guideCoords : `.Guide`
        """
        # print("from guide", self.wavelength)
        if self.holeID not in defaults.VALID_GUIDE_IDS:
            raise CoordIOError(
                "Cannot convert from Guide to (non-guide) wok hole %s" %
                self.holeID
            )

        # make sure the guide wavelength is specified for all coords
        # this may be a little too extreme of a check
        if numpy.sum(self.wavelength - defaults.INST_TO_WAVE["GFA"]) != 0:
            raise CoordIOError(
                "Cannont convert from Guide coords with non-guide wavelength"
            )

        xPix = guideCoords[:, 0]
        yPix = guideCoords[:, 1]
        xBin = guideCoords.xBin
        yBin = guideCoords.yBin

        xTangent = (xPix * xBin - defaults.GFA_CHIP_CENTER) * \
                    defaults.GFA_PIXEL_SIZE / defaults.MICRONS_PER_MM

        yTangent = (yPix * yBin - defaults.GFA_CHIP_CENTER) * \
                    defaults.GFA_PIXEL_SIZE / defaults.MICRONS_PER_MM

        self[:, 0] = xTangent
        self[:, 1] = yTangent
        self[:, 2] = 0  # by definition pixels are in the z=0 tangent plane

        # don't really have to call from raw
        # could just populate proj arrays directly...
        self._fromRaw()

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

        self._fromRaw()

    def _fromRaw(self):
        """Compute projections to xy plane

        """

        rayCenters = _getRayOrigins(
            self.site, self.holeID, self.scaleFactor, self.obsAngle
        )

        for cen, waveCat in zip(rayCenters, ["Apogee", "Boss", "GFA"]):
            arg = numpy.argwhere(
                self.wavelength == defaults.INST_TO_WAVE[waveCat]
            )

            if len(arg) == 0:
                continue

            arg = arg.squeeze()

            _x, _y, _z = self[arg, 0], self[arg, 1], self[arg, 2]
            xProj, yProj, zProj, distProj = conv.proj2XYplane(_x, _y, _z, cen)
            # note zProj is always zero!
            self.xProj[arg] = xProj
            self.yProj[arg] = yProj
            self.distProj[arg] = distProj


class TangentNoProj(Tangent):
    """Class that doesn't compute projections, intended for internal
    use only to eliminate a recursion problem.

    """
    def _fromRaw(self):
        pass
