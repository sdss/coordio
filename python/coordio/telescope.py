import numpy
import pandas as pd
import os

from .coordinate import Coordinate
from .exceptions import CoordIOError, CoordIOWarning
from .utils import sph2Cart, cart2Sph, cart2FieldAngle
from .site import Site
# from .focalPlaneModel import focalPlaneModelDict

# from .sky import Observed, ICRS
# from .site import Site

__all__ = ["Field", "FocalPlane"]

WAVE_TO_INST = {
    5400.0: "Boss",
    6231.0: "GFA",
    16600.0: "Apogee"
}

INST_TO_WAVE = {
    "Boss": 5400.0,
    "GFA": 6231.0,
    "Apogee": 16600.0
}

APO_MAX_FIELD_R = 1.5 # max field radius (deg)
LCO_MAX_FIELD_R = 1 # max field radius (deg)

# read in the focal plane model file
fpModelFile = os.path.join(os.path.dirname(__file__), "etc", "focalPlaneModel.csv")
FP_MODEL = pd.read_csv(fpModelFile, comment="#")


def getModelParams(site, direction, waveCat):
    """Find the right focal plane model to use given site, direction
    and waveCat.

    Parameters
    ------------
    site: str
        "APO" or "LCO"
    direction: str
        "focal" or "field" for the direction of the conversion
    waveCat: str
        "Apogee", "Boss", or "GFA" for the wavelength

    Returns
    ---------
    R: float
        radius of spherical curvature (mm)
    b: float
        z-location of sphere center in focal plane coordinate system
    c0: float
        1st order coeff
    c1: float
        3rd order coeff
    c2: float
        5th order coeff
    c3: float
        7th order coeff
    c4: float
        9th order coeff
    """
    # filter for correct row in the model dataframe
    row = FP_MODEL[
        (FP_MODEL.site == site)
        & (FP_MODEL.direction == direction)
        & (FP_MODEL.waveCat == waveCat)
    ]

    # print("row", row)

    R = float(row.R)
    b = float(row.b)
    c0 = float(row.c0)
    c1 = float(row.c1)
    c2 = float(row.c2)
    c3 = float(row.c3)
    c4 = float(row.c4)
    return R, b, c0, c1, c2, c3, c4


def fieldToFocal(thetaField, phiField, site, waveCat):
    """Convert spherical field coordinates to the modeled spherical
    position on the focal plane.  Origin is the M1 vertex.

    Parameters
    -----------
    thetaField: float or numpy.ndarray
        azimuthal angle of field coordinate (deg)
    phiField: float or numpy.ndarray
        polar angle of field coordinate (deg)
    site: str
        "APO" or "LCO"
    waveCat: str
        "Boss", "Apogee", or "GFA"

    Result
    -------
    xFocal: float or numpy.ndarray
        spherical x focal coord mm
    yFocal: float or numpy.ndarray
       spherical y focal coord mm
    zFocal: float or numpy.ndarray
        spherical z focal coord mm
    R: float
        radius of curvature for sphere
    b: float
        z location of center of sphere
    fieldWarn: bool or boolean array
        True if angle off-axis is large enough to be suspicious
    """

    if site == "APO":
        fieldWarn = numpy.array(phiField) > APO_MAX_FIELD_R
    if site == "LCO":
        fieldWarn = numpy.array(phiField) > LCO_MAX_FIELD_R

    if hasattr(fieldWarn, "__len__"):
        if True in fieldWarn:
            print(  # turn this into warning eventually, problem for tests?
                "Warning! Far off-axis coordinate, conversion may be bogus"
            )
    elif fieldWarn is True:
        print(  # turn this into warning eventually, problem for tests?
            "Warning! Far off-axis coordinate, conversion may be bogus"
        )

    direction = "focal"
    R, b, c0, c1, c2, c3, c4 = getModelParams(site, direction, waveCat)

    phiFocal = c0 * phiField + c1 * phiField**3 + c2 * phiField**5 \
        + c3 * phiField**7 + c4 * phiField**9
    # thetaField = thetaFocal by definition
    # convert back to xyz coords
    rTheta = numpy.radians(thetaField)
    rPhi = numpy.radians(180 - phiFocal)
    xFocal = R * numpy.cos(rTheta) * numpy.sin(rPhi)
    yFocal = R * numpy.sin(rTheta) * numpy.sin(rPhi)
    zFocal = R * numpy.cos(rPhi) + b

    return xFocal, yFocal, zFocal, R, b, fieldWarn


def focalToField(xFocal, yFocal, zFocal, site, waveCat):
    """Convert xyz focal position to unit-spherical field position.
    xyzFocal do not need to lie on the modeled sphere, and the spherical
    radius of curvature is not needed, only the origin of the sphere
    is needed (b).

    Parameters
    -----------
    xFocal: float or 1D array
        x focal coord mm
    yFocal: float or 1D array
        y focal coord mm
    zFocal: float or 1D array
        z focal coord mm, +z points along boresight toward sky.
    site: str
        "APO" or "LCO"
    waveCat: str
        "Boss", "Apogee", or "GFA"

    Result
    -------
    thetaField: float or 1D array
        azimuthal field coordinate (deg)
    phiField: float or 1D array
       polar field coordinate (deg)
    fieldWarn: bool or boolean array
        True if angle off-axis is large enough to be suspicious
    """
    direction = "field"
    R, b, c0, c1, c2, c3, c4 = getModelParams(site, direction, waveCat)
    # note by definition thetaField==thetaFocal
    thetaField = numpy.degrees(numpy.arctan2(yFocal, xFocal))

    # generate focal phis (degree off boresight)
    # angle off-axis from optical axis
    rFocal = numpy.sqrt(xFocal**2 + yFocal**2)
    v = numpy.array([rFocal, zFocal]).T
    v[:, 1] = v[:, 1] - b

    # unit vector pointing toward object on focal plane from circle center
    # arc from vertex to off axis
    v = v / numpy.vstack([numpy.linalg.norm(v, axis=1)] * 2).T
    # FP lands at -Z so, Angle from sphere center towards ground
    downwardZaxis = numpy.array([0, -1])
    # phi angle between ray and optical axis measured from sphere center
    phiFocal = numpy.degrees(numpy.arccos(v.dot(downwardZaxis)))
    phiField = c0 * phiFocal + c1 * phiFocal**3 + c2 * phiFocal**5 \
        + c3 * phiFocal**7 + c4 * phiFocal**9

    if site == "APO":
        fieldWarn = numpy.array(phiField) > APO_MAX_FIELD_R
    if site == "LCO":
        fieldWarn = numpy.array(phiField) > LCO_MAX_FIELD_R

    if hasattr(fieldWarn, "__len__"):
        if True in fieldWarn:
            print(  # turn this into warning eventually, problem for tests?
                "Warning! Far off-axis coordinate, conversion may be bogus"
            )
    elif fieldWarn is True:
        print(  # turn this into warning eventually, problem for tests?
            "Warning! Far off-axis coordinate, conversion may be bogus"
        )

    return thetaField, phiField, fieldWarn


class Field(Coordinate):
    """A representation of Field Coordinates.  A spherical coordinate system
    defined by two angles: theta, phi.  Theta is the angle about the optical
    axis measured from the direction of +RA. Phi is the angle off the optical
    axis of the telescope.  So Phi=0 is the optical axis of the telescope
    increasing from earth to sky.

    In the Cartesian representation on a unit sphere, +x is aligned with +RA,
    +y is aligned with +Dec, and so +z is aligned with the telescope's optical
    axis and points from the earth to sky.

    Parameters
    ------------
    value : numpy.ndarray
        A Nx2 array.  First column is theta, the azimuthal angle from +RA
        through +Dec in deg. Second angle is phi, polar angle angle (off axis
        angle) in deg. Or `.Observed`.  Or `.FocalPlane`.

    field_center : `.Observed`
        An `.Observed` instance containing a single coordinate

    Attributes
    ------------
    x : numpy.ndarray
        unit-spherical x coordinate
    y : numpy.ndarray
        unit-spherical y coordinate
    z : numpy.ndarray
        unit-spherical z coordinate
    x_angle : numpy.ndarray
        zeemax-style x field angle (deg)
    y_angle : numpy.ndarray
        zeemax-style y field angle (deg)
    field_warn: numpy.ndarray
        boolean array indicating suspect conversions from large field angles
    """

    __computed_arrays__ = ["x", "y", "z", "x_angle", "y_angle"]
    __warn_arrays__ = ["field_warn"]
    __extra_params__ = ["field_center"]  # mandatory parameter
    # may want to carry around position angle for all
    # coordinates too through the chain?  Could reduce errors in guiding
    # because direction to north or zenith varies across the field due to...
    # spheres.  For now ignore it?

    def __new__(cls, value, **kwargs):

        field_center = kwargs.get("field_center", None)
        if field_center is None:
            raise CoordIOError("field_center must be passed to Field")
        else:
            if not hasattr(field_center, "coordSysName"):
                raise CoordIOError(
                    "field_center must be an Observed coordinate"
                )
            if field_center.coordSysName != "Observed":
                raise CoordIOError(
                    "field_center must be an Observed coordinate"
                )
            if len(field_center) != 1:
                raise CoordIOError("field_center must contain only one coord")

        if isinstance(value, Coordinate):
            if value.coordSysName == "Observed":
                obj = super().__new__(cls, value, **kwargs)
                obj._fromObserved(value)
            elif value.coordSysName == "FocalPlane":
                # focal plane is a 3D coord sys
                initArray = numpy.zeros((len(value), 2))
                obj = super().__new__(cls, initArray, **kwargs)
                obj._fromFocalPlane(value)
            else:
                raise CoordIOError(
                    "Cannot convert to Field from %s"%value.coordSysName
                )
        else:
            obj = super().__new__(cls, value, **kwargs)
            obj._fromRaw()

        return obj

    def _fromObserved(self, obsCoords):
        """Converts from observed coords to field coords, given the field
        center.  Populates the computed arrays

        Parameters
        -----------
        obsCoords : `.Observed`
        """
        obsCoords = numpy.array(obsCoords)
        # convert alt/az into a spherical sys
        phis = 90 - obsCoords[:, 0]  # alt
        thetas = -1 * obsCoords[:, 1]  # az
        altCenter, azCenter = self.field_center.flatten()
        q = float(self.field_center.pa)  # position angle

        # work in cartesian frame
        coords = sph2Cart(thetas, phis)
        coords = numpy.array(coords).T

        # rotate the xyz coordinate system about z axis
        # such that -y axis is aligned with the azimuthal angle
        # of the field center

        sinTheta = numpy.sin(numpy.radians(90 - azCenter))
        cosTheta = numpy.cos(numpy.radians(90 - azCenter))
        rotTheta = numpy.array([
            [ cosTheta, sinTheta, 0],
            [-sinTheta, cosTheta, 0],
            [        0,        0, 1]
        ])

        coords = rotTheta.dot(coords.T).T

        # rotate the xyz coordinate system about the x axis
        # such that +z points to the field center.

        sinPhi = numpy.sin(numpy.radians(90 - altCenter))
        cosPhi = numpy.cos(numpy.radians(90 - altCenter))
        rotPhi = numpy.array([
            [1,       0,      0],
            [0,  cosPhi, sinPhi],
            [0, -sinPhi, cosPhi]
        ])
        coords = rotPhi.dot(coords.T).T
        # return coords

        # finally rotate about z by the parallactic angle
        # this puts +RA along +X and +DEC along +Y
        cosQ = numpy.cos(numpy.radians(q))
        sinQ = numpy.sin(numpy.radians(q))
        rotQ = numpy.array([
            [ cosQ, sinQ, 0],
            [-sinQ, cosQ, 0],
            [    0,    0, 1]
        ])

        coords = rotQ.dot(coords.T).T

        self.x = coords[:, 0]
        self.y = coords[:, 1]
        self.z = coords[:, 2]
        self.x_angle, self.y_angle = cart2FieldAngle(self.x, self.y, self.z)

        # finally convert back from cartesian to spherical (Field)
        thetaPhi = cart2Sph(self.x, self.y, self.z)
        thetaPhi = numpy.array(thetaPhi).T
        self[:, :] = thetaPhi

    def _fromFocalPlane(self, fpCoords):
        """Convert from FocalPlane coords to Field coords.

        Parameters
        ------------
        fpCoords : `.FocalPlane`
        """
        siteName = fpCoords.site.name.upper()
        argNamePairs = []
        for waveCat in ["Boss", "Apogee", "GFA"]:
            # find which coords are intended for which
            # wavelength
            arg = numpy.argwhere(
                fpCoords.wavelength == INST_TO_WAVE[waveCat]
            ).squeeze()
            argNamePairs.append([arg, waveCat])

        for (arg, waveCat) in argNamePairs:

            xFP = fpCoords[arg,0].squeeze()
            yFP = fpCoords[arg,1].squeeze()
            zFP = fpCoords[arg,2].squeeze()

            if len(xFP) == 0:
                continue

            thetaFocal, phiFocal, fieldWarn = focalToField(
                xFP,
                yFP,
                zFP,
                siteName,
                waveCat
            )
            thetaPhi = numpy.array([thetaFocal, phiFocal]).T
            self[arg] = thetaPhi
            self.field_warn[arg] = fieldWarn

        self._fromRaw()

    def _fromRaw(self):
        """Populates the computed arrays
        """
        self.x, self.y, self.z = sph2Cart(self[:, 0], self[:, 1])
        self.x_angle, self.y_angle = cart2FieldAngle(self.x, self.y, self.z)


class FocalPlane(Coordinate):
    """The focal plane coordinate system is 3D Cartesian with units of mm.
    The origin is the M1 vertex.  +x is aligned with +RA on the image, +y is
    aligned with +Dec on the image, +z points from earth to sky along the
    telescope's optical axis. `.FocalPlane` xy's are physically rotated 180
    degrees with respect to `.Field` xy's because Cassegrain telescope rotates
    the image.  The choice was made to put +x along +RA on the image, rather
    than the sky.

    For each wavelength, a sphereical surface fit has been determined.
    The sphere's center lies along the optical axis at a positive z value.
    The sphere's center is considered the 'ray origin', so all rays traveling
    to the focal plane originate from this location.

    The ray origin and a wavelength-dependent, z-symmetric, distortion model
    is used to convert between `.FocalPlane` and `.Field` coordinates.
    The distortion model is an odd-termed polynomial relating input field
    angle to output focal plane model.

    The ZEMAX models/scripts, and fitting code for this still lives in
    github.com/csayres/sdssconv, but may be eventually moved to this package.

    Parameters
    ------------
    value : numpy.ndarray
        A Nx3 array where [x,y,z] are columns columns. Or `.Field`.  Or `.Wok`.
    wavelength : numpy.ndarray
        A 1D array with he observing wavelength, in angstrom.
        Currently only values of 5400, 6231, and 16600 are valid, corresponding
        to BOSS, Apogee, and sdss-r band (GFA).
    site : `.Site`
        Used to pick the focal plane model to use which is telescope dependent

    Attributes
    -----------
    b : numpy.ndarray
        A 1D array containing the location of the spherical fit's center
        (origin of rays) along the optical axis in mm.  Varies with wavelength.
    R : numpy.ndarray
        1D array containing the radius of curvature for the spherical fit in
        mm.  Varies with wavelength.
    field_warn: numpy.ndarray
        boolean array indicating suspect conversions from large field angles
    """

    __extra_params__ = ["site"]   # mandatory argument
    __extra_arrays__ = ["wavelength"]  # mandatory argument
    __computed_arrays__ = ["b", "R"]
    __warn_arrays__ = ["field_warn"]

    _valid_wavelengths = set([5400., 6231., 16600.])

    def __new__(cls, value, **kwargs):

        site = kwargs.get("site", None)
        if site is None:
            raise CoordIOError("site must be passed to FocalPlane")
        else:
            if not isinstance(site, Site):
                raise CoordIOError(
                    "site must be a coordio.Site"
                )
        wls = kwargs.get("wavelength", None)
        if wls is None:
            raise CoordIOError("must specify corresponding wavelengths")
        else:
            if hasattr(wls, "__len__"):
                wls = numpy.array(wls, dtype=numpy.float64)
                wlSet = set(wls)
            else:
                wls = float(wls)
                wlSet = set([wls])
            if not wlSet.issubset(cls._valid_wavelengths):
                raise CoordIOError(
                    "Invalid wavelength passed to FocalPlane \
                    valid wavelengths are %s"%(str(cls._valid_wavelengths))
                )
        kwargs["wavelength"] = wls

        if isinstance(value, Coordinate):
            if value.coordSysName == "Field":
                # going from 2D -> 3D coord sys
                # expand dimensionality
                arrInit = numpy.zeros((len(value), 3))
                obj = super().__new__(cls, arrInit, **kwargs)
                obj._fromField(value)
            elif value.coordSysName == "Wok":
                # wok is already a 3D coord sys
                obj = super().__new__(cls, value, **kwargs)
                obj._fromWok(value)
            else:
                raise CoordIOError(
                    'Cannot convert to FocalPlane from %s'%value.coordSysName
                )
        else:
            obj = super().__new__(cls, value, **kwargs)
            obj._fromRaw()

        return obj

    def _fromField(self, fieldCoord):
        """ Convert from field coordinates to focal plane coordinates, using
        a pre-fit focal plane model.

        Parameters
        -----------
        fieldCoord : `.Field`
        """
        siteName = self.site.name.upper()
        argNamePairs = []
        for waveCat in ["Boss", "Apogee", "GFA"]:
            # find which coords are intended for which
            # wavelength
            arg = numpy.argwhere(
                self.wavelength == INST_TO_WAVE[waveCat]
            ).squeeze()
            argNamePairs.append([arg, waveCat])

        for (arg, waveCat) in argNamePairs:
            # apply the correct model for the coordinates' wanted
            # wavelength

            thetaField = fieldCoord[arg,0].squeeze()
            phiField = fieldCoord[arg,1].squeeze()

            if len(thetaField) == 0:
                continue

            xFP, yFP, zFP, R, b, fieldWarn = fieldToFocal(
                thetaField,
                phiField,
                siteName,
                waveCat
            )

            xyzFP = numpy.array([xFP, yFP, zFP]).T
            self[arg] = xyzFP
            self.b[arg] = b
            self.R[arg] = R
            self.field_warn[arg] = fieldWarn

    def _fromWok(self, wokCoord):
        """ Convert from wok coordinates to focal plane coordinates.

        Parameters
        -----------
        wokCoord : `.Wok`
        """
        pass

    def _fromRaw(self):
        siteName = self.site.name.upper()
        direction = "focal"
        argNamePairs = []
        for waveCat in ["Boss", "Apogee", "GFA"]:
            arg = numpy.argwhere(
                self.wavelength == INST_TO_WAVE[waveCat]).squeeze()
            argNamePairs.append([arg, waveCat])

        for (arg, waveCat) in argNamePairs:
            R, b, c0, c1, c2, c3, c4 = getModelParams(
                siteName, direction, waveCat
            )
            self.b[arg] = b
            self.R[arg] = R


if __name__ == "__main__":
    pass

