from __future__ import annotations

from typing import TYPE_CHECKING

import numpy

from . import conv, defaults
from .conv import cart2FieldAngle, sph2Cart
from .coordinate import (Coordinate, Coordinate2D, Coordinate3D,
                         verifySite, verifyWavelength)
from .exceptions import CoordIOError


if TYPE_CHECKING:
    from .site import Site
    from .sky import Observed


__all__ = ["Field", "FocalPlane"]


class Field(Coordinate2D):
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

    x: numpy.ndarray
    y: numpy.ndarray
    z: numpy.ndarray
    x_angle: numpy.ndarray
    y_angle: numpy.ndarray
    field_warn: numpy.ndarray
    field_center: Observed

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
                # field is 2D (spherical)
                initArray = numpy.zeros((len(value), 2))
                obj = super().__new__(cls, initArray, **kwargs)
                obj._fromFocalPlane(value)
            else:
                raise CoordIOError(
                    "Cannot convert to Field from %s" % value.coordSysName
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
        altAz = numpy.array(obsCoords)
        alt, az = altAz[:, 0], altAz[:, 1]
        altCenter, azCenter = self.field_center.flatten()
        pa = float(self.field_center.pa)  # position angle
        theta, phi = conv.observedToField(alt, az, altCenter, azCenter, pa)
        self[:, 0] = theta
        self[:, 1] = phi

        self._fromRaw()

    def _fromFocalPlane(self, fpCoords):
        """Convert from FocalPlane coords to Field coords.

        Parameters
        ------------
        fpCoords : `.FocalPlane`
        """
        siteName = fpCoords.site.name.upper()
        for waveCat in ["Boss", "Apogee", "GFA"]:
            # find which coords are intended for which
            # wavelength
            arg = numpy.argwhere(
                fpCoords.wavelength == defaults.INST_TO_WAVE[waveCat]
            )

            if len(arg) == 0:
                continue

            arg = arg.squeeze()

            xFP = numpy.atleast_1d(fpCoords[arg, 0].squeeze())
            yFP = numpy.atleast_1d(fpCoords[arg, 1].squeeze())
            zFP = numpy.atleast_1d(fpCoords[arg, 2].squeeze())

            if hasattr(xFP, "__len__") and len(xFP) == 0:
                continue

            thetaFocal, phiFocal, fieldWarn = conv.focalToField(
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
        # ensure wrapping
        self[:,0] = self[:,0] % 360
        self.x, self.y, self.z = sph2Cart(self[:, 0], self[:, 1])
        self.x_angle, self.y_angle = cart2FieldAngle(self.x, self.y, self.z)


class FocalPlane(Coordinate3D):
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
        A Nx3 array where [x,y,z] are columns. Or `.Field`.  Or `.Wok`.
    wavelength : float or numpy.ndarray
        A 1D array with he observed wavelength, in angstroms.
        Currently only values of 5400, 6231, and 16600 are valid, corresponding
        to BOSS, Apogee, and sdss-r band (GFA).  Defaults to GFA wavelength.
    site : `.Site`
        Used to pick the focal plane model to use which is telescope dependent
    fpScale : float
        Multiplicative scale factor to apply between wok and focal
        coords.  An adjustment to the focal plane model.  Defaults to
        coordio.defaults.FOCAL_SCALE.

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

    __extra_params__ = ["site", "fpScale"]
    __extra_arrays__ = ["wavelength"]
    __computed_arrays__ = ["b", "R"]
    __warn_arrays__ = ["field_warn"]

    b: numpy.ndarray
    R: numpy.ndarray
    field_warn: numpy.ndarray
    wavelength: numpy.ndarray
    site: Site
    fpScale: float

    def __new__(cls, value, **kwargs):

        verifySite(kwargs)

        verifyWavelength(
            kwargs, len(value), strict=True
        )

        fpScale = kwargs.get("fpScale", None)
        if fpScale is None:
            kwargs["fpScale"] = defaults.FOCAL_SCALE

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
        for waveCat in ["Boss", "Apogee", "GFA"]:
            # find which coords are intended for which
            # wavelength
            arg = numpy.argwhere(
                self.wavelength == defaults.INST_TO_WAVE[waveCat]
            )

            if len(arg) == 0:
                continue

            arg = arg.squeeze()

            thetaField = numpy.atleast_1d(fieldCoord[arg, 0].squeeze())
            phiField = numpy.atleast_1d(fieldCoord[arg, 1].squeeze())

            xFP, yFP, zFP, R, b, fieldWarn = conv.fieldToFocal(
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
        self._fromRaw()   # for grabbing fp params, b and R

        pa = wokCoord.obsAngle
        siteName = self.site.name.upper()
        xOff, yOff, zOff, tiltX, tiltY = defaults.getWokOrient(siteName)
        xWok, yWok, zWok = wokCoord[:, 0], wokCoord[:, 1], wokCoord[:, 2]
        xF, yF, zF = conv.wokToFocal(
            xWok, yWok, zWok, pa, xOff,
            yOff, zOff, tiltX, tiltY,
            b=self.b, R=self.R, # set b and R to None if NOT using flat wok model
            fpScale=self.fpScale
        )

        self[:, 0] = xF
        self[:, 1] = yF
        self[:, 2] = zF


    def _fromRaw(self):
        direction = "focal"  # irrelevant, just grabbing sphere params

        for waveCat in ["Boss", "Apogee", "GFA"]:
            arg = numpy.argwhere(
                self.wavelength == defaults.INST_TO_WAVE[waveCat])

            if len(arg) == 0:
                continue

            arg = arg.squeeze()

            R, b, c0, c1, c2, c3, c4 = defaults.getFPModelParams(self.site.name,
                                                                 direction,
                                                                 waveCat)
            self.b[arg] = b
            self.R[arg] = R
