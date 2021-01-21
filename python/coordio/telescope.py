import numpy

from .coordinate import Coordinate
from .exceptions import CoordIOError
from .utils import sph2Cart, cart2Sph, cart2FieldAngle

# from .sky import Observed, ICRS
# from .site import Site

__all__ = ["Field", "FocalPlane"]


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
        An Observed instance containing a single coordinate

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
    """

    __computed_arrays__ = ['x', 'y', 'z', 'x_angle', 'y_angle']

    __extra_params__ = ['field_center']  # mandatory parameter
    # may want to carry around position angle for all
    # coordinates too through the chain?  Could reduce errors in guiding
    # because direction to north or zenith varies across the field due to...
    # spheres.  For now ignore it?

    def __new__(cls, value, **kwargs):

        field_center = kwargs.get('field_center', None)
        if field_center is None:
            raise CoordIOError('field_center must be passed to Field')
        else:
            if not hasattr(field_center, "coordSysName"):
                raise CoordIOError(
                    'field_center must be an Observed coordinate'
                )
            if field_center.coordSysName != 'Observed':
                raise CoordIOError(
                    'field_center must be an Observed coordinate'
                )
            if len(field_center) != 1:
                raise CoordIOError('field_center must contain only one coord')

        obj = super().__new__(cls, value, **kwargs)

        if isinstance(value, Coordinate):
            if value.coordSysName == "Observed":
                obj._fromObserved(value)
            elif value.coordSysName == "FocalPlane":
                obj._fromFocalPlane(value)
            else:
                raise CoordIOError(
                    'Cannot convert to Field from %s'%value.coordSysName
                )
        else:
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

    def _fromFocalPlane(fpCoords):
        """Convert from FocalPlane coords to Field coords.

        Parameters
        ------------
        fpCoords : `.FocalPlane`
        """
        raise NotImplementedError()

    def _fromRaw(self):
        """Populates the computed arrays
        """
        self.x, self.y, self.z = sph2Cart(self[:, 0], self[:, 1])
        self.x_angle, self.y_angle = cart2FieldAngle(self.x, self.y, self.z)


class FocalPlane(Coordinate):
    pass


if __name__ == "__main__":
    pass

