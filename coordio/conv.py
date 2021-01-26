import numpy
import warnings

from . import defaults
from .utils import cart2Sph, sph2Cart
from .exceptions import CoordIOUserWarning


def fieldToObserved(x, y, z, altCenter, azCenter, pa):
    """Convert xyz unit-spherical field coordinates to Alt/Az

    the inverse of observetToField

    Az=0 is north, Az=90 is east

    +x points along +RA, +y points along +Dec, +z points along boresight
    from telescope toward sky

    Parameters
    ------------
    x: scalar or 1D array
        unit-spherical x field coord (aligned with +RA)
    y: scalar or 1D array
        unit-spherical y field coord (aligned with +Dec)
    z: scalar or 1D array
        unit-spherical z coord
    altCenter: float
        altitude in degrees, field center
    azCenter: float
        azimuth in degrees, field center
    pa: float
        position angle of field center

    Returns
    --------
    alt : float or 1D array
        altitude of coordinate in degrees
    az : float or 1D array

    """
    cosQ = numpy.cos(numpy.radians(-1*pa))
    sinQ = numpy.sin(numpy.radians(-1*pa))
    rotQ = numpy.array([
        [ cosQ, sinQ, 0],
        [-sinQ, cosQ, 0],
        [    0,    0, 1]
    ])

    coords = numpy.array(
        [x, y, z]
    ).T

    coords = rotQ.dot(coords.T).T

    sinPhi = numpy.sin(-1*numpy.radians(90-altCenter))
    cosPhi = numpy.cos(-1*numpy.radians(90-altCenter))
    rotPhi = numpy.array([
        [1,       0,      0],
        [0,  cosPhi, sinPhi],
        [0, -sinPhi, cosPhi]
    ])
    coords = rotPhi.dot(coords.T).T

    sinTheta = numpy.sin(-1*numpy.radians(90-azCenter))
    cosTheta = numpy.cos(-1*numpy.radians(90-azCenter))
    rotTheta = numpy.array([
        [ cosTheta, sinTheta, 0],
        [-sinTheta, cosTheta, 0],
        [        0,        0, 1]
    ])

    coords = rotTheta.dot(coords.T).T

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    theta, phi = cart2Sph(x, y, z)

    # convert sph theta, phi to az, alt
    az = -1 * theta
    alt = 90 - phi

    az = az % 360  # wrap to 0,360

    return alt, az


def observedToField(alt, az, altCenter, azCenter, pa):
    """Convert Az/Alt coordinates to cartesian coords on the
    unit sphere with the z axis aligned with field center (boresight)

    Az=0 is north, Az=90 is east

    Resulting Cartesian coordinates have the following
    convention +x points along +RA, +y points along +Dec

    Parameters
    ------------
    alt: float or 1D array
        altitude in degrees, positive above horizon, negative below
    az: float or 1D array
        azimuth in degrees az=0 is north az=90 is east
    altCenter: float
        altitude in degrees, field center
    azCenter: float
        azimuth in degrees, field center
    pa: float
        position angle of field center

    Returns
    --------
    theta : float or 1D array
        azimuthal angle of field coordinate (deg)
    phi : float or 1D array
        polar angle of field coordinat (deg off axis)
    """
    # convert from alt/az to spherical
    phis = 90 - alt  # alt
    thetas = -1 * az  # az

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
    cosQ = numpy.cos(numpy.radians(pa))
    sinQ = numpy.sin(numpy.radians(pa))
    rotQ = numpy.array([
        [ cosQ, sinQ, 0],
        [-sinQ, cosQ, 0],
        [    0,    0, 1]
    ])

    coords = rotQ.dot(coords.T).T

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # finally convert back from cartesian to spherical (Field)
    thetaPhi = cart2Sph(x, y, z)
    thetaPhi = numpy.array(thetaPhi).T
    theta, phi = thetaPhi[:,0], thetaPhi[:,1]
    return theta, phi


def fieldToFocal(thetaField, phiField, site, waveCat):
    """Convert spherical field coordinates to the modeled spherical
    position on the focal plane.  Origin is the M1 vertex.

    Raises warnings for coordinates at large field angles.

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
        fieldWarn = numpy.array(phiField) > defaults.APO_MAX_FIELD_R
    if site == "LCO":
        fieldWarn = numpy.array(phiField) > defaults.LCO_MAX_FIELD_R

    if hasattr(fieldWarn, "__len__"):
        if True in fieldWarn:
            warnings.warn(
                "Warning! Coordinate far off telescope optical axis "\
                "conversion may be bogus",
                CoordIOUserWarning
            )
    elif fieldWarn is True:
        warnings.warn(
            "Warning! Coordinate far off telescope optical axis, "\
            "conversion may be bogus",
            CoordIOUserWarning
        )

    direction = "focal"
    R, b, c0, c1, c2, c3, c4 = defaults.getFPModelParams(site, direction, waveCat)

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

    Raises warnings for coordinates at large field angles.

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
    R, b, c0, c1, c2, c3, c4 = defaults.getFPModelParams(site, direction, waveCat)
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
        fieldWarn = numpy.array(phiField) > defaults.APO_MAX_FIELD_R
    if site == "LCO":
        fieldWarn = numpy.array(phiField) > defaults.LCO_MAX_FIELD_R

    if hasattr(fieldWarn, "__len__"):
        if True in fieldWarn:
            warnings.warn(
                "Warning! Coordinate far off telescope optical axis, conversion may be bogus",
                CoordIOUserWarning
            )
    elif fieldWarn is True:
        warnings.warn(
            "Warning! Coordinate far off telescope optical axis, conversion may be bogus",
            CoordIOUserWarning
        )

    return thetaField, phiField, fieldWarn


def focalToWok(
    xFocal, yFocal, zFocal, positionAngle=0,
    xOffset=0, yOffset=0, zOffset=0, tiltX=0, tiltY=0
):
    """Convert xyz focal coordinates in mm to xyz wok coordinates in mm.

    The origin of the focal coordinate system is the
    M1 vertex. focal +y points toward North, +x points toward E.
    The origin of the wok coordinate system is the wok vertex.  -x points
    toward the boss slithead.  +z points from telescope to sky.

    Tilt is applied about x axis then y axis.


    Parameters
    -------------
    xFocal: scalar or 1D array
        x position of object on focal plane mm
        (+x aligned with +RA on image)
    yFocal: scalar or 1D array
        y position of object on focal plane mm
        (+y aligned with +Dec on image)
    zFocal: scalar or 1D array
        z position of object on focal plane mm
        (+z aligned boresight and increases from the telescope to the sky)
    positionAngle: scalar
        position angle deg.  Angle measured from (image) North through East to wok +y.
        So position angle of 45 deg, wok +y points NE
    site : str
        either "APO" or "LCO"
    xOffset: scalar or None
        x position (mm) of wok origin (vertex) in focal coords
        calibrated
    yOffset: scalar
        y position (mm) of wok origin (vertex) in focal coords
        calibratated
    zOffset: scalar
        z position (mm) of wok origin (vertex) in focal coords
        calibratated
    tiltX: scalar
        tilt (deg) of wok about focal x axis at PA=0
        calibrated
    tiltY: scalar
        tilt (deg) of wok about focal y axis at PA=0
        calibrated

    Returns
    ---------
    xWok: scalar or 1D array
        x position of object in wok space mm
        (+x aligned with +RA on image)
    yWok: scalar or 1D array
        y position of object in wok space mm
        (+y aligned with +Dec on image)
    zWok: scalar or 1D array
        z position of object in wok space mm
        (+z aligned boresight and increases from the telescope to the sky)
    """

    # apply calibrated tilts and translation (at PA=0)
    # where they should be fit
    # tilts are defined https://mathworld.wolfram.com/RotationMatrix.html
    # as coordinate system rotations counter clockwise when looking
    # down the positive axis toward the origin
    coords = numpy.array([xFocal, yFocal, zFocal])
    transXYZ = numpy.array([xOffset, yOffset, zOffset])

    rotX = numpy.radians(tiltX)
    rotY = numpy.radians(tiltY)
    # rotation about z axis is position angle
    # position angle is clockwise positive for rotation measured from
    # north to wok +y (when looking from above the wok)
    # however rotation matrices are positinve for counter-clockwise rotation
    # hence the sign flip that's coming
    rotZ = -1*numpy.radians(positionAngle)

    rotMatX = numpy.array([
        [1, 0, 0],
        [0, numpy.cos(rotX), numpy.sin(rotX)],
        [0, -1*numpy.sin(rotX), numpy.cos(rotX)]
    ])

    rotMatY = numpy.array([
        [numpy.cos(rotY), 0, -1*numpy.sin(rotY)],
        [0, 1, 0],
        [numpy.sin(rotY), 0, numpy.cos(rotY)]
    ])

    # rotates coordinate system
    rotMatZ = numpy.array([
        [numpy.cos(rotZ), numpy.sin(rotZ), 0],
        [-numpy.sin(rotZ), numpy.cos(rotZ), 0],
        [0, 0, 1]
    ])


    # first apply rotation about x axis (x tilt)
    coords = rotMatX.dot(coords)
    # next apply rotation about y axis (y tilt)
    coords = rotMatY.dot(coords)
    # apply rotation about z axis (angle of observation)
    coords = rotMatZ.dot(coords)

    # apply translation
    if hasattr(xFocal, "__len__"):
        # list of coords fed in
        transXYZ = numpy.array([transXYZ]*len(xFocal)).T
        xWok, yWok, zWok = coords - transXYZ
    else:
        # single set of xyz coords fed in
        xWok, yWok, zWok = coords - transXYZ

    return xWok, yWok, zWok


