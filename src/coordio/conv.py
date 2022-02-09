import warnings

import numpy
from skimage.transform import (AffineTransform, EuclideanTransform,
                               SimilarityTransform)

from . import defaults, libcoordio
from .exceptions import CoordIOUserWarning, CoordIOError


# low level, functional conversions

def fieldAngle2Cart(xField, yField):
    """Convert (ZEMAX-style) field angles in degrees to a cartesian point on
    the unit sphere

    Zemax defines field angles like this:
    Positive field angles imply positive slope for the ray in that direction,
    and thus refer to **negative** coordinates on distant objects. ZEMAX converts
    x field angles ( αx ) and y field angles ( αy ) to ray direction cosines
    using the following formulas:

    tanαx = l/n
    tanαy = m/n
    l^2 + m^2 + n^2 = 1

    where l, m, and n are the x, y, and z direction cosines.

    Parameters
    -----------
    xField: scalar or 1D array
        zemax style x field
    yField: scalar or 1D array
        zemax style y field

    Returns
    ---------
    result: list
        [x,y,z] coordinates on unit sphere
    """
    # xField, yField = fieldXY

    # invert field degrees, fields represent slope
    # of ray, so a positivly sloped ray is coming from
    # below its respective plane, the resulting vector
    # from this transform should thus point downward, not up
    tanx = numpy.tan(numpy.radians(-1*xField))
    tany = numpy.tan(numpy.radians(-1*yField))

    # calculate direction cosines
    z = numpy.sqrt(1/(tanx**2 + tany**2 + 1))
    x = tanx * z
    y = tany * z
    # if numpy.isnan(n) or numpy.isnan(l) or numpy.isnan(m):
    #     raise RuntimeError("NaN output [%.2f, %.2f, %.2f] for input [%.2f, %.2f]"%(n, l, m, xField, yField))

    return [x,y,z]


def cart2FieldAngle(x, y, z):
    """Convert cartesian point on unit sphere of sky to (ZEMAX-style) field
    angles in degrees.


    Zemax defines field angles like this:
    Positive field angles imply positive slope for the ray in that direction,
    and thus refer to **negative** coordinates on distant objects. ZEMAX converts
    x field angles ( αx ) and y field angles ( αy ) to ray direction cosines
    using the following formulas:

    tanαx = l/n
    tanαy = m/n
    l^2 + m^2 + n^2 = 1

    where l, m, and n are the x, y, and z direction cosines.

    Parameters
    -----------
    x: scalar or 1D array
    y: scalar or 1D array
    z: scalar or 1D array

    Returns
    ---------
    result: list
        [xField, yField] ZEMAX style field coordinates (degrees)
    """
    # if numpy.abs(numpy.linalg.norm(cartXYZ) - 1) > SMALL_NUM:
    #     raise RuntimeError("cartXYZ must be a vector on unit sphere with L2 norm = 1")
    # l, m, n = x, y, z

    # invert field degrees, fields represent slope
    # of ray, so a positivly sloped ray is coming from
    # below the optical axis, the resulting vector
    # from this transform should thus point downward, not up

    xField = -1 * numpy.degrees(numpy.arctan2(x, z))
    yField = -1 * numpy.degrees(numpy.arctan2(y, z))
    # if numpy.isnan(xField) or numpy.isnan(yField):
    #     raise RuntimeError("NaN output [%.2f, %.2f] for input [%.2f, %.2f, %.2f]"%(xField, yField, cartXYZ[0], cartXYZ[1], cartXYZ[2]))
    return [xField, yField]


def cart2Sph(x, y, z):
    """Convert cartesian coordinates to spherical
    coordinates theta, phi in degrees

    phi is polar angle measure from z axis
    theta is azimuthal angle measured from x axis

    Parameters
    -----------
    x: scalar or 1D array
    y: scalar or 1D array
    z: scalar or 1D array

    Returns
    ---------
    result: list
        [theta, phi] degrees
    """

    # if numpy.abs(numpy.linalg.norm(cartXYZ) - 1) > SMALL_NUM:
    #     raise RuntimeError("cartXYZ must be a vector on unit sphere with L2 norm = 1")
    # if cartXYZ[2] < 0:
    #     raise RuntimeError("z direction must be positive for cartesian field coord")
    # x, y, z = cartXYZ
    theta = numpy.degrees(numpy.arctan2(y, x))
    # wrap theta to be between 0 and 360 degrees
    theta = theta % 360

    phi = numpy.degrees(numpy.arccos(z))
    # if numpy.isnan(theta) or numpy.isnan(phi):
    #     raise RuntimeError("NaN output [%.2f, %.2f] from input [%.2f, %.2f, %.2f]"%(theta, phi, cartXYZ[0], cartXYZ[1], cartXYZ[2]))
    return [theta, phi]


def sph2Cart(theta, phi, r=1):
    """Convert spherical coordinates theta, phi in degrees
    to cartesian coordinates on unit sphere.

    phi is polar angle measure from z axis
    theta is azimuthal angle measured from x axis

    Parameters
    -----------
    theta: scalar or 1D array
        degrees, azimuthal angle
    phi: scalar or 1D array
        degrees, polar angle
    r: scalar
        radius of curvature. Default to 1 for unit sphere

    Returns
    ---------
    result: list
        [x,y,z] coordinates on sphere

    """

    theta, phi = numpy.radians(theta), numpy.radians(phi)
    x = r*numpy.cos(theta) * numpy.sin(phi)
    y = r*numpy.sin(theta) * numpy.sin(phi)
    z = r*numpy.cos(phi)

    # if numpy.isnan(x) or numpy.isnan(y) or numpy.isnan(z):
    #     raise RuntimeError("NaN output [%.2f, %.2f, %.2f] for input [%.2f, %.2f]"%(x, y, z, numpy.degrees(theta), numpy.degrees(phi)))

    return [x, y, z]


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
                "Warning! Coordinate far off telescope optical axis " \
                "conversion may be bogus",
                CoordIOUserWarning
            )
    elif fieldWarn is True:
        warnings.warn(
            "Warning! Coordinate far off telescope optical axis, " \
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
    xOffset=0, yOffset=0, zOffset=0, tiltX=0, tiltY=0,
    fpScale=defaults.FOCAL_SCALE, projectFlat=True
):
    """Convert xyz focal coordinates in mm to xyz wok coordinates in mm.

    The origin of the focal coordinate system is the
    M1 vertex. focal +y points toward North, +x points toward E.
    The origin of the wok coordinate system is the wok vertex.  -x points
    toward the boss slithead.  +z points from telescope to sky.

    Tilt is applied about x axis then y axis.

    Note: currently zFocal irrelevant using a flat wok model.


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
    fpScale: scalar
        radial scale multiplier
    projectFlat: bool
        if true, assume flat wok model

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

    # scale xyWok from dither analysis results
    xWok = xWok / fpScale # from gfa: 0.9998899
    yWok = yWok / fpScale # from gfa: 0.9998899

    if projectFlat:
        # force all z coords to be positioner height
        # necessary for a flat wok model
        # and this should be removed if we go
        # to curved wok
        # !!!!!!!!!!!!!!!!!
        if hasattr(xFocal, "__len__"):
            zWok = numpy.array([defaults.POSITIONER_HEIGHT]*len(xFocal)).T
        else:
            zWok = defaults.POSITIONER_HEIGHT

    return xWok, yWok, zWok


def wokToFocal(
    xWok, yWok, zWok, positionAngle=0,
    xOffset=0, yOffset=0, zOffset=0, tiltX=0, tiltY=0,
    fpScale=defaults.FOCAL_SCALE,
    b=None, R=None
):
    """Convert xyz wok coordinates in mm to xyz focal coordinates in mm.

    The origin of the focal coordinate system is the
    M1 vertex. focal +y points toward North, +x points toward E.
    The origin of the wok coordinate system is the wok vertex.  -x points
    toward the boss slithead.  +z points from telescope to sky.

    Tilt is applied first about x axis then y axis.

    Note: currently zWok is irrelevant using a flat wok model.


    Parameters
    -------------
    xWok: scalar or 1D array
        x position of object in wok space mm
        (+x aligned with +RA on image at PA=0)
    yWok: scalar or 1D array
        y position of object in wok space mm
        (+y aligned with +Dec on image at PA=0)
    zWok: scalar or 1D array
        z position of object in wok space mm
        (+z aligned boresight and increases from the telescope to the sky)
    positionAngle: scalar
        position angle deg.  Angle measured from (image) North through East to wok +y.
        So position angle of 45 deg, wok +y points NE
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
    fpScale: scalar
        radial scale multiplier
    b: float
        z location of the center of curvature, dist from M1 vertex (mm)
    R: float
        radius of curvature (mm)


    Returns
    ---------
    xFocal: scalar or 1D array
        x position of object in focal coord sys mm
        (+x aligned with +RA on image)
    yFocal: scalar or 1D array
        y position of object in focal coord sys mm
        (+y aligned with +Dec on image)
    zFocal: scalar or 1D array
        z position of object in focal coord sys mm
        (+z aligned boresight and increases from the telescope to the sky)
    """

    # scale xyWok from dither analysis results
    xWok = numpy.array(xWok) * fpScale # from gfa: 0.9998899
    yWok = numpy.array(yWok) * fpScale # from gfa: 0.9998899

    # this routine is a reversal of the steps
    # in the function focalToWok, with rotational
    # angles inverted and translations applied in reverse
    coords = numpy.array([xWok, yWok, zWok])

    rotX = numpy.radians(-1*tiltX)
    rotY = numpy.radians(-1*tiltY)
    rotZ = numpy.radians(positionAngle)

    rotMatX = numpy.array([
        [1, 0, 0],
        [0, numpy.cos(rotX), numpy.sin(rotX)],
        [0, -numpy.sin(rotX), numpy.cos(rotX)]
    ])

    rotMatY = numpy.array([
        [numpy.cos(rotY), 0, -numpy.sin(rotY)],
        [0, 1, 0],
        [numpy.sin(rotY), 0, numpy.cos(rotY)]
    ])

    rotMatZ = numpy.array([
        [numpy.cos(rotZ), numpy.sin(rotZ), 0],
        [-numpy.sin(rotZ), numpy.cos(rotZ), 0],
        [0, 0, 1]
    ])

    transXYZ = numpy.array([xOffset, yOffset, zOffset])

    # add offsets for reverse transform
    if hasattr(xWok, "__len__"):
        # list of coords fed in
        transXYZ = numpy.array([transXYZ]*len(xWok)).T
        coords = coords + transXYZ
    else:
        # single set of xyz coords fed in
        coords = coords + transXYZ

    coords = rotMatZ.dot(coords)
    coords = rotMatY.dot(coords)
    xFocal, yFocal, zFocal = rotMatX.dot(coords)

    if b is not None or R is not None:
        # calculate z coords based on xy
        # basically force the point to land on the focal
        # surface for a given wavelength
        # necessary for a flat wok model
        # and this should be removed if we go
        # to curved wok
        # !!!!!!!!!!!!!!!!!
        if R is None or b is None:
            raise CoordIOError("Must specify both b and R if they are not None")
        rFocal = numpy.sqrt(xFocal**2+yFocal**2)
        zFocal = -1*numpy.sqrt(R**2-rFocal**2) + b

    return xFocal, yFocal, zFocal


def _verify3Vector(checkMe, label):
    if not hasattr(checkMe, "__len__"):
        raise RuntimeError("%s must be a 3-vector"%label)
    checkMe = numpy.array(checkMe).squeeze()
    if len(checkMe) != 3:
        raise RuntimeError("%s must be a 3-vector"%label)
    if checkMe.ndim != 1:
        raise RuntimeError("%s must be a 3-vector"%label)
    return checkMe


def repeat_scalar(input, n):
    """Returns an array with an scalar repeated ``n`` times. Does not affect arrays."""

    if numpy.isscalar(input):
        return numpy.repeat(input, n)

    return input


def wokToTangent(xWok, yWok, zWok, b, iHat, jHat, kHat,
                 elementHeight=defaults.POSITIONER_HEIGHT, scaleFac=1,
                 dx=0, dy=0, dz=0):
    """
    Convert an array of wok coordinates to tangent coordinates.

    xyz Wok coords are mm with orgin at wok vertex. +z points from wok toward M2.
    -x points toward the boss slithead

    In the tangent coordinate frame the xy plane is tangent to the wok
    surface at xyz position b.  The origin is set to be elementHeight above
    the wok surface.

    Parameters
    -------------
    xWok: 1D array
        x position of object in wok space mm
        (+x aligned with +RA on image at PA = 0)
    yWok: 1D array
        y position of object in wok space mm
        (+y aligned with +Dec on image at PA = 0)
    zWok: 1D array
        z position of object in wok space mm
        (+z aligned boresight and increases from the telescope to the sky)
    b: 3-element 1D vector
        x,y,z position (mm) of each hole element on wok surface measured in wok coords
    iHat: 3-element 1D vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate x axis for each hole.
    jHat: 3-element 1D vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate y axis for each hole.
    kHat: 3-element 1D vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate z axis for each hole.
    elementHeight: 1D array
        height (mm) of positioner/GFA chip above wok surface
    scaleFac: 1D array
        scale factor to apply to b to account for thermal expansion of wok.
        scale is applied to b radially
    dx: 1D array
        x offset (mm), calibration to capture small displacements of
        tangent x
    dy: 1D array
        y offset (mm), calibration to capture small displacements of
        tangent y
    dz: 1D array
        z offset (mm), calibration to capture small displacements of
        tangent x

    Returns
    ---------
    xTangent: 1D array
        x position (mm) in tangent coordinates
    yTangent: 1D array
        y position (mm) in tangent coordinates
    zTangent: 1D array
        z position (mm) in tangent coordinates
    """

    b = _verify3Vector(b, 'b').tolist()
    iHat = _verify3Vector(iHat, 'iHat').tolist()
    jHat = _verify3Vector(jHat, 'jHat').tolist()
    kHat = _verify3Vector(kHat, 'kHat').tolist()

    xyzWok = numpy.atleast_2d(numpy.array([xWok, yWok, zWok]).T).tolist()

    xyzTangent = libcoordio.wokToTangentArr(
        xyzWok, b, iHat, jHat, kHat,
        elementHeight, scaleFac, dx, dy, dz
    )

    xyzTangent = numpy.array(xyzTangent)
    xTangent = xyzTangent[:, 0]
    yTangent = xyzTangent[:, 1]
    zTangent = xyzTangent[:, 2]

    return xTangent, yTangent, zTangent


def _wokToTangent(xWok, yWok, zWok, b, iHat, jHat, kHat,
                  elementHeight=defaults.POSITIONER_HEIGHT, scaleFac=1,
                  dx=0, dy=0, dz=0):
    """
    Convert from wok coordinates to tangent coordinates.

    xyz Wok coords are mm with orgin at wok vertex. +z points from wok toward M2.
    -x points toward the boss slithead

    In the tangent coordinate frame the xy plane is tangent to the wok
    surface at xyz position b.  The origin is set to be elementHeight above
    the wok surface.

    Parameters
    -------------
    xWok: scalar or 1D array
        x position of object in wok space mm
        (+x aligned with +RA on image at PA = 0)
    yWok: scalar or 1D array
        y position of object in wok space mm
        (+y aligned with +Dec on image at PA = 0)
    zWok: scalar or 1D array
        z position of object in wok space mm
        (+z aligned boresight and increases from the telescope to the sky)
    b: 3-vector
        x,y,z position (mm) of element hole on wok surface measured in wok coords
    iHat: 3-vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate x axis
    jHat: 3-vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate y axis
    kHat: 3-vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate z axis
    elementHeight: scalar
        height (mm) of positioner/GFA chip above wok surface
    scaleFac: scalar
        scale factor to apply to b to account for thermal expansion of wok.
        scale is applied to b radially
    dx: scalar
        x offset (mm), calibration to capture small displacements of
        tangent x
    dy: scalar
        y offset (mm), calibration to capture small displacements of
        tangent y
    dz: scalar
        z offset (mm), calibration to capture small displacements of
        tangent x

    Returns
    ---------
    xTangent: scalar or 1D array
        x position (mm) in tangent coordinates
    yTangent: scalar or 1D array
        y position (mm) in tangent coordinates
    zTangent: scalar or 1D array
        z position (mm) in tangent coordinates
    """
    # check for problems
    b = _verify3Vector(b, "b")
    iHat = _verify3Vector(iHat, "iHat")
    jHat = _verify3Vector(jHat, "jHat")
    kHat = _verify3Vector(kHat, "kHat")

    coords = numpy.array([xWok,yWok,zWok])

    # apply radial scale factor to b
    # assume that z is unaffected by
    # thermal expansion (probably reasonable)
    if scaleFac != 1:
        r = numpy.sqrt(b[0]**2+b[1]**2)
        theta = numpy.arctan2(b[1], b[0])
        r = r*scaleFac
        b[0] = r*numpy.cos(theta)
        b[1] = r*numpy.sin(theta)

    isArr = hasattr(xWok, "__len__")
    calibOff = numpy.array([dx,dy,dz])
    if isArr:
        b = numpy.array([b]*len(xWok)).T
        calibOff = numpy.array([calibOff]*len(xWok)).T

    # offset to wok base position
    coords = coords - b

    # rotate normal to wok surface at point b
    rotNorm = numpy.array([iHat, jHat, kHat], dtype="float64")
    coords = rotNorm.dot(coords)

    # offset xy plane to focal surface
    if isArr:
        coords[2,:] = coords[2,:] - elementHeight
    else:
        coords[2] = coords[2] - elementHeight

    # apply rotational calibration
    # if dRot != 0:
    #     rotZ = numpy.radians(dRot)
    #     rotMatZ = numpy.array([
    #         [numpy.cos(rotZ), numpy.sin(rotZ), 0],
    #         [-numpy.sin(rotZ), numpy.cos(rotZ), 0],
    #         [0, 0, 1]
    #     ])

    #     coords = rotMatZ.dot(coords)

    coords = coords - calibOff

    return coords[0], coords[1], coords[2]


def tangentToWok(xTangent, yTangent, zTangent, b, iHat, jHat, kHat,
                 elementHeight=defaults.POSITIONER_HEIGHT, scaleFac=1,
                 dx=0, dy=0, dz=0):
    """
    Convert from tangent coordinates at b to wok coordinates.

    xyz Wok coords are mm with origin at wok vertex. +z points from wok toward M2.
    -x points toward the boss slithead

    In the tangent coordinate frame the xy plane is tangent to the wok
    surface at xyz position b.  The origin is set to be elementHeight above
    the wok surface.

    Parameters
    -------------
    xTangent: scalar or 1D array
        x position (mm) in tangent coordinates
    yTangent: scalar or 1D array
        y position (mm) in tangent coordinates
    zTangent: scalar or 1D array
        z position (mm) in tangent coordinates
    b: 3-vector
        x,y,z position (mm) of element hole on wok surface measured in wok coords
    iHat: 3-vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate x axis
    jHat: 3-vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate y axis
    kHat: 3-vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate z axis
    elementHeight: scalar
        height (mm) of positioner/GFA chip above wok surface
    scaleFac: scalar
        scale factor to apply to b to account for thermal expansion of wok.
        scale is applied to b radially
    dx: scalar
        x offset (mm), calibration to capture small displacements of
        tangent x
    dy: scalar
        y offset (mm), calibration to capture small displacements of
        tangent y
    dz: scalar
        z offset (mm), calibration to capture small displacements of
        tangent x

    Returns
    ---------
    xWok: scalar or 1D array
        x position of object in wok space mm
        (+x aligned with +RA on image at PA = 0)
    yWok: scalar or 1D array
        y position of object in wok space mm
        (+y aligned with +Dec on image at PA = 0)
    zWok: scalar or 1D array
        z position of object in wok space mm
        (+z aligned boresight and increases from the telescope to the sky)
    """
    b = _verify3Vector(b, "b")
    iHat = _verify3Vector(iHat, "iHat")
    jHat = _verify3Vector(jHat, "jHat")
    kHat = _verify3Vector(kHat, "kHat")
    # format into list of lists
    if hasattr(xTangent, "__len__"):
        xyzTangent = numpy.array([xTangent,yTangent,zTangent]).T.tolist()
        xyzWok = libcoordio.tangentToWokArr(
            xyzTangent, list(b), list(iHat), list(jHat), list(kHat),
            elementHeight, scaleFac, dx, dy, dz)

        xyzWok = numpy.array(xyzWok)
        xWok = xyzWok[:,0]
        yWok = xyzWok[:,1]
        zWok = xyzWok[:,2]
    else:
        xWok, yWok, zWok = libcoordio.tangentToWok(
            [xTangent, yTangent, zTangent], list(b), list(iHat), list(jHat),
            list(kHat), elementHeight, scaleFac, dx, dy, dz
        )
    return xWok, yWok, zWok


def _tangentToWok(xTangent, yTangent, zTangent, b, iHat, jHat, kHat,
                 elementHeight=defaults.POSITIONER_HEIGHT, scaleFac=1,
                 dx=0, dy=0, dz=0):
    """
    Convert from tangent coordinates at b to wok coordinates.

    xyz Wok coords are mm with orgin at wok vertex. +z points from wok toward M2.
    -x points toward the boss slithead

    In the tangent coordinate frame the xy plane is tangent to the wok
    surface at xyz position b.  The origin is set to be elementHeight above
    the wok surface.

    Parameters
    -------------
    xTangent: scalar or 1D array
        x position (mm) in tangent coordinates
    yTangent: scalar or 1D array
        y position (mm) in tangent coordinates
    zTangent: scalar or 1D array
        z position (mm) in tangent coordinates
    b: 3-vector
        x,y,z position (mm) of element hole on wok surface measured in wok coords
    iHat: 3-vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate x axis
    jHat: 3-vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate y axis
    kHat: 3-vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate z axis
    elementHeight: scalar
        height (mm) of positioner/GFA chip above wok surface
    scaleFac: scalar
        scale factor to apply to b to account for thermal expansion of wok.
        scale is applied to b radially
    dx: scalar
        x offset (mm), calibration to capture small displacements of
        tangent x
    dy: scalar
        y offset (mm), calibration to capture small displacements of
        tangent y
    dz: scalar
        z offset (mm), calibration to capture small displacements of
        tangent x

    Returns
    ---------
    xWok: scalar or 1D array
        x position of object in wok space mm
        (+x aligned with +RA on image at PA = 0)
    yWok: scalar or 1D array
        y position of object in wok space mm
        (+y aligned with +Dec on image at PA = 0)
    zWok: scalar or 1D array
        z position of object in wok space mm
        (+z aligned boresight and increases from the telescope to the sky)
    """
    # check for problems
    b = _verify3Vector(b, "b")
    iHat = _verify3Vector(iHat, "iHat")
    jHat = _verify3Vector(jHat, "jHat")
    kHat = _verify3Vector(kHat, "kHat")

    calibOff = numpy.array([dx,dy,dz])
    isArr = hasattr(xTangent, "__len__")
    if isArr:
        calibOff = numpy.array([calibOff]*len(xTangent)).T

    coords = numpy.array([xTangent,yTangent,zTangent])

    # apply calibration offsets
    coords = coords + calibOff

    # apply rotational calibration
    # if dRot != 0:
    #     rotZ = -1*numpy.radians(dRot)
    #     rotMatZ = numpy.array([
    #         [numpy.cos(rotZ), numpy.sin(rotZ), 0],
    #         [-numpy.sin(rotZ), numpy.cos(rotZ), 0],
    #         [0, 0, 1]
    #     ])

    #     coords = rotMatZ.dot(coords)

    # offset xy plane by element height
    if isArr:
        coords[2, :] = coords[2, :] + elementHeight
    else:
        coords[2] = coords[2] + elementHeight

    # rotate normal to wok surface at point b
    # invRotNorm = numpy.linalg.inv(numpy.array([iHat, jHat, kHat], dtype="float64"))
    # transpose is inverse! ortho-normal rows or unit vectors!
    invRotNorm = numpy.array([iHat, jHat, kHat], dtype="float64").T

    coords = invRotNorm.dot(coords)

    # offset to wok base position
    # apply radial scale factor to b
    # assume that z is unaffected by
    # thermal expansion (probably reasonable)
    if scaleFac != 1:
        r = numpy.sqrt(b[0]**2+b[1]**2)
        theta = numpy.arctan2(b[1], b[0])
        r = r * scaleFac
        b[0] = r * numpy.cos(theta)
        b[1] = r * numpy.sin(theta)

    if isArr:
        b = numpy.array([b]*len(xTangent)).T
    coords = coords + b

    return coords[0], coords[1], coords[2]


def proj2XYplane(x, y, z, rayOrigin):
    """Given a point x, y, z orginating from rayOrigin, project
    this point onto the xy plane.

    Parameters
    ------------
    x: scalar or 1D array
        x position of point to be projected
    y: scalar or 1D array
        y position of point to be projected
    z: scalar or 1D array
        z position of point to be projected
    rayOrigin: 3-vector
        [x,y,z] origin of ray(s)

    Returns
    --------
    xProj: scalar or 1D array
        projected x value
    yProj: scalar or 1D array
        projected y value
    zProj: scalar or 1D array
        projected z value (should be zero always!)
    projDist: scalar or 1D array
        projected distance (a proxy for something like focus offset)
    """

    # intersections of lines to planes...
    # http://geomalgorithms.com/a05-_intersect-1.html
    # intersect3D_SegmentPlane
    rayOrigin = _verify3Vector(rayOrigin, "rayOrigin")
    if hasattr(x, "__len__"):
        rayOrigin = numpy.array([rayOrigin] * len(z), dtype="float64").T
        x = numpy.array(x, dtype="float64")
        y = numpy.array(y, dtype="float64")
        z = numpy.array(z, dtype="float64")

    xyz = numpy.array([x, y, z], dtype="float64")

    u = xyz - rayOrigin
    w = rayOrigin
    zHat = numpy.array([0, 0, 1])  # normal to xy plane
    D = zHat.dot(u)
    N = -1 * zHat.dot(w)

    sI = N / D
    xyzProj = rayOrigin + sI * u
    projDist = numpy.linalg.norm(xyz - xyzProj, axis=0)
    # negative projDist for points below xy plane
    projDist *= numpy.sign(z)

    return xyzProj[0], xyzProj[1], xyzProj[2], projDist

# def tangentToPositioner2(
#     xTangent, yTangent, xBeta, yBeta, la=7.4, alphaOffDeg=0, betaOffDeg=0
# ):
#     """
#     Determine alpha/beta positioner angles that place xBeta, yBeta coords in mm
#     at xTangent, yTangent.  New CPP Version.  Should never return NaN values
#     for alpha beta, and instead.

#     note: vectorized option not here yet.

#     todo: include hooks for positioner non-linearity

#     Parameters
#     -------------
#     xTangent: float
#         x position (mm) in tangent coordinates
#     yTangent: float
#         y position (mm) in tangent coordinates
#     xBeta: float
#         x position (mm) in beta arm frame
#     yBeta: float
#         y position (mm) in beta arm frame
#     la: float
#         length (mm) of alpha arm
#     alphaOffDeg: float
#         alpha zeropoint offset (deg)
#     betaOffDeg: float
#         beta zeropoint offset (deg)

#     Returns
#     ---------
#     alphaDegRH: scalar or 1D array
#         alpha angle in degrees for a right-handed robot
#     betaDegRH: scalar or 1D array
#         beta angle in degrees for a right-handed robot
#     alphaDegLH: scalar or 1D array
#         alpha angle in degrees for a right-lefthanded robot
#     betaDegLH: scalar or 1D array
#         beta angle in degrees for a left-handed robot
#     err: scalar or 1D array
#         distance beween fiber and xyWok.  This is non-zero when
#         xyWok is outside the positioners workspace
#     """

#     # C++ wrapped stuff wants lists, not numpy arrays
#         alphaRH, betaRH, alphaLH, betaLH, dist = libcoordio.tangentToPositioner2(
#             [xTangent, yTangent], [xBeta, yBeta],
#             la, alphaOffDeg, betaOffDeg
#         )

#     return alphaRH, betaRH, alphaLH, betaLH, dist


def tangentToPositioner(
    xTangent, yTangent, xBeta, yBeta, la=7.4, alphaOffDeg=0, betaOffDeg=0,
    lefthand=False
):
    """
    Determine alpha/beta positioner angles that place xBeta, yBeta coords in mm
    at xTangent, yTangent.  CPP Version.

    todo: include hooks for positioner non-linearity

    Parameters
    -------------
    xTangent: scalar or 1D array
        x position (mm) in tangent coordinates
    yTangent: scalar or 1D array
        y position (mm) in tangent coordinates
    xBeta: scalar or 1D array
        x position (mm) in beta arm frame
    yBeta: scalar or 1D array
        y position (mm) in beta arm frame
    la: scalar or 1D array
        length (mm) of alpha arm
    alphaOffDeg: scalar
        alpha zeropoint offset (deg)
    betaOffDeg: scalar
        beta zeropoint offset (deg)
    lefthand: boolean
        whether to return a lefthand or righthand parity robot

    Returns
    ---------
    alphaDeg: scalar or 1D array
        alpha angle in degrees
    betaDeg: scalar or 1D array
        beta angle in degrees
    isOK: boolean or 1D boolean array
        True if point physically accessible, False otherwise
    """
    # C++ wrapped stuff wants lists, not numpy arrays
    isArr = hasattr(xTangent, "__len__")
    if isArr:
        xyTangent = numpy.array([xTangent, yTangent]).T.tolist()

        if hasattr(xBeta, "__len__"):
            xyBeta = numpy.array([xBeta, yBeta]).T.tolist()
        else:
            xyBeta = [[xBeta, yBeta]]*len(xyTangent)

        alphaBeta = libcoordio.tangentToPositionerArr(
            xyTangent, xyBeta, la, alphaOffDeg, betaOffDeg, lefthand
        )

        alphaBeta = numpy.array(alphaBeta)
        alpha = alphaBeta[:, 0]
        beta = alphaBeta[:, 1]
    else:
        alpha, beta = libcoordio.tangentToPositioner(
            [xTangent, yTangent], [xBeta, yBeta],
            la, alphaOffDeg, betaOffDeg, lefthand
        )

    isOKAlpha = numpy.isfinite(alpha)
    isOKBeta = numpy.isfinite(beta)
    isOK = isOKAlpha & isOKBeta

    if not isArr:
        isOK = bool(isOK)

    return alpha, beta, isOK


def _tangentToPositioner(
    xTangent, yTangent, xBeta, yBeta, la=7.4, alphaOffDeg=0, betaOffDeg=0
):
    """
    Determine alpha/beta positioner angles that place xBeta, yBeta coords in mm
    at xTangent, yTangent.  Python Version.

    todo: include hooks for positioner non-linearity

    note, didn't add left hand option here...use C++

    Parameters
    -------------
    xTangent: scalar or 1D array
        x position (mm) in tangent coordinates
    yTangent: scalar or 1D array
        y position (mm) in tangent coordinates
    xBeta: scalar or 1D array
        x position (mm) in beta arm frame
    yBeta: scalar or 1D array
        y position (mm) in beta arm frame
    la: scalar or 1D array
        length (mm) of alpha arm
    alphaOffDeg: scalar
        alpha zeropoint offset (deg)
    betaOffDeg: scalar
        beta zeropoint offset (deg)

    Returns
    ---------
    alphaDeg: scalar or 1D array
        alpha angle in degrees
    betaDeg: scalar or 1D array
        beta angle in degrees
    isOK: boolean or 1D boolean array
        True if point physically accessible, False otherwise
    """

    isArr = hasattr(xTangent, "__len__")
    if isArr:
        xTangent = numpy.array(xTangent, dtype="float64")
        yTangent = numpy.array(yTangent, dtype="float64")

    if hasattr(xBeta, "__len__"):
        xBeta = numpy.array(xBeta, dtype="float64")
        yBeta = numpy.array(yBeta, dtype="float64")

    # polar coords jive better for this calculation
    thetaTangent = numpy.arctan2(yTangent, xTangent)
    rTangentSq = xTangent**2 + yTangent**2

    # convert xy Beta to radial coords
    # the origin of the beta coord system is the
    # beta axis of rotation
    thetaBAC = numpy.arctan2(yBeta, xBeta) # radians!
    rBacSq = xBeta**2+yBeta**2

    gamma = numpy.arccos(
        (la**2 + rBacSq - rTangentSq) / (2 * la * numpy.sqrt(rBacSq))
    )
    xi = numpy.arccos(
        (la**2 + rTangentSq - rBacSq) / (2 * la * numpy.sqrt(rTangentSq))
    )

    thetaTangent = numpy.degrees(thetaTangent)
    thetaBAC = numpy.degrees(thetaBAC)
    gamma = numpy.degrees(gamma)
    xi = numpy.degrees(xi)

    alphaDeg = thetaTangent - xi - alphaOffDeg
    betaDeg = 180 - gamma - thetaBAC - betaOffDeg

    # look for nans
    isOKAlpha = numpy.isfinite(alphaDeg)
    isOKBeta = numpy.isfinite(betaDeg)
    isOK = isOKAlpha & isOKBeta

    if not isArr:
        isOK = bool(isOK)

    # handle wrapping? not sure it's necessary
    alphaDeg = alphaDeg % 360
    betaDeg = betaDeg % 360  # should already be there, but...
    # if hasattr(betaDeg, "__len__"):
    #     if True in betaDeg > 180:
    #         raise RuntimeError("problem in alpha/beta conversion! beta > 180")
    # elif betaDeg > 180:
    #     raise RuntimeError("problem in alpha/beta conversion! beta > 180")

    return alphaDeg, betaDeg, isOK


def positionerToTangent(
    alphaDeg, betaDeg, xBeta, yBeta, la=7.4, alphaOffDeg=0, betaOffDeg=0
):
    """
    Determine tangent coordinates (mm) of xBeta, yBeta coords in mm
    from alpha/beta angle.  CPP version.

    todo: include hooks for positioner non-linearity

    Parameters
    -------------
    alphaDeg: scalar or 1D array
        alpha angle in degrees
    betaDeg: scalar or 1D array
        beta angle in degrees
    xBeta: scalar or 1D array
        x position (mm) in beta arm frame
    yBeta: scalar or 1D array
        y position (mm) in beta arm frame
    la: scalar or 1D array
        length (mm) of alpha arm
    alphaOffDeg: scalar
        alpha zeropoint offset (deg)
    betaOffDeg: scalar
        beta zeropoint offset (deg)

    Returns
    ---------
    xTangent: scalar or 1D array
        x position (mm) in tangent coordinates
    yTangent: scalar or 1D array
        y position (mm) in tangent coordinates
    """
    if hasattr(alphaDeg, "__len__"):
        alphaBeta = numpy.array([alphaDeg, betaDeg]).T.tolist()

        if hasattr(xBeta, "__len__"):
            xyBeta = numpy.array([xBeta, yBeta]).T.tolist()
        else:
            xyBeta = [[xBeta, yBeta]] * len(alphaBeta)

        xyTangent = libcoordio.positionerToTangentArr(
            alphaBeta, xyBeta, la, alphaOffDeg, betaOffDeg
        )

        xyTangent = numpy.array(xyTangent)
        xTangent = xyTangent[:, 0]
        yTangent = xyTangent[:, 1]
    else:
        xTangent, yTangent = libcoordio.positionerToTangent(
            [alphaDeg, betaDeg], [xBeta, yBeta],
            la, alphaOffDeg, betaOffDeg
        )
    return xTangent, yTangent


def _positionerToTangent(
    alphaDeg, betaDeg, xBeta, yBeta, la=7.4, alphaOffDeg=0, betaOffDeg=0
):
    """
    Determine tangent coordinates (mm) of xBeta, yBeta coords in mm
    from alpha/beta angle.  Python version.

    todo: include hooks for positioner non-linearity

    Parameters
    -------------
    alphaDeg: scalar or 1D array
        alpha angle in degrees
    betaDeg: scalar or 1D array
        beta angle in degrees
    xBeta: scalar or 1D array
        x position (mm) in beta arm frame
    yBeta: scalar or 1D array
        y position (mm) in beta arm frame
    la: scalar or 1D array
        length (mm) of alpha arm
    alphaOffDeg: scalar
        alpha zeropoint offset (deg)
    betaOffDeg: scalar
        beta zeropoint offset (deg)

    Returns
    ---------
    xTangent: scalar or 1D array
        x position (mm) in tangent coordinates
    yTangent: scalar or 1D array
        y position (mm) in tangent coordinates
    """
    # convert xy Beta to radial coords
    # the origin of the beta coord system is the
    # beta axis of rotation
    if hasattr(xBeta, "__len__"):
        xBeta = numpy.array(xBeta, dtype="float64")
        yBeta = numpy.array(yBeta, dtype="float64")

    thetaBAC = numpy.arctan2(yBeta, xBeta)  # radians!
    rBAC = numpy.sqrt(xBeta**2 + yBeta**2)
    alphaRad = numpy.radians(alphaDeg+alphaOffDeg)
    betaRad = numpy.radians(betaDeg+betaOffDeg)

    cosAlpha = numpy.cos(alphaRad)
    sinAlpha = numpy.sin(alphaRad)
    cosAlphaBeta = numpy.cos(alphaRad + betaRad + thetaBAC)
    sinAlphaBeta = numpy.sin(alphaRad + betaRad + thetaBAC)

    xTangent = la * cosAlpha + rBAC * cosAlphaBeta
    yTangent = la * sinAlpha + rBAC * sinAlphaBeta

    return xTangent, yTangent


def tangentToGuide(xTangent, yTangent, xBin=1, yBin=1):

    xPix = (1 / xBin) * (
        defaults.MICRONS_PER_MM / defaults.GFA_PIXEL_SIZE * xTangent + \
        defaults.GFA_CHIP_CENTER
    )

    yPix = (1 / yBin) * (
        defaults.MICRONS_PER_MM / defaults.GFA_PIXEL_SIZE * yTangent + \
        defaults.GFA_CHIP_CENTER
    )

    return xPix, yPix


def guideToTangent(xPix, yPix, xBin=1, yBin=1):
    xTangent = (
        (xPix * xBin - defaults.GFA_CHIP_CENTER)
        * defaults.GFA_PIXEL_SIZE
        / defaults.MICRONS_PER_MM
    )

    yTangent = (
        (yPix * yBin - defaults.GFA_CHIP_CENTER)
        * defaults.GFA_PIXEL_SIZE
        / defaults.MICRONS_PER_MM
    )

    return xTangent, yTangent


# class FVCUW(object):
#     # right now this is for UWs test bench, fix image scale and use
#     # euclidean transforms
#     defaultScale = 0.026926930620282834
#     defaultRotation = -0.005353542811111436
#     defaultTranslation = numpy.array([-75.69606047, -48.68561274])
#     # transform from CCD pixels to mm
#     def __init__(self):

#         self.tform = SimilarityTransform(
#             scale=self.defaultScale,
#             rotation=self.defaultRotation,
#             translation=self.defaultTranslation
#          )

#     def fit(self, xpix, ypix, xWok, yWok):
#         xypixels = numpy.array([xpix, ypix]).T
#         xyWok = numpy.array([xWok,yWok]).T
#         self.tform.estimate(xypixels, xyWok)


#     def fvcToWok(self, xpix, ypix):
#         xypixels = numpy.array([xpix, ypix]).T
#         xyWok = self.tform(xypixels)
#         return xyWok[:,0], xyWok[:,1]

#     def wokToFVC(self, xWok, yWok):
#         xyWok = numpy.array([xWok, yWok]).T
#         xypix = self.tform.inverse(xyWok)
#         return xypix[:,0], xypix[:,1]
