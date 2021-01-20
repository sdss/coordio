import numpy

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

    xField = -1*numpy.degrees(numpy.arctan2(x, z))
    yField = -1*numpy.degrees(numpy.arctan2(y, z))
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
    try:
        if theta < 0:
            theta += 360
    except:
        # theta is array
        inds = numpy.argwhere(theta < 0)
        theta[inds] = theta[inds] + 360
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
    # theta, phi = thetaPhi
    # while theta < 0:
    #     theta += 360
    # while theta >= 360:
    #     theta -= 360
    # if theta < 0 or theta >= 360:
    #     raise RuntimeError("theta must be in range [0, 360]")

    # if phi < -90 or phi > 90:
    #     raise RuntimeError("phi must be in range [-90, 90]")

    theta, phi = numpy.radians(theta), numpy.radians(phi)
    x = r*numpy.cos(theta) * numpy.sin(phi)
    y = r*numpy.sin(theta) * numpy.sin(phi)
    z = r*numpy.cos(phi)

    # if numpy.isnan(x) or numpy.isnan(y) or numpy.isnan(z):
    #     raise RuntimeError("NaN output [%.2f, %.2f, %.2f] for input [%.2f, %.2f]"%(x, y, z, numpy.degrees(theta), numpy.degrees(phi)))

    return [x, y, z]