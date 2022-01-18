import os

import pandas as pd

from coordio.calibration import Calibration

from .exceptions import CoordIOError


# default/constant values collected here...for now

# a scale factor between xy focal and xy wok
FOCAL_SCALE = 0.9998

MICRONS_PER_MM = 1000
GFA_PIXEL_SIZE = 13.5  # micron
GFA_CHIP_CENTER = 1024  # unbinned pixels

EPOCH = 2451545.0  # J2000
WAVELENGTH = 6231.0  # angstrom, GFA wavelength
VALID_WAVELENGTHS = set([5400., 6231., 16600.])

# wavelengths in angstroms
WAVE_TO_INST = {
    5400.0: "Boss",
    6231.0: "GFA",  # sdss-r band
    16600.0: "Apogee"
}

INST_TO_WAVE = {
    "Boss": 5400.0,
    "GFA": 6231.0,
    "Apogee": 16600.0
}

PLATE_SCALE = {
    "APO": 217.7358,  # mm/deg
    "LCO": 330.275
}

POSITIONER_HEIGHT = 143.1  # mm, distance from wok surface to fiber
ALPHA_LEN = 7.4  # mm, length of the alpha arm (distance between alpha and beta axes)
BETA_LEN = 16.2  # mm, distance from beta axis to top flat edge of the robot
# extracted from ZEMAX model model
# z location of wok vertex in FP coords
APO_WOK_Z_OFFSET_ZEMAX = -776.4791 - POSITIONER_HEIGHT
LCO_WOK_Z_OFFSET_ZEMAX = -993.0665 - POSITIONER_HEIGHT

APO_MAX_FIELD_R = 1.5  # max field radius (deg)
LCO_MAX_FIELD_R = 1.1  # max field radius (deg)

# nominal fiber positions in beta arm frame
MET_BETA_XY = [14.314, 0]
# boss xy position in solid model
BOSS_BETA_XY = [14.965, -0.376]
# apogee xy position in solid model
AP_BETA_XY = [14.965, 0.376]

# default orientation of tangent to a flat wok
IHAT = [0, -1, 0]
JHAT = [1, 0, 0]
KHAT = [0, 0, 1]


def getFPModelParams(site, direction, waveCat):
    """Find the right focal plane model to use given direction and waveCat.

    Parameters
    ----------
    site : str
        The site string.
    direction : str
        "focal" or "field" for the direction of the conversion
    waveCat : str
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
    if direction not in ["focal", "field"]:
        raise CoordIOError(
            "direction must be one of focal or field, got %s" % direction
        )
    if waveCat not in ["Apogee", "Boss", "GFA"]:
        raise CoordIOError(
            "waveCat must be one of Apogee Boss or GFA, got %s" % waveCat
        )

    row = calibration.FP_MODEL.loc[(site, direction, waveCat)]

    R = float(row.R)
    b = float(row.b)
    c0 = float(row.c0)
    c1 = float(row.c1)
    c2 = float(row.c2)
    c3 = float(row.c3)
    c4 = float(row.c4)
    return R, b, c0, c1, c2, c3, c4


def getWokOrient(site):
    """Return the wok orientation from the wok calibration files.

    Returns
    --------
    x : float
        x position of wok vertex in focal plane coords mm
    y : float
        y position of wok vertex in focal plane coords mm
    z : float
        z position of wok vertex in focal plane coords mm
    xTilt : float
        tilt of wok coord sys about focal plane x axis deg
    yTilt : float
        tilt of wok coord sys about focal plane y axis deg
    """

    row = calibration.wokOrient.loc[site]
    x = float(row.x)
    y = float(row.y)
    z = float(row.z)
    xTilt = float(row.xTilt)
    yTilt = float(row.yTilt)

    return x, y, z, xTilt, yTilt


def getHoleOrient(site, holeID):
    """Return orientation of hole position in the wok.

    Parameters
    ----------
    site : str
        The site string.
    holeID : str or list of str
        Either a string with a valid holeID (e.g. R+13C1) or a list of
        valid holeIDs. In the latter case, the returned arrays are
        2D with axis 1 being the holeID dimmension.

    Returns
    --------
    b : numpy.ndarray
        [x,y,z] base position of hole in wok coords (mm)
    iHat : numpy.ndarray
        [x,y,z] unit vector, direction of x Tangent in wok coords
    jHat : numpy.ndarray
        [x,y,z] unit vector, direction of y Tangent in wok coords
    kHat : numpy.ndarray
        [x,y,z] unit vector, direction of z Tangent in wok coords
    """

    wokCoordsH = calibration.wokCoords.loc[(site, holeID)]

    b = wokCoordsH.loc[['xWok', 'yWok', 'zWok']].to_numpy()
    iHat = wokCoordsH.loc[['ix', 'iy', 'iz']].to_numpy()
    jHat = wokCoordsH.loc[['jx', 'jy', 'jz']].to_numpy()
    kHat = wokCoordsH.loc[['kx', 'ky', 'kz']].to_numpy()

    return b, iHat, jHat, kHat


def getPositionerData(site, holeID):
    """Return data specific to a positioner.

    Returns:
    --------
    alphaArmLength : float
        alpha arm length (mm)
    metX : float
        metrology fiber x (mm), in beta arm coordinates
    metY : float
        metrology fiber y (mm), in beta arm coordinates
    apX : float
        apogee fiber y (mm), in beta arm coordinates
    apY : float
        apogee fiber x (mm), in beta arm coordinates
    bossX : float
        boss fiber y (mm), in beta arm coordinates
    bossY : float
        boss fiber x (mm), in beta arm coordinates
    alphaOffset : float
        alpha arm offset at (0, 0), in degrees
    betaOffset : float
        beta arm offset at (0, 0), in degrees
    dx: float
        y offset (mm), calibration to capture small displacements of
        tangent y
    dy: float
        z offset (mm), calibration to capture small displacements of
        tangent x

    """

    if holeID not in calibration.VALID_HOLE_IDS:
        raise CoordIOError("%s is not a valid hole ID" % holeID)

    row = calibration.positionerTable.loc[(site, holeID)]

    aal = float(row.alphaArmLen)
    metX = float(row.metX)
    metY = float(row.metY)
    apX = float(row.apX)
    apY = float(row.apY)
    bossX = float(row.bossX)
    bossY = float(row.bossY)
    alphaOffset = float(row.alphaOffset)
    betaOffset = float(row.betaOffset)
    dx = float(row.dx)
    dy = float(row.dy)

    return aal, metX, metY, apX, apY, bossX, bossY, alphaOffset, betaOffset, dx, dy


def parseDesignRef():
    designRefFile = os.path.join(calibration.paths[0], "fps_DesignReference.txt")
    _row = []
    _col = []
    _xWok = []
    _yWok = []
    _fType = []
    _holeName = []

    with open(designRefFile, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split("#")[0]
        if not line:
            continue
        row, col, x, y, fType = line.split()

        col = int(col)
        row = int(row)

        # make col 1 indexed to match pdf maps
        # and wok hole naming convention
        if row == -99:
            holeName = "F%i" % col
            col = None
            row = None

        elif row <= 0:
            holeName = "R%iC%i" % (row, col + 1)
        else:
            holeName = "R+%iC%i" % (row, col + 1)

        _row.append(row)
        _col.append(col)
        _xWok.append(float(x))
        _yWok.append(float(y))
        _fType.append(fType)
        _holeName.append(holeName)

    d = {}
    d["holeName"] = _holeName
    d["row"] = _row
    d["col"] = _col
    d["xWok"] = _xWok
    d["yWok"] = _yWok
    d["fType"] = _fType

    df = pd.DataFrame(d)
    return df


calibration = Calibration()
