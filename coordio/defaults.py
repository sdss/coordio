import os
import pandas as pd
import numpy
from .coordinate import CoordIOError

# default/constant values collected here...for now
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

POSITIONER_HEIGHT = 143  # mm, distance from wok surface to fiber
# extracted from ZEMAX model model
# z location of wok vertex in FP coords
APO_WOK_Z_OFFSET_ZEMAX = -776.4791 - POSITIONER_HEIGHT
LCO_WOK_Z_OFFSET_ZEMAX = -993.0665 - POSITIONER_HEIGHT

APO_MAX_FIELD_R = 1.5  # max field radius (deg)
LCO_MAX_FIELD_R = 1.1  # max field radius (deg)

# read in the focal plane model file
fpModelFile = os.path.join(os.path.dirname(__file__), "etc", "focalPlaneModel.csv")
FP_MODEL = pd.read_csv(fpModelFile, comment="#")


def getFPModelParams(site, direction, waveCat):
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
    if direction not in ["focal", "field"]:
        raise CoordIOError(
            "direction must be one of focal or field, got %s"%direction
        )
    if waveCat not in ["Apogee", "Boss", "GFA"]:
        raise CoordIOError(
            "waveCat must be one of Apogee Boss or GFA, got %s"%waveCat
        )
    if site not in ["LCO", "APO"]:
        raise CoordIOError("site must be one of APO or LCO, got %s"%site)

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


# read in the wok orientation model file
wokOrientFile = os.path.join(os.path.dirname(__file__), "etc", "wokOrientation.csv")
wokOrient = pd.read_csv(wokOrientFile, comment="#")


def getWokOrient(site):
    """Return the wok orientation given the site.

    Returns
    --------
    x : float
        x position of wok vertex in focal plane coords mm
    x : float
        y position of wok vertex in focal plane coords mm
    x : float
        z position of wok vertex in focal plane coords mm
    xTilt : float
        tilt of wok coord sys about focal plane x axis deg
    yTilt : float
        tilt of wok coord sys about focal plane y axis deg
    """
    if site not in ["LCO", "APO"]:
        raise CoordIOError("site must be one of APO or LCO, got %s"%site)
    row = wokOrient[wokOrient.site == site]
    x = float(row.x)
    y = float(row.y)
    z = float(row.z)
    xTilt = float(row.xTilt)
    yTilt = float(row.yTilt)

    return x, y, z, xTilt, yTilt


wokCoordFile = os.path.join(os.path.dirname(__file__), "etc", "wokCoords.csv")
wokCoords = pd.read_csv(wokCoordFile, comment="#", index_col=0)
VALID_HOLE_IDS = list(set(wokCoords["holeID"]))
VALID_GUIDE_IDS = [ID for ID in VALID_HOLE_IDS if ID.startswith("GFA")]


def getHoleOrient(site, holeID):
    """Return orientation of hole position in the wok

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

    if site not in ["LCO", "APO"]:
        raise CoordIOError("site must be one of APO or LCO, got %s" % site)
    if holeID not in VALID_HOLE_IDS:
        raise CoordIOError("%s is not a valid hole ID" % holeID)

    row = wokCoords[(wokCoords.wokType == site) & (wokCoords.holeID == holeID)]

    b = numpy.array([
        float(row.x),
        float(row.y),
        float(row.z)
    ])

    iHat = numpy.array([
        float(row.ix),
        float(row.iy),
        float(row.iz)
    ])

    jHat = numpy.array([
        float(row.jx),
        float(row.jy),
        float(row.jz)
    ])

    kHat = numpy.array([
        float(row.kx),
        float(row.ky),
        float(row.kz)
    ])

    return b, iHat, jHat, kHat

# read in positioner table
positionerTableFile = os.path.join(
    os.path.dirname(__file__), "etc", "positionerTable.csv"
)
positionerTable = pd.read_csv(positionerTableFile, comment="#", index_col=0)


def getPositionerData(site, holeID):
    """Return data specific to a positioner

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

    """

    if site not in ["LCO", "APO"]:
        raise CoordIOError("site must be one of APO or LCO, got %s"%site)
    if holeID not in VALID_HOLE_IDS:
        raise CoordIOError("%s is not a valid hole ID" % holeID)

    row = positionerTable[
        (positionerTable.wokID == site) & (positionerTable.holeID == holeID)
    ]


    aal = float(row.alphaArmLen)
    metX = float(row.metX)
    metY = float(row.metY)
    apX = float(row.apX)
    apY = float(row.apY)
    bossX = float(row.bossX)
    bossY = float(row.bossY)

    return aal, metX, metY, apX, apY, bossX, bossY


