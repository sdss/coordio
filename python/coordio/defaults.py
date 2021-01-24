import os
import pandas as pd
# default/constant values collected here

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