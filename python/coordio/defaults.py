# default values collected here

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