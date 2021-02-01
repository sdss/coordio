import numpy
from numpy.testing import assert_array_almost_equal
import pytest

from coordio import (Tangent, PositionerBoss, PositionerApogee,
                    PositionerMetrology, Site, defaults, CoordIOUserWarning)

import matplotlib.pyplot as plt

numpy.random.seed(0)

apoSite = Site("APO")
lcoSite = Site("LCO")

# # set time (not used but required for Observed coords)
apoSite.set_time(2458863, scale='TAI')
lcoSite.set_time(2458863, scale='TAI')

modelMetXY = numpy.array([14.314, 0])
# boss xy position in solid model
modelBossXY = numpy.array([14.965, -0.376])
# apogee xy position in solid model
modelApXY = numpy.array([14.965, 0.376])
alphaLen = 7.4 # mm

maxRadSci = numpy.linalg.norm(numpy.array([7.4, 0]) + modelBossXY)
maxRadMet = numpy.linalg.norm(numpy.array([7.4, 0]) + modelMetXY)

minRadSci = numpy.linalg.norm(numpy.array([7.4, 0]) - modelBossXY)
minRadMet = numpy.linalg.norm(numpy.array([7.4, 0]) - modelMetXY)

nCoords = 10000

def sampleAnnulus(nCoords, R1, R2):
    # R1 is max radius, R2 in min radius
    theta = 360 * numpy.random.random(size=nCoords)
    r = numpy.sqrt(numpy.random.random(size=nCoords)*(R1**2-R2**2)+R2**2)
    x = r * numpy.cos(numpy.radians(theta))
    y = r * numpy.sin(numpy.radians(theta))

    return x, y, theta, r



wl = defaults.INST_TO_WAVE["Boss"]


def test_good_coords():
    xMet, yMet, thetaMet, distMet = sampleAnnulus(
        nCoords, 0.999 * maxRadMet, 1.001 * minRadMet
    )
    xSci, ySci, thetaSci, distSci = sampleAnnulus(
        nCoords, 0.999 * maxRadSci, 1.001 * minRadSci
    )


    metXYZ = numpy.zeros((nCoords,3))
    metXYZ[:,0] = xMet
    metXYZ[:,1] = yMet

    sciXYZ = numpy.zeros((nCoords,3))
    sciXYZ[:,0] = xSci
    sciXYZ[:,1] = ySci

    plist = [PositionerApogee, PositionerBoss, PositionerMetrology]
    tlist = [sciXYZ, sciXYZ, metXYZ]

    for site in [apoSite, lcoSite]:
        for holeID in ["R-13C1", "R0C15"]:
            for _Pos, tanXYZ in zip(plist, tlist):
                tc = Tangent(
                    tanXYZ, site=site, holeID=holeID, wavelength=wl
                )
                pc = _Pos(tc, site=site, holeID=holeID)
                assert not True in pc[:, 0] > 360
                assert not True in pc[:, 0] < 0
                assert not True in pc[:, 1] > 180
                assert not True in pc[:, 1] < 0
                assert not True in pc.positioner_warn
                assert numpy.isfinite(numpy.sum(pc))
                _tc = Tangent(pc, site=site, holeID=holeID, wavelength=wl)
                assert_array_almost_equal(tc, _tc, decimal=10)


def test_bad_coords():
    xMet, yMet, thetaMet, distMet = sampleAnnulus(nCoords, 0.9*minRadMet, 0)
    xSci, ySci, thetaSci, distSci = sampleAnnulus(nCoords, 0.9*minRadSci, 0)


    metXYZ = numpy.zeros((nCoords,3))
    metXYZ[:,0] = xMet
    metXYZ[:,1] = yMet

    sciXYZ = numpy.zeros((nCoords,3))
    sciXYZ[:,0] = xSci
    sciXYZ[:,1] = ySci

    plist = [PositionerApogee, PositionerBoss, PositionerMetrology]
    tlist = [sciXYZ, sciXYZ, metXYZ]

    for site in [apoSite, lcoSite]:
        for holeID in ["R-13C1", "R0C15"]:
            for _Pos, tanXYZ in zip(plist, tlist):
                tc = Tangent(
                    tanXYZ, site=site, holeID=holeID, wavelength=wl
                )
                pc = _Pos(tc, site=site, holeID=holeID)
                assert not False in pc.positioner_warn
                assert not numpy.isfinite(numpy.sum(pc))
                with pytest.warns(CoordIOUserWarning):
                    _tc = Tangent(pc, site=site, holeID=holeID, wavelength=wl)


def test_bad_coords2():
    xMet, yMet, thetaMet, distMet = sampleAnnulus(
        nCoords, 2 * maxRadMet, 1.1 * maxRadMet
    )
    xSci, ySci, thetaSci, distSci = sampleAnnulus(
        nCoords, 2 * maxRadSci, 1.1 * maxRadSci
    )


    metXYZ = numpy.zeros((nCoords,3))
    metXYZ[:,0] = xMet
    metXYZ[:,1] = yMet

    sciXYZ = numpy.zeros((nCoords,3))
    sciXYZ[:,0] = xSci
    sciXYZ[:,1] = ySci

    plist = [PositionerApogee, PositionerBoss, PositionerMetrology]
    tlist = [sciXYZ, sciXYZ, metXYZ]

    for site in [apoSite, lcoSite]:
        for holeID in ["R-13C1", "R0C15"]:
            for _Pos, tanXYZ in zip(plist, tlist):
                tc = Tangent(
                    tanXYZ, site=site, holeID=holeID, wavelength=wl
                )
                pc = _Pos(tc, site=site, holeID=holeID)
                assert not False in pc.positioner_warn
                assert not numpy.isfinite(numpy.sum(pc))
                with pytest.warns(CoordIOUserWarning):
                    _tc = Tangent(pc, site=site, holeID=holeID, wavelength=wl)


if __name__ == "__main__":
    test_good_coords()
