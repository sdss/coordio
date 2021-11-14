import numpy
import pandas
import pytest

from coordio import Field, FocalPlane, Observed, Site, Tangent, Wok
from coordio.defaults import (APO_MAX_FIELD_R, LCO_MAX_FIELD_R,
                              MICRONS_PER_MM, calibration)


@pytest.mark.xfail(reason="GFAs not yet handled.")
def test_proj():

    numpy.random.seed(0)

    # from sdssconv.fieldCoords import parallacticAngle
    apoSite = Site("APO")
    lcoSite = Site("LCO")

    # set time (not used but required for Observed coords)
    apoSite.set_time(2458863, scale='TAI')
    lcoSite.set_time(2458863, scale='TAI')

    # start with some field coords and propagate them
    fcAPO = Observed([[80, 120]], site=apoSite)
    fcLCO = Observed([[80, 120]], site=lcoSite)

    nCoords = 200000

    thetaField = numpy.random.uniform(0, 360, size=nCoords)
    phiFieldLCO = numpy.sqrt(numpy.random.uniform(0, LCO_MAX_FIELD_R**2, size=nCoords))
    phiFieldAPO = numpy.sqrt(numpy.random.uniform(0, APO_MAX_FIELD_R**2, size=nCoords))

    fieldAPO = Field(
        numpy.array([thetaField, phiFieldAPO]).T,
        field_center=fcAPO)

    fieldLCO = Field(
        numpy.array([thetaField, phiFieldLCO]).T,
        field_center=fcLCO
    )

    focalAPO = FocalPlane(fieldAPO, site=apoSite)
    focalLCO = FocalPlane(fieldLCO, site=lcoSite)

    obsAngle = 0

    wokAPO = Wok(focalAPO, site=apoSite, obsAngle=obsAngle)
    wokLCO = Wok(focalLCO, site=lcoSite, obsAngle=obsAngle)

    scaleFactor = 1

    # get tangent coords at these locations
    gfaIDs = ["GFA-S1", "GFA-S2", "GFA-S3", "GFA-S4", "GFA-S5", "GFA-S6"]
    robotIDs = [
        "R0C1", "R+13C1", "R+13C14", "R0C27",
        "R-13C14", "R+13C7", "R-13C1"
    ]

    randomIDs = list(numpy.random.choice(calibration.VALID_HOLE_IDS, size=10))

    allIDs = list(set(gfaIDs + robotIDs + randomIDs))

    for wokCoords, site in zip([wokLCO, wokAPO], [lcoSite, apoSite]):
        for holeID in allIDs:
            tc = Tangent(
                wokLCO, site=lcoSite, holeID=holeID,
                scaleFactor=scaleFactor, obsAngle=obsAngle
            )

            df = {
                "x": tc[:, 0],
                "y": tc[:, 1],
                "z": tc[:, 2],
                "r": numpy.sqrt(tc[:, 0]**2 + tc[:, 1]**2),
                "px": tc.xProj,
                "py": tc.yProj,
                "pr": numpy.sqrt(tc.xProj**2 + tc.yProj**2),
                "d": tc.distProj,

            }
            df = pandas.DataFrame(df)
            df = df[df.r < 40] # keep only thinks within 40 mm xy from tangent center

            dx = df.x - df.px
            dy = df.y - df.py
            dispXY = numpy.sqrt(dx**2 + dy**2) * MICRONS_PER_MM
            dispFoc = (df.d - df.z) * MICRONS_PER_MM

            # less than 5 microns diff between direct proj, and actual proj
            assert numpy.max(dispXY) < 7
            assert numpy.max(numpy.abs(dispFoc)) < 1 # micron

            # zffset is no larger than 700 microns from focal surface
            varFoc = (numpy.mean(df.z) - df.z)*MICRONS_PER_MM

            assert numpy.max(numpy.abs(varFoc)) < 300

            # plt.figure()
            # plt.plot(df.r, dispXY, '.')
            # plt.figure()
            # plt.plot(df.r, dispFoc, '.')
            # plt.figure()
            # plt.plot(df.r, df.z*MICRONS_PER_MM, '.')
            # plt.show()
