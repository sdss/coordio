import numpy
import pytest
from numpy.testing import assert_array_almost_equal

from coordio import CoordIOError, Guide, Site, Tangent, calibration, defaults


numpy.random.seed(0)


CCD_SIZE_MM = defaults.GFA_CHIP_CENTER * 2 * defaults.GFA_PIXEL_SIZE / \
           defaults.MICRONS_PER_MM

nCoords = 100
apoSite = Site("APO")
lcoSite = Site("LCO")

# # set time (not used but required for Observed coords)
apoSite.set_time(2458863, scale='TAI')
lcoSite.set_time(2458863, scale='TAI')


def test_guide_cycle():
    tangX = numpy.random.uniform(
        -CCD_SIZE_MM/2, CCD_SIZE_MM/2, size=nCoords
    )
    tangY = numpy.random.uniform(
        -CCD_SIZE_MM/2, CCD_SIZE_MM/2, size=nCoords
    )
    tangZ = numpy.random.uniform(-0.1, 0.1, size=nCoords)

    tangXYZ = numpy.array([tangX, tangY, tangZ]).T

    for site in [apoSite, lcoSite]:
        for holeID in calibration.VALID_GUIDE_IDS:
            scaleFactor = numpy.random.uniform(.99, 1.01)
            binX = numpy.random.choice([1,2])
            binY = numpy.random.choice([1,2])
            obsAngle = numpy.random.uniform(0,360)
            tc = Tangent(
                tangXYZ, site=site, holeID=holeID,
                scaleFactor=scaleFactor, obsAngle=obsAngle
            )
            gc = Guide(tc, binX=binX, binY=binY)

            _tc = Tangent(
                gc, site=site, holeID=holeID,
                scaleFactor=scaleFactor, obsAngle=obsAngle
            )

            assert_array_almost_equal(tc.xProj, _tc.xProj, decimal=10)
            assert_array_almost_equal(tc.xProj, _tc[:,0], decimal=10)
            assert_array_almost_equal(tc.yProj, _tc.yProj, decimal=10)
            assert_array_almost_equal(tc.yProj, _tc[:,1], decimal=10)
            assert_array_almost_equal(_tc[:,2], numpy.zeros(nCoords), decimal=10)
            assert True not in gc.guide_warn


def test_guide_warn():

    tangX = numpy.random.uniform(-CCD_SIZE_MM/2, CCD_SIZE_MM/2, size=nCoords)
    tangY = numpy.random.uniform(-CCD_SIZE_MM/2, CCD_SIZE_MM/2, size=nCoords)
    tangZ = numpy.random.uniform(-0.1, 0.1, size=nCoords)

    badInds = [2, 3, 4, 5]
    # plug in values that will land outside the chip
    tangX[2] = -5*CCD_SIZE_MM

    tangY[3] = 2*CCD_SIZE_MM

    tangX[4] = CCD_SIZE_MM
    tangY[4] = -CCD_SIZE_MM

    tangY[5] = -5*CCD_SIZE_MM

    tangXYZ = numpy.array([tangX, tangY, tangZ]).T

    for site in [apoSite, lcoSite]:
        for holeID in calibration.VALID_GUIDE_IDS:
            scaleFactor = numpy.random.uniform(.99, 1.01)
            binX = numpy.random.choice([1,2])
            binY = numpy.random.choice([1,2])
            obsAngle = numpy.random.uniform(0,360)
            tc = Tangent(
                tangXYZ, site=site, holeID=holeID,
                scaleFactor=scaleFactor, obsAngle=obsAngle
            )
            gc = Guide(tc, xBin=binX, yBin=binY)

            for ii, warn in enumerate(gc.guide_warn):
                if ii in badInds:
                    assert warn == True
                else:
                    assert warn == False


@pytest.mark.xfail(reason="GFAs not yet handled.")
def test_guide_fail():
    wl = defaults.INST_TO_WAVE["Boss"]

    tangX = numpy.random.uniform(0, CCD_SIZE_MM, size=nCoords)
    tangY = numpy.random.uniform(0, CCD_SIZE_MM, size=nCoords)
    tangZ = numpy.random.uniform(-0.1, 0.1, size=nCoords)
    tangXYZ = numpy.array([tangX, tangY, tangZ]).T

    tc = Tangent(
        tangXYZ, site=apoSite, holeID="GFA-S1", wavelength=wl
    )

    with pytest.raises(CoordIOError) as ee:
        # bad wavelength supplied
        Guide(tc)
    assert "non-guide wavelength" in str(ee)

    tc = Tangent(
        tangXYZ, site=apoSite, holeID="R0C1", wavelength=wl
    )

    with pytest.raises(CoordIOError) as ee:
        # bad hole supplied
        Guide(tc)
    assert "wok hole" in str(ee)

    gx = numpy.random.uniform(50, 80, size=nCoords)
    gy = numpy.random.uniform(50, 80, size=nCoords)
    gxy = numpy.array([gx,gy]).T

    gc = Guide(gxy)

    with pytest.raises(CoordIOError) as ee:
        Tangent(gc, site=apoSite, holeID="R0C1")
    assert "non-GFA location" in str(ee)

    with pytest.raises(CoordIOError) as ee:
        Tangent(gc, site=apoSite, holeID="GFA-S1", wavelength=wl)
    assert "non-guide wavelength" in str(ee)
