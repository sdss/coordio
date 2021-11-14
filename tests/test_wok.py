import numpy
import pytest

from coordio import Field, FocalPlane, Observed, Site, Tangent, Wok
from coordio.defaults import APO_MAX_FIELD_R, LCO_MAX_FIELD_R, calibration


@pytest.fixture
def prepare_test():
    numpy.random.seed(0)

    # from sdssconv.fieldCoords import parallacticAngle
    apoSite = Site("APO")
    lcoSite = Site("LCO")

    # set time (not used but required for Observed coords)
    apoSite.set_time(2458863, scale='TAI')
    lcoSite.set_time(2458863, scale='TAI')

    # start with some field coords and propagate them
    nCoords = 2000
    fcAPO = Observed([[80, 120]], site=apoSite)
    fcLCO = Observed([[80, 120]], site=lcoSite)
    thetaField = numpy.random.uniform(0, 360, size=nCoords)
    phiFieldLCO = numpy.random.uniform(0, LCO_MAX_FIELD_R, size=nCoords)
    phiFieldAPO = numpy.random.uniform(0, APO_MAX_FIELD_R, size=nCoords)

    fieldAPO = Field(
        numpy.array([thetaField, phiFieldAPO]).T,
        field_center=fcAPO
    )

    fieldLCO = Field(
        numpy.array([thetaField, phiFieldLCO]).T,
        field_center=fcLCO
    )

    focalAPO = FocalPlane(fieldAPO, site=apoSite)
    focalLCO = FocalPlane(fieldLCO, site=lcoSite)

    yield (apoSite, lcoSite, focalAPO, focalLCO)


def test_focal_wok_cycle(prepare_test):
    apoSite, lcoSite, focalAPO, focalLCO = prepare_test

    for site, focalCoords in zip([lcoSite, apoSite], [focalLCO, focalAPO]):
        obsAngle = numpy.random.uniform(0, 360)
        wokCoords = Wok(focalCoords, site=site, obsAngle=obsAngle)
        _focalCoords = FocalPlane(wokCoords, site=site)
        numpy.testing.assert_array_almost_equal(
            focalCoords, _focalCoords, decimal=10
        )


def test_wok_tangent_cycle(prepare_test):
    apoSite, lcoSite, focalAPO, focalLCO = prepare_test

    for site, focalCoords in zip([lcoSite, apoSite], [focalLCO, focalAPO]):
        obsAngle = numpy.random.uniform(0, 360)
        wokCoords = Wok(focalCoords, site=site, obsAngle=obsAngle)
        site_hole_ids = calibration.positionerTable.loc[site.name].index.tolist()
        for holeID in calibration.VALID_HOLE_IDS:
            if holeID not in site_hole_ids:
                continue
            scaleFactor = numpy.random.uniform(.97, 1.03)
            tangentCoords = Tangent(
                wokCoords, site=site, holeID=holeID,
                scaleFactor=scaleFactor, obsAngle=obsAngle
            )
            _wokCoords = Wok(tangentCoords, site=site, obsAngle=obsAngle)
            numpy.testing.assert_array_almost_equal(
                wokCoords, _wokCoords, decimal=10
            )
