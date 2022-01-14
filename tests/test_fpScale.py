import numpy
import pytest

from coordio import Field, FocalPlane, Observed, Site, Tangent, Wok
from coordio.defaults import APO_MAX_FIELD_R, LCO_MAX_FIELD_R, calibration


def test_fpScaleAPO():
    nCoords = 200
    apoSite = Site("APO")
    apoSite.set_time(2458863, scale='TAI')
    fcAPO = Observed([[80, 120]], site=apoSite)
    phiFieldAPO = numpy.random.uniform(0, APO_MAX_FIELD_R, size=nCoords)
    thetaField = numpy.random.uniform(0, 360, size=nCoords)

    fieldAPO = Field(
        numpy.array([thetaField, phiFieldAPO]).T,
        field_center=fcAPO
    )

    focalAPOInit = FocalPlane(fieldAPO, site=apoSite, fpScale=1)

    wokRadii = []
    for sf in [0.999882, 1, 1.000118]:
        focalAPO = FocalPlane(fieldAPO, site=apoSite, fpScale=sf)
        wokAPO = Wok(focalAPO, site=apoSite, obsAngle=0)
        rad = numpy.linalg.norm(wokAPO[:,:2], axis=1)
        wokRadii.append(rad)
        _focalAPO = FocalPlane(wokAPO, site=apoSite, fpScale=sf)

        # dFocalFocalAPO = focalAPO - focalAPOInit
        # assert numpy.max(numpy.linalg.norm(dFocalFocalAPO, axis=1)) < 1e-10

        # dFocalWokAPO = focalAPO - _focalAPO
        # assert numpy.max(numpy.linalg.norm(dFocalWokAPO, axis=1)) < 1e-10

    wokRadii = numpy.array(wokRadii)
    dwokRadii = numpy.diff(wokRadii, axis=0)
    assert numpy.max(dwokRadii) < 0

if __name__ == "__main__":
    test_fpScaleAPO()