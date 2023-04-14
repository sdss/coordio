from coordio import Observed, Site, Field, FocalPlane
from coordio import CoordIOError, CoordIOUserWarning
from coordio.defaults import APO_MAX_FIELD_R, LCO_MAX_FIELD_R
from coordio.defaults import VALID_WAVELENGTHS
import numpy

import pytest

# from sdssconv.fieldCoords import fieldToFocal, focalToField

numpy.random.seed(0)

SMALL_NUM = 1e-10

# from sdssconv.fieldCoords import parallacticAngle
apoSite = Site("APO")
lcoSite = Site("LCO")

# set time (not used but required for Observed coords)
apoSite.set_time(2458863, scale='TAI')
lcoSite.set_time(2458863, scale='TAI')


def test_observedToField_APO():

    #### start checks along the meridian
    ####################################
    azCen = 180 # (pointing south)
    altCen = 45
    obsCen = Observed([[altCen, azCen]], site=apoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=apoSite)
    field = Field(obsCoord, field_center=obsCen)

    assert abs(float(field.x)) < SMALL_NUM
    assert float(field.y) > 0
    assert numpy.abs(field.flatten()[0]-90) < SMALL_NUM
    assert numpy.abs(field.flatten()[1]-1) < SMALL_NUM

    azCen = 0 # (north)
    altCen = 45 # above north star
    obsCen = Observed([[altCen, azCen]], site=apoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=apoSite)
    field = Field(obsCoord, field_center=obsCen)

    assert abs(float(field.x)) < SMALL_NUM
    assert float(field.y) < 0
    assert numpy.abs(field.flatten()[0]-270) < SMALL_NUM
    assert numpy.abs(field.flatten()[1]-1) < SMALL_NUM

    azCen = 0  # (north)
    altCen = 20  # below north star
    obsCen = Observed([[altCen, azCen]], site=apoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=apoSite)
    field = Field(obsCoord, field_center=obsCen)

    assert abs(float(field.x)) < SMALL_NUM
    assert float(field.y) > 0
    assert numpy.abs(field.flatten()[0]-90) < SMALL_NUM
    assert numpy.abs(field.flatten()[1]-1) < SMALL_NUM

    ##### check field rotation (off meridian)
    #########################################
    # remember +x is eastward
    azCen = 180 + 20 # (south-west)
    altCen = 45
    obsCen = Observed([[altCen, azCen]], site=apoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=apoSite)
    field = Field(obsCoord, field_center=obsCen)

    assert float(field.x) > 0
    assert float(field.y) > 0
    assert field.flatten()[0] > 1
    assert field.flatten()[0] < 89

    # check field rotation (off meridian)
    # remember +x is eastward
    azCen = 180 - 20 # (south-east)
    altCen = 45
    obsCen = Observed([[altCen, azCen]], site=apoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=apoSite)
    field = Field(obsCoord, field_center=obsCen)

    assert float(field.x) < 0
    assert float(field.y) > 0
    assert field.flatten()[0] > 91
    assert field.flatten()[0] < 179


    # check field rotation (off meridian)
    # remember +x is eastward
    azCen = 10 # (slightly east of north)
    altCen = apoSite.latitude - 10
    obsCen = Observed([[altCen, azCen]], site=apoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=apoSite)
    field = Field(obsCoord, field_center=obsCen)

    assert float(field.x) < 0
    assert float(field.y) > 0

    azCen = 10 # (slightly east of north)
    altCen = apoSite.latitude + 10
    obsCen = Observed([[altCen, azCen]], site=apoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=apoSite)
    field = Field(obsCoord, field_center=obsCen)

    assert float(field.x) < 0
    assert float(field.y) < 0


def test_observedToField_LCO():

    #### start checks along the meridian
    ####################################
    azCen = 0 # (north)
    altCen = 45
    obsCen = Observed([[altCen, azCen]], site=lcoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=lcoSite)
    field = Field(obsCoord, field_center=obsCen)

    assert abs(float(field.x)) < SMALL_NUM
    assert float(field.y) < 0

    azCen = 180 # (south)
    altCen = numpy.abs(lcoSite.latitude) + 10 # above SCP, (lat is negative!)
    obsCen = Observed([[altCen, azCen]], site=lcoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=lcoSite)
    field = Field(obsCoord, field_center=obsCen)

    assert abs(float(field.x)) < SMALL_NUM
    assert float(field.y) > 0

    azCen = 180 # (south)
    altCen = numpy.abs(lcoSite.latitude) - 10 # below SCP, (lat is negative!)
    obsCen = Observed([[altCen, azCen]], site=lcoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=lcoSite)
    field = Field(obsCoord, field_center=obsCen)
    assert abs(float(field.x)) < SMALL_NUM
    assert float(field.y) < 0

    ##### check field rotation (off meridian)
    #########################################
    # remember +x is eastward
    azCen = 20 # (north-east)
    altCen = 45
    obsCen = Observed([[altCen, azCen]], site=lcoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=lcoSite)
    field = Field(obsCoord, field_center=obsCen)
    assert float(field.x) < 0
    assert float(field.y) < 0

    # remember +x is eastward
    azCen = 290 # (north-west)
    altCen = 45
    obsCen = Observed([[altCen, azCen]], site=lcoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=lcoSite)
    field = Field(obsCoord, field_center=obsCen)
    assert float(field.x) > 0
    assert float(field.y) < 0

    # # check field rotation (off meridian)
    # # remember +x is eastward
    azCen = 170 # (slightly east of south)
    altCen = lcoSite.latitude - 10
    obsCen = Observed([[altCen, azCen]], site=lcoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=lcoSite)
    field = Field(obsCoord, field_center=obsCen)
    assert float(field.x) < 0
    assert float(field.y) < 0

    azCen = 190 # (slightly west of south)
    altCen = lcoSite.latitude - 10
    obsCen = Observed([[altCen, azCen]], site=lcoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=lcoSite)
    field = Field(obsCoord, field_center=obsCen)
    assert float(field.x) > 0
    assert float(field.y) < 0


def test_raises():
    azCen = 190  # (slightly west of south)
    altCen = lcoSite.latitude - 10
    badCen = Observed([[altCen, azCen], [altCen, azCen]], site=lcoSite)
    obsCoord = Observed([[altCen+1, azCen]], site=lcoSite)
    with pytest.raises(CoordIOError):
        Field(obsCoord)
    with pytest.raises(CoordIOError):
        Field(obsCoord, field_center=[1,2])
    with pytest.raises(CoordIOError):
        Field(obsCoord, field_center=badCen)


def test_field_obs_cycle():
    # try a bunch of pointings make sure the round trip works
    azCens = numpy.random.uniform(1,359, size=30)
    altCens = numpy.random.uniform(0,89, size=30)
    for site in [lcoSite, apoSite]:
        for azCen, altCen in zip(azCens, altCens):
            azCoords = azCen + numpy.random.uniform(-1,1, size=10)
            altCoords = altCen + numpy.random.uniform(-1,1, size=10)
            altAzs = numpy.array([altCoords, azCoords]).T
            obs = Observed(altAzs, site=site)
            fc = Observed([[altCen, azCen]], site=site)
            field = Field(obs, field_center=fc)
            obs1 = Observed(field, site=site)
            # with open("/Users/csayres/Desktop/last.txt", "w") as f:
            #     f.write("%s %.8f %.8f\n"%(site.name, azCen, altCen))
            #     for coord in altAzs:
            #         f.write("%.8f %.8f\n"%(coord[0], coord[1]))

            numpy.testing.assert_array_almost_equal(obs, obs1, decimal=9)
            numpy.testing.assert_array_almost_equal(obs.ra, obs1.ra, decimal=9)
            numpy.testing.assert_array_almost_equal(obs.dec, obs1.dec, decimal=9)
            numpy.testing.assert_array_almost_equal(obs.ha, obs1.ha, decimal=9)
            numpy.testing.assert_array_almost_equal(obs.pa, obs1.pa, decimal=9)

            fieldArr = numpy.array(field.copy())
            field1 = Field(fieldArr, field_center=fc)

            numpy.testing.assert_equal(field, field1)
            for attr in ['x', 'y', 'z', 'x_angle', 'y_angle']:
                numpy.testing.assert_almost_equal(
                    getattr(field, attr), getattr(field1, attr), decimal=9
                )


def test_field_to_focal():
    npts = 10
    phis = numpy.zeros(npts) + 2 # two deg off axis
    thetas = numpy.zeros(npts) + 45
    coordArr = numpy.array([thetas, phis]).T
    wls = numpy.random.choice([5400., 6231., 16600.], size=npts)
    site = apoSite
    fc = Observed([[80, 92]], site=site)
    field = Field(coordArr, field_center=fc)
    fp = FocalPlane(field, wavelength=wls, site=site)

    fpArr = numpy.array(fp)
    fp2 = FocalPlane(fpArr, wavelength=wls, site=site)

    numpy.testing.assert_array_equal(fp, fp2)
    numpy.testing.assert_array_equal(fp.b, fp2.b)
    numpy.testing.assert_array_equal(fp.R, fp2.R)


def test_reasonable_field_focal_cycle():
    arcsecError = 0.01 # tenth of an arcsec on cycle

    nCoords = 100000
    for site, maxField in zip(
        [lcoSite, apoSite], [LCO_MAX_FIELD_R, APO_MAX_FIELD_R]
    ):
        fc = Observed([[80, 120]], site=site)
        thetaField = numpy.random.uniform(0, 360, size=nCoords)
        phiField = numpy.random.uniform(0, maxField, size=nCoords)
        wls = numpy.random.choice(list(VALID_WAVELENGTHS), size=nCoords)
        # wls = 16600
        coordArr = numpy.array([thetaField, phiField]).T
        field = Field(coordArr, field_center=fc)
        assert not True in field.field_warn
        fp = FocalPlane(field, wavelength=wls, site=site)
        assert not True in fp.field_warn
        field2 = Field(fp, field_center=fc)
        assert not True in field2.field_warn

        xyz = numpy.array([field.x, field.y, field.z]).T
        xyz2 = numpy.array([field2.x, field2.y, field2.z]).T

        angErrors = []
        for _xyz, _xyz2 in zip(xyz, xyz2):
            # print("norms", numpy.linalg.norm(_xyz), numpy.linalg.norm(_xyz2))
            dt = _xyz.dot(_xyz2)
            if dt > 1:
                assert dt-1 < 1e-8 # make sure it's basically zero
                dt = 1  # nan in arccos if greater than 1
            # print("xyz", _xyz, _xyz2)
            # print("\n\n")
            err = numpy.degrees(numpy.arccos(dt))*3600
            angErrors.append(err)
        angErrors = numpy.array(angErrors)
        maxErr = numpy.max(angErrors)
        assert maxErr < arcsecError


def test_unreasonable_field_focal_cycle():
    # for lco, apo. lco errors remain small despite large field
    arcsecErrs = [1, 11]

    nCoords = 10000
    for site, maxField, errLim in zip(
        [lcoSite, apoSite], [LCO_MAX_FIELD_R, APO_MAX_FIELD_R], arcsecErrs
    ):
        fc = Observed([[80, 120]], site=site)
        thetaField = numpy.random.uniform(0, 360, size=nCoords)
        phiField = numpy.random.uniform(1.5*maxField, 1.6*maxField, size=nCoords)
        wls =  numpy.random.choice(list(VALID_WAVELENGTHS), size=nCoords)
        # wls = 16600.
        coordArr = numpy.array([thetaField, phiField]).T
        field = Field(coordArr, field_center=fc)
        assert not True in field.field_warn
        with pytest.warns(CoordIOUserWarning):
            fp = FocalPlane(field, wavelength=wls, site=site)
        assert not False in fp.field_warn
        with pytest.warns(CoordIOUserWarning):
            field2 = Field(fp, field_center=fc)
        assert not False in field2.field_warn

        xyz = numpy.array([field.x, field.y, field.z]).T
        xyz2 = numpy.array([field2.x, field2.y, field2.z]).T

        angErrors = []
        for _xyz, _xyz2 in zip(xyz, xyz2):
            # print("norms", numpy.linalg.norm(_xyz), numpy.linalg.norm(_xyz2))
            dt = _xyz.dot(_xyz2)
            if dt > 1:
                assert dt-1 < 1e-8 # make sure it's basically zero
                dt = 1  # nan in arccos if greater than 1
            # print("\n\n")
            err = numpy.degrees(numpy.arccos(dt))*3600
            angErrors.append(err)
        angErrors = numpy.array(angErrors)
        maxErr = numpy.max(angErrors)
        print("maxErr", maxErr)
        # break
        assert maxErr < errLim

        # print(angErrors, angErrors.shape)


def test_invalid_wavelength():
    site = apoSite
    nCoords = 100
    fc = Observed([[80, 120]], site=site)
    thetaField = numpy.random.uniform(0, 360, size=nCoords)
    phiField = numpy.random.uniform(0, 1, size=nCoords)
    wls = 250 # invalid
    coordArr = numpy.array([thetaField, phiField]).T
    field = Field(coordArr, field_center=fc)
    with pytest.raises(CoordIOError):
        fp = FocalPlane(field, wavelength=wls, site=site)
    wls = numpy.random.choice([30, 40, 50], size=nCoords)
    with pytest.raises(CoordIOError):
        fp = FocalPlane(field, wavelength=wls, site=site)


def test_invalid_site():
    site = "junk"
    with pytest.raises(CoordIOError):
        fc = Observed([[80, 120]], site=site)
    site = Site("APO") # time not specified
    with pytest.raises(CoordIOError):
        fc = Observed([[80, 120]], site=site)


def test_focal_plane_closest_wavelength():
    site = apoSite
    thetaField = numpy.random.uniform(0, 360, size=1)
    phiField = numpy.random.uniform(0, 1, size=1)
    coordArr = numpy.array([thetaField, phiField]).T

    fc = Observed([[80, 120]], site=site)
    field = Field(coordArr, field_center=fc)
    wls = 100
    fp = FocalPlane(field, wavelength=wls, site=site, use_closest_wavelength=True)

    assert numpy.allclose(fp.wavelength, [5400.0])


def test_focal_plane_closest_wavelength_array():
    site = apoSite
    nCoords = 3
    thetaField = numpy.random.uniform(0, 360, size=nCoords)
    phiField = numpy.random.uniform(0, 1, size=nCoords)
    coordArr = numpy.array([thetaField, phiField]).T

    fc = Observed([[80, 120]], site=site)
    field = Field(coordArr, field_center=fc)
    wls = numpy.array([100, 6500, 8000])
    fp = FocalPlane(field, wavelength=wls, site=site, use_closest_wavelength=True)

    assert numpy.allclose(fp.wavelength, [5400.0, 6231.0, 6231.0])
