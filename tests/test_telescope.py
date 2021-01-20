from coordio import Observed, Site, Field
import numpy

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

if __name__ == "__main__":
    test_observedToField_APO()

