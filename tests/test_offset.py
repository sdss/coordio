from coordio.utils import object_offset
import numpy
import pytest


mag_limits = {}
mag_limits['bright'] = {}
mag_limits['bright']['Boss'] = numpy.array([12.7, -999.0, 12.7, -999.0, 13.0, 13.0, 13.0, -999.0, -999.0, -999.0])
mag_limits['bright']['Apogee'] = numpy.array([-999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, 7.0, -999.0])

mag_limits['dark'] = {}
mag_limits['dark']['Boss'] = numpy.array([15.0, 15.0, 15.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0])
mag_limits['dark']['Apogee'] = numpy.array([-999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, 7.0, -999.0])


def test_flags():
    # Boss Bright
    offset_min_skybrightness = 1
    waveName = 'Boss'
    lunation = 'bright'
    flags_test = [0, 1, 2, 8, 16, 32]
    test_mags = [mag_limits[lunation][waveName][5] - 2.,
                 mag_limits[lunation][waveName][5] + 2.,
                 -999.,
                 mag_limits[lunation][waveName][5] - 2.,
                 mag_limits[lunation][waveName][5] - 2.,
                 5.]
    skybrightness = [None, None, None, 0.3, None, None]
    can_offset = [None, None, None, None, False, None]
    for flag, mag, sky, can_off in zip(flags_test, test_mags, skybrightness, can_offset):
        delta_ra, delta_dec, offset_flag = object_offset(mag, mag_limits[lunation][waveName], lunation,
                                                         waveName, safety_factor=0.1,
                                                         beta=5, FWHM=1.7, skybrightness=sky,
                                                         offset_min_skybrightness=offset_min_skybrightness,
                                                         can_offset=can_off)
        assert offset_flag == flag

    # Boss dark
    offset_min_skybrightness = 1
    waveName = 'Boss'
    lunation = 'dark'
    flags_test = [0, 1, 2, 8, 16, 32]
    test_mags = [mag_limits[lunation][waveName][1] - 2.,
                 mag_limits[lunation][waveName][1] + 2.,
                 -999.,
                 mag_limits[lunation][waveName][1] - 2.,
                 mag_limits[lunation][waveName][1] - 2.,
                 5.]
    skybrightness = [None, None, None, 0.3, None, None]
    can_offset = [None, None, None, None, False, None]
    for flag, mag, sky, can_off in zip(flags_test, test_mags, skybrightness, can_offset):
        delta_ra, delta_dec, offset_flag = object_offset(mag, mag_limits[lunation][waveName], lunation,
                                                         waveName, safety_factor=0.1,
                                                         beta=5, FWHM=1.7, skybrightness=sky,
                                                         offset_min_skybrightness=offset_min_skybrightness,
                                                         can_offset=can_off)
        assert offset_flag == flag

    # Apogee bright
    offset_min_skybrightness = 1
    waveName = 'Apogee'
    lunation = 'bright'
    flags_test = [0, 1, 2, 8, 16, 32]
    test_mags = [mag_limits[lunation][waveName][8] - 2.,
                 mag_limits[lunation][waveName][8] + 2.,
                 -999.,
                 mag_limits[lunation][waveName][8] - 2.,
                 mag_limits[lunation][waveName][8] - 2.,
                 0.1]
    skybrightness = [None, None, None, 0.3, None, None]
    can_offset = [None, None, None, None, False, None]
    for flag, mag, sky, can_off in zip(flags_test, test_mags, skybrightness, can_offset):
        delta_ra, delta_dec, offset_flag = object_offset(mag, mag_limits[lunation][waveName], lunation,
                                                         waveName, safety_factor=0.1,
                                                         beta=5, FWHM=1.7, skybrightness=sky,
                                                         offset_min_skybrightness=offset_min_skybrightness,
                                                         can_offset=can_off)
        assert offset_flag == flag

    # Apogee dark
    offset_min_skybrightness = 1
    waveName = 'Apogee'
    lunation = 'dark'
    flags_test = [0, 1, 2, 8, 16, 32]
    test_mags = [mag_limits[lunation][waveName][8] - 2.,
                 mag_limits[lunation][waveName][8] + 2.,
                 -999.,
                 mag_limits[lunation][waveName][8] - 2.,
                 mag_limits[lunation][waveName][8] - 2.,
                 0.1]
    skybrightness = [None, None, None, 0.3, None, None]
    can_offset = [None, None, None, None, False, None]
    for flag, mag, sky, can_off in zip(flags_test, test_mags, skybrightness, can_offset):
        delta_ra, delta_dec, offset_flag = object_offset(mag, mag_limits[lunation][waveName], lunation,
                                                         waveName, safety_factor=0.1,
                                                         beta=5, FWHM=1.7, skybrightness=sky,
                                                         offset_min_skybrightness=offset_min_skybrightness,
                                                         can_offset=can_off)
        assert offset_flag == flag

