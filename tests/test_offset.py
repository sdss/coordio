from coordio.utils import object_offset, Moffat2dInterp
import numpy
import pytest


mag_limits = {}
mag_limits['bright'] = {}
mag_limits['bright']['Boss'] = numpy.array([12.7, -999.0, 12.7, -999.0, 13.0, 13.0, 13.0, -999.0, -999.0, -999.0])
mag_limits['bright']['Apogee'] = numpy.array([-999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, 7.0, -999.0])

mag_limits['dark'] = {}
mag_limits['dark']['Boss'] = numpy.array([15.0, 15.0, 15.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0])
mag_limits['dark']['Apogee'] = numpy.array([-999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, 7.0, -999.0])

fmagloss = Moffat2dInterp()


def test_all_flags():
    def  test_flags(flag, mag, mag_limits, lunation, waveName,
                    sky, offset_min_skybrightness, can_off):
        delta_ra, delta_dec, offset_flag = object_offset(mag, mag_limits, lunation,
                                                         waveName, fmagloss=fmagloss,
                                                         skybrightness=sky,
                                                         offset_min_skybrightness=offset_min_skybrightness,
                                                         can_offset=can_off)
        assert offset_flag == flag
        assert delta_dec == 0.
        if flag > 0:
            assert delta_ra == 0.
        else:
            assert delta_ra > 0.

        if can_off is None:
            can_off_arr = None
        else:
            can_off_arr = numpy.array([can_off])
        delta_ra, delta_dec, offset_flag = object_offset(numpy.array([mag]), mag_limits, lunation,
                                                         waveName, fmagloss=fmagloss,
                                                         skybrightness=sky,
                                                         offset_min_skybrightness=offset_min_skybrightness,
                                                         can_offset=can_off_arr)
        assert numpy.all(offset_flag == flag)
        assert numpy.all(delta_dec == 0.)
        if flag > 0:
            assert numpy.all(delta_ra == 0.)
        else:
            assert numpy.all(delta_ra > 0.)
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
        test_flags(flag, mag, mag_limits[lunation][waveName], lunation, waveName,
                   sky, offset_min_skybrightness, can_off)

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
        test_flags(flag, mag, mag_limits[lunation][waveName], lunation, waveName,
                   sky, offset_min_skybrightness, can_off)

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
        test_flags(flag, mag, mag_limits[lunation][waveName], lunation, waveName,
                   sky, offset_min_skybrightness, can_off)

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
        test_flags(flag, mag, mag_limits[lunation][waveName], lunation, waveName,
                   sky, offset_min_skybrightness, can_off)

