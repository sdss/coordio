from coordio.utils import object_offset, Moffat2dInterp, offset_definition
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
                    sky, offset_min_skybrightness, can_off, offset_val):
        # test APO fails with 1D array
        with pytest.raises(ValueError, match='mags must be a 2D numpy.array of shape \\(N, 10\\)'):
            delta_ra, delta_dec, offset_flag = object_offset(mag, mag_limits, lunation,
                                                             waveName, 'APO', fmagloss=fmagloss,
                                                             skybrightness=sky,
                                                             offset_min_skybrightness=offset_min_skybrightness,
                                                             can_offset=can_off)

        # test LCO fails with 1D array
        with pytest.raises(ValueError, match='mags must be a 2D numpy.array of shape \\(N, 10\\)'):
            delta_ra, delta_dec, offset_flag = object_offset(mag, mag_limits, lunation,
                                                             waveName, 'LCO', fmagloss=fmagloss,
                                                             skybrightness=sky,
                                                             offset_min_skybrightness=offset_min_skybrightness,
                                                             can_offset=can_off)

        if can_off is None:
            can_off_arr = None
        else:
            can_off_arr = numpy.array([can_off, can_off])
        # test APO with 2D array
        delta_ra, delta_dec, offset_flag = object_offset(numpy.vstack((mag, mag)), mag_limits, lunation,
                                                         waveName, 'APO', fmagloss=fmagloss, skybrightness=sky,
                                                         offset_min_skybrightness=offset_min_skybrightness,
                                                         can_offset=can_off_arr)
        assert numpy.all(offset_flag == flag)
        assert numpy.all(delta_dec == 0.)
        if flag > 0:
            assert numpy.all(delta_ra == 0.)
        else:
            assert numpy.all(delta_ra > 0.)

        # test LCO with 2D array
        delta_ra, delta_dec, offset_flag = object_offset(numpy.vstack((mag, mag)), mag_limits, lunation,
                                                         waveName, 'LCO', fmagloss=fmagloss, skybrightness=sky,
                                                         offset_min_skybrightness=offset_min_skybrightness,
                                                         can_offset=can_off_arr)
        assert numpy.all(offset_flag == flag)
        assert numpy.all(delta_dec == 0.)
        if flag > 0:
            assert numpy.all(delta_ra == 0.)
        else:
            assert numpy.all(delta_ra > 0.)

        # check offset_valid
        with pytest.raises(ValueError, match='Must provide program to check valid offsets!'):
            delta_ra, delta_dec, offset_flag, valid_offset = object_offset(
                numpy.vstack((mag, mag)), mag_limits, lunation,
                waveName, 'APO', fmagloss=fmagloss, skybrightness=sky,
                offset_min_skybrightness=offset_min_skybrightness,
                can_offset=can_off_arr,
                check_valid_offset=True)
        delta_ra, delta_dec, offset_flag, valid_offset = object_offset(
            numpy.vstack((mag, mag)), mag_limits, lunation,
            waveName, 'APO', fmagloss=fmagloss, skybrightness=sky,
            offset_min_skybrightness=offset_min_skybrightness,
            can_offset=can_off_arr,
            check_valid_offset=True,
            program=numpy.array(['science',
                                'science']))
        assert numpy.all(offset_flag == flag)
        assert numpy.all(delta_dec == 0.)
        if flag > 0:
            assert numpy.all(delta_ra == 0.)
        else:
            assert numpy.all(delta_ra > 0.)
        assert numpy.all(valid_offset == offset_val)
    # Boss Bright
    offset_min_skybrightness = 1
    waveName = 'Boss'
    lunation = 'bright'
    flags_test = [0, 1, 2, 8, 16, 32, 32]
    flags_test = [f if f == 0 else f + 64 for f in flags_test]
    offset_valid = [True, True, True, False, False, False, False]

    test_mags = []
    test_mags.append(numpy.array([m - 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([m + 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([-999.] * len(mag_limits[lunation][waveName])))
    test_mags.append(numpy.array([m - 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([m - 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([5. if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([7. if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags[-1][0] = 5.

    skybrightness = [None, None, None, 0.3, None, None, None]
    can_offset = [None, None, None, None, False, None, None]
    for flag, mag, sky, can_off, offset_val in zip(flags_test, test_mags, skybrightness, can_offset, offset_valid):
        test_flags(flag, mag, mag_limits[lunation][waveName], lunation, waveName,
                   sky, offset_min_skybrightness, can_off, offset_val)

    # Boss dark
    offset_min_skybrightness = 1
    waveName = 'Boss'
    lunation = 'dark'
    flags_test = [0, 1, 2, 8, 16, 32, 32]
    flags_test = [f if f == 0 else f + 64 for f in flags_test]
    offset_valid = [True, True, True, False, False, False, False]
    
    test_mags = []
    test_mags.append(numpy.array([m - 1 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([m + 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([-999.] * len(mag_limits[lunation][waveName])))
    test_mags.append(numpy.array([m - 1 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([m - 1 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([12. if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([14. if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags[-1][0] = 12.

    skybrightness = [None, None, None, 0.3, None, None, None]
    can_offset = [None, None, None, None, False, None, None]
    for flag, mag, sky, can_off, offset_val in zip(flags_test, test_mags, skybrightness, can_offset, offset_valid):
        test_flags(flag, mag, mag_limits[lunation][waveName], lunation, waveName,
                   sky, offset_min_skybrightness, can_off, offset_val)

    # Apogee bright
    offset_min_skybrightness = 1
    waveName = 'Apogee'
    lunation = 'bright'
    flags_test = [0, 1, 2, 8, 16, 32]
    flags_test = [f if f == 0 else f + 64 for f in flags_test]
    offset_valid = [True, True, True, False, False, False]
    
    test_mags = []
    test_mags.append(numpy.array([m - 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([m + 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([-999.] * len(mag_limits[lunation][waveName])))
    test_mags.append(numpy.array([m - 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([m - 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([0.1 if m != -999. else m for m in mag_limits[lunation][waveName]]))

    skybrightness = [None, None, None, 0.3, None, None]
    can_offset = [None, None, None, None, False, None]
    for flag, mag, sky, can_off, offset_val in zip(flags_test, test_mags, skybrightness, can_offset, offset_valid):
        test_flags(flag, mag, mag_limits[lunation][waveName], lunation, waveName,
                   sky, offset_min_skybrightness, can_off, offset_val)

    # Apogee dark
    offset_min_skybrightness = 1
    waveName = 'Apogee'
    lunation = 'dark'
    flags_test = [0, 1, 2, 8, 16, 32]
    flags_test = [f if f == 0 else f + 64 for f in flags_test]
    offset_valid = [True, True, True, False, False, False]
    
    test_mags = []
    test_mags.append(numpy.array([m - 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([m + 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([-999.] * len(mag_limits[lunation][waveName])))
    test_mags.append(numpy.array([m - 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([m - 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags.append(numpy.array([0.1 if m != -999. else m for m in mag_limits[lunation][waveName]]))

    skybrightness = [None, None, None, 0.3, None, None]
    can_offset = [None, None, None, None, False, None]
    for flag, mag, sky, can_off, offset_val in zip(flags_test, test_mags, skybrightness, can_offset, offset_valid):
        test_flags(flag, mag, mag_limits[lunation][waveName], lunation, waveName,
                   sky, offset_min_skybrightness, can_off, offset_val)

    # test engineering design_mode
    # Boss Bright
    offset_min_skybrightness = 1
    waveName = 'Boss'
    lunation = 'bright'
    flags_test = [64]

    test_mags = []
    test_mags.append(numpy.array([m - 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))

    skybrightness = [None]
    can_offset = [None]
    for flag, mag, sky, can_off in zip(flags_test, test_mags, skybrightness, can_offset):
        test_flags(flag, mag, numpy.zeros(10) - 999., lunation, waveName,
                   sky, offset_min_skybrightness, can_off, True)

    # test get combination of all flags
    # Boss Bright
    offset_min_skybrightness = 1
    waveName = 'Boss'
    lunation = 'bright'
    flags_test = [1 + 32]
    flags_test = [f if f == 0 else f + 64 for f in flags_test]

    test_mags = []
    test_mags.append(numpy.array([m + 2 if m != -999. else m for m in mag_limits[lunation][waveName]]))
    test_mags[0][0] = 5.

    skybrightness = [None]
    can_offset = [None]
    for flag, mag, sky, can_off in zip(flags_test, test_mags, skybrightness, can_offset):
        test_flags(flag, mag, mag_limits[lunation][waveName], lunation, waveName,
                   sky, offset_min_skybrightness, can_off, False)


    # test bright neighbor exclusion radius for very bright stars
    offset_min_skybrightness = 1
    waveName = 'Boss'
    lunation = 'bright'
    test_mags = numpy.array([0.1 if m != -999. else m for m in mag_limits[lunation][waveName]])
    r_exclude, _ = offset_definition(numpy.vstack((mag, mag)),
                                     mag_limits[lunation][waveName],
                                     lunation=lunation,
                                     waveName=waveName,
                                     fmagloss=fmagloss,
                                     obsSite='APO')
    assert numpy.all(r_exclude > 0.)

    offset_min_skybrightness = 1
    waveName = 'Apogee'
    lunation = 'bright'
    test_mags = numpy.array([0.1 if m != -999. else m for m in mag_limits[lunation][waveName]])
    r_exclude, _ = offset_definition(numpy.vstack((mag, mag)),
                                     mag_limits[lunation][waveName],
                                     lunation=lunation,
                                     waveName=waveName,
                                     fmagloss=fmagloss,
                                     obsSite='APO')
    assert numpy.all(r_exclude > 0.)
