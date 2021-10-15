#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-21
# @Filename: test_coordinate.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import numpy
import pytest

from coordio import CoordinateError
from coordio.coordinate import Coordinate, Coordinate2D, Coordinate3D


class _TestCoordinate(Coordinate):

    __extra_arrays__ = ['array1', 'array2']
    __extra_params__ = ['param1']
    __warn_arrays__ = ['warn']
    __computed_arrays__ = ['array12']

    def __new__(cls, value, **kwargs):
        obj = super().__new__(cls, value, **kwargs)

        array12 = obj.array1 + obj.array2
        obj.array12 = array12

        # warn if sum is greater than 10
        obj.warn = obj.array12 > 10

        return obj

class _TestCoordinate2D(Coordinate2D):
    pass

class _TestCoordinate3D(Coordinate3D):
    pass

def test_coordinate():

    test_coordinate = _TestCoordinate([[1, 2], [3, 4]])

    assert numpy.array_equal(test_coordinate, [[1, 2], [3, 4]])

    assert test_coordinate.param1 is None

    assert test_coordinate.array1 is not None
    assert test_coordinate.array2 is not None

    assert numpy.array_equal(test_coordinate.array1, [0., 0.])

    test_coordinate = _TestCoordinate([[1, 2],
                                       [3, 4]], param1=1, array1=[5, 6])
    assert test_coordinate.param1 == 1
    assert numpy.array_equal(test_coordinate.array1, [5, 6])


def test_tile():

    test_coordinate = _TestCoordinate([[1, 2], [3, 4]], array1=5)
    assert numpy.array_equal(test_coordinate.array1, [5, 5])


def test_None():

    with pytest.raises(CoordinateError) as ee:
        _TestCoordinate(None)
    assert 'The input array must be NxM' in str(ee)


def test_bad_input():

    with pytest.raises(CoordinateError) as ee:
        _TestCoordinate([[1], [2]])
    assert 'The input array must be NxM' in str(ee)

    with pytest.raises(CoordinateError) as ee:
        _TestCoordinate([[1, 2], [3, 4]], array1=numpy.array([1]))
    assert 'array1 size does not match the coordinate values.' in str(ee)

    with pytest.raises(CoordinateError):
        _TestCoordinate2D([[1,2,3],[4,5,6]])

    with pytest.raises(CoordinateError):
        _TestCoordinate3D([[1,2],[4,5]])

def test_slice():

    test_coordinate = _TestCoordinate([[1, 2],
                                       [3, 4],
                                       [5, 6]], array1=[7, 8, 9])

    sliced = test_coordinate[0:2, :]

    assert isinstance(sliced, _TestCoordinate)
    assert id(sliced) != id(test_coordinate)
    assert (sliced == numpy.array([[1, 2], [3, 4]])).all()
    assert (sliced.array1 == numpy.array([7, 8])).all()
    assert (sliced.array2 == numpy.array([0, 0])).all()

    # Confirm the original array is not modified
    assert (test_coordinate == numpy.array([[1, 2], [3, 4], [5, 6]])).all()
    assert (test_coordinate.array1 == numpy.array([7, 8, 9])).all()


def test_slice_to_numpy():

    test_coordinate = _TestCoordinate([[1, 2],
                                       [3, 4],
                                       [5, 6]], array1=[7, 8, 9])

    sliced = test_coordinate[:, 0]

    assert type(sliced).__name__ == 'ndarray'
    assert (sliced == numpy.array([1, 3, 5])).all()
    assert hasattr(sliced, 'array1') is False


def test_array_2d():

    test_coordinate = _TestCoordinate([[1, 2], [3, 4]],
                                      array1=[7, 8],
                                      array2=numpy.array([[10, 11],
                                                          [12, 13]]))

    assert test_coordinate.array2.ndim == 2

    sliced = test_coordinate[0:1, :]
    assert (sliced.array2 == numpy.array([[10, 11]])).all()


def test_copy():

    test_coordinate = _TestCoordinate([[1, 2],
                                       [3, 4],
                                       [5, 6]], array1=[7, 8, 9])

    test_copy = test_coordinate.copy()

    assert id(test_copy) != id(test_coordinate)
    assert isinstance(test_copy, _TestCoordinate)
    assert (test_coordinate == test_copy).all()
    assert (test_coordinate.array1 == test_copy.array1).all()


def test_more_slicing():
    arr = [1, 2, 3, 4, 5, 6]
    array1 = numpy.array(arr)
    array2 = numpy.array(arr)
    dim = 3
    param1 = "param1"

    coords = numpy.array([arr] * dim).T

    tc = _TestCoordinate(coords, array1=array1, array2=array2, param1=param1)

    with pytest.raises(IndexError):
        tc[:, dim]  # ask for a dim that doesn't exist

    with pytest.raises(IndexError):
        tc[6, :]  # ask for something off the end of the list

    tc1 = tc[2:3,:]
    numpy.testing.assert_equal(tc1, coords[2:3,:])
    numpy.testing.assert_equal(tc1.array1, array1[2:3])
    numpy.testing.assert_equal(tc1.array2, array2[2:3])
    assert tc1.param1 == tc.param1

    tc1 = tc[:,1:3]
    numpy.testing.assert_equal(tc1, coords[:,1:3])

    with pytest.raises(AttributeError):
        tc1.array1

    with pytest.raises(AttributeError):
        tc1.param1

    filtArr = tc.array1 < 4
    tc1 = tc[filtArr]
    numpy.testing.assert_equal(tc1, coords[filtArr])
    numpy.testing.assert_equal(tc1.array1, array1[filtArr])
    numpy.testing.assert_equal(tc1.array2, array2[filtArr])
    assert tc1.param1 == tc.param1

    filtArr = [0,3,5]
    tc1 = tc[filtArr]
    numpy.testing.assert_equal(tc1, coords[filtArr])
    numpy.testing.assert_equal(tc1.array1, array1[filtArr])
    numpy.testing.assert_equal(tc1.array2, array2[filtArr])
    assert tc1.param1 == tc.param1


def test_warn_arr():
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    array1 = numpy.array(arr)
    array2 = numpy.array(arr)
    coords = numpy.array([array1, array1]).T

    array12 = array1 + array2
    warn = array12 > 10

    tc = _TestCoordinate(coords, array1=array1, array2=array2, param1="hi")

    slicer = tc.array1 >= 3
    _tc = tc[slicer]
    _coords = coords[slicer]
    _array1 = array1[slicer]
    _array2 = array2[slicer]
    _array12 = array12[slicer]
    _warn = warn[slicer]

    numpy.testing.assert_equal(_tc, _coords)
    numpy.testing.assert_equal(_tc.array1, _array1)
    numpy.testing.assert_equal(_tc.array2, _array2)
    numpy.testing.assert_equal(_tc.array12, _array12)
    numpy.testing.assert_equal(_tc.warn, _warn)
    assert _tc.param1 == "hi"

    _tc = tc[slicer, :]
    _coords = coords[slicer, :]
    _array1 = array1[slicer]
    _array2 = array2[slicer]
    _array12 = array12[slicer]
    _warn = warn[slicer]

    numpy.testing.assert_equal(_tc, _coords)
    numpy.testing.assert_equal(_tc.array1, _array1)
    numpy.testing.assert_equal(_tc.array2, _array2)
    numpy.testing.assert_equal(_tc.array12, _array12)
    numpy.testing.assert_equal(_tc.warn, _warn)
    assert _tc.param1 == "hi"

    slicer = [2,4,6]
    _tc = tc[slicer]
    _coords = coords[slicer]
    _array1 = array1[slicer]
    _array2 = array2[slicer]
    _array12 = array12[slicer]
    _warn = warn[slicer]
    assert _tc.param1 == "hi"
