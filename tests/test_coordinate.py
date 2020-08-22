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
from coordio.coordinate import Coordinate


class _TestCoordinate(Coordinate):

    __extra_arrays__ = ['array1', 'array2']
    __extra_params__ = ['param1']


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
    assert 'The input array must be NxM, M>=2.' in str(ee)


def test_bad_input():

    with pytest.raises(CoordinateError) as ee:
        _TestCoordinate([[1], [2]])
    assert 'The input array must be NxM, M>=2.' in str(ee)

    with pytest.raises(CoordinateError) as ee:
        _TestCoordinate([[1, 2], [3, 4]], array1=numpy.array([1]))
    assert 'array1 size does not match the coordinate values.' in str(ee)


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