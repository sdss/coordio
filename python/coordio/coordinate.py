#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-17
# @Filename: coordinate.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import numpy

from .exceptions import CoordinateError


class Coordinate(numpy.ndarray):
    r"""Base class to represent a coordinate system.

    `.Coordinate` is a wrapper around the `numpy.ndarray` to represent a
    series of points in a given coordinate system. It handles most of the
    particularities of `subclassing a Numpy array
    <https://numpy.org/doc/stable/user/basics.subclassing.html>`__.

    A `.Coordinate` instance must be a 2D array of dimensions :math:`N\times M`
    in which N is the number of coordinate points, and M are their axis values.
    Extra parameters and arrays can be added to the coordinate object. For
    example, the following code minimally implements an ICRS coordinate system

    ::

        class ICRS(Coordinate):

        __extra_params__ ['name']
        __extra_arrays__ = ['epoch', 'pmra', 'pmdec', 'parallax', 'rvel']

    Now we can instantiate a set of coordinates ::

        >>> icrs = ICRS([[100., 20.], [200., 10], [300., 5.]],
                        pmra=[1, 2, 0], pmdec=[-0.5, 0.5, 0],
                        name='my_coordinates')
        >>> print(icrs)
        ICRS([[100.,  20.],
              [200.,  10.],
              [300.,   5.]])
        >>> print(icrs.pmra)
        [1, 2, 0]
        >>> print(icrs.parallax)
        [0, 0, 0]
        >>> print(icrs.name)
        my_coordinates

    Values in ``__extra_params__`` are propagated to the new instance as
    attributes and can be defined with any user-provided value.
    ``__extra_arrays__`` on the other hand are expected to be arrays with
    the same length as the number of coordinates. If the extra array value
    is not provided when instantiating the coordinate, it is replaced with
    arrays of zeros of size N (the number of coordinates). If you don't want
    the default value to be zeros you'll need to override the
    ``__array_finalize__`` method, for example ::

        def __array_finalize__(self, obj):

            super().__array_finalize__(obj)

            if getattr(obj, 'epoch', None) is None:
                self.epoch += 2451545.0

    And now the default value for ``epoch`` will be 2451545.

    `.Coordinate` takes care of slicing the arrays appropriately. If we slice
    the coordinate along the first axis the resulting object will be another
    coordinate instance in which both the internal array and the extra ones
    are sliced correctly ::

        >>> new_icrs = icrs[0:2, :]
        >>> print(new_icrs)
        ICRS([[100.,  20.],
              [200.,  10.]])
        >>> print(icrs.pmra)
        [1, 2]

    But if we slice it in a way that the resulting array cannot represent the
    coordinate system anymore, a simple Numpy array will be returned, without
    any of the additional information ::

        >>> icrs[:, 0]
        array([100., 200., 300.])

    """

    __extra_arrays__ = []
    __extra_params__ = []
    __computed_arrays__ = []  # values that are computed (not passed)

    def __new__(cls, value, **kwargs):

        obj = numpy.asarray(value).view(cls)

        if len(obj.shape) != 2 or obj.shape[1] < 2:
            raise CoordinateError('The input array must be NxM, M>=2.')

        for param in obj.__extra_params__:
            setattr(obj, param, kwargs.get(param, None))

        for param in obj.__extra_arrays__:
            array = kwargs.get(param, None)
            if array is None:
                # arrays default to zeros, could cause trouble
                # but its nice for things like pm, radVal, etc
                # which should probably default to zeros
                array = numpy.zeros(obj.shape[0], dtype=numpy.float64)
            elif isinstance(array, (numpy.ndarray, list, tuple)):
                array = numpy.array(array)
            else:
                array = numpy.tile(array, obj.shape[0])
            setattr(obj, param, array)

            if array.shape[0] != obj.shape[0]:
                raise CoordinateError(f'{param} size does not match '
                                      'the coordinate values.')

        # create zero arrays for computed arrays
        for param in obj.__computed_arrays__:
            array = kwargs.get(param, None)
            array = numpy.zeros(obj.shape[0], dtype=numpy.float64)
            setattr(obj, param, array)

        return obj

    def __array_finalize__(self, obj):

        if obj is None or not isinstance(obj, numpy.ndarray):
            return obj

        # This is so that copies and slices of array copy the params.
        params = []
        params += self.__extra_arrays__
        params += self.__extra_params__
        params += self.__computed_arrays__
        for param in params:
            setattr(self, param, getattr(obj, param, None))

    def __getitem__(self, sl):

        sliced = super().__getitem__(sl)

        if (not isinstance(sliced, numpy.ndarray) or
                sliced.ndim != 2 or sliced.shape[1] != self.shape[1]):
            return sliced.view(numpy.ndarray)

        for param in self.__extra_arrays__ + self.__computed_arrays__:
            setattr(sliced, param, getattr(sliced, param)[sl[0]])

        for param in self.__extra_params__:
            setattr(sliced, param, getattr(sliced, param))

        return sliced

    @property
    def coordSysName(self):
        return self.__class__.__name__

