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

    __extra_arrays__ = []
    __extra_params__ = []

    def __new__(cls, value, **kwargs):

        obj = numpy.asarray(value).view(cls)

        for param in cls.__extra_params__ + cls.__extra_arrays__:
            if param in kwargs:
                setattr(obj, param, kwargs.get(param, None))

        return obj

    def __array_finalize__(self, obj):

        if obj is None or not isinstance(obj, numpy.ndarray):
            return obj

        if self.shape[1] != 2:
            raise CoordinateError('The input array must be Nx2.')

        for param in self.__extra_params__:
            setattr(self, param, getattr(obj, param, None))

        for param in self.__extra_arrays__:
            array = getattr(obj, param, None)
            if array is None:
                array = numpy.zeros(self.shape[0], dtype=numpy.float64)
            elif isinstance(array, (numpy.ndarray, list, tuple)):
                pass
            else:
                array = numpy.tile(array, self.shape[0])
            setattr(self, param, array)

            if array.ndim < 1:
                raise CoordinateError(f'{param} must be a 1D array.')
            if array.shape[0] != self.shape[0]:
                raise CoordinateError(f'{param} size does not match '
                                      'the coordinate values.')

    def __array_wrap__(self, out_arr, context=None):

        return super().__array_wrap__(self, out_arr, context)

    def __getitem__(self, sl):

        value = self.view(numpy.ndarray).__getitem__(sl)

        if not isinstance(value, numpy.ndarray) or value.ndim != 2:
            return value

        kwargs = {}

        for param in self.__extra_arrays__:
            kwargs[param] = getattr(self, param)[sl]

        for param in self.__extra_params__:
            kwargs[param] = getattr(self, param)

        return self.__class__(value, **kwargs)
