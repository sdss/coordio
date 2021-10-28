#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-17
# @Filename: coordinate.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import warnings
from typing import Type, TypeVar

import numpy

from .defaults import VALID_WAVELENGTHS, WAVELENGTH
from .exceptions import CoordinateError, CoordIOError, CoordIOUserWarning
from .site import Site


T = TypeVar('T', bound="Coordinate", covariant=True)


def verifySite(kwargs, strict=True):
    """Search through kwargs for site parameter, and verify
    that it looks ok.  Otherwise raise a CoordIOError

    Parameters
    ------------
    kwargs : dict
        kwargs passed to Coordinate __new__
    strict : bool
        if True, a time must be specified on the site

    """
    if kwargs.get('site', None) is None:
        raise CoordIOError('Site must be passed')

    else:
        site = kwargs.get('site')
        if not isinstance(site, Site):
            raise CoordIOError('Site must be passed')
        if strict and site.time is None:
            raise CoordIOError(
                "Time of observation must be specified on Site"
            )


def verifyWavelength(kwargs, lenArray, strict=True):
    """Search through kwargs for wavelength parameter.  If not existent,
    return insert GFA wavelength to kwargs (of the right size).  If strict and
    wavelengths do not correspond to Apogee, Boss, or GFA wavelengths, raise a
    CoordIOError

    Parameters
    -----------
    kwargs : dict
        kwargs passed to Coordinate __new__
    lenArray : int
        length of the array to create if wavelength not present in kwargs
    strict : bool
        if strict is True, wavelengths may only be those corresponding to
        Apogee, Boss, or GFA

    """

    wls = kwargs.get("wavelength", None)
    if wls is None:
        # create an array of correct length default to gfa wavelength
        wls = numpy.zeros(lenArray) + WAVELENGTH
        warnings.warn(
            "Warning! Wavelengths not supplied, defaulting to %i angstrom" % WAVELENGTH,
            CoordIOUserWarning
        )
    else:
        if hasattr(wls, "__len__"):
            # array passed
            wls = numpy.array(wls, dtype=numpy.float64)
        else:
            # single value passed
            wls = numpy.zeros(lenArray) + float(wls)
        wlSet = set(wls)
        if strict and not wlSet.issubset(VALID_WAVELENGTHS):
            raise CoordIOError(
                "Invalid wavelength passed to FocalPlane \
                valid wavelengths are %s" % (str(VALID_WAVELENGTHS))
            )

    # modify
    kwargs["wavelength"] = wls


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
    __warn_arrays__ = []  # boolean arrays to indicate warnings
    __coord_dim__ = None

    def __new__(cls: Type[T], value, **kwargs) -> T:
        if value is not None:
            value = value.copy()  # this prevents weirdness

        obj = numpy.asarray(value, dtype=numpy.float64).view(cls)

        if len(obj.shape) != 2 or obj.shape[1] < 2:
            raise CoordinateError('The input array must be NxM')

        if obj.shape[1] < 2:
            raise CoordinateError('The input array must be NxM, M>=2.')

        if obj.__coord_dim__ is not None and obj.shape[1] != obj.__coord_dim__:
            raise CoordinateError(
                'The input array must be Nx%i.' % obj.__coord_dim__
            )

        for kwarg in kwargs:
            if kwarg not in obj.__extra_arrays__ and kwarg not in obj.__extra_params__:
                raise CoordinateError(f'Invalid input argument {kwarg!r}.')

        for param in obj.__extra_params__:
            setattr(obj, param, kwargs.get(param, None))

        # initialize extra arrays arrays
        # set to zeros if not passed
        for param in obj.__extra_arrays__:
            array = kwargs.get(param, None)
            if array is None:
                # arrays default to zeros, could cause trouble
                # but its nice for things like pm, radVal, etc
                # which should probably default to zeros
                array = numpy.zeros(obj.shape[0], dtype=numpy.float64)
            elif isinstance(array, (numpy.ndarray, list, tuple)):
                array = numpy.array(array, dtype=numpy.float64)
            else:
                array = numpy.tile(array, obj.shape[0])
            setattr(obj, param, array)

            if array.shape[0] != obj.shape[0]:
                raise CoordinateError(f'{param} size does not match '
                                      'the coordinate values.')

        # initialize computed arrays to zeros
        for param in obj.__computed_arrays__:
            array = numpy.zeros(obj.shape[0], dtype=numpy.float64)
            setattr(obj, param, array)

        # initialize warning arrays to false
        for param in obj.__warn_arrays__:
            array = numpy.array([False] * obj.shape[0])
            setattr(obj, param, array)

        return obj

    def __array_finalize__(self, obj):
        # self is the new array, obj is the old array
        if obj is None or not isinstance(obj, numpy.ndarray):
            return obj

        # This is so that copies and slices of array copy the params.
        params = []
        params += self.__extra_arrays__
        params += self.__extra_params__
        params += self.__computed_arrays__
        params += self.__warn_arrays__

        for param in params:
            setattr(self, param, getattr(obj, param, None))

    def __getitem__(self, sl):
        """ When a coordinate is sliced, slice the extra arrays too,
        and carry over params, if that's the right thing to do

        Returns
        --------
        sliced : numpy.ndarray or `.Coordinate`
            Depending on how it was sliced...
        """

        # handling everything here is tricky
        # https://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html
        # is helpful
        sliced = super().__getitem__(sl)

        if (not isinstance(sliced, numpy.ndarray) or
                sliced.ndim != 2 or sliced.shape[1] != self.shape[1]):
            return sliced.view(numpy.ndarray)

        arrays2slice = []
        arrays2slice += self.__extra_arrays__
        arrays2slice += self.__computed_arrays__
        arrays2slice += self.__warn_arrays__

        for param in arrays2slice:
            if isinstance(sl, tuple):
                # index looks something like arr[2:4,:]
                setattr(sliced, param, getattr(sliced, param)[sl[0]])
            else:
                # index looks something like arr[arr<4]
                setattr(sliced, param, getattr(sliced, param)[sl])

        return sliced

    @property
    def coordSysName(self):
        return self.__class__.__name__


class Coordinate2D(Coordinate):
    """2 dimensional coordinate system
    """

    __coord_dim__ = 2


class Coordinate3D(Coordinate):
    """3 dimensional coordinate system
    """

    __coord_dim__ = 3
