#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from skimage.transform import SimilarityTransform

from coordio.zhaoburge import fitZhaoBurge, getZhaoBurgeXY


__all__ = ["RoughTransform", "ZhaoBurgeTransform"]


class RoughTransform(object):
    """Apply a rough transformation."""

    def __init__(self, xyCCD, xyWok):

        # scale pixels to mm roughly
        xCCD = xyCCD[:, 0]
        yCCD = xyCCD[:, 1]
        xWok = xyWok[:, 0]
        yWok = xyWok[:, 1]

        self.meanCCDX = numpy.mean(xCCD)
        self.meanCCDY = numpy.mean(yCCD)
        self.stdCCDX = numpy.std(xCCD)
        self.stdCCDY = numpy.std(yCCD)

        self.stdWokX = numpy.std(xWok)
        self.stdWokY = numpy.std(yWok)

    def apply(self, xyCCD):

        xCCD = xyCCD[:, 0]
        yCCD = xyCCD[:, 1]
        roughWokX = (xCCD - self.meanCCDX) / self.stdCCDX * self.stdWokX
        roughWokY = (yCCD - self.meanCCDY) / self.stdCCDY * self.stdWokY

        return numpy.array([roughWokX, roughWokY]).T


class ZhaoBurgeTransform(object):
    """Apply the full Zhao-Burge transformation."""

    def __init__(self, xyCCD, xyWok, polids=None):

        self.polids = numpy.array(polids or [0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29],
                                  dtype=numpy.int16)

        # First fit a transrotscale model
        self.simTrans = SimilarityTransform()
        self.simTrans.estimate(xyCCD, xyWok)

        # Apply the model to the data
        xySimTransFit = self.simTrans(xyCCD)

        # Use ZB polys to get the rest of the way use leave-one out xverification to
        # estimate "unbiased" errors in fit.
        self.unbiasedErrs = []
        for ii in range(len(xyCCD)):
            _xyWok = xyWok.copy()
            _xySimTransFit = xySimTransFit.copy()
            _xyWok = numpy.delete(_xyWok, ii, axis=0)
            _xySimTransFit = numpy.delete(_xySimTransFit, ii, axis=0)
            fitCheck = numpy.array(xySimTransFit[ii, :]).reshape((1, 2))
            destCheck = numpy.array(xyWok[ii, :]).reshape((1, 2))

            polids, coeffs = fitZhaoBurge(
                _xySimTransFit[:, 0],
                _xySimTransFit[:, 1],
                _xyWok[:, 0],
                _xyWok[:, 1],
                polids=self.polids,
            )

            dx, dy = getZhaoBurgeXY(polids, coeffs, fitCheck[:, 0], fitCheck[:, 1])
            zxfit = fitCheck[:, 0] + dx
            zyfit = fitCheck[:, 1] + dy
            zxyfit = numpy.array([zxfit, zyfit]).T
            self.unbiasedErrs.append(destCheck.squeeze() - zxyfit.squeeze())

        self.unbiasedErrs = numpy.array(self.unbiasedErrs)
        self.unbiasedRMS = numpy.sqrt(numpy.mean(self.unbiasedErrs ** 2))

        # Now do the "official fit", using all points

        polids, self.coeffs = fitZhaoBurge(
            xySimTransFit[:, 0],
            xySimTransFit[:, 1],
            xyWok[:, 0],
            xyWok[:, 1],
            polids=self.polids,
        )

        dx, dy = getZhaoBurgeXY(
            polids,
            self.coeffs,
            xySimTransFit[:, 0],
            xySimTransFit[:, 1],
        )

        xWokFit = xySimTransFit[:, 0] + dx
        yWokFit = xySimTransFit[:, 1] + dy
        xyWokFit = numpy.array([xWokFit, yWokFit]).T

        self.errs = xyWok - xyWokFit
        self.rms = numpy.sqrt(numpy.mean(self.errs ** 2))

    def apply(self, xyCCD, zb=True):
        """Apply the transformation to a set of (x, y) coordinates."""

        xySimTransFit = self.simTrans(xyCCD)

        if zb:
            dx, dy = getZhaoBurgeXY(
                self.polids,
                self.coeffs,
                xySimTransFit[:, 0],
                xySimTransFit[:, 1],
            )
            xWokFit = xySimTransFit[:, 0] + dx
            yWokFit = xySimTransFit[:, 1] + dy
            xyWokFit = numpy.array([xWokFit, yWokFit]).T
        else:
            xyWokFit = xySimTransFit

        return xyWokFit
