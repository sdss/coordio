#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy
import pandas
from skimage.transform import SimilarityTransform
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from coordio.zhaoburge import fitZhaoBurge, getZhaoBurgeXY
import coordio


__all__ = ["RoughTransform", "ZhaoBurgeTransform"]


def arg_nearest_neighbor(xyA, xyB):
    """loop over xy list A, find nearest neighbor in list B
    return the indices in list b that best match A
    """
    xyA = numpy.array(xyA)
    xyB = numpy.array(xyB)
    out = []
    distance = []
    for x, y in xyA:
        dist = numpy.sqrt((x - xyB[:, 0])**2 + (y - xyB[:, 1])**2)
        amin = numpy.argmin(dist)
        distance.append(dist[amin])
        out.append(amin)

    return numpy.array(out), numpy.array(distance)


def plot_fvc_assignments(
    filename: str,
    xyFitCentroids: numpy.ndarray,
    xyMetFiber: numpy.ndarray | None = None,
    xyBossFiber: numpy.ndarray | None = None,
    xyApogeeFiber: numpy.ndarray | None = None,
    xyFIF: numpy.ndarray | None = None,
    positionerIDs: list | None = None,
    assoc_used: numpy.ndarray | None = None,
    title: str | None = None,
):
    """
    Plot the results of the transformation.  All xy units are wok MM.

    filename : str
    xyFitCentroids : numpy.ndarray
        locations of centroids transformed to wok space
    xyMetFiber : numpy.ndarray
        expected location of Metrology fiber in wok space given robot arm
        angles
    xyBossFiber : numpy.ndarray
        expected location of BOSS fiber in wok space given robot arm
        angles
    xyApogeeFiber : numpy.ndarray
        expected location of APOGEE fiber in wok space given robot arm
        angles
    xyFIF : numpy.ndarray
        expected location of fiducial fibers in wok space

    """

    plt.figure(figsize=(8, 8))
    textProps = {"ha": "center", "va": "center", "fontsize": 0.15, "fontweight": "bold"}

    if title:
        plt.title(title)

    plt.plot(
        xyFitCentroids[:, 0],
        xyFitCentroids[:, 1],
        "o",
        ms=4,
        markerfacecolor="None",
        markeredgecolor="tab:red",
        markeredgewidth=1,
        label="Centroid",
    )

    if xyMetFiber is not None:
        plt.plot(
            xyMetFiber[:,0],
            xyMetFiber[:,1],
            "x",
            color="tab:blue",
            # markeredgecolor="white",
            # markeredgewidth=0.1,
            alpha=1,
            ms=3,
            label="Expected MET",
        )
        if positionerIDs is not None:
            for (x, y), pid in zip(xyMetFiber, positionerIDs):
                strID = "P" + ("%i"%pid).zfill(4)
                txt = plt.text(x, y, strID, textProps)


    if xyBossFiber is not None:
        plt.plot(
            xyBossFiber[:,0],
            xyBossFiber[:,1],
            "x",
            color="black",
            alpha=1,
            ms=3,
            label="Expected BOSS",
        )
        if positionerIDs is not None:
            for (x, y), pid in zip(xyBossFiber, positionerIDs):
                strID = "P" + ("%i"%pid).zfill(4)
                txt = plt.text(x, y, strID, textProps)
                txt.set_path_effects([PathEffects.withStroke(linewidth=0.08, foreground='w')])

    if xyApogeeFiber is not None:
        plt.plot(
            xyApogeeFiber[:,0],
            xyApogeeFiber[:,1],
            "x",
            color="tab:purple",
            alpha=1,
            ms=3,
            label="Expected APOGEE",
        )
        if positionerIDs is not None:
            for (x, y), pid in zip(xyApogeeFiber, positionerIDs):
                strID = "P" + ("%i"%pid).zfill(4)
                plt.text(x, y, strID, textProps)

    # Overplot fiducials
    if xyFIF is not None:
        plt.plot(
            xyFIF[:,0],
            xyFIF[:,1],
            "D",
            ms=6,
            markerfacecolor="None",
            markeredgecolor="cornflowerblue",
            markeredgewidth=1,
            label="Expected FIF",
        )

    if assoc_used is not None:
        for expected, measured in zip(assoc_used[0], assoc_used[1]):
            plt.plot([expected[0], measured[0]], [expected[1], measured[1]], "-k")

    plt.axis("equal")
    plt.legend()
    plt.xlim([-350, 350])
    plt.ylim([-350, 350])
    plt.xlabel("Wok x (mm)")
    plt.ylabel("Wok y (mm)")
    plt.savefig(filename, dpi=350)
    plt.close()


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

        if polids is None:
            self.polids = numpy.array(
                [0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29], dtype=numpy.int16
            )
        else:
            self.polids = numpy.array(polids, dtype=numpy.int16)

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

def positionerToWok(
        alphaDeg, betaDeg,
        xBeta, yBeta, la,
        alphaOffDeg, betaOffDeg,
        dx, dy, b, iHat, jHat, kHat
    ):

    xt, yt = coordio.conv.positionerToTangent(
        alphaDeg, betaDeg, xBeta, yBeta,
        la, alphaOffDeg, betaOffDeg
    )

    if hasattr(xt, "__len__"):
        zt = numpy.zeros(len(xt))
    else:
        zt = 0

    xw, yw, zw = coordio.conv.tangentToWok(
        xt, yt, zt, b, iHat, jHat, kHat,
        elementHeight=coordio.defaults.POSITIONER_HEIGHT, scaleFac=1,
        dx=dx, dy=dy, dz=0

    )

    return xw, yw, zw

def xyWokFromPosAngles(fullTable, fiberType):
    # pos angle is a pandas DataFrame with columns
    # "positionerID", "alphaReport", "betaReport"

    xWok = []
    yWok = []

    for ii, posRow in fullTable.iterrows():
        alpha = float(posRow.alphaReport)
        beta = float(posRow.betaReport)

        b = numpy.array([posRow.xWok, posRow.yWok, posRow.zWok])
        iHat = numpy.array([posRow.ix, posRow.iy, posRow.iz])
        jHat = numpy.array([posRow.jx, posRow.jy, posRow.jz])
        kHat = numpy.array([posRow.kx, posRow.ky, posRow.kz])
        la = float(posRow.alphaArmLen)
        alphaOffDeg = float(posRow.alphaOffset)
        betaOffDeg = float(posRow.betaOffset)
        dx = float(posRow.dx)
        dy = float(posRow.dy)

        # assert len(posRow) == 1

        if fiberType == "Metrology":
            xBeta = float(posRow.metX)
            yBeta = float(posRow.metY)
        elif fiberType == "Boss":
            xBeta = float(posRow.bossX)
            yBeta = float(posRow.bossY)
        else:
            xBeta = float(posRow.apX)
            yBeta = float(posRow.apY)

        xw, yw, zw = positionerToWok(
            alpha, beta, xBeta, yBeta, la,
            alphaOffDeg, betaOffDeg,
            dx, dy, b, iHat, jHat, kHat
        )

        xWok.append(xw)
        yWok.append(yw)

    fullTable["xWokExpect"] = xWok
    fullTable["yWokExpect"] = yWok

    return fullTable

def transformFromMetData(centroids, fullTable, fiducialCoords, figPrefix="", polids=None):
    """

    Parameters
    ------------
    centroids : pandas.DataFrame
        output of source extractor, xys in CCD coords
    fullTable : pandas.DataFrame
        contains xy wok expected locations of metrology fibers for robots

    """
    xyMetFiber = fullTable[["xWokExpect", "yWokExpect"]].to_numpy()

    xyWokFIF = fiducialCoords[["xWok", "yWok"]].to_numpy()

    # centroids = fitsTableToPandas(ff[7].data)
    # centroids = centroids[centroids.npix > 400]

    # print(len(centroids))

    xyCCD = centroids[["x", "y"]].to_numpy()

    # first do a rough transform
    rt = RoughTransform(xyCCD, xyMetFiber)
    xyWokRough = rt.apply(xyCCD)

    # just grab outer fiducials for first pass
    rWokFIF = numpy.linalg.norm(xyWokFIF, axis=1)
    xyWokOuterFIF = xyWokFIF[rWokFIF > 310]

    # associate the centroids to the outer wok FIDs
    argFound, roughDist = arg_nearest_neighbor(xyWokOuterFIF, xyWokRough)
    assoc_found = [xyWokOuterFIF, xyWokRough[argFound]]

    # plot the rough transform
    plot_fvc_assignments(
        figPrefix + "roughTransform.pdf",
        xyFitCentroids=xyWokRough,
        xyMetFiber=xyMetFiber,
        xyFIF=xyWokFIF,
        assoc_used=assoc_found,
        title="rough scale assoc"
        )

    # use associations from rough transform to fit a full
    # transform using only outer fiducial ring
    # pass zb = false to just fit translation rotation and scale

    xyCCDOuterFIF = xyCCD[argFound]

    ft = ZhaoBurgeTransform(
        xyCCDOuterFIF,
        xyWokOuterFIF,
        polids=polids,
    )

    xyWokMeas = ft.apply(xyCCD, zb=False)

    # associate centroids with all fiducials in grid, now that we're closer
    argFound, roughDist = arg_nearest_neighbor(xyWokFIF, xyWokMeas)
    assoc_found = [xyWokFIF, xyWokMeas[argFound]]


    plot_fvc_assignments(
        figPrefix + "similarityTransform.pdf",
        xyFitCentroids=xyWokMeas,
        xyMetFiber=xyMetFiber,
        xyFIF=xyWokFIF,
        assoc_used=assoc_found,
        title="similarity transform assoc"
    )

    # finally, do the full ZB transform based on all found FIF locations
    xyCCDFIF = xyCCD[argFound]
    ft = ZhaoBurgeTransform(
        xyCCDFIF,
        xyWokFIF,
        polids=polids
    )

    xyWokMeas = ft.apply(xyCCD, zb=True)
    positionerIDs = list(fullTable.positionerID)

    plot_fvc_assignments(
        figPrefix + "zhaoBurgeTransform.pdf",
        xyFitCentroids=xyWokMeas,
        xyMetFiber=xyMetFiber,
        xyFIF=xyWokFIF,
        positionerIDs=positionerIDs,
        title="Zhao-Burge Transform"
    )


    # now associate measured xy locations of fiber
    # for each robot, and measured angles for each robot
    argFound, roughDist = arg_nearest_neighbor(xyMetFiber, xyWokMeas)
    xyWokFiberMeas = xyWokMeas[argFound, :]
    fullTable["xWokMetMeas"] = xyWokFiberMeas[:, 0]
    fullTable["yWokMetMeas"] = xyWokFiberMeas[:, 1]

    # xerr = fullTable.xWokMetMeas - fullTable.xWokExpect
    # yerr = fullTable.yWokMetMeas - fullTable.yWokExpect
    # print("Fiber Position RMS", numpy.sqrt(numpy.mean(xerr**2 + yerr**2))* 1000)

    return ft, fullTable
