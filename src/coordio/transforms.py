#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy
import pandas
import scipy
import sep
from skimage.transform import SimilarityTransform

from .conv import positionerToTangent, tangentToWok, wokToTangent
from .defaults import POSITIONER_HEIGHT, calibration, PLATE_SCALE
from .exceptions import CoordIOError
from .libcoordio import tangentToPositioner, tangentToPositioner2
from .utils import refinexy
from .zhaoburge import fitZhaoBurge, getZhaoBurgeXY


# __all__ = ["RoughTransform", "ZhaoBurgeTransform", "FVCTransformAPO"]


def arg_nearest_neighbor(
    xyA: numpy.ndarray,
    xyB: numpy.ndarray,
    atol: float | None = None,
):
    """Finds the nearest neighbour in list B for each target in list A.

    If the distance between the item in A and the closest element in B is greater
    than ``atol``, a match is not returned.

    Parameters
    ----------
    xyA
        The list we want to match.
    xyB
        The reference table.
    atol
        The maximum allowed distance. `None` to not do any distance checking.

    Returns
    -------
    result
        A tuple with the indices in ``A`` that have been matched, the matching index
        in ``B`` for each matched element in ``A``, and the distance from each
        element in ``A`` to the nearest neighbour in ``B`` (regardless of whether
        that distance is greater than ``atol``).

    """

    xyA = numpy.array(xyA)
    xyB = numpy.array(xyB)

    distances = scipy.spatial.distance.cdist(xyA, xyB)

    min_distances = numpy.array([numpy.min(d) for d in distances])
    indexB = numpy.array([numpy.argmin(d) for d in distances])

    if atol is not None:
        good_matches = numpy.where(min_distances < atol)[0]
    else:
        good_matches = numpy.arange(len(indexB))

    return good_matches, indexB[good_matches], min_distances


def wokToPositioner(
    xWok, yWok,
    xBeta, yBeta, la,
    alphaOffDeg, betaOffDeg,
    dx, dy, b, iHat, jHat, kHat,
    newInvKin=True
):
    """
    Find alpha beta coords of a robot given xyWok coords of a robot's fiber

    note: vectorized option not here yet.  Only right hand solutions
    for now.

    Parameters
    ------------
    xWok : float
        x location of fiber in wok coordinates (mm)
    yWok : float
        y location of fiber in wok coordinates (mm)
    xBeta : float
        x location of fiber in beta coords (mm)
    yBeta : numpy.array or float
        y location of fiber in beta coords (mm)
    la : numpy.array or float
        alpha arm length in mm
    alphaOffDeg : numpy.narray or float
        calibrated alpha arm offset in deg
    betaOffDeg : float
        calibrated beta arm offset in deg
    dx : float
        calibrated robot body x offset in wok coords (mm)
    dy : float
        calibrated robot body y offset in wok coords (mm)
    b: numpy.ndarray
        3-element 1D vector
        x,y,z position (mm) of each hole element on wok
        surface measured in wok coords
    iHat: numpy.ndarray
        3-element 1D vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate x axis for each hole.
    jHat: numpy.ndarray
        3-element 1D vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate y axis for each hole.
    kHat: numpy.ndarray
        3-element 1D vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate z axis for each hole.
    newInvKin : bool
        Use new style inverse kinematics solver


    Returns
    ---------
    alpha
        alpha angle of robot in degrees
    beta
        beta angle of robot in degrees

    """
    xt,yt,zt = wokToTangent(
        xWok, yWok, POSITIONER_HEIGHT, b, iHat, jHat, kHat,
        elementHeight=POSITIONER_HEIGHT, scaleFac=1,
        dx=dx, dy=dy, dz=0
    )

    if newInvKin:
        alpha, beta, alphaLH, betaLH, dist = tangentToPositioner2(
            [xt, yt], [xBeta, yBeta], la, alphaOffDeg, betaOffDeg
        )
    else:
        lefthand = False
        alpha, beta, isOK = tangentToPositioner(
            [xt, yt], [xBeta, yBeta], la, alphaOffDeg, betaOffDeg, lefthand
        )
    return alpha, beta


def positionerToWok(
        alphaDeg, betaDeg,
        xBeta, yBeta, la,
        alphaOffDeg, betaOffDeg,
        dx, dy, b, iHat, jHat, kHat
):
    """
    Find xyWok coords of a fiber given alpha beta coords of a robot

    Parameters
    ------------
    alphaDeg : numpy.ndarray or float
        robot alpha angle coordinate in degrees
    betaDeg : numpy.ndarray or float
        robot beta angle coordinate in degrees
    xBeta : numpy.ndarray or float
        x location of fiber in beta coords (mm)
    yBeta : numpy.array or float
        y location of fiber in beta coords (mm)
    la : numpy.array or float
        alpha arm length in mm
    alphaOffDeg : numpy.narray or float
        calibrated alpha arm offset in deg
    betaOffDeg : numpy.ndarray or float
        calibrated beta arm offset in deg
    dx : numpy.ndarray or float
        calibrated robot body x offset in wok coords (mm)
    dy : numpy.ndarray or float
        calibrated robot body y offset in wok coords (mm)
    b: numpy.ndarray
        3-element 1D vector
        x,y,z position (mm) of each hole element on wok
        surface measured in wok coords
    iHat: numpy.ndarray
        3-element 1D vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate x axis for each hole.
    jHat: numpy.ndarray
        3-element 1D vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate y axis for each hole.
    kHat: numpy.ndarray
        3-element 1D vector
        x,y,z unit vector in wok coords that indicate the direction
        of the tangent coordinate z axis for each hole.


    Returns
    ---------
    xWok
        x loaction of fiber in wok coords (mm)
    yWok
        y location of fiber in wok coords (mm)
    zWok
        z location of fiber in wok coords (mm)


    """

    xt, yt = positionerToTangent(
        alphaDeg, betaDeg, xBeta, yBeta,
        la, alphaOffDeg, betaOffDeg
    )

    if hasattr(xt, "__len__"):
        zt = numpy.zeros(len(xt))
    else:
        zt = 0

    xw, yw, zw = tangentToWok(
        xt, yt, zt, b, iHat, jHat, kHat,
        elementHeight=POSITIONER_HEIGHT, scaleFac=1,
        dx=dx, dy=dy, dz=0

    )

    return xw, yw, zw


def plotFVCResults(
    filename,
    xyFitCentroids,
    xyMetFiber=None,
    xyBossFiber=None,
    xyApogeeFiber=None,
    xyFIF=None,
    positionerIDs=None,
    assoc_used=None,
    xyFitCentroidsUnmatched=None,
    xyFiberWarn=None,
    title=None,
    xyFIFLost=None,
):
    """
    Visualize the results of transformation/association.
    All xy units should be MM.

    filename : str
    xyFitCentroids : numpy.ndarray
        nx2 array, locations of centroids transformed to wok space
    xyMetFiber : numpy.ndarray or None
        nx2 array, expected location of Metrology fiber in wok space
    xyBossFiber : numpy.ndarray or None
         nx2 array, expected location of BOSS fiber in wok space
    xyApogeeFiber : numpy.ndarray or None
        nx2 array, expected location of APOGEE fiber in wok space given robot
        arm angles
    xyFIF : numpy.ndarray or None
        mx2 array, expected location of fiducial fibers in wok space
    positionerIDs : list or None
        lenth n list of integers, if supplied, positioner ids are ploted
        near the points for met, ap, and/or boss fibers
    assoc_used : list or None
        [mx2 array, mx2 array] two sets of xy coords to plot lines between
        for seeing who matched to who (usually fiducials)
    xyFitCentroidsUnmatched : numpy.ndarray or None
        centroids detected but not associated with any robot or fiducial
    xyFiberWarn : numpy.ndarray or None
        fibers with abnormally large errors
    title : str or None
        title for the plot
    xyFIFLost : numpy.ndarray or None
        nx2 array, expected locations of fiducial fibers in wok space that weren't matched

    """

    fig = plt.figure(figsize=(8, 8))
    textProps = {
        "ha": "center", "va": "center",
        "fontsize": 0.15, "fontweight": "bold"
    }

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
            xyMetFiber[:, 0],
            xyMetFiber[:, 1],
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
            xyBossFiber[:, 0],
            xyBossFiber[:, 1],
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
                txt.set_path_effects(
                    [PathEffects.withStroke(linewidth=0.08, foreground='w')]
                )

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

    if xyFitCentroidsUnmatched is not None:
        plt.plot(
            xyFitCentroidsUnmatched[:, 0],
            xyFitCentroidsUnmatched[:, 1],
            "^",
            color="tab:orange",
            alpha=1,
            markerfacecolor="None",
            markeredgewidth=1,
            ms=7,
            label="Unmatched Centroid",
        )

    if xyFiberWarn is not None:
        plt.plot(
            xyFiberWarn[:, 0],
            xyFiberWarn[:, 1],
            "v",
            color="tab:cyan",
            alpha=1,
            markerfacecolor="None",
            markeredgewidth=1,
            ms=7,
            label="Fiber Warning",
        )

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
            label="Matched FIF",
        )

    if xyFIFLost is not None:
        plt.plot(
            xyFIFLost[:,0],
            xyFIFLost[:,1],
            "D",
            ms=6,
            markerfacecolor="None",
            markeredgecolor="orange",
            markeredgewidth=1,
            label="Unmatched FIF",
        )

    if assoc_used is not None:
        for xy1, xy2 in zip(assoc_used[0], assoc_used[1]):
            plt.plot([xy1[0], xy2[0]], [xy1[1], xy2[1]], "-k")

    plt.axis("equal")
    # plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.legend(bbox_to_anchor=(1,1), bbox_transform=fig.transFigure)
    # plt.legend()
    plt.xlim([-350, 350])
    plt.ylim([-350, 350])
    plt.xlabel("Wok x (mm)")
    plt.ylabel("Wok y (mm)")
    plt.savefig(filename, dpi=350)
    plt.close()


class RoughTransform(object):
    """Apply a rough transformation between CCD and Wok Coords.

    Simply subtract mean and scale by variance to put coordinate
    sets into roughly the same space.
    """

    def __init__(self, xyCCD, xyWok):
        """
        xyCCD and xyWok do not need to be a mapping, nor even the same length

        Parameters
        -----------
        xyCCD : numpy.ndarray
            Nx2 array of xy ccd coordinates (pixels)
        xyWok :numpy.ndarray
            Mx2 array of xy wok coordinates (mm)
        """

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
        """
        Convert xyCCD to xyWok (very roughly)

        Parameters
        ------------
        xyCCD : numpy.ndarray
            Nx2 array of xy ccd coordinates (pixels)

        Result
        -------
        xyWok
            estimated location of input coords in wok space

        """

        xCCD = xyCCD[:, 0]
        yCCD = xyCCD[:, 1]
        roughWokX = (xCCD - self.meanCCDX) / self.stdCCDX * self.stdWokX
        roughWokY = (yCCD - self.meanCCDY) / self.stdCCDY * self.stdWokY

        return numpy.array([roughWokX, roughWokY]).T


class ZhaoBurgeTransform(object):
    """Apply a full Zhao-Burge transformation.


    Residuals from a similarity tranform are fit with Zhao-Burge
    2D vector polynomials.
    """

    def __init__(self, xyCCD, xyWok, polids=None, normFactor=1):
        """ Compute a transformation between xyCCD and xyWok using
        Zhao-Burge polynomials

        xyCCD and xyWok need to be a mapping by index

        Parameters
        ------------
        xyCCD : numpy.ndarray
            Nx2 array of xy ccd coordinates (pixels)
        xyWok :numpy.ndarray
            Nx2 array of xy wok coordinates (mm)
        polids : numpy.ndarray or None
            Mx1 array of integers specifying the ZB basis vectors to use.
            If not passed a default basis vector set is used.
        normFactor : float
            Scale xyWok coordinates by this factor.  Intended
            use is for moving coordinates to unit circle. 330 is a good
            choice, but default is 1 to remain backwards compatible
            (no normalization)

        Attributes
        -------------
        polids : numpy.ndarray
            1D integer array containing basis IDs used for ZB transform
        coeffs : numpy.ndarray
            1D float array containing the fit coefficient values for each
            polid in the ZB transform
        normFactor : float
            normalization factor used
        simTrans : skimage.transform.SimilarityTransform
            similarity transform fit between xyCCD and xyWok
        unbiasedErrs : numpy.ndarray
            Nx1 array containing zb fit residuals (mm) at each
            jacknife iteration.  At each iter a single point is left out of the
            fit, and that point is used to measure predictive error of the
            model.
        unbiasedRMS : float
            RMS value of unbiasedErrors (mm)
        errs : numpy.ndarray
            Nx1 array containing zb fit resuduals (mm) without
            jacknife (fit using all available data)
        rms : float
            RMS value of errs (mm)
        """

        if polids is None:
            self.polids = numpy.array(
                [0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29], dtype=numpy.int16
            )
        else:
            self.polids = numpy.array(polids, dtype=numpy.int16)

        if normFactor < 1:
            raise CoordIOError("normFactor must >= 1")

        self.normFactor = normFactor

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
                normFactor=self.normFactor
            )

            dx, dy = getZhaoBurgeXY(
                polids,
                coeffs,
                fitCheck[:, 0],
                fitCheck[:, 1],
                normFactor=self.normFactor
            )
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
            normFactor=self.normFactor
        )

        dx, dy = getZhaoBurgeXY(
            polids,
            self.coeffs,
            xySimTransFit[:, 0],
            xySimTransFit[:, 1],
            normFactor=self.normFactor
        )

        xWokFit = xySimTransFit[:, 0] + dx
        yWokFit = xySimTransFit[:, 1] + dy
        xyWokFit = numpy.array([xWokFit, yWokFit]).T

        self.xyWokIn = xyWok
        self.errs = xyWok - xyWokFit
        self.rms = numpy.sqrt(numpy.mean(self.errs ** 2))

    def apply(self, xyCCD, zb=True):
        """Apply the transformation to a set of (x, y) coordinates.


        Parameters
        -----------
        xyCCD : numpy.ndarray
            Nx2 array of xy ccd coordinates (pixels)
        zb : bool
            If False, only apply the similarity transform
            (no Zhao-Burge higher order components)


        Returns
        ---------
        xyWok
            Nx2 array of wok coordinates
        """

        xySimTransFit = self.simTrans(xyCCD)

        if zb:
            dx, dy = getZhaoBurgeXY(
                self.polids,
                self.coeffs,
                xySimTransFit[:, 0],
                xySimTransFit[:, 1],
                normFactor=self.normFactor
            )
            xWokFit = xySimTransFit[:, 0] + dx
            yWokFit = xySimTransFit[:, 1] + dy
            xyWokFit = numpy.array([xWokFit, yWokFit]).T
        else:
            xyWokFit = xySimTransFit

        return xyWokFit


def xyWokFiberFromPositioner(
        fullTable,
        angleType="Report",
        doMetrology=True,
        doApogee=True,
        doBoss=True
    ):
    """
    Determine xy wok position for a each fiber for each robot.

    Parameters
    ------------
    fullTable : pandas.DataFrame
        A merge on holeID of positionerTable and wokCoords DataFrames,
        with additional columns names specified by alphaColumn and betaColumn
        indicating the robot's arm coords.
    angleType : str
        look for columns alphaXXX betaXXX where XXX=column type
    doMetrology : bool
        if True, compute metrology fiber location
    doApogee : bool
        if True, compute apogee fiber location
    doBoss : bool
        if True, compute boss fiber location

    Returns
    --------
    fullTable : pandas.DataFrame
        the input fullTable with new columns appended
        xWokxxxXXX, yWokxxxXXX where xxx=angleType and
        XXX is one of ["Metrology", "APOGEE","BOSS"]
    """

    # probably can vectorize this whole thing, or at least
    # partially

    fiberNameMap = zip(
        [doMetrology, doApogee, doBoss],
        ["Metrology", "APOGEE", "BOSS"],
        ["met", "ap", "boss"]
    )
    for doFiber, fiberType, colName in fiberNameMap:
        if not doFiber:
            continue
        fiberX = "%sX"%colName
        fiberY = "%sY"%colName
        xWok = []
        yWok = []
        for ii, posRow in fullTable.iterrows():
            alpha = float(posRow["alpha" + angleType])
            beta = float(posRow["beta" + angleType])

            b = numpy.array([posRow.xWok, posRow.yWok, posRow.zWok])
            iHat = numpy.array([posRow.ix, posRow.iy, posRow.iz])
            jHat = numpy.array([posRow.jx, posRow.jy, posRow.jz])
            kHat = numpy.array([posRow.kx, posRow.ky, posRow.kz])
            la = float(posRow.alphaArmLen)
            alphaOffDeg = float(posRow.alphaOffset)
            betaOffDeg = float(posRow.betaOffset)
            dx = float(posRow.dx)
            dy = float(posRow.dy)

            xBeta = float(posRow[fiberX])
            yBeta = float(posRow[fiberY])

            xw, yw, zw = positionerToWok(
                alpha, beta, xBeta, yBeta, la,
                alphaOffDeg, betaOffDeg,
                dx, dy, b, iHat, jHat, kHat
            )

            xWok.append(xw)
            yWok.append(yw)

        fullTable["xWok%s%s" % (angleType, fiberType)] = xWok
        fullTable["yWok%s%s" % (angleType, fiberType)] = yWok

    return fullTable


def alphaBetaFromMetMeas(fullTable, newInvKin=True):
    """
    Determine alpha beta angles from metrology fiber measurement.

    Parameters
    ------------
    fullTable : pandas.DataFrame
        A merge on holeID of positionerTable and wokCoords DataFrames,
        with additional columns names xyWokMeasMetrology.
    newInvKin : bool
        If True, use new inverse kinematic calculation
        which will never return NaN's

    Returns
    --------
    fullTable : pandas.DataFrame
        the input fullTable with new columns appended
        alphaMeas and betaMeas.
    """

    alpha = []
    beta = []
    for ii, posRow in fullTable.iterrows():

        xWok = float(posRow.xWokMeasMetrology)
        yWok = float(posRow.yWokMeasMetrology)

        xBeta = float(posRow.metX)
        yBeta = float(posRow.metY)

        b = numpy.array([posRow.xWok, posRow.yWok, posRow.zWok])
        iHat = numpy.array([posRow.ix, posRow.iy, posRow.iz])
        jHat = numpy.array([posRow.jx, posRow.jy, posRow.jz])
        kHat = numpy.array([posRow.kx, posRow.ky, posRow.kz])
        la = float(posRow.alphaArmLen)
        alphaOffDeg = float(posRow.alphaOffset)
        betaOffDeg = float(posRow.betaOffset)
        dx = float(posRow.dx)
        dy = float(posRow.dy)

        xBeta = float(posRow.metX)
        yBeta = float(posRow.metY)

        _alpha, _beta = wokToPositioner(
            xWok, yWok, xBeta, yBeta, la,
            alphaOffDeg, betaOffDeg,
            dx, dy, b, iHat, jHat, kHat,
            newInvKin=newInvKin
        )

        alpha.append(_alpha)
        beta.append(_beta)

    fullTable["alphaMeas"] = alpha
    fullTable["betaMeas"] = beta

    return fullTable


# hogg fourier basis fit stuff

IMAX = 32 # maximum integer wave number
DELTAK = 2. * numpy.pi / 10000.0 # wave number spacing in inverse pixels
_wcd = os.environ.get('WOKCALIB_DIR', None)
### BIG WARNING ### if two wok calib dirs are present just take the first one
### should eventually move this to calibrations for better handling
beta_x_apo = None
beta_y_apo = None
beta_x_lco = None
beta_y_lco = None
if _wcd is not None:
    for _path in _wcd.split(":"):
        with open(os.path.join(_path, "beta_x.npy"), "rb") as f:
            beta_x = numpy.load(f)
        with open(os.path.join(_path, "beta_y.npy"), "rb") as f:
            beta_y = numpy.load(f)
        if "/lco/" in _path:
            beta_x_lco = beta_x.copy()
            beta_y_lco = beta_y.copy()
        else:
            beta_x_apo = beta_x.copy()
            beta_y_apo = beta_y.copy()
beta_x = None
beta_y = None


# functions to set up design matrices


def fourier_functions(xs, ys):
    n = len(xs)
    assert len(ys) == n
    fxs = numpy.zeros((n, IMAX * 2 + 2))
    fys = numpy.zeros((n, IMAX * 2 + 2))
    iis = numpy.zeros(IMAX * 2 + 2).astype(int)
    for i in range(IMAX+1):
        fxs[:, i * 2] = numpy.cos(i * DELTAK * xs)
        fys[:, i * 2] = numpy.cos(i * DELTAK * ys)
        iis[i * 2] = i
        fxs[:, i * 2 + 1] = numpy.sin((i + 1) * DELTAK * xs)
        fys[:, i * 2 + 1] = numpy.sin((i + 1) * DELTAK * ys)
        iis[i * 2 + 1] = i + 1
    return fxs, fys, iis


def design_matrix(xs, ys):
    fxs, fys, iis = fourier_functions(xs, ys)
    n, p = fxs.shape
    Xbig = (fxs[:, :, None] * fys[:, None, :]).reshape((n, p * p))
    i2plusj2 = (iis[:, None] ** 2 + iis[None, :] ** 2).reshape(p * p)
    return Xbig[:, i2plusj2 <= IMAX ** 2]


def applyNudgeModel(
        x, y, dxythresh=0.75,
        site="APO", beta_x=None, beta_y=None
    ):

    if beta_x is None and beta_y is None:
        if site == "APO":
            beta_x = beta_x_apo
            beta_y = beta_y_apo
        else:
            beta_x = beta_x_lco
            beta_y = beta_y_lco

    X = design_matrix(x,y)
    dx = X @ beta_x
    dy = X @ beta_y

    # clip dxs and dys at thresh value
    dx[dx>dxythresh] = dxythresh
    dy[dy>dxythresh] = dxythresh

    dx[dx<-dxythresh] = -dxythresh
    dy[dy<-dxythresh] = -dxythresh

    # rejectInds = (numpy.abs(dx) > dxythresh) | (numpy.abs(dy) > dxythresh)

    newX = x - dx
    newY = y - dy

    # newX[rejectInds] = x[rejectInds]
    # newY[rejectInds] = y[rejectInds]

    return newX, newY


class FVCTransformAPO(object):
    # polids = [0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29]  # Zhao-Burge basis defaults
    polids = numpy.arange(33)
    zbCoeffs = numpy.array([
        3.30790633e-01, 4.13533378e-01, 1.71951213e-03, -9.90141758e-04,
        -4.48460907e-04, -3.68228879e-06, -2.45423918e-06, 3.06249618e-07,
        1.52595326e-06, -1.73037375e-08, 9.09750974e-09, 1.40858150e-08,
        -8.69986544e-11, 2.25009469e-09, 5.94658320e-12, 5.67065198e-12,
        -1.99111356e-12, 6.33175666e-13, -2.82452951e-12, 1.94818210e-12,
        4.03105186e-14, -2.79909483e-14, -1.46249329e-14, -5.04644513e-15,
        1.41549916e-15, 1.27613775e-15, -8.70792948e-15, 2.00843998e-04,
        -1.05007864e-06, 2.48271534e-07, -9.28099433e-10, 6.13172886e-10,
        -1.31799608e-09
    ])
    zbCoeffsFieldCenter = numpy.array([
        -7.25143224e-02,  -3.89886320e-02,   1.21470501e-02,  -4.36545180e-04,
       5.77221153e-05,  -6.80419618e-06,   3.47687303e-06,  -7.22841296e-06,
       3.36246434e-07,  -7.10874816e-07,   9.00745018e-09,   2.40442049e-08,
       1.67481779e-08,   4.90264973e-09,   1.75289911e-11,   2.59249932e-10,
      -8.03748666e-11,   2.04240068e-10,  -9.05440334e-11,   2.60323630e-10,
       1.12381521e-11,  -8.56685856e-14,  -1.37273897e-13,   1.03610076e-14,
      -6.97673027e-13,   1.77283795e-12,   6.96794275e-13,  -5.97070297e-04,
       1.01954593e-06,  -2.25406706e-06,   2.50511002e-08,  -9.82154214e-09,
      -4.41479884e-09
    ])
    telRotAngRef = 135.4
    rotAngDir = 1
    centType = "zbplus2"
    telescopePlateScale = PLATE_SCALE["APO"]/3600. # mm/arcsec
    nudgeOffX = 0 #1000 # fix nudge model after FVC resize
    site = "APO"
    centroidMinNpix = 100

    def __init__(
        self,
        fvcImgData,
        positionerCoords,
        telRotAng,
        plotPathPrefix=None,
        positionerTable=calibration.positionerTable,
        wokCoords=calibration.wokCoords,
        fiducialCoords=calibration.fiducialCoords,
        telRotAngRef=None,
        polids=None,
        nudgeAdjust=True,
        zbNormFactor=330
    ):
        """
        Parameters
        -------------
        fvcImgData : numpy.ndarray
            raw image data from the fvc
        positionerCoords : pandas.DataFrame
            DataFrame containing alphaReported and betaReported
            coordinate columns and a positionerID column for each robot
            in the FVC image
        telRotAng : float
            telescope rotator angle in mount coordinate degrees (IPA in sdss
            headers)
        plotPathPrefix : str
            base path for plot output.  Plot routines will append a
            ".<plotname>.pdf" suffix to the prefix supplied.  If None,
            plots will not be generated.
        positionerTable : pandas.DataFrame
            positioner calibration table for robots in the FVC image.
            Default is coordio.defaults.calibrations.positionerTable
        wokCoords : pandas.DataFrame
            wok coordinates for the robots in the FVC image
            Default is coordio.defaults.calibrations.wokCoords
        fiducialCoords : pandas.DataFrame
            coordiates of fiducials in the FVC image
            Default is coordio.defaults.calibrations.fiducialCoords
        telRotAngRef : float
            telescope rotator angle in mount coordinate degrees at
            which xyCCD ~= xyWok directions.
            Default is 135.4 (for APO)
        polids : 1D array or None
            list of integers for selecting zhaoburge basis polynomials.
            Default is supplied by class attribute polids
        nudgeAdjust : boolean
            if True, apply nudge correction to CCD detections
        zbNormFactor : float
            scale wok coordinates by this factor for unit circle
            normalization when fitting ZhaoBurge transform.
        """
        self.nudgeAdjust = nudgeAdjust
        self.zbNormFactor = zbNormFactor

        self.fvcImgData = numpy.array(fvcImgData, dtype=numpy.float32)

        # self.fvcImgData = self.fvcImgData[100:-100,:].copy()

        # apply rough bias/background subtraction
        # rough bias/bg subtract
        imbias = numpy.median(self.fvcImgData, axis=0)
        imbias = numpy.outer(
            numpy.ones(self.fvcImgData.shape[0], dtype=numpy.float32),
            imbias
        )
        im = self.fvcImgData - imbias

        self.bkg = sep.Background(im) #self.fvcImgData)
        bkg_image = self.bkg.back()

        self.data_sub = im - bkg_image

        self.positionerTable = positionerTable.reset_index()
        self.wokCoords = wokCoords.reset_index()
        self.fiducialCoords = fiducialCoords.reset_index()
        self.positionerCoords = positionerCoords.reset_index()
        self.telRotAng = telRotAng
        if telRotAngRef is not None:
            self.telRotAngRef = telRotAngRef
        self.plotPathPrefix = plotPathPrefix
        if polids is not None:
            self.polids = polids

        ft = positionerTable.merge(wokCoords, on="holeID").reset_index()
        ft = ft.merge(positionerCoords, on="positionerID")
        self._fullTable = xyWokFiberFromPositioner(
            ft,
            angleType="Report"
        )

        # construct a matrix for rotating centroids based on telescope
        # rotator angle
        self.ccd2WokRot = (telRotAng - self.telRotAngRef)*self.rotAngDir
        sinRot = numpy.sin(numpy.radians(self.ccd2WokRot))
        cosRot = numpy.cos(numpy.radians(self.ccd2WokRot))
        self.rotMat = numpy.array([
            [cosRot, -sinRot],
            [sinRot, cosRot]
        ])

        self.nCentroid_expect = len(positionerTable) + len(fiducialCoords)
        self.nPositioner_expect = len(positionerTable)
        self.maxCounts = numpy.max(self.fvcImgData)

        ########### populated by self.extractCentroids() #######
        self.centroids = None  # pandas.DataFrame

        self.backgroundSigma = None
        self.winposSigma = None
        self.winposBoxSize = None
        self.ccdRotCenXY = None
        self.nCentroidFound = None

        self.simpleSigma = None
        self.simpleBoxSize = None

        ############## populated by self.fit() ####################
        # pandas.DataFrame join w/ centroids every positioner
        # gets closest match (no clipping), but warnings present
        self.positionerTableMeas = None
        # pandas.DataFrame join w/ centroids
        # gets closest match (no clipping), but warnings present
        self.fiducialCoordsMeas = None
        # panas.DataFrame subset of self.centroids without
        # matches to a robot or fiducial
        self.unmatchedCentroids = None

        self.roughTransform = None
        self.similarityTransform = None
        self.fullTransform = None

        self.maxRoughDist = None
        self.maxMidDist = None
        self.maxFinalDist = None
        self.nOuterFIF_expect = None
        self.nOuterFIF_found = None
        self.nFIF_expect = None
        self.nFIF_found = None
        self.fiducialRMS = None
        self.nFiducialWarn = None
        self.positionerRMS = None
        self.positionerRMS_clipped = None
        self.nPositionerWarn = None
        self.positionerWarnList = None

    def getMetadata(self):
        """Get a list of data that can be easily stuffed in a fits
        header.
        """

        metaDataList = [
            ("FVC_NWRN", self.nPositionerWarn, "number of robots out of measurement spec"),
            ("FVC_MAXD", self.maxFinalDist, "distance beyond to consider robot out of spec (mm)"),
            ("FVC_CNTT", self.centType, "centroid type used for fitting"),
            ("FVC_BSIG", self.backgroundSigma, "above background sigma for centroid detection"),
            ("FVC_MNPX", self.centroidMinNpix, "minimum number of pixels for a valid centroid"),
            ("FVC_WSIG", self.winposSigma, "sigma for winpos centroid algorithm"),
            ("FVC_WBSZ", self.winposBoxSize, "box size for winpos centroid algorithm (pix)"),
            ("FVC_SSIG", self.simpleSigma, "sigma for simple centroid algorithm (pix)"),
            ("FVC_SPLM", self.simpleBoxSize, "box size for simple centroid algorithm (pix)"),
            ("FVC_RMS", self.positionerRMS, "robot rms (mm)"),
            ("FVC_FRMS", self.fiducialRMS, "fiducial rms (mm)"),
            ("FVC_CRMS", self.positionerRMS_clipped, "in-spec (outlier-clipped) robot rms (mm)"),
            ("FVC_SCL", self.fullTransform.simTrans.scale, "FVC model fit scale"),
            ("FVC_TRAX", self.fullTransform.simTrans.translation[0], "FVC model fit X translation"),
            ("FVC_TRAY", self.fullTransform.simTrans.translation[1], "FVC model fit Y translation"),
            ("FVC_ROT", numpy.degrees(self.fullTransform.simTrans.rotation), "FVC model fit rotation (deg)"),
            ("FVC_ZNRM", self.zbNormFactor, "ZB normilization factor")
        ]

        # add in ZB coeffs
        for polid, coeff in zip(self.polids, self.fullTransform.coeffs):
            metaDataList.append(
                ("FVC_ZB%i"%polid, coeff, "zhao-burge transform coeff for polid %i"%polid)
            )

        return metaDataList

    def extractCentroids(
        self,
        centroidMinNpix=None,
        backgroundSigma=3.5,
        winposSigma=0.7,
        winposBoxSize=3,
        simpleSigma=5,
        simpleBoxSize=19,
        beta_x=None,
        beta_y=None
    ):
        """
        Find centroids in the fvc image, stores result in
        self.centroids attribute (a pandas.DataFrame)

        Parameters
        -----------
        centroidMinNpix : int
            minimum number of pixels that belong to a bona fide detection
        backgoundSigma : float
            sigma above background that belong to a bona fide detection
        winposSigma : float
            used by sep.winpos.  Gaussian sigma used for weighting pixels.
            Pixels within a circular aperture of radius 4*winposSigma are
            included.
        winposBoxSize : int
            odd integer.  winpos centroids will be searched for in a
            winposBoxSize x winposBoxSize centered on the peak
            pixel.
        """
        if centroidMinNpix is not None:
            self.centroidMinNpix = centroidMinNpix
        self.backgroundSigma = backgroundSigma
        self.winposSigma = winposSigma
        self.winposBoxSize = winposBoxSize
        self.ccdRotCenXY = numpy.array(self.data_sub.shape) / 2.
        self.simpleSigma = simpleSigma
        self.simpleBoxSize = simpleBoxSize

        if winposBoxSize % 2 == 0 or winposBoxSize <= 0:
            raise CoordIOError("winposBoxSize must be a positive odd integer")

        # ccdRotCenXY = numpy.array(ccdRotCenXY).squeeze()
        # if len(ccdRotCenXY) != 2:
        #     raise CoordIOError("ccdRotCenXY must be a 2 element vector")


        objects = sep.extract(
            self.data_sub,
            backgroundSigma,
            err=self.bkg.globalrms,
        )

        # get rid of obvious bogus detections
        objects = objects[objects["npix"] > self.centroidMinNpix]
        # remove detections near edges of chip
        # (causes issues for unlucky winpos detections)
        # objects = objects[objects["x"] > 500]
        # objects = objects[objects["x"] < 7500]
        # objects = objects[objects["y"] > 30]
        # objects = objects[objects["y"] < 5970]

        # update based on CCD distortion model the "true location"
        # of the fibers
        if self.data_sub.shape[1] < 8000 and self.nudgeAdjust:
            # ccd was trimmed
            xNudge, yNudge = applyNudgeModel(objects["x"]+self.nudgeOffX, objects["y"], site=self.site, beta_x=beta_x, beta_y=beta_y)
            xNudge = xNudge - self.nudgeOffX
        else:
            xNudge, yNudge = applyNudgeModel(objects["x"], objects["y"], site=self.site, beta_x=beta_x, beta_y=beta_y)


        # don't fit anything with an absolute correction > 0.75 pixels
        # rejectInds = (numpy.abs(xNudge) > 0.75) | (numpy.abs(yNudge) > 0.75)


        # create mask and re-extract using winpos algorithm
        # maskArr = numpy.ones(self.data_sub.shape, dtype=bool)
        # boxRad = numpy.floor(winposBoxSize/2)
        # boxSteps = numpy.arange(-boxRad, boxRad+1, dtype=int)

        # for ii in range(len(objects)):
        #     _xm = objects["xcpeak"][ii]
        #     _ym = objects["ycpeak"][ii]
        #     for xstep in boxSteps:
        #         for ystep in boxSteps:
        #             maskArr[_ym + ystep, _xm + xstep] = False

        # xNew, yNew, flags = sep.winpos(
        #     self.data_sub,
        #     objects["xcpeak"],
        #     objects["ycpeak"],
        #     sig=winposSigma,
        #     mask=maskArr
        # )

        # refinexy requires a square array
        # trim and adjust results accordingly

        # find the longest dimension
        clipDim = numpy.argmax(self.data_sub.shape)

        if clipDim == 0:
            yOff = int((self.data_sub.shape[0]-self.data_sub.shape[1])/2)
            im = self.data_sub[yOff:-yOff, :].copy()

            xSimple, ySimple = refinexy(
                im, objects["x"], objects["y"]-yOff,
                psf_sigma=self.simpleSigma, cutout=self.simpleBoxSize
            )
            ySimple = ySimple + yOff
        else:
            xOff = int((self.data_sub.shape[1]-self.data_sub.shape[0])/2)
            im = self.data_sub[:, xOff:-xOff].copy()

            xSimple, ySimple = refinexy(
                im, objects["x"]-xOff, objects["y"],
                psf_sigma=self.simpleSigma, cutout=self.simpleBoxSize
            )
            xSimple = xSimple + xOff

        objects = pandas.DataFrame(objects)

        # objects["xWinpos"] = xNew
        # objects["yWinpos"] = yNew
        objects["xSimple"] = xSimple
        objects["ySimple"] = ySimple
        objects["xNudge"] = xNudge
        objects["yNudge"] = yNudge

        # rotate raw centroids by rotator angle
        _centTypes = [
            ["x", "y"],
            # ["xWinpos", "yWinpos"],
            ["xSimple", "ySimple"],
            ["xNudge", "yNudge"]
        ]

        for _centType in _centTypes:
            xy = objects[_centType].to_numpy()
            xyRot = (self.rotMat @ (xy - self.ccdRotCenXY).T).T + self.ccdRotCenXY
            objects[_centType[0]+"Rot"] = xyRot[:,0]
            objects[_centType[1]+"Rot"] = xyRot[:,1]

        objects["centroidID"] = list(range(len(objects)))

        self.centroids = objects
        self.nCentroids = len(objects)

        return self.centroids

    def fit(
        self,
        centType=None,
        maxRoughDist=10,
        maxMidDist=4,
        maxFinalDist=0.5,
        newInvKin=True
    ):
        """
        Calculate xy wok positions of centroids.  Store results
        in self.fullTable (a pandas.DataFrame)

        Parameters
        -----------
        centType : str
            one of "zbplus2", zbplus", "zbminus", "sep", or "nudge", default is "nudge"
        maxRoughDist : float
            Max distance for an outer fiducial match (rough mm)
        maxMidDist : float
            Max distance for all fiducial matches after similarity
            transform based on outer fiducials (~ mm)
        maxFinalDist : float
            Max distance for positioner and fiducial matches
            after the full similarity + ZB transform (mm)
        newInvKin : bool
            If True use new inverse kinematics
        """
        if centType is not None:
            centType = centType.lower()
            self.centType = centType
        self.maxRoughDist = maxRoughDist
        self.maxMidDist = maxMidDist
        self.maxFinalDist = maxFinalDist
        self.newInvKin = newInvKin

        if self.centroids is None:
            raise CoordIOError("Must run extractCentroids before fit")
        if self.centType not in ["zbplus2", "zbplus", "zbminus", "sep", "nudge", "simple"]:
            raise CoordIOError("unknown centType: %s"%str(self.centType))

        xyMetFiber = self._fullTable[
            ["xWokReportMetrology", "yWokReportMetrology"]
        ].to_numpy()

        xyWokFIF = self.fiducialCoords[["xWok", "yWok"]].to_numpy()

        # if self.centType == "winpos":
        #     xyCCDRot = self.centroids[["xWinposRot", "yWinposRot"]].to_numpy()
        #     xyCCD = self.centroids[["xWinpos", "yWinpos"]].to_numpy()
        if self.centType == "sep":
            xyCCDRot = self.centroids[["xRot", "yRot"]].to_numpy()
            xyCCD = self.centroids[["x", "y"]].to_numpy()
        elif self.centType == "simple":
            xyCCDRot = self.centroids[["xSimpleRot", "ySimpleRot"]].to_numpy()
            xyCCD = self.centroids[["xSimple", "ySimple"]].to_numpy()
        elif self.centType in ["nudge", "zbplus", "zbminus", "zbplus2"]:
            xyCCDRot = self.centroids[["xNudgeRot", "yNudgeRot"]].to_numpy()
            xyCCD = self.centroids[["xNudge", "yNudge"]].to_numpy()
        else:
            raise CoordIOError("unknown centroid type passed to fit")

        # first do a rough transform
        self.roughTransform = RoughTransform(
            xyCCDRot,
            numpy.vstack((xyMetFiber, xyWokFIF))
        )
        xyWokRough = self.roughTransform.apply(xyCCDRot)

        # just grab outer fiducials for first pass, they're easier to identify
        # centroids lying at radii > 310 must be outer ring fiducials
        rWokFIF = numpy.linalg.norm(xyWokFIF, axis=1)
        xyWokOuterFIF = xyWokFIF[rWokFIF > 310]

        # associate the centroids to the outer wok FIDs
        xyWokOuterFIF_idx, xyWokRough_idx, distances = arg_nearest_neighbor(
            xyWokOuterFIF,
            xyWokRough,
            maxRoughDist,
        )


        self.nOuterFIF_expect = len(xyWokOuterFIF)
        xyWokOuterFIF = xyWokOuterFIF[xyWokOuterFIF_idx]
        self.nOuterFIF_found = len(xyWokOuterFIF)


        # plot the rough transform
        if self.plotPathPrefix is not None:
            assoc_found = [xyWokOuterFIF, xyWokRough[xyWokRough_idx]]
            plotFVCResults(
                self.plotPathPrefix + ".roughTransform.pdf",
                xyFitCentroids=xyWokRough,
                xyMetFiber=xyMetFiber,
                xyFIF=xyWokFIF,
                assoc_used=assoc_found,
                title="Rough Transform and Outer Fiducial Associations\n" + \
                        "missing %i fiducials"%(
                            self.nOuterFIF_expect - self.nOuterFIF_found
                        )
            )

        # use associations from rough transform to fit a full
        # transform using only outer fiducial ring
        # pass zb = false to just fit translation rotation and scale

        xyCCDOuterFIF = xyCCD[xyWokRough_idx]

        self.similarityTransform = SimilarityTransform()
        self.similarityTransform.estimate(xyCCDOuterFIF, xyWokOuterFIF)
        xyWokMeas = self.similarityTransform(xyCCD)

        # associate centroids with all fiducials in grid, now that we're closer

        xyWokFIF_idx, xyWokMeas_idx, distances = arg_nearest_neighbor(
            xyWokFIF,
            xyWokMeas,
            maxMidDist,
        )

        # find unmatched fiducials
        unmatchedFIF_idx = numpy.array(list(set(range(len(xyWokFIF))) - set(xyWokFIF_idx)), dtype=int)
        xyFIFLost = xyWokFIF[unmatchedFIF_idx]

        self.nFIF_expect = len(xyWokFIF)
        _xyWokFIF = xyWokFIF[xyWokFIF_idx]
        self.nFIF_found = len(_xyWokFIF)

        #plot similarity transform
        if self.plotPathPrefix is not None:
            assoc_found = [_xyWokFIF, xyWokMeas[xyWokMeas_idx]]
            plotFVCResults(
                self.plotPathPrefix + ".similarityTransform.pdf",
                xyFitCentroids=xyWokMeas,
                xyMetFiber=xyMetFiber,
                xyFIF=xyWokFIF,
                assoc_used=assoc_found,
                title="Similarity Transform and Full Fiducial Associations\n" + \
                        "missing %i fiducials"%(
                            self.nFIF_expect - self.nFIF_found
                        ),
                xyFIFLost=xyFIFLost
            )

        # finally, do the full ZB transform based on all found FIF locations
        xyCCDFIF = xyCCD[xyWokMeas_idx]
        self.fullTransform = ZhaoBurgeTransform(
            xyCCDFIF,
            _xyWokFIF,
            polids=self.polids,
            normFactor=self.zbNormFactor
        )

        xyWokMeas = self.fullTransform.apply(xyCCD)

        # update the final measurements for positioners
        # and fiducials based on the fullTransform
        # dont enforce any minimum distance
        xyWokFIF_idx, xyWokMeas_idx, distances = arg_nearest_neighbor(
            xyWokFIF,
            xyWokMeas,
            atol=None,  # no min distance
        )

        fiducialMeas = self.fiducialCoords.iloc[xyWokFIF_idx].reset_index()
        centroidMatched = self.centroids.iloc[xyWokMeas_idx].reset_index()

        fiducialMeas = pandas.concat(
            [fiducialMeas, centroidMatched], axis=1
        )

        fiducialMeas["xWokMeas"] = xyWokMeas[xyWokMeas_idx, 0]
        fiducialMeas["yWokMeas"] = xyWokMeas[xyWokMeas_idx, 1]
        # add distances in match
        fiducialMeas["wokErr"] = distances
        fiducialMeas["wokErrWarn"] = distances > maxFinalDist

        self.fiducialCoordsMeas = fiducialMeas

        dx = self.fiducialCoordsMeas.xWokMeas - self.fiducialCoordsMeas.xWok
        dy = self.fiducialCoordsMeas.yWokMeas - self.fiducialCoordsMeas.yWok
        self.fiducialRMS = numpy.sqrt(numpy.mean(dx**2 + dy**2))
        self.nFiducialWarn = sum(fiducialMeas.wokErrWarn)

        xyWokMet_idx, xyWokMeas_idx, distances = arg_nearest_neighbor(
            xyMetFiber,
            xyWokMeas,
            atol=None, # no min distance
        )

        positionerMeas = self._fullTable.iloc[xyWokMet_idx].reset_index()
        centroidMatched = self.centroids.iloc[xyWokMeas_idx].reset_index()
        positionerMeas = pandas.concat(
            [positionerMeas, centroidMatched], axis=1
        )

        positionerMeas["xWokMeasMetrology"] = xyWokMeas[xyWokMeas_idx, 0]
        positionerMeas["yWokMeasMetrology"] = xyWokMeas[xyWokMeas_idx, 1]
        positionerMeas["wokErr"] = distances
        positionerMeas["wokErrWarn"] = distances > maxFinalDist

        #### apply zb corrections if centype is zbplus or zbminus
        if self.centType.startswith("zb") and self.zbCoeffs is not None:
            _xWok = positionerMeas.xWokMeasMetrology.to_numpy()
            _yWok = positionerMeas.yWokMeasMetrology.to_numpy()
            dx_zb_fit, dy_zb_fit = getZhaoBurgeXY(
                self.polids, self.zbCoeffs, _xWok, _yWok
            )


            if self.centType == "zbminus":
                dx_zb_fit *= -1
                dy_zb_fit *= -1
            elif self.centType == "zbplus2":
                # fit the inner region rWok < 100 mm with another
                # set of zb coeffs determined from dither analysis
                _xWok = _xWok + dx_zb_fit
                _yWok = _yWok + dy_zb_fit
                keep = numpy.sqrt(_xWok**2+_yWok**2) < 100
                dx_zb_fit2, dy_zb_fit2 = getZhaoBurgeXY(
                    self.polids, self.zbCoeffsFieldCenter,
                    _xWok[keep], _yWok[keep]
                )
                # dx2 = numpy.zeros(len(_xWok))
                # dy2 = numpy.zeros(len(_xWok))
                # dx2[keep] = dx_zb_fit2
                # dy2[keep] = dy_zb_fit2
                dx_zb_fit[keep] += dx_zb_fit2
                dy_zb_fit[keep] += dy_zb_fit2

            # multiply by scale (arcsec to mm)
            dx_zb_fit *= self.telescopePlateScale
            dy_zb_fit *= self.telescopePlateScale

        else:
            dx_zb_fit = numpy.zeros(len(positionerMeas))
            dy_zb_fit = numpy.zeros(len(positionerMeas))

        positionerMeas["xWokMeasMetrology"] = positionerMeas["xWokMeasMetrology"] + dx_zb_fit
        positionerMeas["yWokMeasMetrology"] = positionerMeas["yWokMeasMetrology"] + dy_zb_fit
        positionerMeas["xWokAdjMetrology"] = dx_zb_fit
        positionerMeas["yWokAdjMetrology"] = dy_zb_fit


        self.positionerTableMeas = positionerMeas

        dx = self.positionerTableMeas.xWokMeasMetrology - \
             self.positionerTableMeas.xWokReportMetrology
        dy = self.positionerTableMeas.yWokMeasMetrology - \
             self.positionerTableMeas.yWokReportMetrology
        sqerr = dx**2 + dy**2

        self.positionerRMS = numpy.sqrt(numpy.mean(sqerr))
        self.positionerRMS_clipped = numpy.sqrt(
            numpy.mean(sqerr[distances < maxFinalDist])
        )
        self.nPositionerWarn = numpy.sum(distances > maxFinalDist)
        _pwl = positionerMeas[positionerMeas.wokErrWarn]["positionerID"].to_numpy()
        self.positionerWarnList = _pwl


        # find unmatched remaining centroids
        centroidsUsed = list(set(
            list(self.positionerTableMeas.centroidID) + \
            list(self.fiducialCoordsMeas.centroidID)
        ))

        unusedCentroids_idx = ~self.centroids["centroidID"].isin(centroidsUsed)
        self.unmatchedCentroids = self.centroids[unusedCentroids_idx]
        unusedCentroidsXY = xyWokMeas[unusedCentroids_idx]


        xyMetWarn = self.positionerTableMeas[self.positionerTableMeas.wokErrWarn]
        xyMetWarn = xyMetWarn[
            ["xWokReportMetrology", "yWokReportMetrology"]
        ].to_numpy()



        if self.plotPathPrefix is not None:
            pwarnStr = ",".join("%i"%pid for pid in self.positionerWarnList)
            title = "Zhao-Burge Transform of all Detections\n"
            title += "positioner warnings: %s"%pwarnStr

            # plot the full transform
            plotFVCResults(
                self.plotPathPrefix + ".zhaoBurgeTransform.pdf",
                xyFitCentroids=xyWokMeas,
                xyMetFiber=xyMetFiber,
                xyFIF=xyWokFIF,
                positionerIDs=list(self._fullTable.positionerID),
                xyFitCentroidsUnmatched=unusedCentroidsXY,
                xyFiberWarn=xyMetWarn,
                title=title,
                xyFIFLost=xyFIFLost
            )

        # lastly compute measured alpha/beta and measured boss/Ap
        # locations of fibers based on metrology fiber location

        self.positionerTableMeas = alphaBetaFromMetMeas(
            self.positionerTableMeas, newInvKin=newInvKin
        )

        self.positionerTableMeas = xyWokFiberFromPositioner(
            self.positionerTableMeas, angleType="Meas", doMetrology=False
        )

        if self.centType in ["nudge", "zbplus", "zbminus", "zbplus2"]:
            self.positionerTableMeas["xFVC"] = self.positionerTableMeas["xNudge"]
            self.positionerTableMeas["yFVC"] = self.positionerTableMeas["yNudge"]
            self.fiducialCoordsMeas["xFVC"] = self.fiducialCoordsMeas["xNudge"]
            self.fiducialCoordsMeas["yFVC"] = self.fiducialCoordsMeas["yNudge"]
        # elif self.centType == "winpos":
        #     self.positionerTableMeas["xFVC"] = self.positionerTableMeas["xWinpos"]
        #     self.positionerTableMeas["yFVC"] = self.positionerTableMeas["yWinpos"]
        #     self.fiducialCoordsMeas["xFVC"] = self.fiducialCoordsMeas["xWinpos"]
        #     self.fiducialCoordsMeas["yFVC"] = self.fiducialCoordsMeas["yWinpos"]
        elif self.centType == "simple":
            self.positionerTableMeas["xFVC"] = self.positionerTableMeas["xSimple"]
            self.positionerTableMeas["yFVC"] = self.positionerTableMeas["ySimple"]
            self.fiducialCoordsMeas["xFVC"] = self.fiducialCoordsMeas["xSimple"]
            self.fiducialCoordsMeas["yFVC"] = self.fiducialCoordsMeas["ySimple"]
        elif self.centType == "sep":
            self.positionerTableMeas["xFVC"] = self.positionerTableMeas["x"]
            self.positionerTableMeas["yFVC"] = self.positionerTableMeas["y"]
            self.fiducialCoordsMeas["xFVC"] = self.fiducialCoordsMeas["x"]
            self.fiducialCoordsMeas["yFVC"] = self.fiducialCoordsMeas["y"]


class FVCTransformLCO(FVCTransformAPO):
    # polids = [0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29]  # Zhao-Burge basis defaults, best so far

    polids = numpy.array([0, 1, 2, 3, 4, 5, 6, 9, 20, 27, 28, 29, 30])
    zbCoeffs = numpy.array([
        -2.55885196e-02, -9.90203546e-02, -9.24054946e-04,  1.41515096e-04,
          6.04621458e-04,  9.61477805e-07,  1.46716446e-06,  6.09027894e-09,
         -8.62447285e-15, -4.44232617e-04, -4.40341216e-07,  4.87637920e-07,
          1.35481743e-09,
    ])
    zbCoeffsFieldCenter = None
    telRotAngRef = 89  # rotator angle that puts xyWok aligned with xyCCD on FVC image
    rotAngDir = -1
    centType = "nudge"
    telescopePlateScale = PLATE_SCALE["LCO"]/3600. # mm/arcsec
    nudgeOffX = 0  # fix nudge model after FVC resize
    site = "LCO"
    centroidMinNpix = 20
