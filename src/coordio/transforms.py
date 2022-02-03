#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy
import pandas
from skimage.transform import SimilarityTransform
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import sep
import scipy

from .zhaoburge import fitZhaoBurge, getZhaoBurgeXY
from .conv import positionerToTangent, tangentToWok
from .defaults import calibration, POSITIONER_HEIGHT
from .exceptions import CoordIOError


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
    title=None
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
    title : str or None
        title for the plot

    """

    plt.figure(figsize=(8, 8))
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
        for xy1, xy2 in zip(assoc_used[0], assoc_used[1]):
            plt.plot([xy1[0], xy2[0]], [xy1[1], xy2[1]], "-k")

    plt.axis("equal")
    plt.legend()
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

    def __init__(self, xyCCD, xyWok, polids=None):
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

        Attributes
        -------------
        polids : numpy.ndarray
            1D integer array containing basis IDs used for ZB transform
        coeffs : numpy.ndarray
            1D float array containing the fit coefficient values for each
            polid in the ZB transform
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
            )
            xWokFit = xySimTransFit[:, 0] + dx
            yWokFit = xySimTransFit[:, 1] + dy
            xyWokFit = numpy.array([xWokFit, yWokFit]).T
        else:
            xyWokFit = xySimTransFit

        return xyWokFit


def xyWokFiberFromPositioner(
    fullTable, alphaColumn="alphaReport", betaColumn="betaReport"
):
    """
    Determine xy wok position for a each fiber for each robot.

    Parameters
    ------------
    fullTable : pandas.DataFrame
        A merge on holeID of positionerTable and wokCoords DataFrames,
        with additional columns names specified by alphaColumn and betaColumn
        indicating the robot's arm coords.
    alphaColumn : str
        column name in fullTable that should be used as alpha coordinates
    betaColumn : str
        column name in fullTable that should be used as beta coordinates


    Returns
    --------
    fullTable : pandas.DataFrame
        the input fullTable with new columns appended
        xWokExpectXXX, yWokExpectXXX where XXX in ["Metrology", "APOGEE",
        "BOSS"]
    """

    # probably can vectorize this whole thing, or at least
    # partially

    fiberNameMap = zip(
        ["Metrology", "APOGEE", "BOSS"],
        ["met", "ap", "boss"]
    )
    for fiberType, colName in fiberNameMap:
        fiberX = "%sX"%colName
        fiberY = "%sY"%colName
        xWok = []
        yWok = []
        for ii, posRow in fullTable.iterrows():
            alpha = float(posRow[alphaColumn])
            beta = float(posRow[betaColumn])

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

        fullTable["xWokExpect%s" % fiberType] = xWok
        fullTable["yWokExpect%s" % fiberType] = yWok

    return fullTable


class FVCTransformAPO(object):
    polids = [0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29]  # Zhao-Burge basis defaults

    def __init__(
        self,
        fvcImgData,
        positionerCoords,
        telRotAng,
        plotPathPrefix,
        alphaColumn="alphaReport",
        betaColumn="betaReport",
        positionerTable=calibration.positionerTable,
        wokCoords=calibration.wokCoords,
        fiducialCoords=calibration.fiducialCoords,
        telRotAngRef=135.4,
        polids=None
    ):
        """
        Parameters
        -------------
        fvcImgData : numpy.ndarray
            raw image data from the fvc
        positionerCoords : pandas.DataFrame
            DataFrame containing alpha/beta coordinate columns and a
            positionerID column for each robot in the FVC image
        telRotAng : float
            telescope rotator angle in mount coordinate degrees (IPA in sdss
            headers)
        plotPathPrefix : str
            base path for plot output.  Plot routines will append a
            ".<plotname>.pdf" suffix to the prefix supplied.
        alphaColumn : str
            column name in positionerCoords that should be used as alpha
            coordinates.  Default is "alphaReport"
        betaColumn : str
            column name in positionerCoords that should be used as beta
            coordinates Default is "betaReport"
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
        """

        self.fvcImgData = numpy.array(fvcImgData, dtype=numpy.float32)
        self.positionerTable = positionerTable.reset_index()
        self.wokCoords = wokCoords.reset_index()
        self.fiducialCoords = fiducialCoords.reset_index()
        self.positionerCoords = positionerCoords.reset_index()
        self.telRotAng = telRotAng
        self.telRotAngRef = telRotAngRef
        self.plotPathPrefix = plotPathPrefix
        if polids is not None:
            self.polids = polids

        ft = positionerTable.merge(wokCoords, on="holeID").reset_index()
        ft = ft.merge(positionerCoords, on="positionerID")
        self._fullTable = xyWokFiberFromPositioner(
            ft,
            alphaColumn=alphaColumn,
            betaColumn=betaColumn
        )

        # construct a matrix for rotating centroids based on telescope
        # rotator angle
        self.ccd2WokRot = telRotAng - telRotAngRef
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

        self.centroidMinNpix = None
        self.backgroundSigma = None
        self.winposSigma = None
        self.winposBoxSize = None
        self.ccdRotCenXY = None
        self.nCentroidFound = None

        ############## populated by self.fit() ####################
        # pandas.DataFrame join w/ centroids
        self.positionerTableMeas = None
        # pandas.DataFrame join w/ centroids
        self.fiducialCoordsMeas = None
        # pandas.DataFrame subset of self._fullTable
        self.positionerTableMiss = None
        # pandas.DataFrame subset of self.fiducialCoords
        self.fiducialCoordsMiss = None
        # panas.DataFrame subset of self.centroids without
        # matches to a robot or fiducial
        self.unmatchedCentroids = None

        self.roughTransform = None
        self.similarityTransform = None
        self.fullTransform = None

        self.useWinpos = None
        self.maxRoughDist = None
        self.maxMidDist = None
        self.maxFinalDist = None
        self.nOuterFIF_expect = None
        self.nOuterFIF_found = None
        self.nFIF_expect = None
        self.nFIF_found = None
        self.nPositioner_found = None
        self.fiducialRMS = None
        self.positionerRMS = None

    def getMetadata(self):
        """Get a list of data that can be easily stuffed in a fits
        header?

        rot
        rms's
        zb coeffs
        centroid counts
        n found

        """
        pass

    def extractCentroids(
        self,
        centroidMinNpix=100,
        backgroundSigma=3.5,
        winposSigma=0.7,
        winposBoxSize=3,
        ccdRotCenXY=numpy.array([4115, 3092]),
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
            used by sep.winpos.  Gaussian sigma sued for weighting pixels.
            Pixels within a circular aperture of radius 4*winposSigma are
            included.
        winposBoxSize : int
            odd integer.  winpos centroids will be searched for in a
            winposBoxSize x winposBoxSize centered on the peak
            pixel.
        ccdRotCenXY : numpy.ndarray
            [x,y] location of the pixel centered on the rotator.  This
            need only be a rough estimate.
        """
        self.centroidMinNpix = centroidMinNpix
        self.backgroundSigma = backgroundSigma
        self.winposSigma = winposSigma
        self.winposBoxSize = winposBoxSize
        self.ccdRotCenXY = ccdRotCenXY

        if winposBoxSize % 2 == 0 or winposBoxSize <= 0:
            raise CoordIOError("winposBoxSize must be a positive odd integer")

        ccdRotCenXY = numpy.array(ccdRotCenXY).squeeze()
        if len(ccdRotCenXY) != 2:
            raise CoordIOError("ccdRotCenXY must be a 2 element vector")

        bkg = sep.Background(self.fvcImgData)
        bkg_image = bkg.back()

        data_sub = self.fvcImgData - bkg_image

        objects = sep.extract(
            data_sub,
            backgroundSigma,
            err=bkg.globalrms,
        )

        # get rid of obvious bogus detections
        objects = objects[objects["npix"] > centroidMinNpix]

        # create mask and re-extract using winpos algorithm
        maskArr = numpy.ones(data_sub.shape, dtype=bool)
        boxRad = numpy.floor(winposBoxSize/2)
        boxSteps = numpy.arange(-boxRad, boxRad+1, dtype=int)

        for ii in range(len(objects)):
            _xm = objects["xcpeak"][ii]
            _ym = objects["ycpeak"][ii]
            for xstep in boxSteps:
                for ystep in boxSteps:
                    maskArr[_ym + ystep, _xm + xstep] = False

        xNew, yNew, flags = sep.winpos(
            data_sub,
            objects["xcpeak"],
            objects["ycpeak"],
            sig=winposSigma,
            mask=maskArr
        )

        objects = pandas.DataFrame(objects)

        objects["xWinpos"] = xNew
        objects["yWinpos"] = yNew


        objects["x"] = xNew
        objects["y"] = yNew

        # rotate raw centroids by rotator angle
        xy = objects[["x", "y"]].to_numpy()
        xyRot = (self.rotMat @ (xy - ccdRotCenXY).T).T + ccdRotCenXY

        objects["xRot"] = xyRot[:, 0]
        objects["yRot"] = xyRot[:, 1]

        # rotate winpos centroids by rotator angle
        xy = objects[["xWinpos", "yWinpos"]].to_numpy()
        xyRot = (self.rotMat @ (xy - ccdRotCenXY).T).T + ccdRotCenXY

        objects["xWinposRot"] = xyRot[:, 0]
        objects["yWinposRot"] = xyRot[:, 1]

        objects["centroidID"] = list(range(len(objects)))

        self.centroids = objects
        self.nCentroids = len(objects)

    def fit(
        self,
        useWinpos=True,
        maxRoughDist=10,
        maxMidDist=4,
        maxFinalDist=0.5
    ):
        """
        Calculate xy wok positions of centroids.  Store results
        in self.fullTable (a pandas.DataFrame)

        Parameters
        -----------
        useWinpos : bool
            If True, use sep.winpos centroids, else use raw sep.extract
            centroids
        maxRoughDist : float
            Max distance for an outer fiducial match (rough mm)
        maxMidDist : float
            Max distance for all fiducial matches after similarity
            transform based on outer fiducials (~ mm)
        maxFinalDist : float
            Max distance for positioner and fiducial matches
            after the full similarity + ZB transform (mm)
        """
        self.useWinpos = useWinpos
        self.maxRoughDist = maxRoughDist
        self.maxMidDist = maxMidDist
        self.maxFinalDist = maxFinalDist

        if self.centroids is None:
            raise CoordIOError("Must run extractCentroids before fit")

        xyMetFiber = self._fullTable[
            ["xWokExpectMetrology", "yWokExpectMetrology"]
        ].to_numpy()

        xyWokFIF = self.fiducialCoords[["xWok", "yWok"]].to_numpy()

        if useWinpos:
            xyCCD = self.centroids[["xWinposRot", "yWinposRot"]].to_numpy()
        else:
            xyCCD = self.centroids[["xRot", "yRot"]].to_numpy()

        # first do a rough transform
        self.roughTransform = RoughTransform(
            xyCCD,
            numpy.vstack((xyMetFiber, xyWokFIF))
        )
        xyWokRough = self.roughTransform.apply(xyCCD)

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

        assoc_found = [xyWokOuterFIF, xyWokRough[xyWokRough_idx]]


        # plot the rough transform
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

        self.nFIF_expect = len(xyWokFIF)
        _xyWokFIF = xyWokFIF[xyWokFIF_idx]
        self.nFIF_found = len(_xyWokFIF)

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
                    )
        )

        # finally, do the full ZB transform based on all found FIF locations
        xyCCDFIF = xyCCD[xyWokMeas_idx]
        self.fullTransform = ZhaoBurgeTransform(
            xyCCDFIF,
            _xyWokFIF,
            polids=self.polids
        )

        xyWokMeas = self.fullTransform.apply(xyCCD)

        # update the final measurements for positioners
        # and fiducials based on the fullTransform

        xyWokFIF_idx, xyWokMeas_idx, distances = arg_nearest_neighbor(
            xyWokFIF,
            xyWokMeas,
            maxFinalDist,
        )

        fiducialMeas = self.fiducialCoords.iloc[xyWokFIF_idx].reset_index()
        centroidMatched = self.centroids.iloc[xyWokMeas_idx].reset_index()

        fiducialMeas = pandas.concat(
            [fiducialMeas, centroidMatched], axis=1
        )

        fiducialMeas["xWokMeas"] = xyWokMeas[xyWokMeas_idx, 0]
        fiducialMeas["yWokMeas"] = xyWokMeas[xyWokMeas_idx, 1]
        self.fiducialCoordsMeas = fiducialMeas

        dx = self.fiducialCoordsMeas.xWokMeas - self.fiducialCoordsMeas.xWok
        dy = self.fiducialCoordsMeas.yWokMeas - self.fiducialCoordsMeas.yWok
        self.fiducialRMS = numpy.sqrt(numpy.mean(dx**2 + dy**2))

        xyWokMet_idx, xyWokMeas_idx, distances = arg_nearest_neighbor(
            xyMetFiber,
            xyWokMeas,
            maxFinalDist,
        )

        positionerMeas = self._fullTable.iloc[xyWokMet_idx].reset_index()
        centroidMatched = self.centroids.iloc[xyWokMeas_idx].reset_index()
        positionerMeas = pandas.concat(
            [positionerMeas, centroidMatched], axis=1
        )

        positionerMeas["xWokMeasMetrology"] = xyWokMeas[xyWokMeas_idx, 0]
        positionerMeas["yWokMeasMetrology"] = xyWokMeas[xyWokMeas_idx, 1]
        self.positionerTableMeas = positionerMeas

        dx = self.positionerTableMeas.xWokMeasMetrology - \
             self.positionerTableMeas.xWokExpectMetrology
        dy = self.positionerTableMeas.yWokMeasMetrology - \
             self.positionerTableMeas.yWokExpectMetrology
        self.positionerRMS = numpy.sqrt(numpy.mean(dx**2 + dy**2))
        self.nPositioner_found = len(positionerMeas)

        plotFVCResults(
            self.plotPathPrefix + ".zhaoBurgeTransform.pdf",
            xyFitCentroids=xyWokMeas,
            xyMetFiber=xyMetFiber,
            xyFIF=xyWokFIF,
            positionerIDs=list(self._fullTable.positionerID),
            title="Zhao-Burge Transform of all Detections"
        )

        # now associate measured xy locations of fiber
        # for each robot, and measured angles for each robot
        # fullTable = self._fullTable.copy()

        # argFound, roughDist = argNearestNeighbor(xyMetFiber, xyWokMeas)
        # xyWokFiberMeas = xyWokMeas[argFound, :]
        # fullTable["xWokMetMeas"] = xyWokFiberMeas[:, 0]
        # fullTable["yWokMetMeas"] = xyWokFiberMeas[:, 1]

        # # associate the original centroid info with the fullTable
        # # to have everything in one place
        # centroidMatched = self.centroids.iloc[argFound].reset_index()
        # fullTable = pandas.concat([fullTable, centroidMatched], axis=1)

        # self.fullTable = fullTable




