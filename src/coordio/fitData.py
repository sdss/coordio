#!/usr/bin/env python

"""Fit plug plate measurements

History:
2010-06-17 ROwen    2.0: first version using Python instead of Igor
2011-01-13 ROwen    2.1: Show another digit of radial position error.
                    Display radial position error as Pos Err instead of Rad Err.
                    Fit and show quadrupole moment as additional data.
2011-01-14 ROwen    2.2: Graph residuals after removing quadrupole error.
                    Normalize reported quadrupole magnitude (always positive) and angle (range -180 to 180).
                    Fixed name dipole -> quadrupole.
                    Slightly improved quadrupole fitting by using an initial value for quadrupole magnitude.
                    Improve display of long filenames (which are sometimes used for debugging).
2011-02-25 ROwen    Split out of fitPlugPlateMeas.py
                    Changed from functions to classes.
"""
import math
import numpy
import scipy.optimize

RadPerDeg = math.pi / 180.0

class Model(object):
    """A function with coefficients (of type float) that can be fit
    
    Subclasses must override basicApply
    and may wish to override setCoeffs to normalize the values
    and/or accessors to get more convenient versions of the coefficients
    """
    def __init__(self, coeffs):
        """Create a model
        
        Inputs:
        - coeffs: an array of floats (or something that can be converted to one); a copy is made
        """
        self._coeffs = numpy.array(coeffs, dtype=float)
    
    def basicApply(self, coeffs, posArr):
        """Apply the model using the specified coefficients. Subclasses must override.
        
        Inputs:
        - coeffs: an array of coefficients
        - posArr: an array of shape [n,2] of x,y positions
        
        Returns:
        - outPosArr: an array of positions after applying the model
        """
        raise RuntimeError("Subclass must override")
    
    def apply(self, posArr, doInverse=False):
        """Apply the function using the current coefficients to an array of shape [n,2]
        
        Inputs:
        - posArr: an array of shape [n,2] of x,y positions
        
        Returns:
        - adjPosArr: an array of shape [n,2] of x,y positions after applying the model
        """
        if doInverse:
            coeffs = self.getInverseCoeffs()
        else:
            coeffs = self._coeffs
        return self.basicApply(coeffs, posArr)
    
    def applyOne(self, xyPos, doInverse=False):
        """Apply the function using the current coefficients to a single point: a sequence of two floats

        Inputs:
        - xyPos: an x,y position (pair of floats)
        
        Returns:
        - adjXYPos: the x,y position after applying the model
        """
        return self.apply(numpy.array([xyPos], dtype=float), doInverse=doInverse)[0]
    
    def getCoeffs(self):
        """Get a copy of the coefficients
        """
        return numpy.array(self._coeffs, copy=True)
    
    def getInverseCoeffs(self):
        """Get coefficients for inverse transformation
        """
        raise RuntimeError("Subclass must override")
    
    def setCoeffs(self, coeffs):
        """Set the coefficients; subclasses may override to normalize the coefficients
        
        Inputs:
        - coeffs: an array of coefficients; a copy is made
        """
        if len(coeffs) != len(self._coeffs):
            raise RuntimeError("coeffs array has the wrong length")
        self._coeffs = numpy.array(coeffs, dtype=float)


class ModelFit(object):
    """Fit a Model
    """
    def __init__(self, model, measPos, nomPos, doRaise=False, maxFuncEval=5000):
        self.model = model
        self.measPos = measPos
        self.nomPos = nomPos
        
        initialCoeffs = model.getCoeffs()
        fitCoeffs, status = scipy.optimize.leastsq(
            self._computeRadSqError,
            initialCoeffs,
            maxfev = maxFuncEval,
        )
        if status not in list(range(5)):
            if doRaise:
                raise RuntimeError("fit failed")
            else:
                fitCoeffs[:] = numpy.nan
        self.model.setCoeffs(fitCoeffs)

    def _computeRadSqError(self, coeffs):
        """Compute the radial error squared for a particular set of coefficients
        
        Inputs:
        - coeffs: see applyTransRotScale
        - measPos: array of measured x,y positions
        - nomPos: array of nominal x,y positions
        """
        fitPos = self.model.basicApply(coeffs, self.nomPos)
        residPosErr = self.measPos - fitPos
        return residPosErr[:,0]**2 + residPosErr[:,1]**2
    
    def getFitPos(self):
        """Return the fit positions: model.apply(self.nomPos)
        """
        return self.model.apply(self.nomPos)
    
    def getPosError(self):
        """Return the residual position error: measPos - fitPos
        """
        return self.measPos - self.getFitPos()
    
    def getRadError(self):
        """Return the residual radial errors: radius of measPos - fitPos
        """
        return numpy.sqrt(self._computeRadSqError(self.model.getCoeffs()))


class TransRotScaleModel(Model):
    """Model translation, rotation and scale
    
    out x  =  c0  +  in x * (c3  -c2)
        y     c1        y   (c2   c3)
            thus:
            translation = (c0, c1)
            rotation = atan2(c2, c3)
            scale = sqrt(c2^2 + c3^2)
    """
    def __init__(self, coeffs=(0.0, 0.0, 0.0, 1.0)):
        """Create a model.
        
        The default coefficients are for no translation, rotation or scale change
        """
        Model.__init__(self, coeffs)

    def basicApply(self, coeffs, posArr):
        """Compute the fit position given a set of coefficients
        """
        # I want rotVec0, rotVec1,... = rotMat * [vec1, vec2,...]
        # but I don't know how to do it, so use this instead
        # rotVec0, rotVec1,... = [vec1, vec2,...] * rotMatTransposed
        rotMatTransposed = numpy.array(((coeffs[3], coeffs[2]), (-coeffs[2], coeffs[3])), dtype=float)
        return coeffs[0:2] + numpy.dot(posArr, rotMatTransposed)
    
    def getInverseCoeffs(self):
        """Get coefficients for inverse transformation
        """
        trans = (self._coeffs[0], self._coeffs[1])
        magSq = self._coeffs[2]**2 + self._coeffs[3]**2
        invC2 = -self._coeffs[2] / magSq
        invC3 = self._coeffs[3] / magSq
        invMatTransposed = numpy.array(((invC3, invC2), (-invC2, invC3)), dtype=float)
        negInvTrans = numpy.dot(trans, invMatTransposed)
        return (
            -negInvTrans[0],
            -negInvTrans[1],
            invC2,
            invC3,
        )

    def getTransRotScale(self):
        """Return translation, rotation and scale factor
        
        Returns:
        - x, y translation
        - rotation angle, in degrees
        - scale factor
        """
        return (
            self._coeffs[0:2],
            math.atan2(self._coeffs[2], self._coeffs[3]) / RadPerDeg,
            math.sqrt(self._coeffs[2]**2 + self._coeffs[3]**2),
        )
    
    def setTransRotScale(self, xyTrans, rotAng, scale):
        """Set coefficients from translation, rotation and scale factor
        """
        rotAngRad = rotAng * RadPerDeg
        self._coeffs = (
            xyTrans[0],
            xyTrans[1],
            scale * math.sin(rotAngRad),
            scale * math.cos(rotAngRad),
        )


class QuadrupoleModel(Model):
    """Model quadrupole
    
    outXY = inXY [1 + qpMag cos(2 inAng - 2 qpAng)]
            where:
            * qpMag is the quadruple magnitude
            * qpAng is the quadruple angle
          = inXY [1 + (c0 * 1e-3 * cos(2 * (inAng - c1))))
            where:
            * c0 = qpMag * 1e3 (the 1e3 helps the fitter converge)
            * apAng = c1

    WARNING: the inverse transformation is approximated as:
    outXY = inXY [1 - apMag cos(2 inAng - 2 qpAng)]
    which only works for small apMag.
    """
    def __init__(self, coeffs=(0.0, 0.0)):
        """Create a model.
        
        The default coefficients are for no quadrupole
        """
        Model.__init__(self, coeffs)

    def basicApply(self, coeffs, inPos):
        """Apply quadruple correction
        """
        xArr = inPos[:,0]
        yArr = inPos[:,1]
        rSqArr = xArr**2 + yArr**2
        inAng = numpy.arctan2(yArr, xArr)
        qpCorr = 1.0 + (coeffs[0] * 1.0e-3 * numpy.cos(2.0 * (inAng - coeffs[1])))
        qpCorr = numpy.where(rSqArr > 0.0, qpCorr, 1.0)
        return inPos * qpCorr[:,numpy.newaxis]

    def getInverseCoeffs(self):
        """Get coefficients for inverse transformation
        
        Warning: this is an approximation that only works for small quadrupole moments
        """
        # preserve magnitude and rotate angle by pi/2
        return (
            self._coeffs[0],
            self._wrapAngleCoeff(self._coeffs[1] + (math.pi * 0.5)),
        )

    def setCoeffs(self, coeffs):
        """Set coefficients
        """
        self._coeffs = numpy.array(coeffs, dtype=float)
        # make quadrupole magnitude positive, adjusting angle if necessary
        if self._coeffs[0] < 0:
            self._coeffs[0] = -self._coeffs[0]
            self._coeffs[1] += math.pi * 0.5
    
        # wrap angle coefficient
        self._coeffs[1] = self._wrapAngleCoeff(self._coeffs[1])

    def getMagnitudeAngle(self):
        """Get the quadrupole magnitude and angle
        
        Returns:
        - Quadrupole magnitude = c0
        - Quadrupole angle, in degrees = c1 converted from radians to degrees
        """
        return (
            self._coeffs[0] * 1.0e-3,
            self._coeffs[1] / RadPerDeg,
        )
    
    def setMagnitudeAngle(self, mag, ang):
        """Set coeffs from magnitude and angle

        Inputs:
        - mag: quadrupole magnitude
        - ang: quadrupole angle, in degrees
        """
        self._coeffs = (
            mag * 1.0e3,
            ang * RadPerDeg,
        )
    
    def _wrapAngleCoeff(self, angCoeff):
        """Wrap angle coeff into range [-pi/2, pi/2] (since it's degenerate to 1/2 rotation)
        
        Input:
        - angleCoeff: angle coefficient (radians)
        
        Returns:
        - wrapped angle coefficient (radians)
        """
        angCoeff = angCoeff % (2.0 * math.pi) # in range [0, 2 pi]
        if angCoeff > math.pi:
            # remove degeneracy and put in range [0, pi]
            angCoeff -= math.pi
        if angCoeff > math.pi * 0.5:
            # shift range to [-pi/2, pi/2]
            angCoeff -= math.pi
        return angCoeff


def arrayRMS(arr):
    """Return the RMS of an array
    """
    return math.sqrt(numpy.sum(arr**2) / len(arr))


if __name__ == "__main__":
    print("Test TransRotScaleModel")
    trsModel = TransRotScaleModel()
    xyPosList = []
    for xPos in (0.0, 1.0): # -1.5, 0.0, 0.5):
        for yPos in (0.0, 1.0): # (-0.5, 0.0, 1.5):
            xyPosList.append((xPos, yPos))
    xyPosArr = numpy.array(xyPosList, dtype=float)
    
    trans = (-0.5, 0.3)
    rot = 45.0
    scale = 1.2
    sqrt2 = math.sqrt(2.0)
    basicOut = numpy.array((sqrt2, sqrt2)) * scale
    trsModel.setTransRotScale(trans, rot, scale)
    for inPos, desOutPos in (
        ((1.0, 0.0), numpy.array((1.0/sqrt2, 1.0/sqrt2)) * scale + trans),
        ((0.0, 1.0), numpy.array((-1.0/sqrt2, 1.0/sqrt2)) * scale + trans),
        ((-2.0, -2.0), numpy.array((0, -sqrt2)) * 2.0 * scale + trans),
    ):
        outPos = trsModel.applyOne(inPos)
        if not numpy.allclose(outPos, desOutPos):
            print("Apply failed on coeffs=%s, inPos=%s, desOutPos=%s outPos=%s" % (trsModel.getCoeffs(), inPos, desOutPos, outPos))

    for trans in ((0.0, 0.0), (-0.5, 0.3), (0.0, 0.0), (11.9, -35.4)):
        for rot in (90.0, -180.0, -179.0, -90.0, -45.0, 0.0, 45.0, 180.0):
            for scale in (1.0, -0.1, 0.9, 0.99999, 1.0, 1.0001, 1.1, 10.0):
                trsModel.setTransRotScale(trans, rot, scale)
                cnvPosArr = trsModel.apply(xyPosArr)
                invPosArr = trsModel.apply(cnvPosArr, doInverse=True)
                if not numpy.allclose(xyPosArr, invPosArr):
                    print("Round trip failed for trans=%s, rot=%s, scale=%s" % (trans, rot, scale))
                    print("xyPosArr=", xyPosArr)
                    print("cnvPosArr=", cnvPosArr)
                    print("invPosArr=", invPosArr)

    print("Test QuadrupoleModel")
    qpModel = QuadrupoleModel()
    xyPosList = []
    for xPos in (0.0, 1.0): # -1.5, 0.0, 0.5):
        for yPos in (0.0, 1.0): # (-0.5, 0.0, 1.5):
            xyPosList.append((xPos, yPos))
    xyPosArr = numpy.array(xyPosList, dtype=float)

    # to do: add direct test of values
    
    for qpMag in (0.0, 0.001, -0.001):
        for qpAngle in (90.0, -180.0, -179.0, -90.0, -45.0, 0.0, 45.0, 180.0):
            qpModel.setMagnitudeAngle(qpMag, qpAngle)
            cnvPosArr = qpModel.apply(xyPosArr)
            invPosArr = qpModel.apply(cnvPosArr, doInverse=True)
            if not numpy.allclose(xyPosArr, invPosArr):
                print("Round trip failed for qpMag=%s, qpAngle=%s" % (qpMag, qpAngle))
                print("xyPosArr=", xyPosArr)
                print("cnvPosArr=", cnvPosArr)
                print("invPosArr=", invPosArr)
