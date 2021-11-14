import numpy
import pytest


numpy.random.seed(7)
import time

import matplotlib.pyplot as plt
import pandas
import seaborn as sns

from coordio import conv, libcoordio
# from coordio import Site, Wok, Observed, Field, FocalPlane, Tangent
from coordio.defaults import MICRONS_PER_MM, POSITIONER_HEIGHT, calibration


# import matplotlib.pyplot as plt
# import seaborn as sns


# LEFTHAND = True



# get tangent coords at these locations
holeIDs = ["R0C1", "R+13C1", "R+13C14", "R0C27", "R-13C14", "R+13C7", "R-13C1"]

tanCoordList = [
    [22, -4, 0],
    [-4, 180, 2],
    [60, 4, -1]
]

angTol = 1e-5 ## Deg, much less than a fraction of a micron for a positioner

modelMetXY = numpy.array([14.314, 0])
# boss xy position in solid model
modelBossXY = numpy.array([14.965, -0.376])
# apogee xy position in solid model
modelApXY = numpy.array([14.965, 0.376])
la = 7.4


def test_tangentAndPositionerLib():
    alpha = 0
    beta = 0
    xb = 14
    yb = 0
    alphaLen = 7 # equal arm lengths
    alphaOff = 0
    betaOff = 0
    # tz = 0

    tx, ty = libcoordio.positionerToTangent([alpha, beta], [xb,yb], alphaLen, alphaOff, betaOff)

    assert tx == pytest.approx(alphaLen + xb)
    assert ty == pytest.approx(0)

    _alpha, _beta = libcoordio.tangentToPositioner(
        [tx,ty], [xb,yb], alphaLen, alphaOff, betaOff
    )


    assert _alpha == pytest.approx(alpha, rel=angTol, abs=angTol)
    assert _beta == pytest.approx(beta, rel=angTol, abs=angTol)


    betaOff = 5

    tx, ty = libcoordio.positionerToTangent([alpha,beta], [xb,yb], alphaLen, alphaOff, betaOff)

    _alpha, _beta = libcoordio.tangentToPositioner(
        [tx,ty], [xb,yb], alphaLen, alphaOff, betaOff
    )

    assert (_alpha == pytest.approx(alpha) or _alpha == pytest.approx(alpha+360))
    assert _beta == pytest.approx(beta, rel=angTol, abs=angTol)
    assert ty > 1


    betaOff = -5 # its gotta go right handed
    tx, ty = libcoordio.positionerToTangent([alpha,beta], [xb,yb], alphaLen, alphaOff, betaOff)


    _alpha, _beta = libcoordio.tangentToPositioner(
        [tx,ty], [xb,yb], alphaLen, alphaOff, betaOff
    )

    assert _beta == pytest.approx(-2*betaOff)

    betaOff = 0
    alphaOff = 5

    tx, ty = libcoordio.positionerToTangent([alpha,beta], [xb,yb], alphaLen, alphaOff, betaOff)
    assert ty > 1


def test_tangentAndPositionerSafeSingle():

    alphas = numpy.random.uniform(10,350)
    betas = numpy.random.uniform(10, 170)
    yBetas = numpy.random.uniform(-0.5,0.5)
    xBetas = 15
    la = 7.4
    alphaOff = numpy.random.uniform(-1,1)
    betaOff = numpy.random.uniform(-1,1)

    xt1, yt1 = conv._positionerToTangent(
        alphas, betas, xBetas, yBetas, la, alphaOff, betaOff
    )

    xt2, yt2 = conv.positionerToTangent(
        alphas, betas, xBetas, yBetas, la, alphaOff, betaOff
    )


    assert xt1 == pytest.approx(xt2)
    assert yt1 == pytest.approx(yt2)

    a1, b1, isOK1 = conv._tangentToPositioner(
        xt1, yt1, xBetas, yBetas, la, alphaOff, betaOff
    )

    a2, b2, isOK2 = conv.tangentToPositioner(
        xt1, yt1, xBetas, yBetas, la, alphaOff, betaOff
    )

    assert a1 == pytest.approx(a2)
    assert b1 == pytest.approx(b2)
    assert isOK1 == isOK2
    assert isOK1

    assert a1 == pytest.approx(alphas)
    assert b1 == pytest.approx(betas)


def test_tangentAndPositionerSafeArr():
    ## if you're not near the edges of travel,
    # things shouldn't be weird?
    nPts = 5
    alphas = numpy.random.uniform(10,350, nPts)
    betas = numpy.random.uniform(10, 170, nPts)
    yBetas = numpy.random.uniform(-0.5,0.5, nPts)
    xBetas = [15]*nPts
    la = 7.4
    alphaOff = numpy.random.uniform(-1,1)
    betaOff = numpy.random.uniform(-1,1)

    for ii in [1,2]:
        # non-array
        xt1, yt1 = conv._positionerToTangent(
            alphas[ii], betas[ii], xBetas[ii], yBetas[ii], la, alphaOff, betaOff
        )

        xt2, yt2 = conv.positionerToTangent(
            alphas[ii], betas[ii], xBetas[ii], yBetas[ii], la, alphaOff, betaOff
        )


        assert xt1 == pytest.approx(xt2)
        assert yt1 == pytest.approx(yt2)


        a1, b1, isOK1 = conv._tangentToPositioner(
            xt1, yt1, xBetas[ii], yBetas[ii], la, alphaOff, betaOff
        )

        a2, b2, isOK2 = conv.tangentToPositioner(
            xt1, yt1, xBetas[ii], yBetas[ii], la, alphaOff, betaOff
        )


        assert a1 == pytest.approx(a2)
        assert b1 == pytest.approx(b2)
        assert isOK1 == isOK2
        assert isOK1

        assert a1 == pytest.approx(alphas[ii])
        assert b1 == pytest.approx(betas[ii])

    # array
    xt1, yt1 = conv._positionerToTangent(
        alphas, betas, xBetas, yBetas, la, alphaOff, betaOff
    )

    xt2, yt2 = conv.positionerToTangent(
        alphas, betas, xBetas, yBetas, la, alphaOff, betaOff
    )


    assert xt1 == pytest.approx(xt2)
    assert yt1 == pytest.approx(yt2)


    a1, b1, isOK1 = conv._tangentToPositioner(
        xt1, yt1, xBetas, yBetas, la, alphaOff, betaOff
    )

    a2, b2, isOK2 = conv.tangentToPositioner(
        xt1, yt1, xBetas, yBetas, la, alphaOff, betaOff
    )


    # import pdb; pdb.set_trace()
    assert a1 == pytest.approx(a2)
    assert b1 == pytest.approx(b2)
    assert not False in isOK1
    assert not False in isOK2

    assert a1 == pytest.approx(alphas)
    assert b1 == pytest.approx(betas)


def test_unreachableTangent():


    r = 24
    thetas = numpy.random.uniform(0, 2*numpy.pi,20)
    xts = r*numpy.cos(thetas)
    yts = r*numpy.sin(thetas)

    for fx,fy in [modelMetXY, modelBossXY, modelApXY]:
        a1,b1,isOK1 = conv._tangentToPositioner(
            xts,yts,fx,fy,la
        )

        a2,b2,isOK2 = conv.tangentToPositioner(
            xts,yts,fx,fy,la
        )
        assert not True in isOK1
        assert not True in isOK2


def plot_wholeRange():
    # not really a test
    nPts = 100000
    rs = numpy.random.uniform(0, (la+17)**2, nPts)
    thetas = numpy.random.uniform(0, 2*numpy.pi, nPts)
    xts = numpy.sqrt(rs)*numpy.cos(thetas)
    yts = numpy.sqrt(rs)*numpy.sin(thetas)

    ii = 0
    for fx, fy in [modelMetXY, modelBossXY, modelApXY]:

        a1,b1,isOK1 = conv._tangentToPositioner(
            xts,yts,fx,fy,la
        )

        a2,b2,isOK2 = conv.tangentToPositioner(
            xts,yts,fx,fy,la
        )

        fig, axs = plt.subplots(1,2, figsize=(13,8))

        sns.scatterplot(x=xts, y=yts, hue=isOK1, alpha=0.2, ax=axs[0])
        sns.scatterplot(x=xts, y=yts, hue=isOK2, alpha=0.2, ax=axs[1])
        axs[0].set_aspect("equal")
        axs[1].set_aspect("equal")
        fig.suptitle("iter %i"%ii)


        _a1 = a1[isOK1]
        _b1 = b1[isOK1]
        xts1 = xts[isOK1]
        yts1 = yts[isOK1]

        _a2 = a2[isOK2]
        _b2 = b2[isOK2]
        xts2 = xts[isOK2]
        yts2 = yts[isOK2]

        fig, axs = plt.subplots(2,2, figsize=(10,10))
        axs = axs.flatten()

        sns.scatterplot(x=xts1, y=yts1, hue=_a1, alpha=0.2, ax=axs[0])
        sns.scatterplot(x=xts1, y=yts1, hue=_b1, alpha=0.2, ax=axs[1])
        sns.scatterplot(x=xts2, y=yts2, hue=_a2, alpha=0.2, ax=axs[2])
        sns.scatterplot(x=xts2, y=yts2, hue=_b2, alpha=0.2, ax=axs[3])
        axs[0].set_aspect("equal")
        axs[1].set_aspect("equal")
        axs[2].set_aspect("equal")
        axs[3].set_aspect("equal")
        axs[0].set_title("alpha")
        axs[1].set_title("beta")

        fig.suptitle("iter %i"%ii)
        ii += 1
    plt.show()


def plot_degenerateSolns():

    npts = 100000
    alphas = numpy.random.uniform(0, 359.9999, npts)
    betas = numpy.random.uniform(0, 180, npts)


    for fx, fy in [modelMetXY, modelBossXY, modelApXY]:
        tx, ty = conv.positionerToTangent(
            alphas, betas, fx, fy, la
        )

        _alphas, _betas, isOK = conv.tangentToPositioner(
            tx, ty, fx, fy, la
        )
        da = alphas - _alphas
        ida = numpy.abs(da) > 1e-5
        db = betas - _betas
        idb = numpy.abs(db) > 1e-5
        fig = plt.figure()
        sns.scatterplot(x=tx[ida], y=ty[ida], hue=da[ida], s=3, alpha=1)
        plt.axis("equal")
        plt.title("d alpha")
        plt.show()

        fig = plt.figure()
        sns.scatterplot(x=tx[idb], y=ty[idb], hue=db[idb], s=3, alpha=1)
        plt.axis("equal")
        plt.title("d beta")
        plt.show()


def test_wokAndTangent():

    for holeID in holeIDs:
        row = calibration.wokCoords.loc[('APO', holeID)]
        b = [round(row.xWok, 5),
             round(row.yWok, 5),
             round(row.zWok, 5)]
        iHat = [float(row.ix), float(row.iy), float(row.iz)]
        jHat = [float(row.jx), float(row.jy), float(row.jz)]
        kHat = [float(row.kx), float(row.ky), float(row.kz)]

        for tx,ty,tz in tanCoordList:
            scaleFac = numpy.random.uniform(0.9,1.1)
            dx = numpy.random.uniform(-0.01,0.01)
            dy = numpy.random.uniform(-0.01,0.01)
            dz = numpy.random.uniform(-0.01,0.01)

            for addMe in [0, numpy.zeros(4)]:
                _tx = tx + addMe
                _ty = ty + addMe
                _tz = tz + addMe

                wx1,wy1,wz1 = conv.tangentToWok(_tx, _ty, _tz, b, iHat, jHat, kHat,
                    scaleFac=scaleFac, dx=dx, dy=dy, dz=dz)
                wx2,wy2,wz2 = conv._tangentToWok(_tx, _ty, _tz, b, iHat, jHat, kHat,
                    scaleFac=scaleFac, dx=dx, dy=dy, dz=dz)

                assert wx1 == pytest.approx(wx2)
                assert wy1 == pytest.approx(wy2)
                assert wz1 == pytest.approx(wz2)

                tx1,ty1,tz1 = conv.wokToTangent(
                    wx1, wy1, wz1, b, iHat, jHat, kHat,
                    scaleFac=scaleFac, dx=dx, dy=dy, dz=dz)
                tx2,ty2,tz2 = conv._wokToTangent(wx1, wy1, wz1, b, iHat, jHat, kHat,
                    scaleFac=scaleFac, dx=dx, dy=dy, dz=dz)

                assert tx1 == pytest.approx(tx2)
                assert ty1 == pytest.approx(ty2)
                assert tz1 == pytest.approx(tz2)

                assert _tx == pytest.approx(tx2)
                assert _ty == pytest.approx(ty2)
                assert _tz == pytest.approx(tz2)

        # break

def test_lefthand():
    npts = 5
    alphaOff = 0
    betaOff = 0
    la = 7.4
    xBeta = 15
    yBeta = 0

    alphas = numpy.random.uniform(0, 360, npts)
    betas = numpy.random.uniform(0, 180, npts)


    xt, yt = conv.positionerToTangent(
        alphas, betas, xBeta, yBeta, la, alphaOff, betaOff
    )

    # lefthand version
    _alphaLeft, _betaLeft, isOK = conv.tangentToPositioner(
        xt, yt, xBeta, yBeta, la, lefthand=True
    )

    # righthand version
    # lefthand version
    _alphaRight, _betaRight, isOK = conv.tangentToPositioner(
        xt, yt, xBeta, yBeta, la, lefthand=False
    )

    assert numpy.all(_betaLeft > 180)
    assert numpy.all(_betaRight < 180)
    # print(xt, yt)

if __name__ == "__main__":
    # plot_degenerateSolns()
    test_lefthand()

#     tx,ty,tz = numpy.zeros(1000)+22, numpy.zeros(1000)-4, numpy.zeros(1000)



#     import pdb; pdb.set_trace()

#     tstart = time.time()
#     _tx1, _ty1, _tz1 = conv.wokToTangent(wx,wy,wz,b,iHat,jHat,kHat)
#     print("carr", time.time()-tstart)
#     tstart = time.time()
#     _tx2, _ty2, _tz2 = conv._wokToTangent(wx,wy,wz,b,iHat,jHat,kHat)
#     print("narr", time.time()-tstart)


#     import pdb; pdb.set_trace()
#     break


# import pdb; pdb.set_trace()
