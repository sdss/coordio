from coordio import calibration
import numpy
from coordio.conv import wokToPositioner, positionerToWok
import time
import matplotlib.pyplot as plt


def test_wok2positioner(plot=False):
    pt = calibration.positionerTable.reset_index()
    wc = calibration.wokCoords.reset_index()
    mt = pt.merge(wc, on=["site", "holeID"])

    nCoords = 10000
    alphas = numpy.random.uniform(size=nCoords)*360
    betas = numpy.random.uniform(size=nCoords)*178 + 2

    mtSamp = mt.sample(n=nCoords, replace=True)

    xBeta = mtSamp.bossX.to_numpy()
    yBeta = mtSamp.bossY.to_numpy()
    la = mtSamp.alphaArmLen.to_numpy()
    alphaOffDeg = mtSamp.alphaOffset.to_numpy()
    betaOffDeg = mtSamp.betaOffset.to_numpy()
    basePos = mtSamp[["xWok", "yWok", "zWok"]].to_numpy()
    iHat = mtSamp[["ix", "iy", "iz"]].to_numpy()
    jHat = mtSamp[["jx", "jy", "jz"]].to_numpy()
    kHat = mtSamp[["kx", "ky", "kz"]].to_numpy()
    dx = mtSamp.dx.to_numpy()
    dy = mtSamp.dy.to_numpy()

    tstart = time.time()
    xWok, yWok, zWok = positionerToWok(
        alphas, betas, xBeta, yBeta, la, alphaOffDeg, betaOffDeg,
        basePos, iHat, jHat, kHat, dx, dy
    )
    print("took", time.time()-tstart)
    # print(zWok)

    _alphas, _betas = wokToPositioner(
        xWok, yWok, zWok, xBeta, yBeta, la, alphaOffDeg, betaOffDeg,
        basePos, iHat, jHat, kHat, dx, dy
    )


    alphaErr = alphas - _alphas
    betaErr = betas - _betas


    numpy.testing.assert_allclose(alphas, _alphas)
    numpy.testing.assert_allclose(betas, _betas)

    print("std roundtrip", numpy.std(alphaErr), numpy.std(betaErr))

    if plot:
        plt.figure()
        plt.plot(xWok,yWok,'.k')
        plt.axis('equal')

        plt.figure()
        plt.plot(alphas, _alphas, '.k')

        plt.figure()
        plt.plot(betas, _betas, '.k')

        # plt.figure()
        # plt.hist(alphaErr, bins=numpy.linspace(-.01, .01, 100))

        # plt.figure()
        # plt.hist(betaErr, bins=numpy.linspace(-.01, .01, 100))

        plt.show()

    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    test_wok2positioner(True)