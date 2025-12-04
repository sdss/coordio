from coordio import calibration
import numpy
from coordio.conv import wokToPositioner, positionerToWok, positionerToTangent, tangentToWok, wokToTangent
from coordio.libcoordio import tangentToPositioner, tangentToPositioner2
from coordio.defaults import POSITIONER_HEIGHT
import time
import matplotlib.pyplot as plt


def test_wok2positioner(plot=False):
    pt = calibration.positionerTable.reset_index()
    wc = calibration.wokCoords.reset_index()
    mt = pt.merge(wc, on=["site", "holeID"])

    nCoords = 10000
    alphas = numpy.random.uniform(size=nCoords)*360
    # betas = numpy.random.uniform(size=nCoords)*178 + 2
    betas = numpy.random.uniform(size=nCoords)*180

    mtSamp = mt.sample(n=nCoords, replace=True)

    xBeta = mtSamp.metX.to_numpy()
    yBeta = mtSamp.metY.to_numpy()
    la = mtSamp.alphaArmLen.to_numpy()
    alphaOffDeg = mtSamp.alphaOffset.to_numpy() * 0
    betaOffDeg = mtSamp.betaOffset.to_numpy() * 0
    basePos = mtSamp[["xWok", "yWok", "zWok"]].to_numpy()
    iHat = mtSamp[["ix", "iy", "iz"]].to_numpy()
    jHat = mtSamp[["jx", "jy", "jz"]].to_numpy()
    kHat = mtSamp[["kx", "ky", "kz"]].to_numpy()
    dx = mtSamp.dx.to_numpy()
    dy = mtSamp.dy.to_numpy()

    # tstart = time.time()
    xWok, yWok, zWok = positionerToWok(
        alphas, betas, xBeta, yBeta, la, alphaOffDeg, betaOffDeg,
        basePos, iHat, jHat, kHat, dx, dy
    )
    # print("took", time.time()-tstart)
    # print(zWok)

    _alphas, _betas = wokToPositioner(
        xWok, yWok, zWok, xBeta, yBeta, la, alphaOffDeg, betaOffDeg,
        basePos, iHat, jHat, kHat, dx, dy
    )


    alphaErr = alphas - _alphas
    betaErr = betas - _betas


    numpy.testing.assert_allclose(alphas, _alphas)
    numpy.testing.assert_allclose(betas, _betas)

    # print("std roundtrip", numpy.std(alphaErr), numpy.std(betaErr))

    if plot:
        plt.figure()
        plt.plot(xWok,yWok,'.k')
        plt.axis('equal')

        plt.figure()
        plt.plot(alphas, alphaErr, '.k')

        plt.figure()
        plt.plot(betas, betaErr, '.k')

        # plt.figure()
        # plt.hist(alphaErr, bins=numpy.linspace(-.01, .01, 100))

        # plt.figure()
        # plt.hist(betaErr, bins=numpy.linspace(-.01, .01, 100))

        plt.show()

    # import pdb; pdb.set_trace()


# def test_newroundtrip(plot=False):

#     pt = calibration.positionerTable.reset_index()
#     wc = calibration.wokCoords.reset_index()
#     mt = pt.merge(wc, on=["site", "holeID"])

#     nCoords = 10000
#     alphas = numpy.random.uniform(size=nCoords)*360
#     # betas = numpy.random.uniform(size=nCoords)*178 + 2
#     betas = numpy.random.uniform(size=nCoords)*180
#     mtSamp = mt.sample(n=nCoords, replace=True)
#     iHat = [0,-1, 0]
#     jHat = [1, 0, 0]
#     kHat = [0, 0, 1]
#     dx = 0
#     dy = 0
#     la = 7.4
#     xBeta = 14.314
#     yBeta = 0
#     b = [0,0,0]


#     _dalpha = []
#     _dbeta = []
#     _alpha = []
#     _beta = []
#     _alphaOff = []
#     _betaOff = []

#     # for alpha, beta in zip(alphas,betas):
#     for alpha, beta in zip([90]*16, [0.0001,1,2,3,4,5.00001,6,7,8,9,10,11,12,13,14,15]):
#         print("one alphabeta", alpha, beta)
#         # for betaOffDeg in numpy.linspace(-5,5,50):
#         for betaOffDeg in [-5]:
#             # for alphaOffDeg in numpy.linspace(-5,5,50):
#             for alphaOffDeg in [0]:

#                 xt, yt = positionerToTangent(
#                     alpha, beta, xBeta, yBeta,
#                     la, alphaOffDeg, betaOffDeg
#                 )

#                 lefthand = False
#                 # _a, _b = tangentToPositioner(
#                 #     [xt, yt], [xBeta, yBeta], la, alphaOffDeg, betaOffDeg, lefthand
#                 # )
#                 _a, _b, alphaLH, betaLH, dist = tangentToPositioner2(
#                     [xt, yt], [xBeta, yBeta], la, alphaOffDeg, betaOffDeg
#                 )
#                 _alpha.append(alpha)
#                 _beta.append(beta)
#                 _dalpha.append(_a-alpha)
#                 _dbeta.append(_b-beta)
#                 _alphaOff.append(alphaOffDeg)
#                 _betaOff.append(betaOffDeg)

#     plt.figure()
#     plt.plot(_beta, _dbeta, '.k')

#     # plt.figure()
#     # plt.plot(_alpha, _dalpha, '.k')

#     plt.show()

#                 # print("output", output)
#                 # import pdb; pdb.set_trace()












if __name__ == "__main__":
    test_newroundtrip()
    # test_wok2positioner(True)