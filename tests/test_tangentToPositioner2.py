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


@pytest.mark.xfail()
def test_sameSolution():
    alphaLen = 7.4
    metXY = [14.314, 0.0]
    apXY = [14.965,0.376]
    bossXY = [14.965, -0.376]

    # safe zones
    for theta in numpy.random.uniform(2,350, 20):
        r = numpy.random.uniform(7.5, 22) # mm radius
        alphaOff = numpy.random.uniform(-0.1, 0.1)
        betaOff = numpy.random.uniform(-0.1, 0.1)
        x = r*numpy.cos(numpy.radians(theta))
        y = r*numpy.sin(numpy.radians(theta))

        for fx, fy in [metXY, apXY, bossXY]:
            a1R, b1R, a1L, b1L, err = libcoordio.tangentToPositioner2(
                [x,y], [fx, fy], alphaLen, alphaOff, betaOff
            )

            a2, b2 = libcoordio.tangentToPositioner(
                [x,y], [fx, fy], alphaLen, alphaOff, betaOff
            )

            assert numpy.abs(a1R-a2) < 1e-10
            assert numpy.abs(b1R-b2) < 1e-10
            assert err == 0


def test_outsideDonut():
    alphaLen = 7.4
    metXY = [14.314, 0.0]

    for theta in numpy.random.uniform(2,350, 5):
        r = metXY[0] + alphaLen + 0.01 # mm radius
        alphaOff = 0
        betaOff = 0
        x = r*numpy.cos(numpy.radians(theta))
        y = r*numpy.sin(numpy.radians(theta))

        fx,fy = metXY

        a1R, b1R, a1L, b1L, err = libcoordio.tangentToPositioner2(
            [x,y], [fx, fy], alphaLen, alphaOff, betaOff
        )

        a2, b2 = libcoordio.tangentToPositioner(
            [x,y], [fx, fy], alphaLen, alphaOff, betaOff
        )

        assert numpy.isnan(a2)
        assert numpy.isnan(b2)
        assert b1R == 0
        assert b1L == 0
        assert err > 0

def test_insideDonut():
    alphaLen = 7.4
    metXY = [14.314, 0.0]

    for theta in numpy.random.uniform(2,350, 5):
        r = metXY[0] + alphaLen + 0.01 # mm radius
        alphaOff = 0
        betaOff = 0
        x = r*numpy.cos(numpy.radians(theta))
        y = r*numpy.sin(numpy.radians(theta))

        fx,fy = metXY

        a1R, b1R, a1L, b1L, err = libcoordio.tangentToPositioner2(
            [x,y], [fx, fy], alphaLen, alphaOff, betaOff
        )

        a2, b2 = libcoordio.tangentToPositioner(
            [x,y], [fx, fy], alphaLen, alphaOff, betaOff
        )

        assert numpy.isnan(a2)
        assert numpy.isnan(b2)
        assert b1R == 0
        assert b1L == 0
        assert err > 0
            # print(a1R-a2, b1R-b2)




if __name__ == "__main__":
    test_outsideDonut()
