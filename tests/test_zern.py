import numpy
import pytest

from coordio import CoordIOError
from coordio.zernike import ZernFit, gradZern, orthoZern


numpy.random.seed(0)

nPts = 1000
badX = numpy.random.uniform(size=nPts)
badY = numpy.random.uniform(size=nPts)

goodX = badX * numpy.sqrt(2) / 2
goodY = badY * numpy.sqrt(2) / 2

zernOrder = 3
largeOrder = 21


def test_ortho():
    dx, dy = orthoZern(goodX, goodY, zernOrder)
    assert dx.shape[0] == nPts
    assert dy.shape[0] == nPts
    assert dx.shape[1] == zernOrder
    assert dy.shape[1] == zernOrder
    with pytest.raises(CoordIOError):
        dx, dy = orthoZern(goodX, goodY, largeOrder)
    with pytest.raises(CoordIOError):
        dx, dy = orthoZern(badX, badY, zernOrder)


def test_grad():
    dx, dy = gradZern(goodX, goodY, zernOrder)
    assert dx.shape[0] == nPts
    assert dy.shape[0] == nPts
    assert dx.shape[1] > zernOrder
    assert dy.shape[1] > zernOrder

    with pytest.raises(CoordIOError):
        dx, dy = gradZern(badX, badY, zernOrder)

    dx, dy = gradZern(goodX, goodY, largeOrder)
    assert dx.shape[0] == nPts
    assert dy.shape[0] == nPts
    assert dx.shape[1] > largeOrder
    assert dy.shape[1] > largeOrder


# Fails sometimes but not always.
@pytest.mark.xfail()
def test_fit():
    zf = ZernFit(goodX, goodY, badX, badY, orders=20, method="ortho")
    xFit, yFit = zf.apply(goodX, goodY)
