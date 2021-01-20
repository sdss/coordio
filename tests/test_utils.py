import numpy

from coordio.utils import cart2FieldAngle, fieldAngle2Cart, cart2Sph, sph2Cart


SMALL_NUM = SMALL_NUM = 1e-9
nPts = 100000
thetas = numpy.random.uniform(size=nPts) * 2 * numpy.pi
# maximum of 4 degrees off axis
phis = numpy.radians(numpy.random.uniform(size=nPts) * 4)
r = 1
xs = r * numpy.cos(thetas) * numpy.sin(phis)
ys = r * numpy.sin(thetas) * numpy.sin(phis)
zs = r * numpy.cos(phis)


def test_phiConversion(verbose=False):
    for fieldAngle in numpy.linspace(-10,10,100):
        x,y,z = fieldAngle2Cart(fieldAngle, 0)
        theta, phi = cart2Sph(x,y,z)
        if verbose:
            print(numpy.abs(phi-fieldAngle))
        else:
            assert numpy.abs(phi-numpy.abs(fieldAngle)) < SMALL_NUM

        x,y,z = fieldAngle2Cart(0, fieldAngle)
        theta, phi = cart2Sph(x,y,z)
        if verbose:
            print(numpy.abs(phi-fieldAngle))
        else:
            assert numpy.abs(phi-numpy.abs(fieldAngle)) < SMALL_NUM


def test_cartField():
    # pick some random points on the unit sphere near the +Z cap
    for theta, phi, x, y, z in zip(thetas, phis, xs, ys, zs):
        xField, yField = cart2FieldAngle(x, y, z)
        if theta < numpy.pi / 2 or theta > 3 * numpy.pi / 2:
            assert xField < 0
        else:
            assert xField > 0

        if theta < numpy.pi:
            assert yField < 0
        else:
            assert yField > 0

        xSolve, ySolve, zSolve = fieldAngle2Cart(xField, yField)
        # print(xSolve - x, ySolve - y, zSolve - z)

        assert numpy.abs(xSolve-x) < SMALL_NUM
        assert numpy.abs(ySolve-y) < SMALL_NUM
        assert numpy.abs(zSolve-z) < SMALL_NUM


def test_cartFieldCycle():
    # check round trippage
    inds = numpy.random.choice(range(nPts), size=100)
    for ind in inds:
        x = xs[ind]
        y = ys[ind]
        z = zs[ind]
        _x = xs[ind]
        _y = ys[ind]
        _z = zs[ind]
        for ii in range(100):
            # print("ii", ii)
            xField, yField = cart2FieldAngle(_x,_y,_z)
            _x, _y, _z = fieldAngle2Cart(xField, yField)
            # repeated round trips require extra numerical buffer
            assert numpy.abs(_x-x) < SMALL_NUM
            assert numpy.abs(_y-y) < SMALL_NUM
            assert numpy.abs(_z-z) < SMALL_NUM


def test_sphCartCycle():
    # check round trippage
    inds = numpy.random.choice(range(nPts), size=100)
    for ind in inds:
        x = xs[ind]
        y = ys[ind]
        z = zs[ind]
        theta = numpy.degrees(thetas[ind])
        phi = numpy.degrees(phis[ind])
        _x = xs[ind]
        _y = ys[ind]
        _z = zs[ind]
        for ii in range(100):
            _theta, _phi = cart2Sph(_x, _y, _z)
            # print("[%.4f, %.4f] == %.5e, %.5e"%(theta, phi, phiTheta[0]-theta, phiTheta[1]-phi))
            assert numpy.abs(_theta - theta) < SMALL_NUM
            assert numpy.abs(_phi - phi) < SMALL_NUM
            _x, _y, _z = sph2Cart(_theta, _phi)
            assert numpy.abs(_x-x) < SMALL_NUM
            assert numpy.abs(_y-y) < SMALL_NUM
            assert numpy.abs(_z-z) < SMALL_NUM