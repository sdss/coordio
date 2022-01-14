import numpy

from coordio.conv import (fieldToFocal, focalToWok, proj2XYplane,
                          sph2Cart, cart2Sph, fieldAngle2Cart, cart2FieldAngle)

SMALL_NUM = 2e-9
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

def test_fieldToFocal():
    # test zero field
    for obs in ["APO", "LCO"]:
        for waveCat in ["Apogee", "Boss", "GFA"]:
            x,y,z = 0,0,1
            theta, phi = cart2Sph(x,y,z)
            # check on axis
            _x,_y,_z,R,b,w = fieldToFocal(theta, phi, obs, waveCat)
            assert numpy.abs(_x) < SMALL_NUM
            assert numpy.abs(_y) < SMALL_NUM
            # check array on axis
            x = [0,0,0]
            y = [0,0,0]
            z = [1,1,1]
            theta, phi = cart2Sph(x,y,z)
            _x,_y,_z,R,b,w = fieldToFocal(theta, phi, obs, waveCat)
            assert numpy.max(numpy.abs(_x)) < SMALL_NUM
            assert numpy.max(numpy.abs(_y)) < SMALL_NUM

            theta = 0
            phi = 0.5 # 0.5 degrees off axis
            x,y,z = sph2Cart(theta, phi)
            assert x > 0
            assert numpy.abs(y) < SMALL_NUM
            _x,_y,_z,R,b,w = fieldToFocal(theta, phi, obs, waveCat)
            assert _x > 0
            assert numpy.abs(_y) < SMALL_NUM
            assert _z < 0

            theta = 90
            phi = 0.5 # 0.5 degrees off axis
            x,y,z = sph2Cart(theta, phi)
            assert numpy.abs(x) < SMALL_NUM
            assert y > 0
            assert z > 0
            _x,_y,_z,R,b,w = fieldToFocal(theta, phi, obs, waveCat)
            assert _y > 0
            assert numpy.abs(_x) < SMALL_NUM
            assert _z < 0

            theta = 180
            phi = 0.5 # 0.5 degrees off axis
            x,y,z = sph2Cart(theta, phi)
            assert numpy.abs(y) < SMALL_NUM
            assert x < 0
            assert z > 0
            _x,_y,_z,R,b,w = fieldToFocal(theta, phi, obs, waveCat)
            assert _x < 0
            assert numpy.abs(_y) < SMALL_NUM
            assert _z < 0

            theta = 270
            phi = 0.5 # 0.5 degrees off axis
            x,y,z = sph2Cart(theta, phi)
            assert numpy.abs(x) < SMALL_NUM
            assert y < 0
            assert z > 0
            _x,_y,_z,R,b,w = fieldToFocal(theta, phi, obs, waveCat)
            assert _y < 0
            assert numpy.abs(_x) < SMALL_NUM
            assert _z < 0

            theta = 300
            phi = 0.5 # 0.5 degrees off axis
            x,y,z = sph2Cart(theta, phi)
            assert x > 0
            assert y < 0
            assert z > 0
            _x,_y,_z,R,b,w = fieldToFocal(theta, phi, obs, waveCat)
            assert _y < 0
            assert _x > 0
            assert _z < 0

            theta = 380
            phi = 0.5 # 0.5 degrees off axis
            x,y,z = sph2Cart(theta, phi)
            assert x > 0
            assert y > 0
            assert z > 0
            _x,_y,_z,R,b,w = fieldToFocal(theta, phi, obs, waveCat)
            assert _y > 0
            assert _x > 0
            assert _z < 0

            theta = -20
            phi = 0.5 # 0.5 degrees off axis
            x,y,z = sph2Cart(theta, phi)
            assert x > 0
            assert y < 0
            assert z > 0
            _x,_y,_z,R,b,w = fieldToFocal(theta, phi, obs, waveCat)
            assert _y < 0
            assert _x > 0
            assert _z < 0


def test_focalToWok():
    zOff = -100
    xOff = 0
    yOff = 0
    xTilt = 0
    yTilt = 0
    positionAngle = 0

    xFocal, yFocal, zFocal = 0, 0, 0
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt,
        fpScale=1, projectFlat=False
    )

    assert xWok == 0
    assert yWok == 0
    assert zWok == -1 * zOff

    positionAngle = 90  # +y wok aligned with +x FP

    xFocal, yFocal, zFocal = 10, 0, 0

    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt,
        fpScale=1, projectFlat=False
    )
    assert yWok == xFocal
    assert numpy.abs(xWok) < SMALL_NUM

    positionAngle = -90 # +y wok aligned with -x FP

    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt,
        fpScale=1, projectFlat=False
    )

    assert yWok == -1*xFocal
    assert numpy.abs(xWok) < SMALL_NUM

    # test translation
    xFocal, yFocal, zFocal = 0, 0, 0
    xOff = 10
    positionAngle = 0
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt,
        fpScale=1, projectFlat=False
    )

    assert xWok == -1*xOff
    assert numpy.abs(yWok) < SMALL_NUM

    yOff = 10
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt,
        fpScale=1, projectFlat=False
    )

    assert xWok == -1*xOff
    assert yWok == -1*yOff

    positionAngle = 45
    xOff, yOff = 10, 10
    xFocal, yFocal, zFocal = 0,0,0
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt,
        fpScale=1, projectFlat=False
    )
    assert xWok == yWok
    assert xWok < 0

    positionAngle = 45
    xOff, yOff = -10, 10
    xFocal, yFocal, zFocal = 0,0,0
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt,
        fpScale=1, projectFlat=False
    )
    assert xWok == -1*yWok
    assert xWok > 0

    b = 0.5*numpy.sqrt(2*10**2)
    a = numpy.sqrt(2*b**2)
    positionAngle = 45
    xOff, yOff, zOff = 10, 10, 0
    xTilt, yTilt = 0, 0
    xFocal, yFocal, zFocal = b, b, 0

    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt,
        fpScale=1, projectFlat=False
    )
    assert numpy.abs(xWok + a) < SMALL_NUM
    assert numpy.abs(yWok) < SMALL_NUM

    # test tilts
    positionAngle =0
    xOff, yOff, zOff = 0, 0, 0
    xFocal, yFocal, zFocal = 1, 0, 0
    xTilt = 1
    yTilt = 0
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt,
        fpScale=1, projectFlat=False
    )
    assert xWok == xFocal
    assert yWok == yFocal
    assert zWok == zFocal

    positionAngle =0
    xOff, yOff, zOff = 0, 0, 0
    xFocal, yFocal, zFocal = 0, 1, 0
    xTilt = 1
    yTilt = 0
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt,
        fpScale=1, projectFlat=False
    )

    assert xWok == 0
    assert yWok < 1
    assert yWok > 0
    assert zWok < 0

    positionAngle =0
    xOff, yOff, zOff = 0, 0, 0
    xFocal, yFocal, zFocal = 1, 1, 0
    xTilt = 1
    yTilt = 4
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt,
        fpScale=1, projectFlat=False
    )

    assert xWok > 0
    assert xWok < 1
    assert yWok < 1
    assert yWok > 0
    assert yWok < 1
    assert zWok > 0

    positionAngle =0
    xOff, yOff, zOff = 0, 0, 0
    xFocal, yFocal, zFocal = 1, 1, 0
    xTilt = 1
    yTilt = -1
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt,
        fpScale=1, projectFlat=False
    )

    assert xWok > 0
    assert xWok < 1
    assert yWok < 1
    assert yWok > 0
    assert yWok < 1
    assert zWok < 0


def test_xyProj():
    rayOrigin = [0,0,1]
    r = 0.5
    thetas = numpy.linspace(-numpy.pi, numpy.pi) # put in arctan2 domain
    x = r*numpy.cos(thetas)
    y = r*numpy.sin(thetas)
    z = [0.5]*len(x)
    px, py, pz, ps = proj2XYplane(x,y,z, rayOrigin)
    # print(ps)
    mags = numpy.sqrt(px**2+py**2)
    assert numpy.max(numpy.abs(mags-1)) < SMALL_NUM
    assert numpy.max(numpy.abs(pz)) < SMALL_NUM

    _thetas = numpy.arctan2(py, px)
    assert numpy.max(numpy.abs(_thetas-thetas)) < SMALL_NUM

    r = 1.5
    x = r*numpy.cos(thetas)
    y = r*numpy.sin(thetas)
    z = [-0.5]*len(x)
    _px, _py, _pz, _ps = proj2XYplane(x,y,z, rayOrigin)

    assert numpy.max(numpy.abs(px-_px)) < SMALL_NUM
    assert numpy.max(numpy.abs(py-_py)) < SMALL_NUM
    assert numpy.max(numpy.abs(_pz)) < SMALL_NUM

    _thetas = numpy.arctan2(_py, _px)
    assert numpy.max(numpy.abs(_thetas-thetas)) < SMALL_NUM

    rayOrigin = [0.5, 0.5, 100]
    x = 0.5
    y = 0.5
    z = 1
    px, py, pz, ps = proj2XYplane(x,y,z, rayOrigin)
    assert numpy.abs(px-x) < SMALL_NUM
    assert numpy.abs(py-y) < SMALL_NUM
    assert numpy.abs(pz) < SMALL_NUM

    rayOrigin = [-1, 0, 2]
    x = 0
    y = 0
    z = 1

    px, py, pz, ps = proj2XYplane(x,y,z, rayOrigin)
    assert numpy.abs(px-1) < SMALL_NUM
    assert numpy.abs(py) < SMALL_NUM
    assert numpy.abs(pz) < SMALL_NUM


def test_xyProj2():
    xExpect = 1
    yExpect = 0
    zExpect = 0
    projExpect = numpy.sqrt(2*0.5**2)
    rayOrigin = numpy.array([0,0,1])
    x, y, z = 0.5, 0, 0.5
    px, py, pz, pd = proj2XYplane(x,y,z, rayOrigin)

    assert numpy.abs(xExpect-px) < SMALL_NUM
    assert numpy.abs(yExpect-py) < SMALL_NUM
    assert numpy.abs(zExpect-pz) < SMALL_NUM
    assert numpy.abs(projExpect-pd) < SMALL_NUM


    x, y, z = 1.5, 0, -0.5
    px, py, pz, pd = proj2XYplane(x,y,z, rayOrigin)

    assert numpy.abs(xExpect-px) < SMALL_NUM
    assert numpy.abs(yExpect-py) < SMALL_NUM
    assert numpy.abs(zExpect-pz) < SMALL_NUM
    assert numpy.abs(-1*projExpect-pd) < SMALL_NUM

    x = [0.5, 1.5]
    y = [0, 0]
    z = [0.5, -0.5]

    px, py, pz, pd = proj2XYplane(x,y,z, rayOrigin)

    assert numpy.abs(xExpect - px[0]) < SMALL_NUM
    assert numpy.abs(yExpect - py[0]) < SMALL_NUM
    assert numpy.abs(zExpect - pz[0]) < SMALL_NUM
    assert numpy.abs(projExpect - pd[0]) < SMALL_NUM

    assert numpy.abs(xExpect - px[1]) < SMALL_NUM
    assert numpy.abs(yExpect - py[1]) < SMALL_NUM
    assert numpy.abs(zExpect - pz[1]) < SMALL_NUM
    assert numpy.abs(-1 * projExpect - pd[1]) < SMALL_NUM
