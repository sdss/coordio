import numpy

from .exceptions import CoordIOError


def unitDiskify(x, y, scaleR=None):
    """
    Convert x y to unit disk, return the radial scale factor

    # note scaleR is the max radius of the disk in original units
    # divide by this to put things on a unit disk...
    """
    r = numpy.sqrt(x**2 + y**2)
    theta = numpy.arctan2(y, x)
    if scaleR is None:
        scaleR = numpy.max(r)
    _r = r / scaleR
    _x = numpy.cos(theta) * _r
    _y = numpy.sin(theta) * _r
    return _x, _y, scaleR


def unDiskify(x, y, scaleR):
    """
    Convert x y to unit disk, return the radial scale factor
    """
    r = numpy.sqrt(x**2 + y**2) * scaleR
    theta = numpy.arctan2(y, x)

    _x = numpy.cos(theta) * r
    _y = numpy.sin(theta) * r
    return _x, _y


def _orthoZern(x, y, zernOrder=20):
    """
    Return zernike gradient in x and y, zernOrder is the order of the polynomial.


    transcribed from: https://doi.org/10.1364/JOSAA.35.000840
    typo was fixed by author of paper (s#17 and a few more were wrong)
    but private correspondence fixed that.  Orthonormalized zernike polynomials
    in x/y.

    Parameters
    -----------
    x : float
        unit disk normalized x coordinate
    y: float
        unit disk normalized y coordinate
    zernOrder : int
        number of zernike coefficients to return in range (1-20)

    Returns
    ---------
    result : 2 element numpy.ndarray
        xy vector
    """

    out = numpy.array([
        [1, 0],  # s2
        [0, 1],  # s3
        numpy.sqrt(3) * numpy.array([x, y]),  # s4
        numpy.sqrt(3) * numpy.array([y, x]),  # s5
        numpy.sqrt(3) * numpy.array([x, -y]),  # s6
        numpy.array([6 * x * y, 3 * x**2 + 9 * y**2 - 2]) / numpy.sqrt(3), # s7
        numpy.array([9 * x**2 + 3 * y**2 - 2, 6 * x * y]) / numpy.sqrt(3), # s8

        # s9
        numpy.array([
            12 * x * y,
            6 * x**2 - 12 * y**2 + 1
        ]) / (2 * numpy.sqrt(2)),

        # s10
        numpy.array([12 * x**2 - 6 * y**2 - 1, -12 * x * y]) / numpy.sqrt(8),

        # s11
        numpy.sqrt(21 / 62) * numpy.array([x, y]) * (15 * x**2 + 15 * y**2 - 7),

        # s12
        numpy.sqrt(7) * numpy.array([
            x * (10 * x**2 - 3),
            y * (3 - 10 * y**2)
        ]) / 2,

        # s13
        numpy.sqrt(21 / 38) * numpy.array([
            x * (15 * x**2 + 5 * y**2 - 4),
            y * (5 * x**2 + 15 * y**2 - 4)
        ]),

        # s14
        numpy.array([
            x * (35 * x**2 - 27 * y**2 - 6),
            y * (-27 * y**2 + 35 * x**2 - 6)
        ]) / numpy.sqrt(5/62),

        # s15
        numpy.sqrt(35 / 3) * numpy.array([
            y * (3 * x**2 - y**2),
            x * (x**2 - 3 * y**2)
        ]),

        # s16
        numpy.array([
            315 * (x**2 + y**2) * (5 * x**2 + y**2) - 30 * (33 * x**2 + 13 * y**2) + 83,
            60 * x * y * (21 * (x**2 + y**2) - 13)
        ]) / (2 * numpy.sqrt(1077)),

        # s17
        numpy.array([
            60 * x * y * (21 * (x**2 + y**2) - 13),
            315 * (x**2 + y**2) * (x**2 + 5 * y**2) - 30 * (13 * x**2 + 33 * y**2) + 83
        ]) / (2 * numpy.sqrt(1077)),

        # s18
        3 * numpy.array([
            140 * (860 * x**4 - 45 * x**2 * y**2 - 187 * y**4) - 30 * (1685 * x**2 - 522 * y**2) + 1279,
            -40 * x * y * (105 * x**2 + 2618 * y**2 - 783)
        ]) / (2 * numpy.sqrt(2489214)),

        # s19
        3 * numpy.array([
            40 * x * y * (2618 * x**2 + 105 * y**2 - 783),
            140 * (187 * x**4 + 45 * x**2 * y**2 - 860 * y**4) - 30 * (522 * x**2 - 1685 * y**2) - 1279
        ]) / (2 * numpy.sqrt(2489214)),

        #  s20
        (1 / 16) * numpy.sqrt(7 / 13557143) * numpy.array([
            60 * (10948 * x**4 - 7830 * x**2 * y**2 + 2135 * y**4 - 3387 * x**2 - 350 * y**2) + 11171,
            -1200 * x * y * (261 * x**2 - 427 * y**2 + 35)
        ]),

        # s21
        (1 / 16) * numpy.sqrt(7 / 13557143) * numpy.array([
            1200 * x * y * (427 * x**2 - 261 * y**2 + 35),
            60 * (2135 * x**4 - 7830 * x**2 * y**2 + 10948 * y**4 - 350 * x**2 - 3387 * y**2) + 11171
        ])
    ])

    # note computing everything up to 20 (the max coeffs provided)
    # but only returning the order requested
    out = out[:zernOrder]

    return out


def orthoZern(x, y, zernOrder=20):
    """
    Compute the orthonormailzed zernike gradients in x and y,
    zernOrder is the order of the polynomial.


    transcribed from: https://doi.org/10.1364/JOSAA.35.000840
    typo was fixed by author of paper (s#17 and a few more were wrong)
    but private correspondence fixed that.  Orthonormalized zernike polynomials
    in x/y.

    Parameters
    -----------
    x : numpy.ndarray
        unit disk normalized x coordinate
    y: numpy.ndarray
        unit disk normalized y coordinate
    zernOrder : int
        number of zernike coefficients to return in range (1-20)

    Returns
    ---------
    dUdx : numpy.ndarray
        gradient in x direction, matrix shape n x zernOrder
    dUdy : numpy.ndarray
        gradient in y direction, matrix shape n x zernOrder
    """

    maxRad = numpy.max(numpy.sqrt(x**2 + y**2))
    if maxRad > 1:
        raise CoordIOError("x y coords must be on unit disk")

    if zernOrder > 20:
        raise CoordIOError("maximum zernike order is 20")

    nPts = len(x)
    dx = numpy.zeros((nPts, zernOrder))
    dy = numpy.zeros((nPts, zernOrder))
    for ii, (_x, _y) in enumerate(zip(x, y)):
        dxy = _orthoZern(_x, _y, zernOrder)
        dx[ii, :] = dxy[:, 0]
        dy[ii, :] = dxy[:, 1]

    return dx, dy


def gradZern(x, y, zernOrder=20):
    """
    Computes zernike polynomial value and gradients in x/y.

    This routine isn't orthonormalized so seems to perform not quite as well
    as the orthoZern routine.  However this routine allows one to fit an
    arbitrarily large number of coefficients, while the other routine only
    allows 20.

    from https://doi.org/10.1364/OE.26.018878
     Pseudo-code to calculate unit-normalized Zernike polynomials and their x,y derivatives

       Numbering scheme:
       Within a radial order, sine terms come first
               ...
             sin((n-2m)*theta)   for m = 0,..., [(n+1)/2]-1
               ...
                1                for n even, m = n/2
               ...
             cos((n-2m)*theta)   for m = [n/2]+1,...,n
               ...

       INPUT:
       x, y normalized (x,y) coordinates in unit circle
       zernOrder: Maximum Zernike radial order

       OUTPUT:
       Zern[...]   array to receive value of each Zernike polynomium at (x,y)
       dUdx[...]   array to receive each derivative dU/dx at (x,y)
       dUdy[...]   array to receive each derivative dU/dy at (x,y)

    Parameters
    -----------
    x : float or numpy.ndarray
        unit disk normalized x coordinate
    y: float or numpy.ndarray
        unit disk normalized y coordinate
    zernOrder : int
        number of radial orders (total orders will be greater!)

    Returns
    --------
    dUdx : numpy.ndarray
        gradient in x direction, matrix shape n x totalOrder
        where totalOrder > zernOrder
    dUdy : numpy.ndarray
        gradient in y direction, matrix shape n x totalOrder
        where totalOrder > zernOrder

    """
    maxRad = numpy.max(numpy.sqrt(x**2 + y**2))
    if maxRad > 1:
        raise CoordIOError("x y coords must be on unit disk")

    nTerms = numpy.sum(numpy.arange(zernOrder + 2)) + 1
    Zern = numpy.zeros(nTerms)
    dUdx = numpy.zeros(nTerms)
    dUdy = numpy.zeros(nTerms)

    if hasattr(x, "__len__"):
        # x, y are vectors
        Zern = numpy.array([Zern] * len(x)).T
        dUdx = numpy.array([dUdx] * len(x)).T
        dUdy = numpy.array([dUdy] * len(x)).T

    # double x, y, Zern[*], dUdx[*], dUdy[*]

    # int nn, mm, kndx, jbeg, jend, jndx, even, nn1, nn2
    # int jndx1, jndx11, jndx2, jndx21
    # double pval, qval

    # pseudocode is 1 indexed, ugh, stick with it
    # and modify later

    Zern[1] = 1.                                  # (0,0)
    dUdx[1] = 0.                                  # (0,0)
    dUdy[1] = 0.                                  # (0,0)

    Zern[2] = y                                   # (1,0)
    Zern[3] = x                                   # (1,1)
    dUdx[2] = 0.                                  # (1,0)
    dUdx[3] = 1.                                  # (1,1)
    dUdy[2] = 1.                                  # (1,0)
    dUdy[3] = 0.                                  # (1,1)

    kndx = 1                # index for term from 2 orders down
    jbeg = 2                # start index for current radial order
    jend = 3                # end index for current radial order
    jndx = 3                # running index for current Zern
    even = -1
    #  Outer loop in radial order index
    for nn in range(2, zernOrder + 1):  # 1 indexed
        even = -even          # parity of radial index
        jndx1 = jbeg           # index for 1st ascending series in x
        jndx2 = jend           # index for 1st descending series in y
        jndx11 = jndx1 - 1      # index for 2nd ascending series in x
        jndx21 = jndx2 + 1      # index for 2nd descending series in y
        jbeg = jend + 1       # end of previous radial order +1
        nn2 = nn // 2
        nn1 = (nn - 1) // 2
        #  Inner loop in azimuthal index
        for mm in range(0, nn + 1):  # 1 indexed
            jndx += 1                  # increment running index for current Zern

            if (mm == 0):
                Zern[jndx] = x * Zern[jndx1] + y * Zern[jndx2]
                dUdx[jndx] = Zern[jndx1] * nn
                dUdy[jndx] = Zern[jndx2] * nn

            elif (mm == nn):
                Zern[jndx] = x * Zern[jndx11] - y * Zern[jndx21]
                dUdx[jndx] = Zern[jndx11] * nn
                dUdy[jndx] = -Zern[jndx21] * nn

            elif ((even > 0) and (mm == nn2)):              # logical “AND”
                Zern[jndx] = 2. * (x * Zern[jndx1] + y * Zern[jndx2]) - Zern[kndx]
                dUdx[jndx] = 2. * nn * Zern[jndx1] + dUdx[kndx]
                dUdy[jndx] = 2. * nn * Zern[jndx2] + dUdy[kndx]
                kndx += 1                        # increment kndx

            elif ((even < 0) and (mm == nn1)):              # logical “AND”
                qval = Zern[jndx2] - Zern[jndx21]
                Zern[jndx] = x * Zern[jndx11] + y * qval - Zern[kndx]
                dUdx[jndx] = Zern[jndx11] * nn + dUdx[kndx]
                dUdy[jndx] = qval * nn + dUdy[kndx]
                kndx += 1                        # increment kndx

            elif ((even < 0) and (mm == nn1 + 1)):            # logical “AND”
                pval = Zern[jndx1] + Zern[jndx11]
                Zern[jndx] = x * pval + y * Zern[jndx2] - Zern[kndx]
                dUdx[jndx] = pval * nn + dUdx[kndx]
                dUdy[jndx] = Zern[jndx2] * nn + dUdy[kndx]
                kndx += 1                        # increment kndx

            else:
                pval = Zern[jndx1] + Zern[jndx11]
                qval = Zern[jndx2] - Zern[jndx21]
                Zern[jndx] = x * pval + y * qval - Zern[kndx]
                dUdx[jndx] = pval * nn + dUdx[kndx]
                dUdy[jndx] = qval * nn + dUdy[kndx]
                kndx += 1                        # increment kndx

            jndx11 = jndx1                   # update indices
            jndx1 += 1
            jndx21 = jndx2
            jndx2 -= 1
            # End of inner azimuthal loop

        jend = jndx
        # print("jend", jend)
        # End of outer radial order loop

    # throw out first term (zero indexing didn't populate it)
    Zern = Zern[1:]  # not returned but computed if we want it later?
    dUdx = dUdx[1:]
    dUdy = dUdy[1:]

    return dUdx.T, dUdy.T


class ZernFit(object):
    def __init__(self, xMeas, yMeas, xExpect,
                 yExpect, orders=20, method="ortho", scaleR=None):
        """
        Basic least squares fitter for zernike gradients.  Translation,
        rotation and scale are expected to be removed from x/y Meas prior to
        zernike fitting

        Parameters
        ------------
        xMeas : numpy.ndarray
            x measurements
        yMeas : numpy.ndarray
            y measurements
        xExpect : numpy.ndarray
            expected (true) x coordinates
        yExpect : numpy.ndarray
            expected (true) y coordinates
        orders : int
            number of orders to include in fit. If method == "grad" these are
            radial order of zernike's, so the total number of coeffs will be >
            input orders.
        method : str
            either "ortho" or "grad".  Ortho uses method from:
            https://doi.org/10.1364/JOSAA.35.000840
            Grad uses method from:
            https://doi.org/10.1364/OE.26.018878
        scaleR : float or None
            if not None, apply this radial scale to xyMeas input
            to put coords on unit disk
        """
        self.orders = orders
        self.xMeas = xMeas
        self.yMeas = yMeas
        self.xExpect = xExpect
        self.yExpect = yExpect
        self.xErr = xExpect - xMeas
        self.yErr = yExpect - yMeas

        # force all measurements to unit disk
        # scale expected locations by same factor
        self.xMeasUnit, self.yMeasUnit, self.rScale = unitDiskify(
            xMeas, yMeas, scaleR
        )
        self.xExpectUnit, self.yExpectUnit, _junk = unitDiskify(
            xExpect, yExpect, self.rScale
        )
        self.xErrUnit = self.xExpectUnit - self.xMeasUnit
        self.yErrUnit = self.yExpectUnit - self.yMeasUnit

        if method == "ortho":
            self._zernFunc = orthoZern
        elif method == "grad":
            self._zernFunc = gradZern
        else:
            raise CoordIOError("method must be either ortho or grad")

        # zx/zy are shape n x orders
        self.zx, self.zy = self._zernFunc(
            self.xMeasUnit, self.yMeasUnit, self.orders
        )

        # stack them and do the fit
        self.zxyStack = numpy.vstack((self.zx, self.zy))
        self.exyStack = numpy.hstack((self.xErrUnit, self.yErrUnit))

        # solve Ax=b for x (the coefficients)
        self.coeff, resid, rank, s = numpy.linalg.lstsq(
            self.zxyStack, self.exyStack
        )

    def apply(self, xMeas, yMeas):
        # apply the fit to measured data
        xMeasUnit, yMeasUnit, _junk = unitDiskify(xMeas, yMeas, self.rScale)
        zx, zy = self._zernFunc(xMeasUnit, yMeasUnit, self.orders)
        dx = zx @ self.coeff
        dy = zy @ self.coeff

        xFitUnit = xMeasUnit + dx
        yFitUnit = yMeasUnit + dy

        # recale back up from unit disk
        xFit, yFit = unDiskify(xFitUnit, yFitUnit, self.rScale)
        return xFit, yFit


