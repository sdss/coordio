import ctypes
import importlib
import os

import numpy
import pandas
from scipy.interpolate import interp1d

from . import defaults
from .site import Site
from .sky import ICRS, Observed
from .telescope import Field, FocalPlane
from .wok import Wok

# Get simplexy C function
mod_path = os.path.join(os.path.dirname(__file__), 'libdimage')
success = False
for suffix in importlib.machinery.EXTENSION_SUFFIXES:
    try:
        libPath = mod_path + suffix
        if os.path.exists(libPath):
            success = True
            break
    except OSError:
        pass
if success is False:
    raise OSError('Could not find a valid libdimage extension.')


libdimage = ctypes.cdll.LoadLibrary(libPath)
# simplexy_function = libdimage.simplexy


def wokCurveAPO(r):
    """Curve of the wok at APO at radial position r

    Parameters
    -----------
    r : scalar or 1D array
        radius (cylindrical coords) mm

    Returns:
    ---------
    result : scalar or 1D array
        z (height) of wok surface in mm (0 at vertex)


    Curve that was specified for machining the APO wok, decided by Colby J.
    """
    A = 9199.322517101522
    return A - numpy.sqrt(A**2 - r**2)


def wokCurveLCO(r):
    """Curve of the wok LCO at radial position r

    Parameters
    -----------
    r : scalar or 1D array
        radius (cylindrical coords) mm

    Returns:
    ---------
    result : scalar or 1D array
        z (height) of wok surface in mm (0 at vertex)

    Curve that was specified for machining the LCO wok, decided by Colby J.
    """
    A = 0.000113636363636
    B = 0.0000000129132231405
    C = 0.0000012336318
    return A * r**2 / (1 + numpy.sqrt(1 - B * r**2)) + C * r**2


def radec2wokxy(ra, dec, coordEpoch, waveName, raCen, decCen, obsAngle,
                obsSite, obsTime, focalScale=None, pmra=None, pmdec=None, parallax=None,
                radVel=None, pressure=None, relativeHumidity=0.5,
                temperature=10, fullOutput=False, darLambda=None):
    r"""
    Convert from ra, dec ICRS coords to a flat-wok XY in mm.  At obsAngle=0
    wok +y is a aligned with +dec, and wok +x is aligned with +ra

    Question for José, do we need to enforce a time scale?  I think everything
    defaults to UTC.

    Parameters
    ------------
    ra : numpy.ndarray
        Right ascension in ICRS, degrees
    dec : numpy.ndarray
        Declination in ICRS, degrees
    coordEpoch : float
        A TDB Julian date, the epoch for the input
        coordinates (from the catalog). Defaults to J2000.
    waveName : str, or numpy.ndarray
        Array elements must be "Apogee", "Boss", or "GFA" strings
    raCen : float
        Right ascension of field center, in degrees
    decCen : float
        Declination of field center, in degrees.
    obsAngle : float
        Position angle for observation.  Angle is measured from North
        through East to wok +y. So obsAngle of 45 deg, wok +y points NE.
    obsSite : str
        Either "APO" or "LCO"
    obsTime : float
        TDB Julian date.  The time at which these coordinates will be observed
        with the FPS.
    focalScale : float or None
        Scale factor for conversion between focal and wok coords, found
        via dither analysis.  Defaults to value in defaults.SITE_TO_SCALE
    pmra : numpy.ndarray
        A 1D array with the proper motion in the RA axis for the N targets,
        in milliarcsec/yr. Must be a true angle, i.e, it must include the
        ``cos(dec)`` term.  Defaults to 0.
    pmdec : numpy.ndarray
        A 1D array with the proper motion in the RA axis for the N targets,
        in milliarcsec/yr.  Defaults to 0.
    parallax : numpy.ndarray
        A 1D array with the parallax for the N targets, in milliarcsec.
        Defaults to 0.
    radVel : numpy.ndarray
        A 1D array with the radial velocity in km/s, positive when receding.
        Defaults to 0.
    pressure : float
        The atmospheric pressure at the site, in millibar (same as hPa). If
        not provided the pressure will be calculated using the altitude
        :math:`h` and the approximate expression

        .. math::

            p \sim -p_0 \exp\left( \dfrac{g h M} {T_0 R_0} \right)

        where :math:`p_0` is the pressure at sea level, :math:`M` is the molar
        mass of the air, :math:`R_0` is the universal gas constant, and
        :math:`T_0=288.16\,{\rm K}` is the standard sea-level temperature.
    relativeHumidity : float
        The relative humidity, in the range :math:`0-1`. Defaults to 0.5.
    temperature : float
        The site temperature, in degrees Celsius. Defaults to
        :math:`10^\circ{\rm C}`.
    fullOutput : Bool
        If True all intermediate coords are returned

    Returns
    ---------
    xWok : numpy.ndarray
        x wok coordinate, mm
    yWok : numpy.ndarray
        y wok coordinate, mm
    fieldWarn : numpy.ndarray
        boolean array.  Where True the coordinate converted should be eyed with
        suspicion.  (It came from very far off axis).
    hourAngle : float
        hour angle of field center in degrees
    positionAngle : float
        position angle of field center in degrees

    if fullOutput=True additionally returns:

    xFocal : numpy.ndarray
        x focal coordinate, mm
    yFocal : numpy.ndarray
        y focal coordinate, mm
    thetaField : numpy.ndarray
        azimuthal field coordinate, deg +RA through +Dec
    phiField : numpy.ndarray
        polar field coordinate, deg off axis
    altitude :  numpy.ndarray
        altitude coordinate, deg
    azimuth : numpy.ndarray
        azimuth coordinate N=0, E=90, deg
    altFieldCen : float
        altitude of field center, deg
    azFieldCen : float
        azimuth of field center, deg

    """
    nCoords = len(ra)

    # first grab the correct wavelengths for fibers
    wavelength = numpy.zeros(nCoords)

    if isinstance(waveName, str):
        # same wl for all coords
        wavelength += defaults.INST_TO_WAVE[waveName]
    else:
        assert len(waveName) == nCoords
        for ii, ft in enumerate(waveName):
            wavelength[ii] = defaults.INST_TO_WAVE[ft]

    site = Site(
        obsSite, pressure=pressure,
        temperature=temperature, rh=relativeHumidity
    )
    site.set_time(obsTime)

    # first determine the field center in observed coordinates
    # use the guide wavelength for field center
    # epoch not needed, no propermotions, etc (josé verify?)
    icrsCen = ICRS([[raCen, decCen]])
    obsCen = Observed(icrsCen, site=site, wavelength=defaults.INST_TO_WAVE["GFA"])

    radec = numpy.array([ra, dec]).T

    if pmra is not None:
        pmra[numpy.isnan(pmra)] = 0

    if pmdec is not None:
        pmdec[numpy.isnan(pmdec)] = 0

    if parallax is not None:
        parallax[numpy.isnan(parallax)] = 0

    icrs = ICRS(
        radec, epoch=coordEpoch, pmra=pmra, pmdec=pmdec,
        parallax=parallax, rvel=radVel
    )

    # propogate propermotions, etc
    icrs = icrs.to_epoch(obsTime, site=site)
    if focalScale is None:
        focalScale = defaults.SITE_TO_SCALE[obsSite]

    # check to see if darLambda was supplied, and if so
    # use that wavelength (just for DAR not telescope distortion model)
    darWavelength = wavelength.copy()
    if darLambda is not None:
        if hasattr(darLambda, "__len__"):
            assert len(darLambda) == nCoords
            darWavelength = darLambda
        else:
            darWavelength.fill(darLambda)
    obs = Observed(icrs, site=site, wavelength=darWavelength)
    field = Field(obs, field_center=obsCen)
    focal = FocalPlane(field,
                       wavelength=wavelength,
                       site=site,
                       fpScale=focalScale,
                       use_closest_wavelength=True)
    wok = Wok(focal, site=site, obsAngle=obsAngle)

    output = (
        wok[:, 0], wok[:, 1], focal.field_warn,
        float(obsCen.ha), float(obsCen.pa)
    )

    if fullOutput:
        output += (
            focal[:, 0], focal[:, 1],
            field[:, 0], field[:, 1],
            obs[:, 0], obs[:, 1],
            obsCen[0][0], obsCen[0][1]
        )
    return output


def wokxy2radec(xWok, yWok, waveName, raCen, decCen, obsAngle,
                obsSite, obsTime, focalScale=None, pressure=None, relativeHumidity=0.5,
                temperature=10):
    r"""
    Convert from flat-wok XY (mm) to ra, dec ICRS coords (deg)

    Question for José, do we need to enforce a time scale?  I think everything
    defaults to UTC.

    Parameters
    ------------
    xWok : numpy.ndarray
        X wok coordinates, mm
    yWok : numpy.ndarray
        Y wok coordinates, mm
    waveName : str, or numpy.ndarray
        Array elements must be "Apogee", "Boss", or "GFA" strings
    raCen : float
        Right ascension of field center, in degrees
    decCen : float
        Declination of field center, in degrees.
    obsAngle : float
        Position angle for observation.  Angle is measured from North
        through East to wok +y. So obsAngle of 45 deg, wok +y points NE.
    obsSite : str
        Either "APO" or "LCO"
    obsTime : float
        TDB Julian date.  The time at which these coordinates will be observed
        with the FPS.
    focalScale : float or None
        Scale factor for conversion between focal and wok coords, found
        via dither analysis.  Defaults to value in defaults.SITE_TO_SCALE
    pressure : float
        The atmospheric pressure at the site, in millibar (same as hPa). If
        not provided the pressure will be calculated using the altitude
        :math:`h` and the approximate expression

        .. math::

            p \sim -p_0 \exp\left( \dfrac{g h M} {T_0 R_0} \right)

        where :math:`p_0` is the pressure at sea level, :math:`M` is the molar
        mass of the air, :math:`R_0` is the universal gas constant, and
        :math:`T_0=288.16\,{\rm K}` is the standard sea-level temperature.
    relativeHumidity : float
        The relative humidity, in the range :math:`0-1`. Defaults to 0.5.
    temperature : float
        The site temperature, in degrees Celsius. Defaults to
        :math:`10^\circ{\rm C}`.

    Returns
    ---------
    xWok : numpy.ndarray
        x wok coordinate, mm
    yWok : numpy.ndarray
        y wok coordinate, mm
    fieldWarn : numpy.ndarray
        boolean array.  Where True the coordinate converted should be eyed with
        suspicion.  (It came from very far off axis).
    """
    nCoords = len(xWok)

    # first grab the correct wavelengths for fibers
    wavelength = numpy.zeros(nCoords)

    if isinstance(waveName, str):
        # same wl for all coords
        wavelength += defaults.INST_TO_WAVE[waveName]
    else:
        assert len(waveName) == nCoords
        for ii, ft in enumerate(waveName):
            wavelength[ii] = defaults.INST_TO_WAVE[ft]

    site = Site(
        obsSite, pressure=pressure,
        temperature=temperature, rh=relativeHumidity
    )
    site.set_time(obsTime)

    # first determine the field center in observed coordinates
    # use the guide wavelength for field center
    # epoch not needed, no propermotions, etc (josé verify?)
    icrsCen = ICRS([[raCen, decCen]])
    obsCen = Observed(icrsCen, site=site, wavelength=defaults.INST_TO_WAVE["GFA"])

    # hack in z wok position of 143 mm
    # could do a little better and estimate the curve of focal surface
    xyzWok = numpy.zeros((nCoords, 3))
    rWok = numpy.sqrt(xWok**2 + yWok**2)

    # this is not totally correct
    # but probably better than modeling as flat
    # project the xy positions onto the 3d surface
    # and add 143 (the positioner height)
    if obsSite == "APO":
        zWok = defaults.POSITIONER_HEIGHT + wokCurveAPO(rWok)
    else:
        zWok = defaults.POSITIONER_HEIGHT + wokCurveLCO(rWok)

    xyzWok[:, 0] = xWok
    xyzWok[:, 1] = yWok
    xyzWok[:, 2] = zWok

    wok = Wok(xyzWok, site=site, obsAngle=obsAngle)
    if focalScale is None:
        focalScale = defaults.SITE_TO_SCALE[obsSite]
    focal = FocalPlane(wok,
                       wavelength=wavelength,
                       site=site,
                       fpScale=focalScale,
                       use_closest_wavelength=True)
    field = Field(focal, field_center=obsCen)
    obs = Observed(field, site=site, wavelength=wavelength)
    icrs = ICRS(obs, epoch=obsTime)

    return icrs[:, 0], icrs[:, 1], field.field_warn


def fitsTableToPandas(recarray):
    d = {}
    if hasattr(recarray, "names"):
        names = recarray.names
    else:
        names = recarray.dtype.names
    for name in names:
        key = name
        if hasattr(key, "decode"):
            key = key.decode()
        d[key] = recarray[name].byteswap().newbyteorder()

    df = pandas.DataFrame(d)
    # decode binary data into strings, if present
    # https://stackoverflow.com/questions/40389764/how-to-translate-bytes-objects-into-literal-strings-in-pandas-dataframe-pytho
    # str_df = df.select_dtypes([numpy.object])
    # if len(str_df) > 0:
    #     str_df = str_df.stack().str.decode('utf-8').unstack()
    #     for col in str_df:
    #         df[col] = str_df[col]
    return df


def simplexy(image, psf_sigma=1., plim=8., dlim=1., saddle=3., maxper=1000,
             maxnpeaks=5000):
    """Determines positions of stars in an image.
    Parameters
    ----------
    image : numpy.float32
        2-D ndarray
    psf_sigma : float
        sigma of Gaussian PSF to assume (default 1 pixel)
    plim : float
        significance to select objects on (default 8)
    dlim : float
        tolerance for closeness of pairs of objects (default 1 pixel)
    saddle : float
        tolerance for depth of saddle point to separate sources
        (default 3 sigma)
    maxper : int
        maximum number of children per parent (default 1000)
    maxnpeaks : int
        maximum number of stars to find total (default 100000)
    Returns
    -------
    (x, y, flux) : (numpy.float32, numpy.float32, numpy.float32)
         ndarrays with pixel positions and peak pixel values of stars
    Notes
    -----
    Calls simplexy.c in libdimage.so
    copied directly from: https://github.com/blanton144/dimage
    """

    # Create image pointer
    if(image.dtype != numpy.float32):
        image_float32 = image.astype(numpy.float32)
        image_ptr = image_float32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        image_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    nx = image.shape[0]
    ny = image.shape[1]
    psf_sigma_ptr = ctypes.c_float(psf_sigma)
    plim_ptr = ctypes.c_float(plim)
    dlim_ptr = ctypes.c_float(dlim)
    saddle_ptr = ctypes.c_float(saddle)
    maxper_ptr = ctypes.c_int(maxper)
    maxnpeaks_ptr = ctypes.c_int(maxnpeaks)

    x = numpy.zeros(maxnpeaks, dtype=numpy.float32)
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    y = numpy.zeros(maxnpeaks, dtype=numpy.float32)
    y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    flux = numpy.zeros(maxnpeaks, dtype=numpy.float32)
    flux_ptr = flux.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sigma = ctypes.c_float(0.)
    npeaks = ctypes.c_int(0)

    libdimage.simplexy(
        image_ptr, nx, ny, psf_sigma_ptr, plim_ptr,
        dlim_ptr, saddle_ptr, maxper_ptr, maxnpeaks_ptr,
        ctypes.byref(sigma), x_ptr, y_ptr, flux_ptr,
        ctypes.byref(npeaks)
    )

    npeaks = npeaks.value
    x = x[0:npeaks]
    y = y[0:npeaks]
    flux = flux[0:npeaks]

    return (x, y, flux)


def refinexy(image, x, y, psf_sigma=2., cutout=19):
    """Refines positions of stars in an image.
    Parameters
    ----------
    image : numpy.float32
        2-D ndarray
    x : numpy.float32
        1-D ndarray of rough x positions
    y : numpy.float32
        1-D ndarray of rough y positions
    psf_sigma : float
        sigma of Gaussian PSF to assume (default 2 pixels)
    cutout : int
        size of cutout used, should be odd (default 19)
    Returns
    -------
    xr : ndarray of numpy.float32
        refined x positions
    yr : ndarray of numpy.float32
        refined y positions
    Notes
    -----
    Calls drefine.c in libdimage.so
    copied directly from: https://github.com/blanton144/dimage
    """

    # Create image pointer
    if(image.dtype != numpy.float32):
        image_float32 = image.astype(numpy.float32)
        image_ptr = image_float32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        image_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    nx = image.shape[0]
    ny = image.shape[1]
    psf_sigma_ptr = ctypes.c_float(psf_sigma)

    ncen = len(x)
    ncen_ptr = ctypes.c_int(ncen)
    cutout_ptr = ctypes.c_int(cutout)
    xrough = numpy.float32(x)
    xrough_ptr = xrough.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    yrough = numpy.float32(y)
    yrough_ptr = yrough.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    xrefined = numpy.zeros(ncen, dtype=numpy.float32)
    xrefined_ptr = xrefined.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    yrefined = numpy.zeros(ncen, dtype=numpy.float32)
    yrefined_ptr = yrefined.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    libdimage.drefine(
        image_ptr, nx, ny,
        xrough_ptr, yrough_ptr, xrefined_ptr, yrefined_ptr,
        ncen_ptr, cutout_ptr, psf_sigma_ptr
    )

    return (xrefined, yrefined)


class MoffatLossProfile(object):
    """
    class for calculating fiber loss based on Moffat profile

    Parameters
    ----------
    offset: numpy.array
        fiber offset

    beta: float
        Power index of the Moffat profile

    FWHM: float
        seeing

    rfiber: float
        radius of the fiber
    """
    def __init__(self, offset, beta, FWHM, rfiber=1):
        self.offset = offset
        self.beta = beta
        self.FWHM = FWHM
        self.alpha = self.FWHM / (2. * numpy.sqrt(2 ** (1 / self.beta) - 1))
        self.amplitude = 1. / self.moffat_norm(1.)  # this is the amplitude needed to normalize the Moffat profile
        self.rfiber = rfiber

    def moffat_1D(self, r, amplitude=None):  # unit flux at the center
        """
        Function computing Moffat 1D profile

        Parameters
        ------------
        r: float or numpy.array
            Distance from the centre of the profile

        Returns
        -------
        moff_prof: float or numpy.array
            1D Moffat profile.
        """
        if amplitude is None:
            moff_prof = self.amplitude * (1 + (r / self.alpha) ** 2) ** (-self.beta)
        else:
            moff_prof = amplitude * (1 + (r / self.alpha) ** 2) ** (-self.beta)
        return moff_prof

    def moffat_norm(self, amplitude):
        """
        Function computing the normalized the Moffat 1D profile.

        Parameters
        -----------
        amplitude: float
            Amplitude of the profile

        Returns
        -------
        moff_prof_norm: float
            Normalised Moffat profile.
        """

        xmin, xmax, xstep = -7., 7., 0.05  # fixed ln radius steps
        # x is the ln radius in units of the r / alpha
        steps = numpy.arange(xmin, xmax, xstep)
        r = self.alpha * numpy.exp(steps)
        norm = numpy.sum(numpy.exp(2 * steps) * self.moffat_1D(r, amplitude=amplitude))
        moff_prof_norm = 2 * numpy.pi * self.alpha ** 2 * norm * xstep
        return moff_prof_norm

    def flux_loss(self, offset):
        """
        Function computing the flux loss obtained by moving the fiber
        across the source in a certain direction.

        Prameters
        ---------
        offset: numpy.array
            fiber offset

        Returns
        --------
        norm: numpy.array
            the flux loss
        """
        x = numpy.arange(-self.rfiber, self.rfiber, self.rfiber / 50)    # set up a 100x100 Cartesian grid across the fiber
        y = numpy.arange(-self.rfiber, self.rfiber, self.rfiber / 50)
        X, Y = numpy.meshgrid(x, y)
        r = numpy.sqrt(X **2 + Y ** 2)
        norm = numpy.zeros(len(offset))
        r_ev = (r <= self.rfiber)
        for i in range(len(offset)):
            r_moffat = numpy.sqrt((X - offset[i]) ** 2 + Y ** 2)
            norm[i] = numpy.sum((self.rfiber / 50) ** 2 *
                                self.moffat_1D(r_moffat[r_ev]))
        return norm

    def func_magloss(self):
        """
        Function computing the magnitude loss obtained by moving the fiber
        across the source in a certain direction. Note that the magnitude
        loss with offset = 0 is not 0, and should coincide with the difference
        between PSF and fiber magnitudes.

        Prameters
        ----------

        Returns
        --------
        magloss: numpy.array
            the magnitude loss
        """
        magloss = numpy.zeros(len(self.offset))
        magloss = -2.5 * numpy.log10(self.flux_loss(self.offset)) \
                   + 2.5 * numpy.log10(self.flux_loss(numpy.array([0.]))[0])
        return magloss


class Moffat2dInterp(object):
    """
    Create the dict of 1D interpolations function
    for a moffat profile offset for various FWHMs.
    Object returned includes two dicts, one for APO
    and one for LCO for the various fiber sizes.
    """
    def __init__(self, Noffset=None, FWHM=None, beta=None):
        if Noffset is None:
            Noffset = 1500
        if FWHM is None:
            FWHM = [1., 1.3, 1.5, 1.7, 1.9]
        if beta is None:
            beta = {'APO': 5., 'LCO': 2.}
        rfibers = {'APO': 1., 'LCO': 1.33 / 2}
        offsets = numpy.zeros((len(FWHM), Noffset))
        FWHMs = numpy.zeros((len(FWHM), Noffset))
        for i, f in enumerate(FWHM):
            FWHMs[i, :] = f
            offsets[i, :] = numpy.linspace(0, 30, Noffset)

        magloss = numpy.zeros((FWHMs.shape[0], Noffset))

        fmagloss = {}
        for obs, rfiber in zip(rfibers.keys(), rfibers.values()):
            fmagloss[obs] = {}
            if isinstance(beta, dict):
                b = beta[obs]
            else:
                b = beta
            for i, f in enumerate(FWHMs[:, 0]):
                magloss[i, :] = MoffatLossProfile(offsets[i, :], b, f, rfiber=rfiber).func_magloss()
                fmagloss[obs][f] = interp1d(magloss[i, :], offsets[i, :])
        self.fmagloss = fmagloss
        self.beta_interp2d = beta
        self.FWHM_interp2d = FWHM

    def __call__(self, magloss, FWHM, obsSite):
        """
        The cal to return the offset value based on the desired
        magloss.

        Parameters
        -----------
        magloss: float or numpy.array
            The desired magnitude loss for the object(s)

        FWHM: float
            The FWHM for the moffat profile that has been calculated
            on the init of the object

        obsSite: str
            The observatory of the observation. Should either be
            'APO' or 'LCO'.

        Returns
        -------
        r: float or numpy.array
            The offset to get the desired magloss in arcseconds.
        """
        r = self.fmagloss[obsSite][FWHM](magloss)
        return r


def offset_definition(mag, mag_limits, lunation, waveName, obsSite, fmagloss=None,
                      safety_factor=0., beta=None, FWHM=None, skybrightness=None,
                      offset_min_skybrightness=None, can_offset=None,
                      use_type='bright_neigh', mag_limit_ind=None):
    """
    Returns the offset needed for object with mag to be
    observed at mag_limit.
    This is for the piecewise appromixation used based on:
    https://wiki.sdss.org/pages/viewpage.action?pageId=100173069

    Parameters
    ----------
    mag: float or numpy.array
        The magniutde(s) of the objects. For BOSS should be
        Gaia G-band and for APOGEE should be 2MASS H-band.

    mag_limits: numpy.array
        Magnitude limits for the designmode of the design.
        This should be an array of length N=10 where indexes
        correspond to magntidues: [g, r, i, z, bp, gaia_g, rp, J, H, K].
        This matches the apogee_bright_limit_targets_min or
        boss_bright_limit_targets_min (depending on instrument) from
        targetdb.DesignMode for the design_mode of the design.

    lunation: str:
        If the designmode is bright time ('bright') or dark
        time ('dark')

    waveName: str
        Instrument for the fibers offset definition being applied
        to. Either 'Boss' or 'Apogee'.

    obsSite: str
        The observatory of the observation. Should either be
        'APO' or 'LCO'.

    fmagloss: object
        Moffat2dInterp class with the lookup table
        for doing the moffat profile inversion. If None,
        then table is calculated at function call.

    safety_factor: float
        Factor to add to mag_limit. Should equal zero for
        bright neighbor checks (i.e. remain at default).

    beta: float
        Power index of the Moffat profile. If None, default set in code.

    FWHM: float
        seeing for the Moffat profile. If None, default set in code.

    skybrightness: float
        Sky brightness for the field cadence. Only set if
        want to check for offset_flag TOO_DARK (8).

    offset_min_skybrightness: float
        Minimum sky brightness for the offset. Only set if
        want to check for offset_flag TOO_DARK (8).

    can_offset: boolean or numpy.array
        can_offset value from targetdb for the target(s) to be
        offset. Only set if
        want to check for offset_flag NO_CAN_OFFSET (16).

    use_type: str
        Defines type for definition use. 'bright_neigh' is for bright
        neighbor check and auto sets mag_limit from mag_limits array.
        'offfset' is for offsetting and the index from mag_limits array
        is set by mag_limit_ind.

    mag_limit_ind: int
        when used with use_type='offfset', then this sets what index
        from mag_limits array is set as the magnitude limit. Otherwise,
        for bright neighbor check this is set in code.

    Returns
    -------
    r: float or numpy.array
        offset radius in arcseconds around object(s)

    offset_flag: int or numpy.array
        bitmask for how offset was set. Flags are:
            - 0: offset applied normally (i.e. when mag <= mag_limit)
            - 1: no offset applied because mag > mag_limit
            - 2: no offset applied because magnitude was null value.
            - 8: No offset applied because sky brightness is <=
                 minimum offset sky brightness
            - 16: no offsets applied because can_offset = False
            - 32: no offset applied because mag <= offset_bright_limit
                  (offset_bright_limit is G = 6 for Boss bright time and
                   G = 13 for Boss dark time, and
                   H = 1 for Apogee).
    """
    # define Null cases for targetdb.magnitude table
    cases = [-999, -9999, 999,
             0.0, numpy.nan, 99.9, None]
    # set magntiude limit for instrument and lunation
    if use_type == 'bright_neigh':
        if waveName == 'Apogee':
            # 2MASS H
            mag_limit = mag_limits[8]
        elif lunation == 'bright':
            # Gaia G
            mag_limit = mag_limits[5]
        else:
            # SDSS r
            mag_limit = mag_limits[1]

        # for bright_neigh, need exclusion radius for all stars
        offset_bright_limit = -9999.
    else:
        mag_limit = mag_limits[mag_limit_ind]

        # only set real mag_limits for offsets
        if waveName == 'Boss':
            if lunation == 'bright':
                offset_bright_limit = 6.
            else:
                offset_bright_limit = 13.
        else:
            offset_bright_limit = 1.
    # get magloss function
    if fmagloss is None:
        fmagloss = Moffat2dInterp(beta=beta, FWHM=[FWHM])
    # assign correct FWHM
    if FWHM is None:
        if obsSite == 'APO':
            FWHM = 1.7
        elif obsSite == 'LCO':
            FWHM = 1.
        else:
            raise ValueError('obsSite must be APO or LCO.')
    if beta is None:
        beta = {'APO': 5., 'LCO': 2.}
    if isinstance(mag, float) or isinstance(mag, int):
        # make can_offset always True if not supplied
        if can_offset is None:
            can_offset = True
        offset_flag = 0
        if mag <= mag_limit and mag not in cases and can_offset and mag > offset_bright_limit:
            # linear portion in the wings
            r_wings = ((mag_limit + safety_factor) - mag - 8.2) / 0.05
            # linear portion in transition area
            r_trans = ((mag_limit + safety_factor) - mag - 4.5) / 0.25
            # core area
            if beta != fmagloss.beta_interp2d or FWHM not in fmagloss.FWHM_interp2d:
                fmagloss = Moffat2dInterp(beta=beta, FWHM=[FWHM])
                r_core = fmagloss((mag_limit + safety_factor) - mag, FWHM, obsSite)
            else:
                r_core = fmagloss((mag_limit + safety_factor) - mag, FWHM, obsSite)
            # tom's old conservative core function
            # r_core = 1.5 * ((mag_limit + safety_factor) - mag) ** 0.8
            # exlusion radius is the max of each section
            r = numpy.nanmax([r_wings, r_trans, r_core])
        else:
            r = 0.
            if mag > mag_limit:
                offset_flag += 1
            elif mag in cases:
                offset_flag += 2
            elif not can_offset:
                offset_flag += 16
            else:
                offset_flag += 32
        if skybrightness is not None and offset_min_skybrightness is not None:
            if skybrightness <= offset_min_skybrightness:
                offset_flag += 8
                r = 0.
    else:
        # make can_offset always True if not supplied
        if can_offset is None:
            can_offset = numpy.zeros(mag.shape, dtype=bool) + True
        # create empty arrays for each portion
        r_wings = numpy.zeros(mag.shape)
        r_trans = numpy.zeros(mag.shape)
        r_core = numpy.zeros(mag.shape)
        # only do calc for valid mags and can_offsets for offset
        # to avoid warning
        mag_valid = ((mag <= mag_limit) & (~numpy.isin(mag, cases)) &
                     (~numpy.isnan(mag)) & (can_offset) & (mag > offset_bright_limit))
        # set flags
        offset_flag = numpy.zeros(mag.shape, dtype=int)
        offset_flag[mag > mag_limit] += 1
        offset_flag[(numpy.isin(mag, cases)) | (numpy.isnan(mag))] += 2
        offset_flag[~can_offset] += 16
        offset_flag[(mag <= offset_bright_limit) & ~((numpy.isin(mag, cases)) | (numpy.isnan(mag)))] += 32
        # linear portion in the wings
        r_wings[mag_valid] = ((mag_limit + safety_factor) - mag[mag_valid] - 8.2) / 0.05
        # linear portion in transition area
        r_trans[mag_valid] = ((mag_limit + safety_factor) - mag[mag_valid] - 4.5) / 0.25
        # core area
        if beta != fmagloss.beta_interp2d or FWHM not in fmagloss.FWHM_interp2d:
            fmagloss = Moffat2dInterp(beta=beta, FWHM=[FWHM])
            r_core[mag_valid] = fmagloss((mag_limit + safety_factor) - mag[mag_valid],
                                         FWHM, obsSite)
        else:
            r_core[mag_valid] = fmagloss((mag_limit + safety_factor) - mag[mag_valid],
                                         FWHM, obsSite)
        # tom's old conservative core function
        # r_core[mag_valid] = 1.5 * ((mag_limit + safety_factor) - mag[mag_valid]) ** 0.8
        # exlusion radius is the max of each section
        r = numpy.nanmax(numpy.column_stack((r_wings,
                                             r_trans,
                                             r_core)),
                         axis=1)
        if skybrightness is not None and offset_min_skybrightness is not None:
            if skybrightness <= offset_min_skybrightness:
                offset_flag[:] += 8
                r[:] = 0.
    return r, offset_flag


def object_offset(mags, mag_limits, lunation, waveName, obsSite, fmagloss=None,
                  safety_factor=None, beta=None, FWHM=None, skybrightness=None,
                  offset_min_skybrightness=None, can_offset=None):
    """
    Returns the offset needed for object with mag to be
    observed at mag_limit. Currently assumption is all offsets
    are set in positive RA direction.

    Parameters
    ----------
    mags: numpy.array
        The magniutdes of the objects. Should be a 2D, Nx10 array, where
        N is number of objects and length of 10 index should correspond
        to magntidues: [g, r, i, z, bp, gaia_g, rp, J, H, K].

    mag_limits: numpy.array
        Magnitude limits for the designmode of the design.
        This should be an array of length N=10 where indexes
        correspond to magntidues: [g, r, i, z, bp, gaia_g, rp, J, H, K].
        This matches the apogee_bright_limit_targets_min or
        boss_bright_limit_targets_min (depending on instrument) from
        targetdb.DesignMode for the design_mode of the design.

    lunation: str:
        If the designmode is bright time ('bright') or dark
        time ('dark')

    waveName: str
        Instrument for the fibers offset definition being applied
        to. Either 'Boss' or 'Apogee'.

    obsSite: str
        The observatory of the observation. Should either be
        'APO' or 'LCO'.

    fmagloss: object
        Moffat2dInterp class with the lookup table
        for doing the moffat profile inversion. If None,
        then table is calculated at function call.

    safety_factor: float
        Factor to add to mag_limit. If None, default set in code.

    beta: float
        Power index of the Moffat profile. If None, default set in code.

    FWHM: float
        seeing for the Moffat profile. If None, default set in code.

    skybrightness: float
        Sky brightness for the field cadence. Only set if
        want to check for offset_flag TOO_DARK (8).

    offset_min_skybrightness: float
        Minimum sky brightness for the offset. Only set if
        want to check for offset_flag TOO_DARK (8).

    can_offset: boolean or numpy.array
        can_offset value from targetdb for the target(s) to be
        offset. Only set if
        want to check for offset_flag NO_CAN_OFFSET (16).

    Returns
    -------
    delta_ra: numpy.array
        offset in RA in arcseconds around object(s) in
        numpy array of length N

    delta_dec: numpy.array
        offset in Decl. in arcseconds around object(s) in
        numpy array of length N

    offset_flag: numpy.array
        bitmask for how offset was set in
        numpy array of length N. Flags are:
            - 0: offset applied normally (i.e. when mag <= mag_limit)
            - 1: no offset applied because mag > mag_limit
            - 2: no offset applied because magnitude was null value.
            - 8: offsets should not be used as sky brightness is <=
                 minimum offset sky brightness
            - 16: no offsets applied because can_offset = False
            - 32: no offset applied because mag <= offset_bright_limit
                  (offset_bright_limit is G = 6 for Boss bright time and
                   G = 13 for Boss dark time, and
                   H = 1 for Apogee).
            - 64: no offset applied because no valid magnitude limits
    """
    # check if 2D array
    if len(mags.shape) != 2:
        raise ValueError('mags must be a 2D numpy.array of shape (N, 10)')
    if mags.shape[1] != 10:
        raise ValueError('mags must be a 2D numpy.array of shape (N, 10)')
    # add default values for offset function if None supplied
    if safety_factor is None:
        if lunation == 'bright':
            safety_factor = 0.5
        else:
            safety_factor = 1.0
    if FWHM is None:
        if obsSite == 'APO':
            FWHM = 1.7
        elif obsSite == 'LCO':
            FWHM = 1.
        else:
            raise ValueError('obsSite must be APO or LCO.')
    if beta is None:
        beta = {'APO': 5., 'LCO': 2.}
    delta_ras = numpy.zeros(mags.shape)
    offset_flags = numpy.zeros(mags.shape)
    for i in range(len(mag_limits)):
        if mag_limits[i] != -999.:
            delta_ras[:, i], offset_flags[:, i] = offset_definition(mags[:, i], mag_limits, lunation, waveName,
                                                                    obsSite, fmagloss=fmagloss,
                                                                    safety_factor=safety_factor, beta=beta,
                                                                    FWHM=FWHM, skybrightness=skybrightness,
                                                                    offset_min_skybrightness=offset_min_skybrightness,
                                                                    can_offset=can_offset,
                                                                    use_type='offset', mag_limit_ind=i)
        else:
            # make artificially less than 0 so this doesnt get chosen
            # for max when setting offset_flag
            delta_ras[:, i] = numpy.zeros(len(mags[:, i]))
            offset_flags[:, i] = numpy.zeros(len(mags[:, i])) + 64
    # retain all flags when delta_ra = 0 for all checks
    def unique_offset_flags(flags):
        unq_flags = []
        poss_flags = [1, 2, 8, 16, 32, 64]
        for f in flags:
            for pf in poss_flags:
                if pf & int(f):
                    unq_flags.append(pf)
        total_flags = numpy.sum(numpy.unique(unq_flags))
        return numpy.zeros(flags.shape) + total_flags
    try:
        offset_flags[numpy.all(delta_ras == 0., 1)] = numpy.apply_along_axis(unique_offset_flags,
                                                                             1,
                                                                             offset_flags[numpy.all(delta_ras == 0., 1)])
    except ValueError:
        pass
    # use max offset
    delta_ra = numpy.max(delta_ras, axis=1)
    ind_max = numpy.argmax(delta_ras, axis=1)
    offset_flag = numpy.array([offset_flags[i, j] for i, j in enumerate(ind_max)],
                              dtype=int)
    delta_dec = numpy.zeros(len(delta_ra))
    return delta_ra, delta_dec, offset_flag

def _offset_radec(ra=None, dec=None, delta_ra=0., delta_dec=0.):
    """Offsets ra and dec according to specified amount. From Mike's
    robostrategy.Field object

    Parameters
    ----------
    ra : numpy.float64 or ndarray of numpy.float64
    right ascension, deg
    dec : numpy.float64 or ndarray of numpy.float64
        declination, deg
    delta_ra : numpy.float64 or ndarray of numpy.float64
        right ascension direction offset, arcsec
    delta_dec : numpy.float64 or ndarray of numpy.float64
        declination direction offset, arcsec

    Returns
    -------
    offset_ra : numpy.float64 or ndarray of numpy.float64
        offset right ascension, deg
    offset_dec : numpy.float64 or ndarray of numpy.float64
        offset declination, deg

    Notes
    -----
    Assumes that delta_ra, delta_dec are in proper coordinates; i.e.
    an offset of delta_ra=1 arcsec represents the same angular separation
    on the sky at any declination.
    Carefully offsets in the local directions of ra, dec based on
    the local tangent plane (i.e. does not just scale delta_ra by
    1/cos(dec))
    """
    deg2rad = numpy.pi / 180.
    arcsec2rad = numpy.pi / 180. / 3600.
    x = numpy.cos(dec * deg2rad) * numpy.cos(ra * deg2rad)
    y = numpy.cos(dec * deg2rad) * numpy.sin(ra * deg2rad)
    z = numpy.sin(dec * deg2rad)
    ra_x = - numpy.sin(ra * deg2rad)
    ra_y = numpy.cos(ra * deg2rad)
    ra_z = 0.
    dec_x = - numpy.sin(dec * deg2rad) * numpy.cos(ra * deg2rad)
    dec_y = - numpy.sin(dec * deg2rad) * numpy.sin(ra * deg2rad)
    dec_z = numpy.cos(dec * deg2rad)
    xoff = x + (ra_x * delta_ra + dec_x * delta_dec) * arcsec2rad
    yoff = y + (ra_y * delta_ra + dec_y * delta_dec) * arcsec2rad
    zoff = z + (ra_z * delta_ra + dec_z * delta_dec) * arcsec2rad
    offnorm = numpy.sqrt(xoff**2 + yoff**2 + zoff**2)
    xoff = xoff / offnorm
    yoff = yoff / offnorm
    zoff = zoff / offnorm
    decoff = numpy.arcsin(zoff) / deg2rad
    raoff = ((numpy.arctan2(yoff, xoff) / deg2rad) + 360.) % 360.
    return(raoff, decoff)


def gaia_mags2sdss_gri(gaia_g, gaia_bp=None, gaia_rp=None, gaia_bp_rp=None):
    '''
    Stolen from Tom Dwelly's dither_work:

    https://github.com/sdss/dither_work/blob/main/python/dither_work/utils.py#L75

    There are some fairly accurate+reliable G,BP-RP -> SDSS gri
    transforms available in the literature. Evans et al (2018,
    https://www.aanda.org/articles/aa/full_html/2018/08/aa32756-18/aa32756-18.html)
    give transforms for main sequence stars. Paraphrased below:

    sdss_g_from_gdr2 = phot_g_mean_mag_gdr2 - (0.13518 - 0.46245*bp_rp_gdr2 +
                                               -0.25171*bp_rp_gdr2**2 + 0.021349*bp_rp_gdr2**3)

    sdss_r_from_gdr2 = phot_g_mean_mag_gdr2 - (-0.12879 + 0.24662*bp_rp_gdr2 +
                                               -0.027464*bp_rp_gdr2**2 - 0.049465*bp_rp_gdr2**3)

    sdss_i_from_gdr2 = phot_g_mean_mag_gdr2 - (-0.29676 + 0.64728*bp_rp_gdr2 +
                                               -0.10141*bp_rp_gdr2**2)

    '''
    if (gaia_bp is not None) and (gaia_rp is not None) and (gaia_bp_rp is None):
        gaia_bp_rp = gaia_bp - gaia_rp
    elif (gaia_bp is None) and (gaia_rp is None) and (gaia_bp_rp is not None):
        pass
    else:
        raise Exception("Error - you must supply either bp and rp or just bp-rp")

    sdss_g = gaia_g - (0.13518 - 0.46245 * gaia_bp_rp -
                       0.25171 * gaia_bp_rp**2 + 0.021349 * gaia_bp_rp**3)
    sdss_r = gaia_g - (-0.12879 + 0.24662 * gaia_bp_rp -
                       0.027464 * gaia_bp_rp**2 - 0.049465 * gaia_bp_rp**3)
    sdss_i = gaia_g - (-0.29676 + 0.64728 * gaia_bp_rp -
                       0.10141 * gaia_bp_rp**2)

    return sdss_g, sdss_r, sdss_i
