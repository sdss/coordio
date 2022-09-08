import numpy
import pandas
import ctypes
import importlib
import os

from .sky import ICRS, Observed
from .telescope import Field, FocalPlane
from .wok import Wok
from .site import Site
from . import defaults

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
                temperature=10):
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

    icrs = ICRS(
        radec, epoch=coordEpoch, pmra=pmra, pmdec=pmdec,
        parallax=parallax, rvel=radVel
    )

    # propogate propermotions, etc
    icrs = icrs.to_epoch(obsTime, site=site)
    if focalScale is None:
        focalScale = defaults.SITE_TO_SCALE[obsSite]

    obs = Observed(icrs, site=site, wavelength=wavelength)
    field = Field(obs, field_center=obsCen)
    focal = FocalPlane(field, wavelength=wavelength, site=site, fpScale=focalScale)
    wok = Wok(focal, site=site, obsAngle=obsAngle)

    output = (
        wok[:, 0], wok[:, 1], focal.field_warn,
        float(obsCen.ha), float(obsCen.pa)
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
    focal = FocalPlane(wok, wavelength=wavelength, site=site, fpScale=focalScale)
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
