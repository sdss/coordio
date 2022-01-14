import numpy
import pandas

from .sky import ICRS, Observed
from .telescope import Field, FocalPlane
from .wok import Wok
from .site import Site
from . import defaults


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
    """
    A = 0.000113636363636
    B = 0.0000000129132231405
    C = 0.0000012336318
    return A * r**2 / (1 + numpy.sqrt(1 - B * r**2)) + C * r**2


def radec2wokxy(ra, dec, coordEpoch, waveName, raCen, decCen, obsAngle,
                obsSite, obsTime, pmra=None, pmdec=None, parallax=None,
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

    obs = Observed(icrs, site=site, wavelength=wavelength)
    field = Field(obs, field_center=obsCen)
    focal = FocalPlane(field, wavelength=wavelength, site=site)
    wok = Wok(focal, site=site, obsAngle=obsAngle)

    output = (
        wok[:, 0], wok[:, 1], focal.field_warn,
        float(obsCen.ha), float(obsCen.pa)
    )
    return output


def wokxy2radec(xWok, yWok, waveName, raCen, decCen, obsAngle,
                obsSite, obsTime, pressure=None, relativeHumidity=0.5,
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
    focal = FocalPlane(wok, wavelength=wavelength, site=site)
    field = Field(focal, field_center=obsCen)
    obs = Observed(field, site=site, wavelength=wavelength)
    icrs = ICRS(obs, epoch=obsTime)

    return icrs[:, 0], icrs[:, 1], field.field_warn


def fitsTableToPandas(recarray):
    d = {}
    for name in recarray.names:
        d[name] = recarray[name].byteswap().newbyteorder()
    return pandas.DataFrame(d)
