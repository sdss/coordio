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


class MoffatFluxLoss(object):
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
        self.rfiber = rfiber

    def moffat_1D(self, amplitude, r, alpha, beta):  # unit flux at the center
        """
        Function computing Moffat 1D profile
        
        Parameters
        ------------
        amplitude: float
            Amplitude of the profile

        r: float or numpy.array
            Distance from the centre of the profile

        alpha: float
            Scale parameter of the Moffat profile, depends on seeing

        beta: float
            Power index of the Moffat profile
        
        Returns
        -------
        moff_prof: float or numpy.array
            1D Moffat profile.
        """
        moff_prof = amplitude * (1 + (r / alpha) ** 2) ** (-beta)
        return moff_prof

    def moffat_norm(self, amplitude, alpha, beta): 
        """
        Function computing the normalized the Moffat 1D profile.
        
        Parameters
        -----------
        amplitude: float
            Amplitude of the profile

        alpha: float
            Scale parameter of the Moffat profile, depends on seeing

        beta: float
            Power index of the Moffat profile
        
        Returns
        -------
        moff_prof_norm: float
            Normalised Moffat profile. 
        """
        
        xmin, xmax, xstep = -7., 7., 0.05  # fixed ln radius steps
        # x is the ln radius in units of the r / alpha
        steps = numpy.arange(xmin, xmax, xstep)
        r = alpha * numpy.exp(steps)
        norm = numpy.sum(numpy.exp(2 * steps) * self.moffat_1D(amplitude, r, alpha, beta))
        moff_prof_norm = 2 * numpy.pi * alpha ** 2 * norm * xstep 
        return moff_prof_norm

    def flux_loss(self, offset, alpha, beta):
        """
        Function computing the flux loss obtained by moving the fiber
        across the source in a certain direction.
        
        Prameters
        ---------
        offset: numpy.array
            fiber offset

         alpha: float
            Scale parameter of the Moffat profile, depends on seeing

        beta: float
            Power index of the Moffat profile

        Returns
        --------
        norm: numpy.array
            the flux loss 
        """
        amplitude = 1. / self.moffat_norm(1., alpha, beta)  # this is the amplitude needed to normalize the Moffat profile
        x = numpy.arange(-self.rfiber, self.rfiber, self.rfiber / 50)    # set up a 100x100 Cartesian grid across the fiber
        y = numpy.arange(-self.rfiber, self.rfiber, self.rfiber / 50)
        X, Y = numpy.meshgrid(x, y)
        r = numpy.sqrt(X **2 + Y ** 2)
        norm = numpy.zeros(len(offset))
        for i in range(len(offset)):
            r_moffat = numpy.sqrt((X - offset[i]) ** 2 + Y ** 2)
            norm[i] = numpy.sum((self.rfiber / 50) ** 2 *
                                self.moffat_1D(amplitude, r_moffat[r <= self.rfiber], alpha, beta))                
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
        seeing_alpha = self.FWHM / (2. * numpy.sqrt(2 ** (1 / self.beta) - 1))
        magloss = -2.5 * numpy.log10(self.flux_loss(self.offset, seeing_alpha, self.beta)) \
                   + 2.5 * numpy.log10(self.flux_loss(numpy.array([0.]), seeing_alpha, self.beta)[0])
        return magloss


def offset_definition(mag, mag_limit, lunation, waveName, safety_factor=0.):
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

    mag_limit: float
        Magnitude limit for the designmode of the design.
        For BOSS should be r-band limit and for APOGEE should
        be H-band limit.

    lunation: str:
        If the designmode is bright time ('bright') or dark
        time ('dark')

    waveName: str
        Instrument for the fibers offset definition being applied
        to. Either 'Boss' or 'Apogee'.

    safety_factor: float
        Factor to add to mag_limit. Should equal zero for
        bright neighbor checks (i.e. remain at default).

    Returns
    -------
    r: float or numpy.array
        offset radius in arcseconds around object(s)

    offset_flag: int or numpy.array
        Flag for how offset was set. 0 indicates offset applied
        normally (i.e. when mag <= mag_limit), 1 indicates no offset applied
        because mag > mag_limit and 2 indiciates no offset applied
        because magnitude was null value.
    """
    # define Null cases for targetdb.magnitude table
    cases = [-999, -9999, 999,
             0.0, numpy.nan, 99.9, None]
    if isinstance(mag, float):
        if mag <= mag_limit and mag not in cases:
            # linear portion in the wings
            r_wings = ((mag_limit + safety_factor) - mag - 8.2) / 0.05
            # linear portion in transition area
            r_trans = ((mag_limit + safety_factor) - mag - 4.5) / 0.25
            # core area
            # do dark core for apogee or dark
            if lunation == 'dark' or waveName == 'Apogee':
                r_core = 1.5 * ((mag_limit + safety_factor) - mag) ** 0.8
            else:
                r_core = 1.75 * ((mag_limit + safety_factor) - mag) ** 0.6
            # exlusion radius is the max of each section
            r = max(r_wings, r_trans, r_core)
            offset_flag = 0
        else:
            r = 0.
            if mag > mag_limit:
                offset_flag = 1
            else:
                offset_flag = 2
    else:
        # create empty arrays for each portion
        r_wings = numpy.zeros(mag.shape)
        r_trans = numpy.zeros(mag.shape)
        r_core = numpy.zeros(mag.shape)
        # only do calc for valid mags for offset
        # to avoid warning
        mag_valid = (mag <= mag_limit) & (~numpy.isin(mag, cases)) & (~numpy.isnan(mag))
        # set flags
        offset_flag = numpy.zeros(mag.shape, dtype=int)
        offset_flag[mag > mag_limit] = 1
        offset_flag[(numpy.isin(mag, cases)) | (numpy.isnan(mag))] = 2
        # linear portion in the wings
        r_wings[mag_valid] = ((mag_limit + safety_factor) - mag[mag_valid] - 8.2) / 0.05
        # linear portion in transition area
        r_trans[mag_valid] = ((mag_limit + safety_factor) - mag[mag_valid] - 4.5) / 0.25
        # core area
        # core area
        # do dark core for apogee or dark
        if lunation == 'dark' or waveName == 'Apogee':
            r_core[mag_valid] = 1.5 * ((mag_limit + safety_factor) - mag[mag_valid]) ** 0.8
        else:
            r_core[mag_valid] = 1.75 * ((mag_limit + safety_factor) - mag[mag_valid]) ** 0.6
        # exlusion radius is the max of each section
        r = numpy.nanmax(numpy.column_stack((r_wings,
                                             r_trans,
                                             r_core)),
                         axis=1)
    return r, offset_flag


def object_offset(mag, mag_limit, lunation, waveName, safety_factor=0.1):
    """
    Returns the offset needed for object with mag to be
    observed at mag_limit. Currently assumption is all offsets
    are set in positive RA direction.

    Parameters
    ----------
    mag: float or numpy.array
        The magniutde(s) of the objects. For BOSS should be
        Gaia G-band and for APOGEE should be 2MASS H-band.

    mag_limit: float
        Magnitude limit for the designmode of the design.
        For BOSS should be r-band limit and for APOGEE should
        be H-band limit.

    lunation: str:
        If the designmode is bright time ('bright') or dark
        time ('dark')

    waveName: str
        Instrument for the fibers offset definition being applied
        to. Either 'Boss' or 'Apogee'.

    safety_factor: float
        Factor to add to mag_limit.

    offset_flag: int or numpy.array
        Flag for how offset was set. 0 indicates offset applied
        normally (i.e. when mag <= mag_limit), 1 indicates no offset applied
        because mag > mag_limit and 2 indiciates no offset applied
        because magnitude was null value.

    Returns
    -------
    delta_ra: float or numpy.array
        offset in RA in arcseconds around object(s)

    delta_dec: float or numpy.array
        offset in Decl. in arcseconds around object(s)
    """
    delta_ra, offset_flag = offset_definition(mag, mag_limit, lunation, waveName,
                                              safety_factor=safety_factor)
    if isinstance(delta_ra, float):
        delta_dec = 0.
    else:
        delta_dec = numpy.zeros(len(delta_ra))
    return delta_ra, delta_dec, offset_flag
