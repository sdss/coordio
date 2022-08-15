import numpy
import pandas
from scipy.interpolate import interp1d

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
    for a moffat profile offset for various FWHMs
    """
    def __init__(self, Noffset=None, FWHM=None, beta=None):
        if Noffset is None:
            Noffset = 1000
        if FWHM is None:
            FWHM = [1.3, 1.5, 1.7, 1.9]
        if beta is None:
            beta = 5
        offsets = numpy.zeros((len(FWHM), Noffset))
        FWHMs = numpy.zeros((len(FWHM), Noffset))
        for i, f in enumerate(FWHM):
            FWHMs[i, :] = f
            offsets[i, :] = numpy.linspace(0, 20, Noffset)

        magloss = numpy.zeros((FWHMs.shape[0], Noffset))

        fmagloss = {}
        for i, f in enumerate(FWHMs[:, 0]):
            magloss[i, :] = MoffatLossProfile(offsets[i, :], beta, f).func_magloss()
            fmagloss[f] = interp1d(magloss[i, :], offsets[i, :])
        self.fmagloss = fmagloss
        self.beta_interp2d = beta
        self.FWHM_interp2d = FWHM

    def __call__(self, magloss, FWHM):
        r = self.fmagloss[FWHM](magloss)
        return r


def offset_definition(mag, mag_limits, lunation, waveName, fmagloss=None,
                      safety_factor=0., beta=5, FWHM=1.7, skybrightness=None,
                      offset_min_skybrightness=None, can_offset=None):
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

    fmagloss: object
        Moffat2dInterp class with the lookup table
        for doing the moffat profile inversion. If None,
        then table is calculated at function call.

    safety_factor: float
        Factor to add to mag_limit. Should equal zero for
        bright neighbor checks (i.e. remain at default).

    beta: float
        Power index of the Moffat profile

    FWHM: float
        seeing for the Moffat profile

    skybrightness: float
        Sky brightness for the field cadence. Only set if
        want to check for offset_flag TOO_DARK (8).

    offset_min_skybrightness: float
        Minimum sky brightness for the offset. Only set if
        want to check for offset_flag TOO_DARK (8).

    can_offset: boolean or np.array
        can_offset value from targetdb for the target(s) to be
        offset. Only set if
        want to check for offset_flag NO_CAN_OFFSET (16).

    Returns
    -------
    r: float or numpy.array
        offset radius in arcseconds around object(s)

    offset_flag: int or numpy.array
        Flag for how offset was set. Flags are:
            - 0: offset applied normally (i.e. when mag <= mag_limit)
            - 1: no offset applied because mag > mag_limit
            - 2: no offset applied because magnitude was null value.
            - 8: No offset applied because sky brightness is <=
                 minimum offset sky brightness
            - 16: no offsets applied because can_offset = False
            - 32: no offset applied because mag <= offset_bright_limit
                  (offset_bright_limit is G = 6 for Boss and
                   H = 1 for Apogee).
    """
    # define Null cases for targetdb.magnitude table
    cases = [-999, -9999, 999,
             0.0, numpy.nan, 99.9, None]
    if waveName == 'Boss':
        offset_bright_limit = 6.
    else:
        offset_bright_limit = 1.
    # set magntiude limit for instrument and lunation
    if waveName == 'Apogee':
        # 2MASS H
        mag_limit = mag_limits[8]
    elif lunation == 'bright':
        # Gaia G
        mag_limit = mag_limits[5]
    else:
        # SDSS r
        mag_limit = mag_limits[1]
    # get magloss function
    if fmagloss is None:
        fmagloss = Moffat2dInterp(beta=beta, FWHM=[FWHM])
    if isinstance(mag, float) or isinstance(mag, int):
        # make can_offset always True if not supplied
        if can_offset is None:
            can_offset = True
        if mag <= mag_limit and mag not in cases and can_offset and mag > offset_bright_limit:
            # linear portion in the wings
            r_wings = ((mag_limit + safety_factor) - mag - 8.2) / 0.05
            # linear portion in transition area
            r_trans = ((mag_limit + safety_factor) - mag - 4.5) / 0.25
            # core area
            # do dark core for apogee or dark
            if lunation == 'bright' or waveName == 'Apogee':
                if beta != fmagloss.beta_interp2d or FWHM not in fmagloss.FWHM_interp2d:
                    fmagloss = Moffat2dInterp(beta=beta, FWHM=[FWHM])
                    r_core = fmagloss((mag_limit + safety_factor) - mag, FWHM)
                else:
                    r_core = fmagloss((mag_limit + safety_factor) - mag, FWHM)
            else:
                r_core = 1.5 * ((mag_limit + safety_factor) - mag) ** 0.8
            # exlusion radius is the max of each section
            r = numpy.nanmax([r_wings, r_trans, r_core])
            offset_flag = 0
        else:
            r = 0.
            if mag > mag_limit:
                offset_flag = 1
            elif mag in cases:
                offset_flag = 2
            elif not can_offset:
                offset_flag = 16
            else:
                offset_flag = 32
        if skybrightness is not None and offset_min_skybrightness is not None:
            if skybrightness <= offset_min_skybrightness:
                offset_flag = 8
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
        offset_flag[mag > mag_limit] = 1
        offset_flag[(numpy.isin(mag, cases)) | (numpy.isnan(mag))] = 2
        offset_flag[~can_offset] = 16
        offset_flag[(mag <= offset_bright_limit) & ~((numpy.isin(mag, cases)) | (numpy.isnan(mag)))] = 32
        # linear portion in the wings
        r_wings[mag_valid] = ((mag_limit + safety_factor) - mag[mag_valid] - 8.2) / 0.05
        # linear portion in transition area
        r_trans[mag_valid] = ((mag_limit + safety_factor) - mag[mag_valid] - 4.5) / 0.25
        # core area
        # core area
        # do dark core for apogee or dark
        if lunation == 'bright' or waveName == 'Apogee':
            if beta != fmagloss.beta_interp2d or FWHM not in fmagloss.FWHM_interp2d:
                fmagloss = Moffat2dInterp(beta=beta, FWHM=[FWHM])
                r_core[mag_valid] = fmagloss((mag_limit + safety_factor) - mag[mag_valid],
                                             FWHM)
            else:
                r_core[mag_valid] = fmagloss((mag_limit + safety_factor) - mag[mag_valid],
                                             FWHM)
        else:
            r_core[mag_valid] = 1.5 * ((mag_limit + safety_factor) - mag[mag_valid]) ** 0.8
        # exlusion radius is the max of each section
        r = numpy.nanmax(numpy.column_stack((r_wings,
                                             r_trans,
                                             r_core)),
                         axis=1)
        if skybrightness is not None and offset_min_skybrightness is not None:
            if skybrightness <= offset_min_skybrightness:
                offset_flag[:] = 8
                r[:] = 0.
    return r, offset_flag


def object_offset(mag, mag_limits, lunation, waveName, fmagloss=None,
                  safety_factor=0., beta=5, FWHM=1.7, skybrightness=None,
                  offset_min_skybrightness=None, can_offset=None):
    """
    Returns the offset needed for object with mag to be
    observed at mag_limit. Currently assumption is all offsets
    are set in positive RA direction.

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

    fmagloss: object
        Moffat2dInterp class with the lookup table
        for doing the moffat profile inversion. If None,
        then table is calculated at function call.

    safety_factor: float
        Factor to add to mag_limit.

    beta: float
        Power index of the Moffat profile

    FWHM: float
        seeing for the Moffat profile

    skybrightness: float
        Sky brightness for the field cadence. Only set if
        want to check for offset_flag TOO_DARK (8).

    offset_min_skybrightness: float
        Minimum sky brightness for the offset. Only set if
        want to check for offset_flag TOO_DARK (8).

    can_offset: boolean or np.array
        can_offset value from targetdb for the target(s) to be
        offset. Only set if
        want to check for offset_flag NO_CAN_OFFSET (16).

    Returns
    -------
    delta_ra: float or numpy.array
        offset in RA in arcseconds around object(s)

    delta_dec: float or numpy.array
        offset in Decl. in arcseconds around object(s)

    offset_flag: int or numpy.array
        Flag for how offset was set. Flags are:
            - 0: offset applied normally (i.e. when mag <= mag_limit)
            - 1: no offset applied because mag > mag_limit
            - 2: no offset applied because magnitude was null value.
            - 8: offsets should not be used as sky brightness is <=
                 minimum offset sky brightness
            - 16: no offsets applied because can_offset = False
            - 32: no offset applied because mag <= offset_bright_limit
                  (offset_bright_limit is G = 6 for Boss and
                   H = 1 for Apogee).
    """
    delta_ra, offset_flag = offset_definition(mag, mag_limits, lunation, waveName,
                                              fmagloss=fmagloss,
                                              safety_factor=safety_factor, beta=beta,
                                              FWHM=FWHM, skybrightness=skybrightness,
                                              offset_min_skybrightness=offset_min_skybrightness,
                                              can_offset=can_offset)
    if isinstance(delta_ra, float):
        delta_dec = 0.
    else:
        delta_dec = numpy.zeros(len(delta_ra))
    return delta_ra, delta_dec, offset_flag
