from __future__ import annotations

import pathlib
import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy
import pandas
import scipy.ndimage
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import fit_wcs_from_points
from scipy.spatial import KDTree
from skimage.registration import phase_cross_correlation

from coordio.conv import guideToTangent, tangentToWok

from . import Site, calibration, defaults
from .astrometry import astrometrynet_quick
from .conv import tangentToGuide, wokToTangent
from .coordinate import Coordinate, Coordinate2D
from .exceptions import CoordIOError, CoordIOUserWarning
from .extraction import sextractor_quick
from .sky import ICRS, Observed
from .telescope import Field, FocalPlane
from .utils import radec2wokxy
from .wok import Wok


__all__ = [
    "Guide",
    "GuiderFitter",
    "umeyama",
    "radec_to_gfa",
    "gfa_to_radec",
    "cross_match",
]


warnings.filterwarnings("ignore", module="astropy.wcs.wcs")
warnings.filterwarnings("ignore", category=FITSFixedWarning)


class Guide(Coordinate2D):
    """Guide coordinates are 2D cartesian xy coordinates.  Units are pixels.
    The origin is the lower left corner of the lower left pixel when looking
    at the CCD through the camera window.  Thus guide images are taken in
    "selfie-mode".  So the coordinate (0.5, 0.5) is the center of the lower
    left pixel. -y points (generally) toward the wok vertex.  True rotations
    and locations of GFAs in the wok are measured/calibrated after
    installation and on-sky characterization.

    Parameters
    -----------
    value : numpy.ndarray
        A Nx2 array where [x,y] are columns.  Or `.Tangent`
    xBin : int
        CCD x bin factor, defaults to 1
    yBin : int
        CCD y bin factor, defaults to 1

    Attributes
    -----------
    guide_warn : numpy.ndarray
        boolean array indicating coordinates outside the CCD FOV
    """

    __extra_params__ = ["xBin", "yBin"]
    __warn_arrays__ = ["guide_warn"]

    xBin: int
    yBin: int
    guide_warn: numpy.ndarray

    def __new__(cls, value, **kwargs):

        xBin = kwargs.get("xBin", None)
        if xBin is None:
            kwargs["xBin"] = 1

        yBin = kwargs.get("yBin", None)
        if yBin is None:
            kwargs["yBin"] = 1

        if isinstance(value, Coordinate):
            if value.coordSysName == "Tangent":
                # going from 3D -> 2D coord sys
                # reduce dimensionality
                arrInit = numpy.zeros((len(value), 2))
                obj = super().__new__(cls, arrInit, **kwargs)
                obj._fromTangent(value)
            else:
                raise CoordIOError(
                    "Cannot convert to Guide from %s" % value.coordSysName
                )
        else:
            obj = super().__new__(cls, value, **kwargs)
            obj._fromRaw()

        return obj

    def _fromTangent(self, tangentCoords):
        """Convert from tangent coordinates to guide coordinates"""
        if tangentCoords.holeID not in calibration.VALID_GUIDE_IDS:
            raise CoordIOError(
                "Cannot convert from wok hole %s to Guide" % tangentCoords.holeID
            )
        # make sure the guide wavelength is specified for all coords
        # this may be a little too extreme of a check
        badWl = numpy.sum(tangentCoords.wavelength - defaults.INST_TO_WAVE["GFA"]) != 0
        if badWl:
            raise CoordIOError(
                "Cannont convert to Guide coords from non-guide wavelength"
            )

        # use coords projected to tangent xy plane
        # could almost equivalently use tangentCoords[:,0:2]
        # and we may want to ditch the projection eventually if its
        # unnecessary

        # note, moved this calculation into
        # conv.tangentToGuide, may want to just use that one
        xT = tangentCoords.xProj
        yT = tangentCoords.yProj

        xPix = (1 / self.xBin) * (
            defaults.MICRONS_PER_MM / defaults.GFA_PIXEL_SIZE * xT
            + defaults.GFA_CHIP_CENTER
        )

        yPix = (1 / self.yBin) * (
            defaults.MICRONS_PER_MM / defaults.GFA_PIXEL_SIZE * yT
            + defaults.GFA_CHIP_CENTER
        )

        self[:, 0] = xPix
        self[:, 1] = yPix

        self._fromRaw()

    def _fromRaw(self):
        """Find coords that may be out of the FOV and tag them"""

        # note this might be wrong when xBin is 3
        maxX = defaults.GFA_CHIP_CENTER * 2 / self.xBin
        maxY = defaults.GFA_CHIP_CENTER * 2 / self.yBin
        xPix = self[:, 0]
        yPix = self[:, 1]

        arg = numpy.argwhere((xPix < 0) | (xPix > maxX) | (yPix < 0) | (yPix > maxY))

        if len(arg) == 0:
            # everything in range...
            return

        arg = arg.squeeze()
        self.guide_warn[arg] = True


def gfa_to_wok(observatory: str, x_pix: float, y_pix: float, gfa_id: int):
    """Converts from a GFA pixel to wok coordinates."""

    gfa_row = calibration.gfaCoords.loc[(observatory, gfa_id), :]

    b = gfa_row[["xWok", "yWok", "zWok"]].to_numpy().squeeze()
    iHat = gfa_row[["ix", "iy", "iz"]].to_numpy().squeeze()
    jHat = gfa_row[["jx", "jy", "jz"]].to_numpy().squeeze()
    kHat = gfa_row[["kx", "ky", "kz"]].to_numpy().squeeze()

    xt, yt = guideToTangent(x_pix, y_pix)
    zt = 0

    return tangentToWok(xt, yt, zt, b, iHat, jHat, kHat)  # type: ignore


def gfa_to_radec(
    observatory: str,
    x_pix: float,
    y_pix: float,
    gfa_id: int,
    bore_ra: float,
    bore_dec: float,
    position_angle: float = 0,
    offset_ra: float = 0,
    offset_dec: float = 0,
    offset_pa: float = 0,
    obstime: float | None = None,
    icrs: bool = False,
):
    """Converts from a GFA pixel to observed RA/Dec. Offsets are in arcsec."""

    site = Site(observatory)
    site.set_time(obstime)

    wavelength = defaults.INST_TO_WAVE["GFA"]

    wok_coords = gfa_to_wok(observatory, x_pix, y_pix, gfa_id)

    bore_ra += offset_ra / 3600.0 / numpy.cos(numpy.radians(bore_dec))
    bore_dec += offset_dec / 3600.0
    position_angle -= offset_pa / 3600.0

    boresight_icrs = ICRS([[bore_ra, bore_dec]])
    boresight = Observed(
        boresight_icrs,
        site=site,
        wavelength=wavelength,
    )

    wok = Wok([wok_coords], site=site, obsAngle=position_angle)
    focal = FocalPlane(wok, wavelength=wavelength, site=site)
    field = Field(focal, field_center=boresight)
    observed = Observed(field, wavelength=wavelength, site=site)

    if icrs is False:
        return (observed.ra[0], observed.dec[0])
    else:
        icrs = ICRS(observed)
        return icrs[0]


def radec_to_gfa(
    observatory: str,
    ra: float | numpy.ndarray,
    dec: float | numpy.ndarray,
    gfa_id: int,
    bore_ra: float,
    bore_dec: float,
    position_angle: float = 0,
    offset_ra: float = 0,
    offset_dec: float = 0,
    offset_pa: float = 0,
    obstime: float | None = None,
):
    """Converts from observed RA/Dec to GFA pixel, taking into account offsets."""

    ra = numpy.atleast_1d(ra)
    dec = numpy.atleast_1d(dec)

    site = Site(observatory)
    site.set_time(obstime)
    assert site.time

    bore_ra += offset_ra / 3600.0 / numpy.cos(numpy.radians(bore_dec))
    bore_dec += offset_dec / 3600.0
    position_angle -= offset_pa / 3600.0

    xwok, ywok, *_ = radec2wokxy(
        ra,
        dec,
        None,
        "GFA",
        bore_ra,
        bore_dec,
        position_angle,
        observatory,
        site.time.jd,
        focalScale=1.0,  # Guider scale is relative to 1
    )

    zwok = numpy.array([defaults.POSITIONER_HEIGHT] * len(xwok))

    gfa_coords = calibration.gfaCoords.loc[observatory]
    gfa_coords.reset_index(inplace=True)

    gfa_row = gfa_coords[gfa_coords.id == gfa_id]

    b = gfa_row[["xWok", "yWok", "zWok"]].to_numpy().squeeze()
    iHat = gfa_row[["ix", "iy", "iz"]].to_numpy().squeeze()
    jHat = gfa_row[["jx", "jy", "jz"]].to_numpy().squeeze()
    kHat = gfa_row[["kx", "ky", "kz"]].to_numpy().squeeze()

    xt, yt, _ = wokToTangent(xwok, ywok, zwok, b, iHat, jHat, kHat)
    x_ccd, y_ccd = tangentToGuide(xt, yt)

    return x_ccd, y_ccd


def umeyama(X, Y):
    """Rigid alignment of two sets of points in k-dimensional Euclidean space.

    Given two sets of points in correspondence, this function computes the
    scaling, rotation, and translation that define the transform TR that
    minimizes the sum of squared errors between TR(X) and its corresponding
    points in Y.  This routine takes O(n k^3)-time.

    Parameters
    ----------
    X
        A ``k x n`` matrix whose columns are points.
    Y
        A ``k x n`` matrix whose columns are points that correspond to the
        points in X

    Returns
    -------
    c,R,t
        The  scaling, rotation matrix, and translation vector defining the
        linear map TR as ::

            TR(x) = c * R * x + t

        such that the average norm of ``TR(X(:, i) - Y(:, i))`` is
        minimized.

    Copyright: Carlo Nicolini, 2013

    Code adapted from the Mark Paskin Matlab version from
    http://openslam.informatik.uni-freiburg.de/data/svn/tjtf/trunk/matlab/ralign.m

    See paper by Umeyama (1991)

    """

    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)

    Xc = X - numpy.tile(mx, (n, 1)).T
    Yc = Y - numpy.tile(my, (n, 1)).T

    sx = numpy.mean(numpy.sum(Xc * Xc, 0))

    Sxy = numpy.dot(Yc, Xc.T) / n

    U, D, V = numpy.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = V.T.copy()

    r = numpy.linalg.matrix_rank(Sxy)
    S = numpy.eye(m)

    if r < (m - 1):
        raise CoordIOError("Not enough independent measurements")

    if numpy.linalg.det(Sxy) < 0:
        S[-1, -1] = -1
    elif r == m - 1:
        if numpy.linalg.det(U) * numpy.linalg.det(V) < 0:
            S[-1, -1] = -1

    R = numpy.dot(numpy.dot(U, S), V.T)
    c = numpy.trace(numpy.dot(numpy.diag(D), S)) / sx
    t = my - c * numpy.dot(R, mx)

    return c, R, t


def cross_match(
    measured_xy: numpy.ndarray,
    reference_xy: numpy.ndarray,
    reference_radec: numpy.ndarray,
    x_size: int,
    y_size: int,
    cross_corrlation_shift: bool = True,
    blur: float = 5,
    distance_upper_bound: int = 10,
    **kwargs,
):
    """Determines the shift between two sets of points using cross-correlation.

    Constructs 2D images from reference and measured data sets and calculates
    the image shift using cross-correlation registration. It then associated
    reference to measured detections using nearest neighbours and builds a
    WCS using the reference on-sky positions.

    Parameters
    ----------
    measured_xy
        A 2D array with the x and y coordinates of each measured point.
    reference_xy
        A 2D array with the x and y coordinates of each reference point.
    reference_radec
        A 2D array with the ra and dec coordinates of each reference point.
    x_size
        Size of the image for 2D cross-correlation.
    y_size
        Size of the image for 2D cross-correlation.
    blur
        The sigma, in pixels, of the Gaussian kernel used to convolve the images.
    distance_upper_bound
        Maximum distance, in pixels, for KD tree nearest neighbours.
    kwargs
        Other arguments to pass to ``skimage.registration.phase_cross_correlation``.

    Returns
    -------
    wcs
        A tuple with the WCS of the solution and the translation invariant normalized
        RMS error between reference and moving image (see ``phase_cross_correlation``).

    """

    # Create and blur the reference and measured images.
    ref_image = numpy.zeros((y_size, x_size), numpy.float32)
    ref_image[reference_xy.astype(int)[:, 1], reference_xy.astype(int)[:, 0]] = 1
    if blur > 0:
        ref_image = scipy.ndimage.gaussian_filter(ref_image, blur)

    meas_image = numpy.zeros((y_size, x_size), numpy.float32)
    meas_image[measured_xy.astype(int)[:, 1], measured_xy.astype(int)[:, 0]] = 1
    if blur > 0:
        meas_image = scipy.ndimage.gaussian_filter(meas_image, blur)

    # Calculate the shift and error.
    error: float
    if cross_corrlation_shift:
        shift, error, _ = phase_cross_correlation(ref_image, meas_image, **kwargs)
    else:
        shift = numpy.array([0.0, 0.0])
        error = numpy.nan

    # Apply shift.
    measured_shift = measured_xy + shift[::-1]

    # Associate measured to reference using KD tree.
    tree = KDTree(reference_xy)
    dd, ii = tree.query(measured_shift, k=1, distance_upper_bound=distance_upper_bound)

    # Reject measured objects without a close neighbour.
    # KDTree.query() assigns an index larger than the initial set of points when
    # the closest neighbour has distance > distance_upper_bound.
    valid = dd < len(reference_xy)
    ii_valid = ii[valid]

    # Select valid measured object and their corresponding RA/Dec references.
    measured_xy_valid = measured_xy[valid]
    reference_radec_valid = reference_radec[ii_valid, :]

    if len(reference_radec_valid) < 3 or len(measured_xy_valid) < 3:
        return None, 0

    # Build the WCS.
    reference_skycoord_valid = SkyCoord(
        ra=reference_radec_valid[:, 0],
        dec=reference_radec_valid[:, 1],
        unit="deg",
    )

    wcs = fit_wcs_from_points(
        (measured_xy_valid[:, 0], measured_xy_valid[:, 1]),
        reference_skycoord_valid,
    )

    assert isinstance(wcs, WCS)

    return wcs, error


@dataclass
class GuiderFit:
    """Results of a guider data fit."""

    gfa_wok: pandas.DataFrame
    astro_wok: pandas.DataFrame
    delta_ra: float
    delta_dec: float
    delta_rot: float
    delta_scale: float
    xrms: float
    yrms: float
    rms: float
    only_radec: bool = False


class GuiderFitter:
    """Fits guider data, returning the measured offsets."""

    def __init__(self, observatory: str):

        self.observatory = observatory.upper()

        self.plate_scale = defaults.PLATE_SCALE[self.observatory]
        self.pixel_size = defaults.GFA_PIXEL_SIZE
        self.pixel_scale = 1.0 / self.plate_scale * self.pixel_size * 3600 / 1000

        self.reference_pixel: tuple[int, int] = (1024, 1024)

        self.gfa_wok_coords = self._gfa_to_wok()
        self.astro_data: pandas.DataFrame | None = None

        self.result: GuiderFit | None = None

    def reset(self):
        """Clears the astrometric data."""

        self.astro_data = None
        self.result = None

    def _gfa_to_wok(self):
        """Converts GFA reference points to wok coordinates."""

        if self.observatory not in calibration.gfaCoords.index.get_level_values(0):
            raise CoordIOError(
                f"GFA coordinates for {self.observatory} are missing. "
                "Did you load the correct calibrations?"
            )

        gfaCoords = calibration.gfaCoords.loc[self.observatory]

        wok_coords = {}

        for gfa_id in range(1, 7):

            camera_coords: pandas.DataFrame = gfaCoords.loc[gfa_id]

            b = camera_coords[["xWok", "yWok", "zWok"]].to_numpy().squeeze()
            iHat = camera_coords[["ix", "iy", "iz"]].to_numpy().squeeze()
            jHat = camera_coords[["jx", "jy", "jz"]].to_numpy().squeeze()
            kHat = camera_coords[["kx", "ky", "kz"]].to_numpy().squeeze()

            xt, yt = guideToTangent(self.reference_pixel[0], self.reference_pixel[1])
            zt = 0

            wok_coords[gfa_id] = tangentToWok(xt, yt, zt, b, iHat, jHat, kHat)

        df = pandas.DataFrame.from_dict(wok_coords).transpose()
        df.index.name = "gfa_id"
        df.columns = ["xwok", "ywok", "zwok"]

        return df

    def add_astro(
        self,
        camera: str | int,
        ra: float,
        dec: float,
        obstime: float,
        x_pixel: float,
        y_pixel: float,
    ):
        """Adds an astrometric measurement."""

        if isinstance(camera, str):
            match = re.match(r".*([1-6]).*", camera)
            if not match:
                raise CoordIOError(f"Cannot understand camera {camera!r}.")
            camera = int(match.group(1))

        new_data = (camera, ra, dec, obstime, x_pixel, y_pixel)

        if self.astro_data is None:
            self.astro_data = pandas.DataFrame(
                [new_data],
                columns=["gfa_id", "ra", "dec", "obstime", "x", "y"],
            )
        else:
            self.astro_data.loc[len(self.astro_data.index)] = new_data

    def add_wcs(
        self,
        camera: str | int,
        wcs: WCS,
        obstime: float,
        pixels: numpy.ndarray | None = None,
    ):
        """Adds a camera measurement from a WCS solution."""

        if pixels is None:
            coords: Any = wcs.pixel_to_world(*self.reference_pixel)

            ra = coords.ra.value
            dec = coords.dec.value

            self.add_astro(camera, ra, dec, obstime, *self.reference_pixel)

        else:
            ras, decs = wcs.all_pix2world(pixels[:, 0], pixels[:, 1], 0)
            for ii in range(len(ras)):
                self.add_astro(
                    camera,
                    ras[ii],
                    decs[ii],
                    obstime,
                    pixels[ii, 0],
                    pixels[ii, 1],
                )

    def add_gimg(self, path: str | pathlib.Path, pixels: numpy.ndarray | None = None):
        """Processes a raw GFA image, runs astrometry.net, adds the solution."""

        data = fits.getdata(str(path))
        header = fits.getheader(str(path), 1)

        if header["OBSERVAT"] != self.observatory:
            raise CoordIOError("GFA image is from a different observatory.")

        if "RAFIELD" not in header or "DECFIELD" not in header:
            raise CoordIOError("GFA image does not have RAFIELD or DECFIELD.")

        ra_field = header["RAFIELD"]
        dec_field = header["DECFIELD"]
        pa_field = header["FIELDPA"]

        sources = sextractor_quick(data)

        camera_id = int(header["CAMNAME"][3])
        radec_centre = gfa_to_radec(
            self.observatory,
            self.reference_pixel[0],
            self.reference_pixel[1],
            camera_id,
            ra_field,
            dec_field,
            position_angle=pa_field,
        )

        wcs = astrometrynet_quick(
            sources,
            radec_centre[0],
            radec_centre[1],
            self.pixel_scale,
        )

        if wcs is None:
            raise CoordIOError("astrometry.net could not solve image.")

        obstime = Time(header["DATE-OBS"], format="iso").jd

        self.add_wcs(camera_id, wcs, obstime, pixels=pixels)

        return wcs

    def add_proc_gimg(
        self,
        path: str | pathlib.Path,
        pixels: numpy.ndarray | None = None,
    ):
        """Adds an astrometric solution from a ``proc-gimg`` image."""

        hdu = fits.open(str(path))

        is_proc = len(hdu) > 1 and "SOLVED" in hdu[1].header
        if not is_proc:
            raise CoordIOError(f"{path!s} is not a proc-gimg image.")

        header = hdu[1].header

        if header["OBSERVAT"] != self.observatory:
            raise CoordIOError("GFA image is from a different observatory.")

        if not header.get("SOLVED", False):
            raise CoordIOError(f"Image {path!s} has not been astrometrically solved.")

        wcs = WCS(header)
        obstime = Time(header["DATE-OBS"], format="iso").jd

        self.add_wcs(header["CAMNAME"], wcs, obstime, pixels=pixels)

    def fit(
        self,
        field_ra: float,
        field_dec: float,
        field_pa: float = 0.0,
        offset: tuple[float, float, float] | numpy.ndarray = (0.0, 0.0, 0.0),
        scale_rms: bool = False,
        only_radec: bool = False,
    ):
        """Performs the fit and returns a `.GuiderFit` object.

        The fit is performed using `.umeyama`.

        Parameters
        ----------
        field_ra
            The field centre RA.
        field_dec
            The field centre declination.
        field_pa
            The field centre position angle.
        offset
            The offset in RA, Dec, PA to apply.
        scale_rms
            Whether to correct the RMS using the measured scale factor.
        only_radec
            If `True`, fits only translation. Useful when only one or two cameras
            are solving. In this case the rotation and scale offsets will still be
            calculated using the Umeyama model, but the ra and dec offsets are
            overridden with a simple translation. The `.GuiderFit` object will
            have `only_radec=True`.

        """

        offset_ra, offset_dec, offset_pa = offset

        if self.astro_data is None:
            raise CoordIOError("Astro data has not been set.")

        camera_ids: list[int] = []
        xwok_gfa: list[float] = []
        ywok_gfa: list[float] = []
        xwok_astro: list[float] = []
        ywok_astro: list[float] = []

        for d in self.astro_data.itertuples():

            camera_id: int = d.gfa_id

            camera_ids.append(camera_id)

            x_wok, y_wok, _ = gfa_to_wok(self.observatory, d.x, d.y, camera_id)
            xwok_gfa.append(x_wok)
            ywok_gfa.append(y_wok)

            _xwok_astro, _ywok_astro, *_ = radec2wokxy(
                [d.ra],
                [d.dec],
                None,
                "GFA",
                field_ra - offset_ra / numpy.cos(numpy.deg2rad(d.dec)) / 3600.0,
                field_dec - offset_dec / 3600.0,
                field_pa - offset_pa / 3600.0,
                self.observatory,
                d.obstime,
                focalScale=1.0,  # Guider scale is relative to 1
            )

            xwok_astro += _xwok_astro.tolist()
            ywok_astro += _ywok_astro.tolist()

        # X: GFA to wok coordinates as a 2D array
        # Y: ICRS to wok coordinates as a 2D array
        X = numpy.array([xwok_gfa, ywok_gfa])
        Y = numpy.array([xwok_astro, ywok_astro])

        try:
            c, R, t = umeyama(X, Y)
        except ValueError:
            if only_radec is True:
                warnings.warn(
                    "Cannot fit using Umeyama. Assuming unitary rotation and scale.",
                    CoordIOUserWarning,
                )
                c = 1.0
                R = numpy.array([[1, 0], [1, 0]])
            else:
                return False

        if only_radec is True:
            # Override the translation component with a simple average translation
            t = (Y - X).mean(axis=1).T

        # delta_x and delta_y only align with RA/Dec if PA=0. Otherwise we need to
        # project using the PA.
        pa_rad = numpy.deg2rad(field_pa - offset_pa / 3600.0)
        delta_ra = t[0] * numpy.cos(pa_rad) + t[1] * numpy.sin(pa_rad)
        delta_dec = -t[0] * numpy.sin(pa_rad) + t[1] * numpy.cos(pa_rad)

        # Convert to arcsec and round up
        delta_ra = float(numpy.round(delta_ra / self.plate_scale * 3600.0, 3))
        delta_dec = float(numpy.round(delta_dec / self.plate_scale * 3600.0, 3))

        delta_rot = -numpy.rad2deg(numpy.arctan2(R[1, 0], R[0, 0])) * 3600.0
        delta_rot = float(numpy.round(delta_rot, 1))

        delta_scale = float(numpy.round(c, 6))

        if scale_rms:
            xwok_astro = list(numpy.array(xwok_astro) / delta_scale)
            ywok_astro = list(numpy.array(ywok_astro) / delta_scale)

        delta_x = (numpy.array(xwok_gfa) - numpy.array(xwok_astro)) ** 2
        delta_y = (numpy.array(ywok_gfa) - numpy.array(ywok_astro)) ** 2

        xrms = numpy.sqrt(numpy.sum(delta_x) / len(delta_x))
        yrms = numpy.sqrt(numpy.sum(delta_y) / len(delta_y))
        rms = numpy.sqrt(numpy.sum(delta_x + delta_y) / len(delta_x))

        # Convert to arcsec and round up
        xrms = float(numpy.round(xrms / self.plate_scale * 3600.0, 3))
        yrms = float(numpy.round(yrms / self.plate_scale * 3600.0, 3))
        rms = float(numpy.round(rms / self.plate_scale * 3600.0, 3))

        astro_wok = pandas.DataFrame.from_records(
            {"gfa_id": camera_ids, "xwok": xwok_astro, "ywok": ywok_astro}
        )

        gfa_wok = pandas.DataFrame.from_records(
            {"gfa_id": camera_ids, "xwok": xwok_gfa, "ywok": ywok_gfa}
        )

        self.result = GuiderFit(
            gfa_wok,
            astro_wok,
            delta_ra,
            delta_dec,
            delta_rot,
            delta_scale,
            xrms,
            yrms,
            rms,
            only_radec=only_radec,
        )

        return self.result
