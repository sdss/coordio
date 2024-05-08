from __future__ import annotations

import pathlib
import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy
import numpy.typing as nt
import pandas
import scipy.ndimage
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import fit_wcs_from_points
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import sigmaclip
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, EuclideanTransform

from coordio.conv import guideToTangent, tangentToWok

from . import Site, calibration, defaults
from .astrometry import astrometrynet_quick
from .conv import tangentToGuide, wokToTangent
from .coordinate import Coordinate, Coordinate2D
from .exceptions import CoordIOError, CoordIOUserWarning
from .extraction import sextractor_quick
from .sky import ICRS, Observed
from .telescope import Field, FocalPlane
from .utils import radec2wokxy, wokxy2radec, gaia_mags2sdss_gri
from .wok import Wok
from .transforms import arg_nearest_neighbor


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
        return ICRS(observed)[0]


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

    cameras: list[int]
    gfa_wok: pandas.DataFrame
    astro_wok: pandas.DataFrame
    coeffs: list[float]
    delta_ra: float
    delta_dec: float
    delta_rot: float
    delta_scale: float
    xrms: float
    yrms: float
    rms: float
    rms_data: pandas.DataFrame
    fit: pandas.DataFrame
    fit_rms: pandas.DataFrame
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
        header = hdu[1].header

        is_proc = len(hdu) > 1 and "SOLVED" in header
        if not is_proc:
            raise CoordIOError(f"{path!s} is not a proc-gimg image.")

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
        cameras: list[int] | None = None,
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
        use_detections: list[bool] = []
        used_cameras: set[int] = set([])

        for d in self.astro_data.itertuples():
            camera_id: int = int(d.gfa_id)
            if cameras is None or camera_id in cameras:
                used_cameras.add(camera_id)

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

            # Use these detections fit unless we are excluding the camera.
            use_detections.append(cameras is None or camera_id in cameras)

        # Create arrays with all detections, then with only selected ones.
        X_all: nt.NDArray[numpy.float32] = numpy.array([xwok_gfa, ywok_gfa])
        Y_all: nt.NDArray[numpy.float32] = numpy.array([xwok_astro, ywok_astro])
        X_fit = X_all[:, use_detections]  # GFA to wok coordinates as a 2D array
        Y_fit = Y_all[:, use_detections]  # ICRS to wok coordinates as a 2D array

        try:
            c, R, t = umeyama(X_fit, Y_fit)
        except ValueError:
            if only_radec is True:
                warnings.warn(
                    "Cannot fit using Umeyama. Assuming unitary rotation and scale.",
                    CoordIOUserWarning,
                )
                c = 1.0
                R = numpy.array([[1, 0], [1, 0]])
                t = numpy.array([0.0, 0.0])
            else:
                return False

        if only_radec is True:
            # Override the translation component with a simple average translation
            t = (Y_fit - X_fit).mean(axis=1).T

        coeffs = [c, R, t]

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

        detection_id = numpy.arange(len(xwok_astro)) + 1

        astro_wok = pandas.DataFrame.from_records(
            {
                "gfa_id": camera_ids,
                "detection_id": detection_id,
                "xwok": xwok_astro.copy(),
                "ywok": ywok_astro.copy(),
                "selected": numpy.array(use_detections).astype(int),
            },
            index=["gfa_id", "detection_id"],
        )

        gfa_wok = pandas.DataFrame.from_records(
            {
                "gfa_id": camera_ids,
                "detection_id": detection_id,
                "xwok": xwok_gfa.copy(),
                "ywok": ywok_gfa.copy(),
                "selected": numpy.array(use_detections).astype(int),
            },
            index=["gfa_id", "detection_id"],
        )

        rms_df = self.calculate_rms(
            gfa_wok,
            astro_wok,
            scale=c if scale_rms else 1,
            cameras=cameras,
        )

        fit_data = c * R @ X_all + t[numpy.newaxis].T
        fit_df = pandas.DataFrame.from_records(
            {
                "gfa_id": camera_ids,
                "detection_id": detection_id,
                "xwok": fit_data[0, :],
                "ywok": fit_data[1, :],
                "selected": numpy.array(use_detections).astype(int),
            },
            index=["gfa_id", "detection_id"],
        )
        fit_rms_df = self.calculate_rms(fit_df, astro_wok, cameras=cameras)

        self.result = GuiderFit(
            list(used_cameras),
            gfa_wok,
            astro_wok,
            coeffs,
            delta_ra,
            delta_dec,
            delta_rot,
            delta_scale,
            rms_df.loc[0].xrms,
            rms_df.loc[0].yrms,
            rms_df.loc[0].rms,
            rms_df,
            fit_df,
            fit_rms_df,
            only_radec=only_radec,
        )

        return self.result

    def calculate_rms(
        self,
        gfa_wok: pandas.DataFrame,
        astro_wok: pandas.DataFrame,
        scale: float = 1,
        cameras: list[int] | None = None,
    ):
        """Calculates the RMS of the measurements.

        Parameters
        ----------
        gfa_wok
            GFA to wok coordinates data. The data frame contains three columns:
            ``gfa_id``, ``xwok``, and ``ywok``.
        astro_wok
            Astrometric solution to wok coordinates data. The data frame contains
            three columns: ``gfa_id``, ``xwok``, and ``ywok``.
        scale
            Factor by which to scale coordinates.
        cameras
            The cameras to use to calculate the global RMS.

        Returns
        -------
        rms
            A data frame with columns ``gfa_id``, ``xrms``, ``yrms``, ``rms``
            for the x, y, and combined RMS measurements for each camera. An
            additional ``gfa_id=0`` is added with the RMS measurements for all
            GFA cameras combined. RMS values are returned in mm on wok coordinates.

        """

        def calc_rms(gfa_cam: pandas.DataFrame):
            gfa_id = gfa_cam.index.values[0][0]
            astro_cam = astro_wok.loc[gfa_id]

            delta = gfa_cam - astro_cam
            xrms = numpy.sqrt(numpy.mean(delta.xwok**2))
            yrms = numpy.sqrt(numpy.mean(delta.ywok**2))
            rms = numpy.sqrt(numpy.mean(delta.xwok**2 + delta.ywok**2))

            return pandas.Series([xrms, yrms, rms], index=["xrms", "yrms", "rms"])

        astro_wok = astro_wok.copy()
        gfa_wok = gfa_wok.copy()
        gfa_wok.loc[:, ["xwok", "ywok"]] *= scale

        rms_df = gfa_wok.groupby("gfa_id").apply(calc_rms)

        if cameras is None:
            delta = gfa_wok - astro_wok
        else:
            delta = gfa_wok.loc[cameras] - astro_wok.loc[cameras]
        xrms = numpy.sqrt(numpy.mean(delta.xwok**2))
        yrms = numpy.sqrt(numpy.mean(delta.ywok**2))
        rms = numpy.sqrt(numpy.mean(delta.xwok**2 + delta.ywok**2))

        rms_df.loc[0, :] = (xrms, yrms, rms)

        return rms_df

    def plot_fit(self, filename: str | pathlib.Path):
        """Plot the fit results."""

        if self.result is None:
            raise RuntimeError("fit() needs to be run before plot_fit().")

        fig, ax = plt.subplots(2, 3)
        fig.tight_layout()

        iax = 0
        jax = 0
        direction = 1
        for cam_id in range(1, 7):
            one_arcsec = self.plate_scale / 3600.0

            camera_note = None
            color = "k"

            if cam_id in self.result.fit.index:
                X = self.result.fit.loc[cam_id].xwok
                Y = self.result.fit.loc[cam_id].ywok

                U = self.result.astro_wok.loc[cam_id].xwok
                V = self.result.astro_wok.loc[cam_id].ywok
                U = U - X
                V = V - Y

                rms_mm = float(self.result.fit_rms.loc[cam_id].rms)
                rms = numpy.round(rms_mm / self.plate_scale * 3600, 4)

                if self.result.cameras and cam_id not in self.result.cameras:
                    camera_note = "Not fit"
                    color = "r"

            else:
                X = Y = U = V = []
                rms = -999
                camera_note = "Disabled / not solved"
                color = "r"

            q = ax[iax][jax].quiver(
                X,
                Y,
                U,
                V,
                color=color,
                units="xy",
                angles="xy",
                scale_units="inches",
                scale=0.15,
            )

            if cam_id == 1:
                ax[iax][jax].quiverkey(q, 0.2, 0.9, one_arcsec, label="1 arcsec")

            text1 = ax[iax][jax].text(
                0.5,
                0.95,
                f"RMS: {rms}",
                color=color,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax[iax][jax].transAxes,
            )
            text1.set_bbox(dict(facecolor="white", alpha=1, edgecolor="none"))

            if camera_note:
                text2 = ax[iax][jax].text(
                    0.5,
                    0.90,
                    camera_note,
                    color=color,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax[iax][jax].transAxes,
                )
                text2.set_bbox(dict(facecolor="white", alpha=1, edgecolor="none"))

            ax[iax][jax].set_title(f"GFA{cam_id}", color=color)

            jax += direction
            if jax == 3:
                jax = -1
                iax = 1
                direction = -1

        fig.savefig(str(filename))


class SolvePointing:
    """
    Determines the best-fit telescope boresight pointing from a set of GFA
    images.  Requires specifying a guess or reference pointing as a starting
    point.  The reference pointing can be assumed from the gfa image headers.

    Parameters
    -------------
    raCen : float | None = None
        RA of field center in degrees.  If None, user must specify pt_source
    decCen : float | None = None
        Dec of field center in degrees.  If None, user must specify pt_source
    paCen : float | None = None
        PA of field center in degrees, If None, user must specify pt_src
    pt_source : str | None = None
        either "design" or "telescope".  Design will solve using robostrategy
        intented field center, telescope will solve using the reported
        sky position of the telescope boresight. Data pulled from gcam headers.
        If raCen, decCen, and/or paCen are specified, those values are used
        instead of data from image headers
    offset_ra : float = 0
        offset in ra to add to the supplied field center (true arcseconds)
    offset_dec : float = 0
        offset in dec to add to the supplied field center (arcseconds)
    offset_pa : float = 0
        offset in pa to subtract from supplied field angle (arcseconds)
    scale : float | None = None
        scale factor for the field
    db_conn_st : str
        connection string for the db
    db_tab_name : str
        table in database with gaia info
    """
    def __init__(
        self,
        raCen: float | None = None,
        decCen: float | None = None,
        paCen: float | None = None,
        scale: float | None = None,
        pt_source: str | None = None,
        offset_ra: float = 0,
        offset_dec: float = 0,
        offset_pa: float = 0,
        db_conn_st: str = "postgresql://sdss_user@operations.sdss.org/sdss5db",
        db_tab_name: str = "catalogdb.gaia_dr2_source"
    ):

        if pt_source is None and None in [raCen, decCen, paCen]:
            raise RuntimeError(
                "must specify either pt_source or raCen, decCen, paCen"
            )
        if raCen is None or decCen is None or paCen is None:
            if pt_source not in ["telescope", "design"]:
                raise RuntimeError(
                    "pt_source must be either 'telescope' or 'design'"
                )

        self._raCen = raCen
        self._decCen = decCen
        self._paCen = paCen
        self.pt_source = pt_source
        self.offset_ra = offset_ra
        self.offset_dec = offset_dec
        self.offset_pa = offset_pa
        self.scale = scale
        self.db_conn_st = db_conn_st
        self.db_tab_name = db_tab_name

        self.GAIA_EPOCH = 2457206

        # populated by add_gimg
        self.fieldInit = False
        self.observatory = None
        self.raCenRef = None
        self.decCenRef = None
        self.paCenRef = None
        self.obsTimeRef = None
        self.ipa = None
        self.imgNum = None
        self.gfaHeaders = {}
        self.gfaWCS = {}
        self.allCentroids = pandas.DataFrame()

        # populated or modified by solve
        self.allGaia = pandas.DataFrame()
        self.skyDistThresh = None  # arcseconds for a match hit
        self.raCenMeas = None
        self.decCenMeas = None
        self.paCenMeas = None
        self.scaleMeas = None
        self.altCenMeas = None
        self.azCenMeas = None
        # self.fit_rms = None
        self.matchedSources = pandas.DataFrame()
        self.translation = numpy.array([0.0, 0.0])
        self.rotation = 0.0

    def getMetadata(self):
        """Get a list of data that can be easily stuffed in a fits
        header.
        """

        metaDataList = [
            ("SOL_RA", self.raCenMeas, "solved RA of pointing (deg)"),
            ("SOL_DEC", self.decCenMeas, "solved Dec of pointing (deg)"),
            ("SOL_PA", self.paCenMeas, "solved PA of pointing (deg)"),
            ("SOL_SCL", self.scaleMeas, "solved scale factor"),
            ("SOL_ALT", self.altCenMeas, "solved alt of pointing (deg)"),
            ("SOL_AZ", self.azCenMeas, "solved az of pointing (deg)"),
            ("SOL_GRMS", self.guide_rms, "guide rms between reference and solved (mm)"),
            ("SOL_FRMS", self.fit_rms, "rms of fit (mm)"),
            ("REF_RA", self.raCenRef, "reference RA (deg)"),
            ("REF_DEC", self.decCenRef, "reference Dec (deg)"),
            ("REF_PA", self.paCenRef, "reference PA (deg)"),
            ("REF_SCL", self.scale, "reference scale factor"),
            ("TAI_MID", self.obsTimeRef.mjd * 24 * 60 * 60, "tai seconds at middle of exposure"),
            # ("N_WCS", len(self.gfaWCS), "number of GFAs with astronet solutions"),
            ("N_STARS", len(self.matchedSources), "number of stars used in fit"),
            ("N_GFAS", len(set(self.matchedSources.gfaNum)), "number of GFAs used in fit"),
            ("NITR_WCS", self.nIterWCS, "number of iterations for wcs solve"),
            ("NITR_ALL", self.nIterAll, "number of iterations for full solve")
        ]

        return metaDataList

    def median_zeropoint(self, gfaNum):
        _df = self.matchedSources[self.matchedSources.gfaNum == gfaNum]
        return numpy.nanmedian(_df.zp)

    @property
    def plate_scale(self):
        # mm per deg
        return defaults.PLATE_SCALE[self.observatory]

    # @property
    # def pixel_scale(self):
    #     # arcsec per pixel
    #     return 1.0 / self.plate_scale * defaults.GFA_PIXEL_SIZE * 3600 / 1000

    @property
    def wokDistThresh(self):
        # mm
        return self.skyDistThresh / 3600. * self.plate_scale

    @property
    def xyWokPredictRef(self):
        output = radec2wokxy(
            self.matchedSources.ra.to_numpy(),
            self.matchedSources.dec.to_numpy(), self.GAIA_EPOCH,
            "GFA", self.raCenRef, self.decCenRef, self.paCenRef,
            self.observatory, self.obsTimeRef.jd, focalScale=self.scaleMeas,
            pmra=self.matchedSources.pmra.to_numpy(),
            pmdec=self.matchedSources.pmdec.to_numpy(),
            parallax=self.matchedSources.parallax.to_numpy(), fullOutput=True
        )

        xPred = output[0]
        yPred = output[1]
        return xPred, yPred

    def fitWCS(self, gfaNum):
        # return a dictionary keyed by GFA ID

        _df = self.matchedSources[self.matchedSources.gfaNum==gfaNum]

        if len(_df) < 3:
            # require at least 3 stars for a wcs solution?
            return None

        xy = _df[["x", "y"]].to_numpy()
        raDec = _df[["ra", "dec"]].to_numpy()

        # Build the WCS.
        reference_skycoord_valid = SkyCoord(
            ra=raDec[:, 0],
            dec=raDec[:, 1],
            unit="deg",
        )

        wcs = fit_wcs_from_points(
            (xy[:, 0], xy[:, 1]),
            reference_skycoord_valid,
        )

        return wcs


    def rms_df(self, fit=False):
        """
        returns rms for expected vs solved pointing
        ----------
        A data frame with columns ``gfa_id``, ``xrms``, ``yrms``, ``rms``
        for the x, y, and combined RMS measurements for each camera. An
        additional ``gfa_id=0`` is added with the RMS measurements for all
        GFA cameras combined. RMS values are returned in mm on wok coordinates.
        """
        df = self.matchedSources.copy()
        if not fit:
            # reference rms
            xPred, yPred = self.xyWokPredictRef
            df["xWokPred"] = xPred
            df["yWokPred"] = yPred

        _gfa_id = []
        _xrms = []
        _yrms = []
        _rms = []

        for gfaNum in list(set(df.gfaNum)):
            _df = df[df.gfaNum==gfaNum]
            dx = _df.xWokPred - _df.xWokMeas
            dy = _df.yWokPred - _df.yWokMeas

            _gfa_id.append(gfaNum)
            _xrms.append(numpy.sqrt(numpy.mean(dx**2)))
            _yrms.append(numpy.sqrt(numpy.mean(dy**2)))
            _rms.append(numpy.sqrt(numpy.mean(dx**2+dy**2)))

        # finally add total rms under gfa_id = 0
        dx = df.xWokPred - df.xWokMeas
        dy = df.yWokPred - df.yWokMeas

        _gfa_id.append(0)
        _xrms.append(numpy.sqrt(numpy.mean(dx**2)))
        _yrms.append(numpy.sqrt(numpy.mean(dy**2)))
        _rms.append(numpy.sqrt(numpy.mean(dx**2+dy**2)))

        out = pandas.DataFrame({
            "gfa_id": _gfa_id,
            "xrms": _xrms,
            "yrms": _yrms,
            "rms": _rms,
        })
        # out.set_index("gfa_id")

        return out.set_index("gfa_id")

    @property
    def guide_rms(self):
        # RMS error between stars in reference frame and solved frame
        # the measured scale is applied to remove a scale dependence
        # on rms
        xPred, yPred = self.xyWokPredictRef

        xFit = self.matchedSources.xWokPred.to_numpy()
        yFit = self.matchedSources.yWokPred.to_numpy()

        rms = numpy.sqrt(
            numpy.mean((xPred - xFit)**2 + (yPred - yFit)**2)
        )
        return rms

    @property
    def fit_rms(self):
        return numpy.sqrt(numpy.mean(self.matchedSources.dr**2))

    @property
    def guide_rms_sky(self):
        # arcsec
        return self.guide_rms / self.plate_scale * 3600

    @property
    def fit_rms_sky(self):
        # arcsec
        return self.fit_rms / self.plate_scale * 3600

    @property
    def used_cameras(self):
        gfaNums = set(self.matchedSources.gfaNum)
        return sorted(list(gfaNums))

    @property
    def n_stars_used(self):
        return len(self.matchedSources)

    @property
    def gfa_wok(self):
        gfa_id = self.matchedSources.gfaNum.to_numpy(),
        detection_id = numpy.arange(len(gfa_id))
        x_wok = self.matchedSources.xWokMeas.to_numpy(),
        y_wok = self.matchedSources.yWokMeas.to_numpy(),
        selected = numpy.ones(len(gfa_id))
        df = pandas.DataFrame(
            {
                "gfa_id": gfa_id,
                "detection_id": detection_id,
                "xwok": x_wok,
                "ywok": y_wok,
                "selected": selected,
            },
            index=["gfa_id", "detection_id"],
        )
        return df

    @property
    def astro_wok(self):
        gfa_id = self.matchedSources.gfaNum.to_numpy()
        detection_id = numpy.arange(len(gfa_id))
        x_wok, y_wok = self.xyWokPredictRef
        selected = numpy.ones(len(gfa_id))
        df = pandas.DataFrame.from_records(
            {
                "gfa_id": gfa_id,
                "detection_id": detection_id,
                "xwok": x_wok,
                "ywok": y_wok,
                "selected": selected,
            },
            index=["gfa_id", "detection_id"],
        )
        return df

    @property
    def fit_data(self):
        gfa_id = self.matchedSources.gfaNum.to_numpy()
        detection_id = numpy.arange(len(gfa_id))
        x_wok = self.matchedSources.xWokPred.to_numpy()
        y_wok = self.matchedSources.yWokPred.to_numpy()
        selected = numpy.ones(len(gfa_id))
        df = pandas.DataFrame.from_records(
            {
                "gfa_id": gfa_id,
                "detection_id": detection_id,
                "xwok": x_wok,
                "ywok": y_wok,
                "selected": selected,
            },
            index=["gfa_id", "detection_id"],
        )
        return df

    @property
    def guider_fit(self):
        rms_df = self.rms_df()
        fit_rms_df = self.rms_df(fit=True)

        result = GuiderFit(
            list(self.used_cameras),
            self.gfa_wok,
            self.astro_wok,
            self.coeffs,
            self.delta_ra,
            self.delta_dec,
            self.delta_rot,
            self.delta_scale,
            rms_df.loc[0].xrms,
            rms_df.loc[0].yrms,
            rms_df.loc[0].rms,
            rms_df,
            self.fit_data,
            fit_rms_df,
            only_radec=False,
        )

        return result

    @property
    def coeffs(self):
        rotMat = numpy.array([
            [numpy.cos(self.rotation), -numpy.sin(self.rotation)],
            [numpy.sin(self.rotation), numpy.cos(self.rotation)]
        ])
        return [self.scaleMeas, rotMat, self.translation]

    @property
    def delta_ra(self):
        cosDec = numpy.cos(numpy.radians(self.decCenMeas))
        return (self.raCenMeas - self.raCenRef) * 3600 * cosDec

    @property
    def delta_dec(self):
        return (self.decCenMeas - self.decCenRef) * 3600

    @property
    def delta_rot(self):
        return (self.paCenMeas - self.paCenRef) * 3600

    @property
    def delta_scale(self):
        return self.scaleMeas

    def pix2wok(self, x, y, gfaNum):
        g = calibration.gfaCoords.loc[(self.observatory, gfaNum), :]
        zt = numpy.zeros(len(x))
        b = g[["xWok", "yWok", "zWok"]].to_numpy()
        iHat = g[["ix", "iy", "iz"]].to_numpy()
        jHat = g[["jx", "jy", "jz"]].to_numpy()
        kHat = g[["kx", "ky", "kz"]].to_numpy()

        xt, yt = guideToTangent(x, y)

        xw, yw, zw = tangentToWok(
            xt, yt, zt,
            b, iHat, jHat, kHat
        )
        return xw, yw

    def wok2pix(self, x, y, gfaNum):
        g = calibration.gfaCoords.loc[(self.observatory, gfaNum), :]
        zw = numpy.zeros(len(x))
        b = g[["xWok", "yWok", "zWok"]].to_numpy()
        iHat = g[["ix", "iy", "iz"]].to_numpy()
        jHat = g[["jx", "jy", "jz"]].to_numpy()
        kHat = g[["kx", "ky", "kz"]].to_numpy()

        xt, yt, zt = wokToTangent(
            x, y, zw,
            b, iHat, jHat, kHat
        )

        xyPix = tangentToGuide(xt,yt) #,y_0=float(g.y_0), y_1=float(g.y_1))
        return xyPix

    def gfa2radec(self, gfaNum):
        """ get the ra/dec of ccd center, if wcs is available use that """
        xw, yw = self.pix2wok(numpy.array([1024]), numpy.array([1024]), gfaNum)
        ra, dec, fieldWarn = wokxy2radec(
            numpy.array(xw), numpy.array(yw), "GFA", self.raCenMeas,
            self.decCenMeas, self.paCenMeas, self.observatory,
            self.obsTimeRef.jd, focalScale=self.scaleMeas
        )
        return ra[0], dec[0]

    def initializeField(self, observatory, imgNum, hdr):
        self.observatory = observatory

        if self.pt_source == "design":
            _raCen = hdr["RAFIELD"]
            _decCen = hdr["DECFIELD"]
            _paCen = hdr["FIELDPA"]
        elif self.pt_source == "telescope":
            # warning assumes APO for now
            # will need to modify for LCO
            _raCen = hdr["RA"]  # RA is ObjNetPos, RADEG is ObjPos (tcc kws)
            _decCen = hdr["DEC"]
            _paCen = hdr["ROTPOS"] # wont work for LCO no ROTPOS header

        # if no field center was specified, use field center from
        # header
        if self._raCen is None:
            self._raCen = _raCen
        if self._decCen is None:
            self._decCen = _decCen
        if self._paCen is None:
            self._paCen = _paCen

        # apply offsets if specified
        ddec = self.offset_dec / 3600.
        self.decCenRef = self._decCen - ddec
        dra = self.offset_ra / 3600. / numpy.cos(numpy.radians(self.decCenRef))
        self.raCenRef = self._raCen - dra
        self.paCenRef = self._paCen - self.offset_pa / 3600.

        # initalize the starting point for the iteration at the reference
        # pointing
        self.raCenMeas = self.raCenRef
        self.decCenMeas = self.decCenRef
        self.paCenMeas = self.paCenRef
        if self.scale is None:
            self.scale = defaults.SITE_TO_SCALE[self.observatory]
        self.scaleMeas = self.scale

        self.tStart = Time(hdr["DATE-OBS"], format="iso", scale="tai")
        self.exptime = hdr["EXPTIMEN"]
        dt = TimeDelta(self.exptime/2.0, format="sec", scale="tai")
        # choose exposure end as reference time because
        # it better matches the telescope headers (eg for a pointing model)
        self.obsTimeRef = self.tStart + dt
        # dt = TimeDelta(hdr["EXPTIME"]/2., format="sec", scale="tai")
        # self.obsTimeMid = self.tStart + dt
        # # tcc need this in seconds
        # self.taiMid = self.obsTimeRef.mjd * 24 * 60 * 60
        self.imgNum = imgNum
        self.ipa = hdr["IPA"]
        if self.observatory == "APO":
            self.fieldCenAltRef = hdr["ALT"]
            self.fieldCenAzRef = hdr["AZ"]
        else:
            self.fieldCenAltRef = None
            self.fieldCenAzRef = None
        self.initField = True

    def add_gimg(
        self,
        img_path: str | pathlib.Path,
        centroids: pandas.DataFrame | None = None,
        wcs: WCS | None = None,
        gain: float | None = None
    ):
        """
        Add gfa data to be used in fitting.  Don't add a gfa you don't want
        incuded in fitting.  Strict checking is done to make sure all images
        added have the same exposure number.  Expects SDSS style naming and
        headers, should work for both gimg*.fits and proc-gimg*.fits files.

        Parameters
        --------------
        img_path
            path to a gimg or proc-gimg file
        centroids
            sep style extracted parameters in pandas.DataFrame form.
            Additionally if a "CENTROIDS" extension is present in the fits
            file, those will be used.  If not present, centroids are extracted
            from the data.
        wcs
            a WCS object
        gain
            electrons per adu for this camera
        """
        img_path = str(img_path)
        ff = fits.open(img_path)
        if centroids is None or len(centroids)==0:
            # extract centroids
            centroids = sextractor_quick(ff[1].data)
        if len(centroids) == 0:
            # no centroids found skip this gfa
            ff.close()
            return

        hdr = dict(ff[1].header)

        tokens = img_path.strip(".fits").split("/")[-1].split("-")
        imgNum = int(tokens[-1])
        gfaNum = int(tokens[-2].strip("gfa").strip("n").strip("s"))
        self.gfaHeaders[gfaNum] = hdr

        if not self.fieldInit:
            if tokens[-2].endswith("n"):
                observatory = "APO"
            else:
                observatory = "LCO"
            self.initializeField(observatory, imgNum, hdr)

        if imgNum != self.imgNum:
            raise RuntimeError(
                "May not add gfa files with differing img numbers"
            )

        # fwhm = 2 * (numpy.log(2) * (centroids.a**2 + centroids.b**2))**0.5
        # centroids["fwhm"] = fwhm

        centroids["gfaNum"] = gfaNum
        # remove saturated sources
        # centroids = centroids[centroids.peak < 55000].reset_index(drop=True)
        # calculate wok coordinates for each centroid
        xWokMeas, yWokMeas = self.pix2wok(
            centroids.x.to_numpy(),
            centroids.y.to_numpy(),
            gfaNum
        )
        centroids["xWokMeas"] = xWokMeas
        centroids["yWokMeas"] = yWokMeas

        if wcs is not None:
            self.gfaWCS[gfaNum] = wcs
            # use wcs to calculate on-sky locations
            # of centroids note these are off by 0.5
            # but leaving so that cherno default rotation and fudge factor
            # don't need to change
            xyCents = centroids[["x", "y"]].to_numpy()
            raDecMeas = numpy.array(wcs.pixel_to_world_values(xyCents))
            centroids["raMeas"] = raDecMeas[:, 0]
            centroids["decMeas"] = raDecMeas[:, 1]
            # import pdb; pdb.set_trace()

        centroids["gain"] = gain

        self.allCentroids = pandas.concat(
            [self.allCentroids, centroids], ignore_index=True
        )

        # if wcs is not None:
        #     self.gfaWCS[gfaNum] = WCS(open(wcs).read())

        ff.close()

    def getGaiaSources(self, ra, dec, radius=0.16, magLimit=18):
        import time; tstart=time.time()
        query = "SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, bp_rp"
        query += " FROM %s" % self.db_tab_name
        query += " WHERE q3c_radial_query"
        query += "(ra, dec, %.4f, %.4f, %.4f)" % (ra, dec, radius)
        query += " AND phot_g_mean_mag < %.2f" % magLimit
        # print(query)
        df = pandas.read_sql(query, self.db_conn_st)
        # print("getGaiaSources took", time.time()-tstart)
        return df.dropna().reset_index(drop=True)

        # self.allGaia = pandas.concat(allGaia)
        # print("got", len(self.allGaia), " sources")

    def _updateZP(self):
        xWokPredRef, yWokPredRef = self.xyWokPredictRef
        self.matchedSources["xWokPredRef"] = xWokPredRef
        self.matchedSources["yWokPredRef"] = yWokPredRef
        sdss_g, sdss_r, sdss_i = gaia_mags2sdss_gri(
            gaia_g=self.matchedSources.phot_g_mean_mag.to_numpy(),
            gaia_bp_rp=self.matchedSources.bp_rp.to_numpy()
        )
        self.matchedSources["sdss_r"] = sdss_r

        # when re-analyzing old images, aperflux is not available in the
        # centroids table unless you re-extract sources.  For this case, just
        # replace it with flux which has always been there.
        if "aperflux" in list(self.matchedSources.columns):
            flux = self.matchedSources.aperflux
        else:
            flux = self.matchedSources.flux

        zp = 2.5 * numpy.log10(
            flux * self.matchedSources.gain / self.exptime
        ) + sdss_r
        self.matchedSources["zp"] = zp

    def _matchWCS(self):
        """Note, only works if there was at least 1 wcs solution provided!
        """
        # allGaia = []
        # for gfaNum, wcs in self.gfaWCS.items():
        #     ra, dec = wcs.pixel_to_world_values([[1024,1024]])[0]
        #     # print("wcs ra/decs", gfaNum, ra,dec)
        #     gaiaDF = self.getGaiaSources(ra,dec)
        #     gaiaDF["gfaNum"] = gfaNum
        #     allGaia.append(gaiaDF)
        # self.allGaia = pandas.concat(allGaia)
        # self.allGaia = self.allGaia[self.allGaia.gfaNum==1]

        raDecGaia = self.allGaia[["ra", "dec"]].to_numpy()
        # na values in centroids mean no wcs soln was present, so drop them
        cents = self.allCentroids.dropna().reset_index(drop=True)
        # cents = cents[cents.gfaNum==1]
        raDecMeas = cents[["raMeas", "decMeas"]].to_numpy()

        matches, indices, minDists = arg_nearest_neighbor(raDecMeas, raDecGaia)

        gaiaNN = self.allGaia.iloc[indices].reset_index(drop=True)

        matched = pandas.concat([cents, gaiaNN], axis=1)
        matched = matched.loc[:, ~matched.columns.duplicated()].copy()

        # only keep matches closer than specified threshold
        dra = (matched.ra - matched.raMeas)
        dra = dra * numpy.cos(numpy.radians(matched.dec))
        ddec = (matched.dec - matched.decMeas)
        matched["drSky"] = numpy.sqrt(dra**2 + ddec**2) * 3600

        matched = matched[matched.drSky < self.skyDistThresh].reset_index(drop=True)

        # plt.figure(figsize=(8,8))
        # plt.plot(self.allGaia.ra, self.allGaia.dec, 'o', mfc="none", mec="red")
        # plt.plot(matched.raMeas, matched.decMeas, 'x', color="black")
        # plt.savefig("skymatch_%i.png"%self.nIterWCS, dpi=200)
        # plt.close("all")

        # for gfaNum in list(set(matched.gfaNum)):
        #     plt.figure(figsize=(8,8))
        #     ag = self.allGaia[self.allGaia.gfaNum==gfaNum]
        #     mm = matched[matched.gfaNum==gfaNum]
        #     plt.plot(ag.ra, ag.dec, 'o', mfc="none", mec="red")
        #     plt.plot(mm.raMeas, mm.decMeas, 'x', color="black")
        #     plt.savefig("skymatch_%i_gfa%i.png"%(self.nIterWCS, gfaNum), dpi=200)
        #     plt.close("all")

        # import pdb; pdb.set_trace()
        output = radec2wokxy(
            matched.ra.to_numpy(), matched.dec.to_numpy(), self.GAIA_EPOCH,
            "GFA", self.raCenMeas, self.decCenMeas, self.paCenMeas,
            self.observatory, self.obsTimeRef.jd, focalScale=self.scaleMeas,
            pmra=matched.pmra.to_numpy(), pmdec=matched.pmdec.to_numpy(),
            parallax=matched.parallax.to_numpy(), fullOutput=True
        )

        xPred = output[0]
        yPred = output[1]
        fieldWarn = output[2]
        ha = output[3]
        pa = output[4]
        xFocal = output[5]
        yFocal = output[6]
        thetaField = output[7]
        phiField = output[8]
        altPred = output[9]
        azPred = output[10]
        self.altCenMeas = output[11]
        self.azCenMeas = output[12]

        matched["xWokPred"] = xPred
        matched["yWokPred"] = yPred
        matched["dx"] = matched.xWokMeas - matched.xWokPred
        matched["dy"] = matched.yWokMeas - matched.yWokPred
        matched["dr"] = numpy.sqrt(matched.dx**2 + matched.dy**2)

        # plt.figure(figsize=(8,8))
        # plt.plot(matched.xWokPred, matched.yWokPred, 'o', mfc="none", mec="red")
        # plt.plot(matched.xWokMeas, matched.yWokMeas, 'x', color="black")
        # plt.savefig("wok_wcs_%i.png"%self.nIterWCS, dpi=200)
        # plt.close("all")

        # for gfaNum in list(set(matched.gfaNum)):
        #     plt.figure(figsize=(8,8))
        #     mm = matched[matched.gfaNum==gfaNum]
        #     plt.figure(figsize=(8,8))
        #     plt.plot(mm.xWokPred, mm.yWokPred, 'o', mfc="none", mec="red")
        #     plt.plot(mm.xWokMeas, mm.yWokMeas, 'x', color="black")
        #     plt.savefig("wok_wcs_%i_gfa%i.png"%(self.nIterWCS, gfaNum), dpi=200)
        #     plt.close("all")


        self.matchedSources = matched
        self._updateZP()
        # import pdb; pdb.set_trace()
        # print("sources matched", len(self.matchedSources))

    def _matchWok(self):
        output = radec2wokxy(
            self.allGaia.ra.to_numpy(), self.allGaia.dec.to_numpy(), self.GAIA_EPOCH,
            "GFA", self.raCenMeas, self.decCenMeas, self.paCenMeas,
            self.observatory, self.obsTimeRef.jd, focalScale=self.scaleMeas,
            pmra=self.allGaia.pmra.to_numpy(), pmdec=self.allGaia.pmdec.to_numpy(),
            parallax=self.allGaia.parallax.to_numpy(), fullOutput=True
        )

        xPred = output[0]
        yPred = output[1]
        fieldWarn = output[2]
        ha = output[3]
        pa = output[4]
        xFocal = output[5]
        yFocal = output[6]
        thetaField = output[7]
        phiField = output[8]
        altPred = output[9]
        azPred = output[10]
        self.fieldCenAltMeas = output[11]
        self.fieldCenAzMeas = output[12]

        self.allGaia["xWokPred"] = xPred
        self.allGaia["yWokPred"] = yPred

        xyWokMeas = self.allCentroids[["xWokMeas", "yWokMeas"]].to_numpy()
        xyWokPredict = self.allGaia[["xWokPred", "yWokPred"]].to_numpy()
        matches, indices, minDists = arg_nearest_neighbor(xyWokMeas, xyWokPredict)

        gaiaNN = self.allGaia.iloc[indices].reset_index(drop=True)
        matched = pandas.concat([self.allCentroids, gaiaNN], axis=1)
        matched = matched.loc[:, ~matched.columns.duplicated()].copy()

        # reject matches above the threshold
        matched["dx"] = matched.xWokMeas - matched.xWokPred
        matched["dy"] = matched.yWokMeas - matched.yWokPred
        matched["dr"] = numpy.sqrt(matched.dx**2 + matched.dy**2)

        # plt.figure()
        # plt.hist(matched.dr, bins=200)
        # plt.savefig("wokdist.png")
        # plt.close("all")


        # for gfaNum in range(1,7):
        #     g1 = matched[matched.gfaNum==gfaNum]
        #     plt.figure(figsize=(8,8))
        #     plt.plot(g1.xWokPred, g1.yWokPred, 'o', mfc="none", mec="red")
        #     plt.plot(g1.xWokMeas, g1.yWokMeas, 'x', color="black")
        #     plt.savefig("wokmatch_%i_gfa%i.png"%(self.nIterAll, gfaNum), dpi=200)
        #     plt.close("all")

        # plt.figure(figsize=(8,8))
        # plt.plot(matched.xWokPred, matched.yWokPred, 'o', mfc="none", mec="red")
        # plt.plot(matched.xWokMeas, matched.yWokMeas, 'x', color="black")
        # plt.savefig("wokmatch_%i.png"%self.nIterAll, dpi=200)
        # plt.close("all")

        goodMatches = matched[matched.dr < self.wokDistThresh]
        # import pdb; pdb.set_trace()
        self.matchedSources = goodMatches.reset_index(drop=True)
        self._updateZP()

    def _iter(self):
        # matched_df = self.matchedSources

        nGFAS = len(set(self.matchedSources.gfaNum))

        # x = matched_df.xWokMeas.to_numpy()
        # y = matched_df.yWokMeas.to_numpy()
        # dx = xPred - x
        # dy = yPred - y
        # measRMS = numpy.sqrt(numpy.mean(dx**2 + dy**2)) * 1000
        # print("meas rms", measRMS)

        # plt.figure(figsize=(8,8))
        # plt.quiver(x,y,dx,dy,angles="xy",units="xy", width=.2)
        # plt.axis("equal")
        # plt.show()
        xyMeas = self.matchedSources[["xWokMeas", "yWokMeas"]].to_numpy()
        xyPred = self.matchedSources[["xWokPred", "yWokPred"]].to_numpy()

        if nGFAS > 1:
            st = SimilarityTransform()
            st.estimate(xyPred, xyMeas)
            xyFit = st(xyPred)
            dxy = xyFit - xyMeas
            dRot = st.rotation
            dTrans = st.translation
            # print("dTrans", dTrans, numpy.degrees(dRot), st.scale)
            dScale = st.scale
            self.rotation += dRot
            self.translation += dTrans
        else:
            dTrans = numpy.mean(xyMeas - xyPred, axis=0)
            dxy = xyPred + dTrans - xyMeas
            dRot = 0
            dScale = 1

        # print(st.translation, numpy.degrees(st.rotation), st.scale)

        # self.fit_rms = numpy.sqrt(
        #     numpy.mean(numpy.sum(dxy**2, axis=1))
        # )
        # print("%.8f"%self.fit_rms, len(self.matchedSources), set(self.matchedSources.gfaNum))

        # plt.figure(figsize=(8,8))
        # plt.quiver(x,y,dx,dy,angles="xy", units="xy", width=.2)
        # plt.axis("equal")

        # update field center
        self.paCenMeas += numpy.degrees(dRot)
        rotRad = numpy.radians(self.paCenMeas)
        cosRot = numpy.cos(rotRad)
        sinRot = numpy.sin(rotRad)
        # rotate by PA to get offset directions along ra/dec
        rotMat = numpy.array([
            [cosRot, sinRot],
            [-sinRot, cosRot]
        ])
        dra, ddec = rotMat @ dTrans
        self.decCenMeas -= ddec / self.plate_scale
        dra = dra / self.plate_scale
        dra = dra / numpy.cos(numpy.radians(self.decCenMeas))
        self.raCenMeas -= dra
        self.scaleMeas /= dScale
        # print("len allGaia", len(self.allGaia))

        # plt.show()

    def solve(
        self,
        skyDistThresh: float = 3,  # arcseconds
    ):
        # print("field initial solve\n------------\n")
        self.skyDistThresh = skyDistThresh
        # initialize field center to
        # user supplied reference


        self.nIterWCS = 0
        self.nIterAll = 0
        if len(self.gfaWCS) > 0:
            # match wcs gets gaia sources for matching based on
            # wcs if available
            allGaia = []
            for gfaNum, wcs in self.gfaWCS.items():
                ra, dec = wcs.pixel_to_world_values([[1024,1024]])[0]
                # print("wcs ra/decs", gfaNum, ra,dec)
                gaiaDF = self.getGaiaSources(ra,dec)
                gaiaDF["gfaNum"] = gfaNum
                allGaia.append(gaiaDF)
            self.allGaia = pandas.concat(allGaia)

            lastRMS = None
            self._matchWCS()
            for ii in range(10):
                # maximum of 10 iters
                self._iter()
                self._matchWCS()
                self.nIterWCS += 1
                # print("wcs fit_rms", ii, len(self.matchedSources), self.fit_rms)
                if lastRMS is not None:
                    if numpy.abs(lastRMS - self.fit_rms) < 0.003:
                        break
                lastRMS = self.fit_rms

        if len(self.gfaHeaders) != len(self.gfaWCS):
            # at least one chip missing a WCS,
            # add in gfa's without WCS solution  and re-fit
            gaiaGFAs = list(set(self.allGaia.gfaNum))
            for gfaNum in self.gfaHeaders.keys():
                if gfaNum in gaiaGFAs:
                    continue  # already got gaia sources for this GFA
                ra, dec = self.gfa2radec(gfaNum)
                gaiaDF = self.getGaiaSources(ra, dec)
                gaiaDF["gfaNum"] = gfaNum
                self.allGaia = pandas.concat([self.allGaia, gaiaDF])

            lastRMS = None
            self._matchWok()
            for ii in range(10):
                # print("coordio iter", ii)

                self._iter()
                self._matchWok()
                self.nIterAll += 1
                # print("coordion fit_rms", ii, len(self.matchedSources), self.fit_rms)
                if lastRMS is not None:
                    if numpy.abs(lastRMS - self.fit_rms) < 0.003:
                        break
                lastRMS = self.fit_rms

        return self.guider_fit

    def reSolve(
        self,
        imgNum: int,
        obsTimeRef: Time,
        exptime: float,
        newCentroids: pandas.DataFrame,
        skyDistThresh: float = 3,  # arcseconds
    ):
        """ warning only use if:
                imgNum = self.imgNum + 1 (next image in sequence)
                previous guide rms was small (eg < 1 arcsecond)
                pointing reference is same as previous frame
                same gfas included as previous frame

        centroids needs (at least) columns x,y,peak,gfaNum
        date_obs, exptime come from GFA header

        this skips all db queries and relies on pre-existing
        guide star table and doesn't require any pre-existing WCS
        solutions
        """
        self.skyDistThresh = skyDistThresh
        self.imgNum = imgNum
        self.obsTimeRef = obsTimeRef
        self.exptime = exptime

        # self.saved_nstars = self.n_stars_used
        # self.saved_fit_rms = self.fit_rms
        # self.saved_used_cameras = self.used_cameras

        # print("resolving field")

        dfList = []
        for gfaNum, group in newCentroids.groupby("gfaNum"):
            # group = group[group.peak < 55000].reset_index(drop=True)
            # calculate wok coordinates for each centroid
            xWokMeas, yWokMeas = self.pix2wok(
                group.x.to_numpy(),
                group.y.to_numpy(),
                gfaNum
            )
            group["xWokMeas"] = xWokMeas
            group["yWokMeas"] = yWokMeas
            dfList.append(group)

        self.allCentroids = pandas.concat(dfList).reset_index(drop=True)

        lastRMS = None
        self.nIterAll = 0
        self.nIterWCS = 0

        self._matchWok()
        for ii in range(10):
            # print("coordio iter", ii)
            self._iter()
            self._matchWok()
            self.nIterAll += 1
            # print("coordion fit_rms", ii, len(self.matchedSources), self.fit_rms)
            if lastRMS is not None:
                if numpy.abs(lastRMS - self.fit_rms) < 0.003:
                    break
            lastRMS = self.fit_rms

        return self.guider_fit


            # compute wok position of each centroid
        # return self.result




