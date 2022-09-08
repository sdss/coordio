from __future__ import annotations

import pathlib
import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy
import pandas
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning

from coordio.conv import guideToTangent, tangentToWok

from . import Site, calibration, defaults
from .astrometry import astrometrynet_quick
from .coordinate import Coordinate, Coordinate2D
from .exceptions import CoordIOError
from .extraction import sextractor_quick
from .sky import ICRS, Observed
from .telescope import Field, FocalPlane
from .utils import radec2wokxy
from .wok import Wok


__all__ = ["Guide", "GuiderFitter", "umeyama"]


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
):
    """Converts from a GFA pixel to observed RA/Dec."""

    site = Site(observatory)
    site.set_time()

    wavelength = defaults.INST_TO_WAVE["GFA"]

    wok_coords = gfa_to_wok(observatory, x_pix, y_pix, gfa_id)

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

    return (observed.ra[0], observed.dec[0])


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

    def add_astro(self, camera: str | int, ra: float, dec: float, obstime: float):
        """Adds an astrometric measurement."""

        if isinstance(camera, str):
            match = re.match(r".*([1-6]).*", camera)
            if not match:
                raise CoordIOError(f"Cannot understand camera {camera!r}.")
            camera = int(match.group(1))

        new_data = (camera, ra, dec, obstime)

        if self.astro_data is None:
            self.astro_data = pandas.DataFrame(
                [new_data],
                columns=["gfa_id", "ra", "dec", "obstime"],
            )
            self.astro_data.set_index("gfa_id", inplace=True)
        else:
            self.astro_data.loc[camera] = new_data[1:]  # type: ignore

    def add_wcs(self, camera: str | int, wcs: WCS, obstime: float):
        """Adds a camera measurement from a WCS solution."""

        coords: Any = wcs.pixel_to_world(*self.reference_pixel)

        ra = coords.ra.value
        dec = coords.dec.value

        self.add_astro(camera, ra, dec, obstime)

    def add_gimg(self, path: str | pathlib.Path):
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

        self.add_wcs(camera_id, wcs, obstime)

        return wcs

    def add_proc_gimg(self, path: str | pathlib.Path):
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

        self.add_wcs(header["CAMNAME"], wcs, obstime)

    def fit(
        self,
        field_ra: float,
        field_dec: float,
        field_pa: float = 0.0,
        offset: tuple[float, float, float] | numpy.ndarray = (0.0, 0.0, 0.0),
        scale_rms: bool = False,
    ):
        """Performs the fit and returns a `.GuiderFit` object.

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

            camera_id: int = d.Index

            camera_ids.append(camera_id)
            xwok_gfa.append(self.gfa_wok_coords.loc[camera_id, "xwok"])  # type: ignore
            ywok_gfa.append(self.gfa_wok_coords.loc[camera_id, "ywok"])  # type: ignore

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

        X = numpy.array([xwok_gfa, ywok_gfa])
        Y = numpy.array([xwok_astro, ywok_astro])
        try:
            c, R, t = umeyama(X, Y)
        except ValueError:
            return False

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

        delta_x = (numpy.array(xwok_gfa) - numpy.array(xwok_astro)) ** 2  # type: ignore
        delta_y = (numpy.array(ywok_gfa) - numpy.array(ywok_astro)) ** 2  # type: ignore

        xrms = numpy.sqrt(numpy.sum(delta_x) / len(delta_x))
        yrms = numpy.sqrt(numpy.sum(delta_y) / len(delta_y))
        rms = numpy.sqrt(numpy.sum(delta_x + delta_y) / len(delta_x))

        # Convert to arcsec and round up
        xrms = float(numpy.round(xrms / self.plate_scale * 3600.0, 3))
        yrms = float(numpy.round(yrms / self.plate_scale * 3600.0, 3))
        rms = float(numpy.round(rms / self.plate_scale * 3600.0, 3))

        astro_wok = pandas.DataFrame.from_records(
            {"gfa_id": camera_ids, "xwok": xwok_astro, "ywok": ywok_astro}
        ).set_index("gfa_id")

        gfa_wok = pandas.DataFrame.from_records(
            {"gfa_id": camera_ids, "xwok": xwok_gfa, "ywok": ywok_gfa}
        ).set_index("gfa_id")

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
        )

        return self.result
