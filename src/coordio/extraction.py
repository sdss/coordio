#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-08-22
# @Filename: extraction.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import numpy
import pandas
import seaborn
import sep
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian1D
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import NDArray


__all__ = ["sextractor_quick"]


seaborn.set_color_codes("deep")


def sextractor_quick(
    data: NDArray,
    threshold: float = 5.0,
    clean: bool = True,
    minarea: int = 10,
    return_background: bool = False,
) -> pandas.DataFrame | tuple[pandas.DataFrame, NDArray]:
    """Runs a quick extraction using SExtractor (sep).

    Parameters
    ----------
    data
        The array from which to extract sources.
    threshold
        The threshold for detections, in sigmas above the background.
    clean
        Applies some reasonable cuts to select only star-like objects
        (eccentricity greater than 0.05 and less than 0.8).
    minarea
        Minimum number of pixels in the detected regions.
    return_background
        Whether to return the background image.

    Returns
    -------
    extractions
        A data frame with the output of ``sep.extract``. If
        ``return_background=True`` a second element is returned with
        the background image.

    """

    data = data.astype("f8")

    back = sep.Background(data)
    background: NDArray[numpy.float32] = back.back()
    rms = back.globalrms

    stars = sep.extract(data - background, threshold, err=rms, minarea=minarea)

    df = pandas.DataFrame(stars)

    df["ecc"] = numpy.sqrt(df.a**2 - df.b**2) / df.a

    if clean:
        filter = (df.cpeak < 60000) & (df.ecc < 0.8) & (df.ecc > 0.05)
        df = df.loc[filter]

    if return_background:
        return df, background

    return df


def get_marginal(data: NDArray, x: int, y: int, box_size: int, axis: int = 0):
    """Returns the marginal distribution around a point.

    Parameters
    ----------
    data
        The image array.
    x
        The x coordinate of the centre of the box. If not an integer, it's
        truncated.
    y
        The y coordinate of the centre of the box.
    box_size
        The size of the box around the ``x,y`` coordinates. Either an
        odd number or the closes odd number will be used.
    axis
        The axis of ``data`` along which to collapse the image.

    Returns
    -------
    marginal
        The normalised marginal distribution of the box around ``x,y``
        summed along axis ``axis``.

    """

    box_size = int(box_size)
    if box_size % 2 == 0:
        box_size += 1

    x = int(x)
    y = int(y)

    marginal = data[
        y - box_size // 2 : y + box_size // 2 + 1,
        x - box_size // 2 : x + box_size // 2 + 1,
    ]
    marginal = marginal.sum(axis=axis)
    marginal /= marginal.max()

    return marginal


def fit_gaussian_to_marginal(
    data: NDArray,
    x: int,
    y: int,
    box_size: int,
    axis: int = 0,
    sigma_0: float | None = None,
):
    """Fits the marginal distribution with a 1D Gaussian.

    Parameters
    ----------
    data
        The image array.
    x
        The x coordinate of the centre of the box. If not an integer, it's
        truncated.
    y
        The y coordinate of the centre of the box.
    box_size
        The size of the box around the ``x,y`` coordinates. Either an
        odd number or the closes odd number will be used.
    axis
        The axis of ``data`` along which to collapse the image.
    sigma_0
        An initial estimate of the sigma of the Gaussian to fit.

    Returns
    -------
    fit
        A tuple containing the mean of the fitted Gaussian, the fitted sigma,
        and the RMS of the residuals between the original data and the fit.

    """

    box_size = int(box_size)
    if box_size % 2 == 0:
        box_size += 1

    x = int(x)
    y = int(y)

    marginal = get_marginal(data, x, y, box_size, axis=axis)
    xx = numpy.arange(marginal.size)

    fitter = LevMarLSQFitter()
    model = Gaussian1D(2, box_size / 2, sigma_0)

    gg = fitter(model, xx, marginal)

    rms = numpy.sqrt(numpy.sum((marginal - gg(xx)) ** 2) / (len(xx) - 1))

    if axis == 0:
        mean = y + gg.mean - box_size // 2
    else:
        mean = x + gg.mean - box_size // 2

    return (mean, gg.stddev.value, rms)


def extract_marginal(
    data: NDArray,
    threshold: float = 5.0,
    box_size: int = 51,
    sigma_0: float | None = None,
    exclude_border: bool = True,
    sextractor_quick_options: dict = {},
    plot: pathlib.Path | str | None = None,
    n_rows_plot: int = 9,
    plot_title: str | None = None,
):
    """Extracts regions from an image and fit them using the marginal distribution.

    Parameters
    ----------
    data
        The image array.
    threshold
        The threshold for detections, in sigmas above the background.
    box_size
        The size of the box around the ``x,y`` coordinates. Either an
        odd number or the closes odd number will be used.
    sigma_0
        An initial estimate of the sigma of the Gaussian to fit.
    exclude_border
        Reject regions that are closer to the border of the image than
        ``box_size``.
    sextractor_quick_options
        Additional parameters to pass to `.sectractor_quick`.
    plot
        If `None`, no plots will be produced. Otherwise it must
        be the path where to write the PDF with the plot.
    n_rows_plot
        How many regions per page to plot.
    plot_title
        A title to be added to the plot.

    Returns
    -------
    extractions
        A data frame with the extracted regions. It includes all the
        usual columns from ``sep.extract`` plus ``x1``, ``xstd``, ``xrms``
        (and same for ``y``) with the mean, sigma, and RMS for the given
        axis obtained by fitting the marginal distribution (see
        `.fit_gaussian_to_marginal`).

    """

    data = data.astype("f8")

    sextractor_quick_options.pop("threshold", None)
    sextractor_quick_options.pop("return_background", None)

    detections, back = sextractor_quick(
        data,
        threshold=threshold,
        return_background=True,
        **sextractor_quick_options,
    )

    assert isinstance(detections, pandas.DataFrame)
    assert isinstance(back, numpy.ndarray)

    if len(detections) == 0:
        return detections

    if exclude_border:
        detections = detections.loc[
            (detections.x > box_size // 2)
            & (detections.x < data.shape[1] - box_size // 2)
            & (detections.y > box_size // 2)
            & (detections.y < data.shape[0] - box_size // 2),
            :,
        ]

    back = sep.Background(data)
    sub = data - back.back()

    if len(detections) > 0:
        for axis in [1, 0]:
            ax = "x" if axis == 1 else "y"

            fit_df = detections.apply(
                lambda d: pandas.Series(
                    fit_gaussian_to_marginal(
                        sub,
                        d.x,
                        d.y,
                        box_size,
                        axis=axis,
                        sigma_0=sigma_0,
                    ),
                    index=[f"{ax}1", f"{ax}std", f"{ax}rms"],
                ),
                axis=1,
            )

            detections = pandas.concat([detections, fit_df], axis=1)

    else:
        # Add new columns. If there are no detections at least the columns will exist
        # on an empty data frame and the overall shape won't change.
        detections[["x1", "xstd", "xrms", "y1", "ystd", "yrms"]] = numpy.nan

    if plot is not None:
        if not isinstance(plot, pathlib.Path) and not isinstance(plot, str):
            raise TypeError("plot must be the path to the output file.")

        plot = pathlib.Path(plot)
        plot.parent.mkdir(exist_ok=True, parents=True)

        with plt.ioff():
            with PdfPages(str(plot)) as pdf:

                figure, ax = plt.subplots(figsize=(8.5, 8.5))
                ax.imshow(
                    sub,
                    origin="lower",
                    vmin=sub.mean() - sub.std(),
                    vmax=sub.mean() + sub.std(),
                    cmap="gray",
                )

                if len(detections) > 0:
                    ax.scatter(
                        detections.x1,
                        detections.y1,
                        marker="x",  # type:ignore
                        c="r",
                    )

                    for reg, row in detections.iterrows():
                        ax.annotate(
                            str(reg),
                            (row.x1 + 15, row.y1 + 15),
                            fontsize=10,
                            color="k",
                        )

                ax.set_title(plot_title or "", fontsize=15, pad=20)

                pdf.savefig(figure)
                plt.close(figure)

                for ii in range(0, len(detections), n_rows_plot):
                    sample = detections.iloc[ii : ii + n_rows_plot]
                    if len(sample) == 0:
                        break
                    _plot_one_page(pdf, sub, sample, box_size, n_rows_plot)

            plt.close("all")

            plt.clf()
            plt.cla()

    return detections


def _plot_one_page(
    pdf: PdfPages,
    data: NDArray,
    df: pandas.DataFrame,
    box_size: int,
    n_rows: int,
):
    """Plots one page of marginal distributions."""

    figure, ax = plt.subplots(n_rows, 2, figsize=(8.5, 11))

    ii = 0
    for region, row in df.iterrows():

        for col_ax, axis in enumerate(["x", "y"]):

            marginal = get_marginal(
                data,
                row.x1,
                row.y1,
                box_size,
                axis=0 if axis == "y" else 1,
            )

            xx = numpy.arange(marginal.size)
            ax[ii][col_ax].plot(
                xx,
                marginal,
                color="k",
                ls="dotted",
            )

            if axis == "x":
                pos = row.x1
                rms = row.xrms
                stddev = row.xstd
            else:
                pos = row.y1
                rms = row.yrms
                stddev = row.ystd

            mean = pos - int(pos) + marginal.size // 2

            model = Gaussian1D(1, mean, stddev)
            ax[ii][col_ax].plot(
                xx,
                model(xx),
                color="r",
                ls="solid",
            )

            title = ""
            if axis == "x":
                title += f"Region {region}: "
            title += rf"$\overline{{{axis}}}={pos:.1f},\ $"
            title += rf"$\sigma_{axis}={stddev:.2f},\ $"
            title += f"RMS={rms:.3f}"

            ax[ii][col_ax].set_title(title, fontsize=10)
            ax[ii][col_ax].tick_params(labelsize=10)
            ax[ii][col_ax].tick_params(labelsize=10)

        ii += 1

    if ii < n_rows - 1:
        for jj in range(ii, n_rows):
            ax[jj][0].axis("off")
            ax[jj][1].axis("off")

    figure.subplots_adjust(hspace=0.6, top=0.97, bottom=0.03, left=0.05, right=0.95)

    pdf.savefig(figure)
    plt.close(figure)
