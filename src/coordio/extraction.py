#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-08-22
# @Filename: extraction.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import numpy
import pandas
import sep


__all__ = ["sextractor_quick"]


def sextractor_quick(data: numpy.ndarray, threshold: float = 5.0, clean: bool = True):
    """Runs a quick extraction using SExtractor (sep).

    Parameters
    ----------
    data
        The array from which to extract sources.
    clean
        Applies some reasonable cuts to select only star-like objects.

    """

    data = data.astype("f8")

    back = sep.Background(data)
    rms = back.globalrms

    stars = sep.extract(data - back.back(), threshold, err=rms)

    df = pandas.DataFrame(stars)

    if clean:
        ecc = numpy.sqrt(df.a**2 - df.b**2) / df.a
        filter = (df.cpeak < 60000) & (ecc < 0.7)
        df = df.loc[filter]

    return df
