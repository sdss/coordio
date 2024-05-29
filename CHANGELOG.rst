.. _coordio-changelog:

==========
Change Log
==========

1.11.2 (2024-05-29)
------------------
* limit gaia source queries to 250 brightest stars per gfa (speeds up crowded field guiding)

1.11.1 (2024-05-08)
------------------
* remove print statements

1.11.0 (2024-05-08)
------------------
* Add new guide fitter (SolvePointing class), implements a gaia based iterative fitter for pointing solutions.
* Add sped up array based conversions between wok and positioner coords.

1.10.0 (2024-03-07)
------------------

* Change default focal plane scale factor to 1.0003 after new IMB installed
* Add set of zbplus coefficients (probably bad ones, but whatever) to the transforms.FVCTransformLCO class.

1.9.3 (2024-01-27)
------------------

* `#23 <https://github.com/sdss/coordio/pull/23>`__: added ``Observer.fromEquatorial()`` classmethod to initialise a new ``Observer`` from an array of topocentric RA and Dec coordinates. Alternatively a set to HA and Dec coordinates can be provided if ``hadec=True``.


1.9.2 (2024-01-15)
------------------

* Handle import of ``transforms`` if ``WOKCALIB_DIR`` has not been defined.


1.9.1 (2024-01-15)
------------------

* No changes since 1.9.1b1. Tagging as full release.


1.9.1b1 (2024-01-10)
--------------------

* Change to use the ``sdss-sep`` package and wheels.


1.9.0 (2023-09-29)
-------------------

* Change nudge x offset back to zero.  fps_calibrations now has new nudge models in place built from trimmed FVC images.
* Modify nudge model clipping behavior, currently all corrections with abs(corr)>.75 pixels will be applied with corrections clipped at +/- 0.75


1.8.1 (2023-07-28)
------------------

* Add ZB normalization factor option for FVC fitting.  Defaults to 330 mm in FVC transform classes.


1.8.0 (2023-07-24)
------------------

* Update offset functions to include following functionalities:
  * New defaults for offset function parameters
  * Observatory dependent offsetting as default
  * Offsets are calculated using all magnitudes for a source and the maximum is used as the returned offset


1.7.3 (2023-04-28)
------------------

* Also calculate measured RMS using only cameras that were used for the fit.


1.7.2 (2023-04-27)
------------------

* Calculate global fit RMS using only cameras that were used for the fit.


1.7.1 (2023-04-25)
------------------

* Update default ``safety_factor`` for bright and dark time.
* Calculate guider fit RMS.


1.7.0 (2023-04-15)
------------------

* Adjust nudge model for new FVC frame sizes at APO and LCO.
* Added binding for the SOFA ``iauRefco`` function.
* Allow to call ``FocalPlane()`` with ``use_closest_wavelength`` and an arbitrary wavelength to use the ZEMAX model with the closest wavelength.


1.6.1 (2023-01-15)
------------------

* In ``get_marginal()`` allow to normalise the distribution using an input value. Otherwise use a narrow regions around the centre of the box.
* Several improvements to Gaussian fitting in ``fit_gaussian_to_marginal()``. Mark marginal fit as bad if the fit is unsuccessful.
* Improve plotting of detections in ``extract_marginal()``
* Sort detections by flux and allow to cap the number of sources returned.
* Require that ``tnpix`` be larger than ``minarea`` in ``sextractor_quick()``


1.6.0 (2023-01-05)
------------------

* Implement ``zbplus2`` FVC transform.
* Handle Hogg coefficients by loading from the first ``wokcalib`` dir on path.


1.5.2 (2023-01-02)
------------------

Several changes to ``extraction.extract_marginal()``:

* Fix a potential problem in plot box.
* Deal with no detections in ``extract_marginal()``.
* Use gray colourmap in ``extract_marginal()`` plotting.


1.5.1 (2022-12-26)
------------------

* Make ``seaborn`` a dependency since it's needed by some of the extraction routines.


1.5.0 (2022-12-26)
------------------

* No major changes since 1.5.0b1. Tagging as full release.


1.5.0b1 (2022-11-07)
--------------------

* Added additional tools for extraction and fitting the marginal distribution.
* `#17 <https://github.com/sdss/coordio/pull/17>`__: implementation of the offset function.


1.4.5 (2022-10-20)
------------------

* Add guider tools for cross-matching with catalogue data.


1.4.4 (2022-09-15)
------------------

* Add an ``only_radec`` option to ``GuiderFitter.fit()`` to only fit RA/Dec (pure translation).


1.4.3 (2022-09-11)
------------------

* Use ``focalScale=1`` in ``GuiderFitter``.
* Update the ``solve-field`` command options when calling ``AstrometryNet.run_async()``.


1.4.2 (2022-09-08)
------------------

* Add default scale factors for APO and LCO for radec2wokxy and wokxy2radec


1.4.1 (2022-08-31)
------------------

* Use astropy 5 and numpy 1.23 for Python>=3.8.


1.4.0 (2022-08-31)
------------------

* Add dimage (Blanton's) simplexy and refinexy for centroiding if wanted
* Add nudge option for centroiding based on CCD static distortion model
* Default to 33 term ZB basis and nudge centroiding for FVC
* Fix a bug in which the object epoch for an ``ICRS`` coordinate would not change when ``ICRS.to_epoch()`` was called.
* Tweaks to ``FVCTransformLCO`` parameters based on telescope data.
* Moved astrometry.net and guider fitting tools from ``cherno`` to ``coordio.guide``.
* Added a ``coordio.extraction.sextractor_quick()`` function for simple extraction with background subtraction using ``sep``.
* Change ``defaults.FOCAL_SCALE`` to 1. It may be removed in the future.


1.3.1 (2022-04-24)
------------------

* Updated release action in GitHub to build wheels for manylinux and macOS.


1.3.0 (2022-04-21)
------------------

* Add ``FVCTransformAPO`` class.
* Change the base URL for the IERS bulletins.
* Pass the ``fpsScale`` parameter to ``wokToFocal`` when creating focal coordinates from wok coordinates.


1.2.1 (2022-01-26)
------------------

* Add ``fpScale`` parameter to adjust the scale of the focal plane. Default value is 0.9998.
* Modify default behavior between focal plane and wok to assume a flat wok.


1.2.0 (2022-01-04)
------------------

* Add ``fiberAssignment`` to ``Calibration``.
* Add new implementation of ``tangentToPositioner``.
* Add GFA coordinates to calibrations.
* Add plate scale defaults for APO and LCO.


1.1.3 (2021-11-14)
----------------

* When ``Calibration`` does not have any files, the data frames are set to empty instead of ``None``.


1.1.2 (2021-11-14)
----------------

* Use measured alpha and beta offsets when transforming from tangent to positioner.
* Replace error in ``iauPmsafe`` with warning.
* Add ``RoughTransform`` and ``ZhaoBurgeTransform`` (#11).
* Undo changes to ``wokToTangentArr``. Reverted to only supporting one holeID per array (#11).
* Add a ``Calibration`` class to store all active calibrations, allowing for concatenation of different site calibrations (#12).


1.1.1 (2021-10-28)
-------------------
C++ implementation of wok, tangent, positioner transforms. Improvements to packaging.


1.0.0  (2021-05-01)
--------------------

First tagged version
