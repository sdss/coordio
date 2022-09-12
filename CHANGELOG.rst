.. _coordio-changelog:

==========
Change Log
==========

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
