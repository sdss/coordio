.. _coordio-changelog:

==========
Change Log
==========

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
