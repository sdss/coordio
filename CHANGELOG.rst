.. _coordio-changelog:

==========
Change Log
==========

Next version
------------

* Add ``fiberAssignment`` to ``Calibration``.


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
