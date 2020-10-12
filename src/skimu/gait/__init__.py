"""
IMU Gait Analysis (:mod:`skimu.gait`)
=====================================

.. currentmodule:: skimu.gait

Pipeline gait processing
------------------------

.. autosummary::
    :toctree: generated/

    Gait

.. _event-level-gait-metrics:

Event Level Gait Metrics
------------------------

.. autosummary::
    :toctree: generated/

    StrideTime
    StanceTime
    SwingTime
    StepTime
    InitialDoubleSupport
    TerminalDoubleSupport
    DoubleSupport
    SingleSupport
    StepLength
    StrideLength
    GaitSpeed
    Cadence
    IntraStepCovarianceV
    IntraStrideCovarianceV
    HarmonicRatioV

.. _bout-level-gait-metrics:

Bout Level Gait Metrics
-----------------------

.. autosummary::
    :toctree: generated/

    PhaseCoordinationIndex
    GaitSymmetryIndex
    StepRegularityV
    StrideRegularityV
    AutocovarianceSymmetryV
    RegularityIndexV

Background Information
----------------------

::

|      IC             FC      IC             FC      IC             FC
|      i-1            i-1     i+1            i+1     i+3            i+3
| L    |--------------|       |--------------|       |--------------|
| R               |--------------|        |--------------|
|                 i              i        i+2            i+2
|                 IC             FC       IC             FC
|      time --->

General terminology:

- Initial Contact (IC): the first contact of a foot with the ground, also "Heel Strike"
- Final Contact (FC): the last contact of a foot with the ground, also "Toe Off"
- Stride: between ICs of the same foot, eg IC(i) to IC(i+2), IC(i+1) to IC(i+3)

Per stride terminology:

- Step: between ICs of opposite feet, eg IC(i) to IC(i+1), IC(i+1) to IC(i+2). There
  are 2 steps to every stride
- Stance: when the foot is in contact with the ground, eg IC(i) to FC(i)
- Swing: when the foot is not in contact with the ground, eg FC(i) to IC(i+2)
- Double Support: when both feet are in contact with the ground simultaneously
- Single Support: when only 1 foot is in contact with the ground

Adding Custom Gait Metrics
--------------------------

A modular system for computing gait metrics is employed to aid in the addition of custom gait
metrics. Two base classes exist depending on what type of metric is being added:

- `gait_metrics.EventMetric` for per- step/stride metrics
- `gait_metrics.BoutMetric` for per-bout metrics

New metrics should be subclasses of either of these base classes, which provide some basic
functionality behind the scenes. The definition and initialization is very straight-forward

.. code-block:: python

    class NewMetric(gait_metrics.EventMetric):
        def __init__(self):
            super().__init__('new metric', depends=[gait_metrics.StrideTime])

`__init__` should take no arguments, and its call to the super method has 1 required and 1
optional argument: the name for the metric (this will appear in the output as "PARAM:{name}"
or "BOUTPARAM:{name}" for bout metrics, and references to any other metrics it depends upon.
In this case, the calculation of the new metric would reference the computed stride time.

To implement the feature computation, the `_predict` method should be defined

.. code-block:: python

    class NewMetric(gait_metrics.EventMetric):
        ...
        def _predict(self, dt, leg_length, gait, gait_aux):
            mask, mask_ofst= self._predict_init(gait, True, 1)

            key = 'PARAM:stride time'
            gait[self.k_][mask] = gait[key][mask_ofst] - gait[key][mask]

The arguments to the `_predict` method are as follows:

- `dt`: sampling period in seconds
- `leg_length`: leg length in meters, if provided to the `Gait` process. Otherwise `None`
- `gait`: dictionary containing all the end result gait metrics. Where the newly metric will be
  stored as well. Not returned, just modified in place. Has several keys that are defined before
  any metrics are calculated - `'IC'`, `'FC'`, `'FC opp foot'` and `'delta h'`
- `gait_aux`: dictionary containing the acceleration (3D, key: `'accel'`), vertical acceleration
  axis (`'vert axis'`), vertical velocity (`'vert velocity'`) and vertical position
  (`'vert position'`) for each bout of gait. Additionally contains a mapping from individual steps
  to bouts (`'inertial data i'`).

There are a few convenience functionalities, namely the `_predict_init` function, and the
`self.k_` attribute. The `_predict_init` function optionally does up to two things:

1. Initialize the results (`self.k_`) to an array of `nan` values, if the second argument is `True`
2. Create `mask` and `mask_ofst`, which can be used to index into the per event values. Combined,
   they work together to provide similar functionality to something like `x[ofst:] - x[:-ofst]`
   where `ofst` (third argument) is either `1` or `2`. They additionally account for steps that
   are not valid/at the end of bouts where values would be nonsensical.

The `self.k_` attribute simply stores the full name of the metric for easy/shorthand access.
Finally, if the custom metric has a basic symmetry value computed by subtracting sequential
values, this can be quickly implemented by adding the decorator `gait_metrics.basic_symmetry`
above the `_predict` definition.
"""
from skimu.gait.gait import Gait
from skimu.gait import gait
from skimu.gait.gait_metrics import *
from skimu.gait import gait_metrics
from skimu.gait import train_classifier
