"""
Scikit IMU (:mod:`skimu`)
=======================================

.. currentmodule:: skimu

Pipeline Processing
-------------------

.. autosummary::
    :toctree: generated/

    Pipeline

Utility Functions
-----------------

.. autosummary::
    :toctree: generated/

    utility.compute_window_samples
    utility.get_windowed_view
"""
from skimu.version import __version__

from skimu.pipeline import Pipeline
from skimu import utility

from skimu import gait
from skimu import sit2stand
from skimu import features
from skimu import read

__skimu_version__ = __version__
__all__ = ['Pipeline', 'gait', 'sit2stand', 'read', 'features', 'utility', '__skimu_version__']