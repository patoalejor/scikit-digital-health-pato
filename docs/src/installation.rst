Installation
============

Both PyPI and Conda-Forge should have pre-built wheels for major distributions.

.. tabbed:: conda

    ::

        conda install -c conda-forge scikit-digital-health

.. tabbed:: pip
    :selected:

    ::

        pip install scikit-digital-health

.. tabbed:: pip from GitHub

    ::

        pip install git+ssh://git@github.com/PfizerRD/scikit-digital-health.git
        # install from a specific tag
        pip install git+ssh://git@github.com/PfizerRD/scikit-digital-health@0.9.10
        # install from a specific branch
        pip install git+ssh://git@github.com/PfizerRD/scikit-digital-health@development

Run-time requirements
^^^^^^^^^^^^^^^^^^^^^

- numpy >=1.17.2
- scipy >=1.3.1
- pandas >=1.0.0
- lightgbm >=2.3.0
- pywavelets
- scikit-learn
- h5py
- matplotlib
- packaging
- pyyaml
- importlib_resources [Python < 3.7]

Windows Notes
^^^^^^^^^^^^^

Windows users might need to install additional dependencies, even if not building from source.

https://wiki.python.org/moin/WindowsCompilers has more details, but generally users should expect
to have to install a Microsoft Visual C++ redistributable package. For Python >3.5 this should be
14.0 or greater.

The 2015 redistributable update 3 can be found here: https://www.microsoft.com/en-us/download/details.aspx?id=53587

Building from Source
====================
In order to build from source, the following are required:

- C compiler (currently has only been tested with GCC)
- Fortran compiler (currently only been tested with GFortran)
- `libzip <https://libzip.org/>`_
- setuptools
- numpy >=1.17.2

If you are using Conda, libzip can easily be installed via::

    conda install -c conda-forge libzip

C/Fortran Compilers
^^^^^^^^^^^^^^^^^^^

NumPy only supports specific compilers on each platform. To get a list of available
compilers, when in the `scikit-digital-health` directory, running::

    # Get a list of platform supported/installed C compilers
    python setup.py build_ext --help-compiler
    # Get a list of platform supported/installed Fortran compilers
    python setup.py build_ext --help-fcompiler


Testing the Build
^^^^^^^^^^^^^^^^^

In order to test a build, you will need to clone or download the source code from
`GitHub <https://github.com/PfizerRD/scikit-digital-health>`_, as the tests are isolated
from access to the package source code. Additionally, the following are requirements
to run the test suite:

- pytest
- psutil
- tables
- coverage [optional, for running test coverage]

From the top level `scikit-digital-health` directory, run::

    pytest test/

The output should look similar to::

    ============================================================================
    platform darwin -- Python 3.10.2, pytest-7.0.1, pluggy-1.0.0
    rootdir: /Users/lukasadamowicz/Documents/Packages/scikit-digital-health
    collected 434 items

    test/activity/test_activity_core.py .....                             [  1%]
    test/activity/test_activity_endpoints.py ................             [  4%]
    # ... rest of test results ...

    =========================== warnings summary ===============================
    # ... warnings content ...
    ========= 427 passed, 7 skipped, 61 warnings in 6.30s ======================

