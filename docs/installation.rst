Installation
=============

Installation via pip
--------------------

The easiest way to install CVD Risk Calculator is using pip:

.. code-block:: bash

   pip install cvd-risk

Installation with optional dependencies
---------------------------------------

For development work, documentation, or performance optimization:

.. code-block:: bash

   # Development dependencies
   pip install cvd-risk[dev]

   # Documentation dependencies
   pip install cvd-risk[docs]

   # Performance optimization dependencies
   pip install cvd-risk[performance]

   # All dependencies
   pip install cvd-risk[dev,docs,performance]

Development installation
------------------------

To install from source in development mode:

.. code-block:: bash

   git clone https://github.com/m-aljasem/PyCVDRisk.git
   cd cvd-risk
   pip install -e ".[dev]"

Requirements
------------

- Python 3.10 or higher
- NumPy 1.24.0 or higher
- Pandas 2.0.0 or higher
- Pydantic 2.0.0 or higher

