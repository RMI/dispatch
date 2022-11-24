Dispatch: A simple and efficient electricity dispatch model
=======================================================================================


.. image:: https://github.com/rmi-electricity/dispatch/workflows/tox-pytest/badge.svg
   :target: https://github.com/rmi-electricity/dispatch/actions?query=workflow%3Atox-pytest
   :alt: Tox-PyTest Status

.. image:: https://github.com/rmi-electricity/dispatch/workflows/docs/badge.svg
   :target: https://rmi-electricity.github.io/dispatch/
   :alt: GitHub Pages Status

.. image:: https://coveralls.io/repos/github/rmi-electricity/dispatch/badge.svg
   :target: https://coveralls.io/github/rmi-electricity/dispatch

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black>
   :alt: Any color you want, so long as it's black.

.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat
   :target: https://pycqa.github.io/isort/
   :alt: Imports: isort

.. readme-intro

Description
=======================================================================================

``Dispatch`` is a hugely simplified production cost model. It takes in a
portfolio of dispatchable and storage resources and asks, how can these resources
be dispatched to meet [net] load? And how much would it cost?

It contains no optimization, security constraints, or formal unit commitment. It
attempts a loose approximation of a production cost model by applying the following
procedure in each hour:

1. Iterate through operating dispatchable plants in order of their marginal cost and
   adjust their output to meet load, limited by their ramp rate.
2. If there is excess energy from renewables, use it to charge storage. If there is
   still unmet load, discharge storage.
3. If there is still unmet load, iterate through non-operating plants in order of
   their start-up cost and turn them on if they are needed to meet load, limited by
   their ramp rate.

For more information about how the model works and how to use it, please see the
`model documentation <https://rmi-electricity.github.io/dispatch/>`__.

Installation
=======================================================================================

Dispatch can be installed and used in it's own environment or installed into another
environment using pip. To install it using pip:

.. code-block:: bash

   pip install git+https://github.com/rmi-electricity/dispatch.git

Or from the dev branch:

.. code-block:: bash

   pip install git+https://github.com/rmi-electricity/dispatch.git@dev


To create an environment for Dispatch, navigate to the repo folder in terminal and run:

.. code-block:: bash

   mamba update mamba
   mamba env create --name dispatch --file environment.yml

If you get a ``CondaValueError`` that the prefix already exists, that means an
environment with the same name already exists. You must remove the old one before
creating the new one:

.. code-block:: bash

   mamba update mamba
   mamba env remove --name dispatch
   mamba env create --name dispatch --file environment.yml
