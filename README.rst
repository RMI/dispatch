Dispatch: A simple and efficient electricity dispatch model
=======================================================================================

.. readme-intro

.. image:: https://github.com/rmi-electricity/dispatch/workflows/tox-pytest/badge.svg
   :target: https://github.com/rmi-electricity/dispatch/actions?query=workflow%3Atox-pytest
   :alt: Tox-PyTest Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black>
   :alt: Any color you want, so long as it's black.


Description
=======================================================================================

* TK long description

Installation
=======================================================================================
Dispatch can be installed and used in it's own environment or installed into another
environment using pip. To create an environment for Dispatch, navigate to the repo
folder in terminal and run:

.. code-block:: console

   $ mamba update mamba
   $ mamba env create --name dispatch --file environment.yml

If you get a ``CondaValueError`` that the prefix already exists, that means an
environment with the same name already exists. You must remove the old one before
creating the new one:

.. code-block:: console

   $ mamba update mamba
   $ mamba env remove --name dispatch
   $ mamba env create --name dispatch --file environment.yml
