Dispatch: A simple and efficient electricity dispatch model
=======================================================================================


.. image:: https://github.com/rmi/dispatch/workflows/tox-pytest/badge.svg
   :target: https://github.com/rmi/dispatch/actions?query=workflow%3Atox-pytest
   :alt: Tox-PyTest Status

.. image:: https://github.com/rmi/dispatch/workflows/docs/badge.svg
   :target: https://rmi.github.io/dispatch/
   :alt: GitHub Pages Status

.. image:: https://coveralls.io/repos/github/RMI/dispatch/badge.svg?branch=main
   :target: https://coveralls.io/github/RMI/dispatch?branch=main

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black>
   :alt: Any color you want, so long as it's black.

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. readme-intro

Description
=======================================================================================

``Dispatch`` is a hugely simplified production cost model. It takes in a
portfolio of dispatchable, fixed-output and storage resources and asks, how can these
resources be dispatched to meet [net] load? And how much would it cost?

It contains no optimization, security constraints, or formal unit commitment. It
attempts a loose approximation of a production cost model by applying the following
procedure in each hour:

1. Augment or diminish load to be met by operating dispatchable plants based on state
   of charge relative to storage reserve. The storage reserve can be dynamically set
   each hour based on ramping requirements over the next 24 hours.
2. Iterate through operating dispatchable plants in order of their marginal cost and
   adjust their output to meet load, limited by their ramp rate.
3. If there is excess energy from renewables, use it to charge storage. If there is
   still unmet load, discharge storage but holding some in state of charge in reserve.
4. If there is still unmet load, iterate through non-operating plants in order of
   their start-up cost and turn them on if they are needed to meet load, limited by
   their ramp rate.
5. If there is still unmet load, use any reserve state of charge to meet load.

For more information about how the model works and how to use it, please see the
`model documentation <https://rmi.github.io/dispatch/>`__.

Installation
=======================================================================================

Dispatch can be installed and used in it's own environment or installed into another
environment using pip. To install it using pip:

.. code-block:: bash

   pip install git+https://github.com/rmi/dispatch.git


As a dependency in a project
-------------------------------------------
To add it as a dependency in a project add
``"rmi.dispatch @ git+https://github.com/rmi/dispatch.git"`` to the
``dependency`` section of ``pyproject.toml``.
