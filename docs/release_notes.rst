=======================================================================================
Dispatch Release Notes
=======================================================================================

.. _release-v0-2-0:

---------------------------------------------------------------------------------------
0.2.0 (2022-XX-XX)
---------------------------------------------------------------------------------------

What's New?
^^^^^^^^^^^
*   :py:class:`dispatch.model.DispatchModel` now uses ``__slots__``
*   New ``to_disk`` and ``from_disk`` methods that allow a
    :py:class:`dispatch.model.DispatchModel` object to be saved to disk and recreated
    from a file. This uses a ``zip`` of many ``parquet`` files for size and to avoid
    ``pickle`` being tied to a particular module layout.
*   Methods to calculate hourly cost for historical and redispatch.
*   Method to simplify aggregating hourly generator-level data to less granular
    frequencies and asset specificity.

Bug Fixes
^^^^^^^^^
*   ...

Known Issues
^^^^^^^^^^^^
*   ...

.. _release-v0-1-0:

---------------------------------------------------------------------------------------
0.1.0 (2022-08-23)
---------------------------------------------------------------------------------------

What's New?
^^^^^^^^^^^
*   A dispatch model with no RMI dependencies and in its own repository!
*   Repository built off of
    `catalyst-cooperative.cheshire <https://github.com/catalyst-cooperative/cheshire>`_
    that uses cool tools like ``tox``, ``sphinx``, etc.

Bug Fixes
^^^^^^^^^
*   It's good to make a note of any known bugs that are fixed by the release, and refer
    to the relevant issues.
*   `mypy <https://github.com/python/mypy>`_ is disabled because of error described in :issue:`1`.

Known Issues
^^^^^^^^^^^^
*   :py:class:`dispatch.model.DispatchModel` only set up to work properly with
    `patio-model <https://github.com/rmi-electricity/patio-model>`_.
*   Test thoroughness is lacking.
*   No substantive readme or documentation.


..
    Examples so I don't forget
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
    * You can refer to the relevant pull request using the ``pr`` role: :pr:`1`
    * Don't hesitate to give shoutouts to folks who contributed like :user:`arengel`
    * You can link to issues that were closed like this: :issue:`2,3,4`
