=======================================================================================
Release Notes
=======================================================================================

.. _release-v0-3-0:

---------------------------------------------------------------------------------------
0.3.0 (2022-XX-XX)
---------------------------------------------------------------------------------------

What's New?
^^^^^^^^^^^
*   :meth:`.DispatchModel.to_file` can create an output with summary
    outputs.
*   Adopting :mod:`.pandera` for metadata and validation using
    :class:`.Validator` to organize and specialize data input
    checking.
*   Adding cost component details and capacity data to
    :meth:`.DispatchModel.operations_summary`.
*   We now automatically apply ``operating_date`` and ``retirement_date`` from
    :attr:`.DispatchModel.dispatchable_plant_specs` to
    :attr:`.DispatchModel.dispatchable_profiles` using
    :func:`.apply_op_ret_date`.
*   Added validation and processing for :attr:`.DispatchModel.re_plant_specs` and
    :attr:`.DispatchModel.re_profiles`, as well as :meth:`.DispatchModel.re_summary`
    to, when the data is provided create a summary of renewable operations analogous
    to :meth:`.DispatchModel.operations_summary`.
*   Added :meth:`.DispatchModel.storage_summary` to create a summary of storage
    operations analogous to :meth:`.DispatchModel.operations_summary`.
*   Added :meth:`.DispatchModel.full_output` to create the kind of outputs needed by
    Optimus and other post-dispatch analysis tools built on the three

Known Issues
^^^^^^^^^^^^
*   :meth:`.DispatchModel.re_summary` and :meth:`.DispatchModel.storage_summary` have
    operations cost data.
*   :meth:`.DispatchModel.re_summary` and :meth:`.DispatchModel.storage_summary` have
    no tests.
*   There is still no nice way to include nuclear and hydro resources.


.. _release-v0-2-0:

---------------------------------------------------------------------------------------
0.2.0 (2022-09-15)
---------------------------------------------------------------------------------------

What's New?
^^^^^^^^^^^
*   :class:`.DispatchModel` now uses ``__slots__``
*   New :meth:`.DispatchModel.to_file` and :meth:`.DispatchModel.from_file` methods
    that allow a :class:`.DispatchModel` object to be saved to disk and recreated
    from a file. This uses a ``zip`` of many ``parquet`` files for size and to avoid
    ``pickle`` being tied to a particular module layout.
*   Methods to calculate hourly cost for historical and redispatch.
*   Method to simplify aggregating hourly generator-level data to less granular
    frequencies and asset specificity.
*   Storage resources can now be added to the portfolio over time based on their
    ``operating_date`` in ``storage_specs``.
*   When using :meth:`.DispatchModel.from_fresh`, ``operating_date`` and
    ``retirement_date`` columns in ``dispatchable_plant_specs`` determine the period
    during dispatch that a generator may operate. This provides a straightforward
    method for having the portfolio you wish to dispatch change over time.
*   Cleanup and rationalization of :meth:`.DispatchModel.to_file` and
    :meth:`.DispatchModel.from_file` methods.
*   Updates to system for storing and processing marginal cost data. This is now a
    separate argument to :meth:`.DispatchModel.__init__` rather than a
    messy confusing part of ``dispatchable_plant_specs``. This is now consistent with
    how ``patio`` prepares and stores the data.

Bug Fixes
^^^^^^^^^
*   :meth:`.DispatchModel.to_file` and
    :meth:`.DispatchModel.from_file` now properly deal with
    internal data stored in both :class:`pandas.DataFrame` and :class:`pandas.Series`.

Known Issues
^^^^^^^^^^^^
*   Tests are still pretty rudimentary.

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
*   `mypy <https://github.com/python/mypy>`_ is disabled because of error described
    in :issue:`1`.

Known Issues
^^^^^^^^^^^^
*   :class:`.DispatchModel` only set up to work properly with
    `patio-model <https://github.com/rmi-electricity/patio-model>`_.
*   Test thoroughness is lacking.
*   No substantive readme or documentation.


..
    Examples so I don't forget
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
    * You can refer to the relevant pull request using the ``pr`` role: :pr:`1`
    * Don't hesitate to give shoutouts to folks who contributed like :user:`arengel`
    * You can link to issues that were closed like this: :issue:`2,3,4`
