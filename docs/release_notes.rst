=======================================================================================
Release Notes
=======================================================================================

.. _release-v0-6-0:

---------------------------------------------------------------------------------------
0.6.0 (2023-XX-XX)
---------------------------------------------------------------------------------------

What's New?
^^^^^^^^^^^
*  Initial implementation of dynamic storage reserves. If storage does not have a
   reserve or the reserve is ``0.0``, we calculate a dynamic reserve based on the next
   24 hours of net load using :eq:`reserve`. The value of ``coeff`` can be set manually
   when creating :class:`.DispatchModel`, or automatically (default) by trying a number
   of values between 0.0 and 1.5 and selecting the one that best minimizes deficit and
   curtailment, using :func:`dispatch.engine.choose_best_coefficient`.

.. math::
   :label: reserve

      ramp &= \frac{max(load_{h+1}, ..., load_{h+24})}{load_h} - 1

      reserve &= 1 - e^{-coeff \times ramp}

*  Additional deficit and curtailment metrics in
   :meth:`.DispatchModel.system_level_summary`.
*  :meth:`.DispatchModel.dispatchable_summary` and :meth:`.DispatchModel.full_output`
   now include all columns provided in ``<x>_specs``.
*  Enable multiple storage generators under a single ``plant_id_eia`` so long as they
   don't share a ``plant_id_eia`` with a renewable generator.
*  :meth:`.DispatchModel.system_level_summary` allows for the roll-up of storage
   facilities to create higher-level metrics using the ``storage_rollup`` parameter.
*  :meth:`.DispatchModel.hourly_data_check` allows exploration of hours preceding and
   including both deficit and curtailment hours.
*  :class:`.DispatchModel` now accepts ``dispatchable_cost`` with monthly frequency.
*  Added :meth:`.DispatchModel.set_metadata` to safely set metadata.
   :meth:`.DispatchModel.__getattr__` allows retrieving metadata using attribute access.
*  Added ``marginal_for_startup_rank`` flag to :func:`.dispatch_engine` and
   :class:`.DispatchModel` ``config`` to use marginal cost rank to order generators for
   startup rather than startup cost. Given costs from :mod:`rmi.gencost` have very low
   startup cost relative to marginal costs, marginal costs can be a better metric for
   committing units.
*  Update build tool versions in ``pyproject.toml`` and ``environment.yml``.
*  Address :mod:`pandas` :class:`DeprecationWarning` in :meth:`pandas.DataFrame.groupby`
   and :func:`pandas.value_counts`.
*  Remove local pytest and blackdoc hooks from pre-commit.
*  Switch from black to ruff format for autoformatting.
*  New storage features to represent more types of storage:

   *  Storage charge rate can be different from discharge rate, to use, provide
      ``charge_mw`` column in ``storage_specs``.
   *  ``charge_eff`` and ``discharge_eff`` will replace ``roundtrip_eff`` in
      ``storage_specs`` to enable finer-grained control of when losses occur.
      Previously ``roundtrip_eff`` was effectively treated as the charge efficiency and
      there were no discharge losses.

*  Add support for Python 3.12.
*  New :meth:`.DispatchModel.redispatch_lambda` and
   :meth:`.DispatchModel.historical_lambda` to calculate hourly marginal costs.
*  Updates to narrative description of dispatch logic in ``approach.rst``.
*  Attempts to reduce memory use and unnecessary allocations in
   :func:`.zero_profiles_outside_operating_dates` and by validate inputs inplace.
*  Create new :meth:`.DispatchModel.system_summary_core` that includes only a subset of
   the columns in :meth:`.DispatchModel.system_level_summary` that can be produced
   faster and excludes storage metrics.

Bug Fixes
^^^^^^^^^
*  Fixed a bug where ``operating_date`` and ``retirement_date`` did not apply to
   dispatchable resources, where ``no_limit`` is ``True``.
*  Fixed a bug where ``capacity_mw`` for storage and renewables was not zero in outputs
   before their operating date. We now compare the modeled year to the operating year
   rather than the date. This provides the expected output when outputs are aggregated
   annually as is the typical case.
*  Fixed renewable name maps for plotting in :const:`dispatch.constants.PLOT_MAP`.
*  Added a second fallback method for determining the frequency of cost data in
   :meth:`.Validator.dispatchable_cost`. While this isn't needed within the model
   anymore, it is used to determine if there is missing data.
*  Fixed a bug where profile index validation failed because :mod:`pandera`
   :class:`pandas.DatetimeIndex` type validation was applied inconsistently between
   :class:`pandas.DataFrame` and :class:`pandas.Series`.
*  Fixed a bug where the ``capacity_mw`` column returned by
   :meth:`.DispatchModel.re_summary` and :meth:`.DispatchModel.storage_summary`
   was zero in the first year of operations when ``freq='YS'``.

.. _release-v0-5-0:

---------------------------------------------------------------------------------------
0.5.0 (2023-05-15)
---------------------------------------------------------------------------------------

What's New?
^^^^^^^^^^^
*  Added checks to make sure that the datetime indexes of ``load_profile``,
   ``dispatchable_profiles``, and ``re_profiles`` match.
*  Prep for deprecating :meth:`.DispatchModel.from_patio` and
   :meth:`.DispatchModel.from_fresh`.
*  Extracted :func:`.calculate_generator_output` from :func:`.dispatch_engine` to make
   the latter easier to read and to more easily test the former's logic.
*  Many updates to internal variable names in :mod:`.engine` to make the code easier to
   read.
*  Renamed :func:`.apply_op_ret_date` to
   :func:`.zero_profiles_outside_operating_dates` for clarity, use of the former name
   will be removed in the future.
*  Code cleanup along with adoption of ruff and removal of bandit, flake8, isort, etc.
*  Added the ability to specify in ``dispatchable_specs`` via a ``no_limit`` column
   that a generator not limited to its historical hourly output by the model without
   affecting historical dispatch data.
*  Added the ability to specify in ``dispatchable_specs`` via a ``min_uptime`` column
   the minimum number of hours a generator must have been operating before it can start
   ramping down.
*  Adjusted process of determining the provisional deficit used to dispatch currently
   operating generators. Previously, we adjusted our target for dispatchable generation
   based on the assumption we would want to use up all storage state of charge before
   dispatching operating generators. We now set the provisional deficit so that we hold
   2x ``reserve`` state of charge in reserve. If state of charge is below ``reserve``,
   we increase the provisional deficit in order to replenish the reserve.
*  Changed battery discharge so that only a part of storage can be used before
   dispatchable start-up, only down to the ``reserve``. After dispatchable start-up,
   storage is dispatched a second time in case a deficit remains, in this part of the
   sequence, all storage state of charge can be used.
*  dispatch now works with Python 3.11 using newly released :mod:`numba` version 0.57.
*  dispatch now works with :mod:`pandas` 2.0.


.. _release-v0-4-0:

---------------------------------------------------------------------------------------
0.4.0 (2023-01-25)
---------------------------------------------------------------------------------------

What's New?
^^^^^^^^^^^
*  Tests for :func:`.engine.dispatch_engine`, :func:`.copy_profile`.
*  :meth:`.DispatchModel.hourly_data_check` to help in checking for dispatch errors,
   and running down why deficits are occuring.
*  :class:`.DispatchModel` now takes ``load_profile`` that resources will be
   dispatched against. If ``re_profiles`` and ``re_plant_specs`` are not provided,
   this should be a net load profile. If they are provided, this *must* be a gross
   load profile, or at least, gross of those RE resources. These calculations are done
   by :meth:`.DispatchModel.re_and_net_load`.
*  :class:`.DispatchModel` now accepts (and requires) raw DC ``re_profiles``, it
   determines actual renewable output using capacity data and ilr provided in
   ``re_plant_specs``. This will allow :class:`.DispatchModel` to model DC-coupled
   RE+Storage facilities that can charge from otherwise clipped generation. The
   calculations for the amount of charging from DC-coupled RE is in
   :meth:`.DispatchModel.dc_charge`.
*  Updates to :func:`.engine.dispatch_engine` and :func:`.engine.validate_inputs` to
   accommodate DC-coupled RE charging data. Storage can now be charged from
   DC-coupled RE in addition to the grid. This includes tracking ``gridcharge``
   in addition to ``charge``, where the latter includes charging from the grid
   and DC-coupled RE.
*  All output charging metrics use the ``gridcharge`` data because from the grid's
   perspective, this is what matters. ``discharge`` data does not distinguish,
   so in some cases net charge data may be positive, this reflects RE generation
   run through the battery that otherwise would have been curtailed.
*  :class:`.DataZip`, a subclass of :class:`zipfile.ZipFile` that has methods for
   easily reading and writing :class:`pandas.DataFrame` as ``parquet`` and
   :class:`dict` as ``json``. This includes storing column names separately that
   cannot be included in a ``parquet``.
*  Extracted :func:`.engine.charge_storage` and
   :func:`.engine.make_rank_arrays` from :func:`.engine.dispatch_engine`. This
   allows easier unit testing and, in the former case, makes sure all charging is
   implemented consistently.
*  Added plotting functions :meth:`.DispatchModel.plot_output` to visualize columns
   from :meth:`.DispatchModel.full_output` and updated
   :meth:`.DispatchModel.plot_period` to display data by generator if ``by_gen=True``.
   :meth:`.DispatchModel.plot_year` can now display the results with daily or hourly
   frequency.
*  For renewables, ``plant_id_eia`` no longer need by unique, now for renewables,
   ``plant_id_eia`` and ``generator_id`` must be jointly unique. In cases where a
   single ``plant_id_eia`` has two renewable generator's as well as storage,
   :meth:`.DispatchModel.dc_charge` assumes excess renewable generation from the
   several generators can be combined to charge the facility's storage.
*  ``re_plant_specs``, ``dispatchable_specs``, and ``storage_specs``, now allow zeros
   for ``capacity_mw`` and ``duration_hrs``.
*  :class:`.DataZip`, :meth:`.DispatchModel.to_file`, and
   :meth:`.DispatchModel.from_file` now support :class:`io.BytesIO` as ``file``
   or ``path``. This now allows any object that implements ``to_file``/``from_file``
   methods using :class:`.DataZip`, to be written into and recovered from another
   :class:`.DataZip`.
*  Added the ability to specify in ``dispatchable_specs`` via an ``exclude`` column
   that a generator not be dispatched by the model without affecting historical
   dispatch data.
*  Migrating :class:`.DataZip` functionality to :class:`etoolbox.datazip.DataZip`.
*  Updates to constants to allow Nuclear and Conventional Hydroelectric to be properly
   displayed in plots.
*  Updates to ``re_plant_specs``, its validation, and
   :meth:`.DispatchModel.re_and_net_load` for a new column, ``interconnect_mw``, that
   allows interconnection capacity for a renewable facility to independent of its
   capacity. By default, this is the same as ``capacity_mw`` but can be reduced to
   reflect facility-specific transmission / interconnection constraints. If the
   facility has storage, storage can be charged by the constrained excess.
*  Added ``compare_hist`` argument to :meth:`.DispatchModel.plot_period` which creates
   panel plot showing both historical dispatch and redispatch for the period.
*  :meth:`.DispatchModel.plot_output` adds a row facet to show both historical and
   redispatch versions of the requested data if available.
*  Cleanup of configuration and packaging files. Contents of ``setup.cfg`` and
   ``tox.ini`` moved to ``pyproject.toml``.
*  Added the ability to specify FOM for renewables in ``re_plant_specs`` via an
   optional ``fom_per_kw`` column. This allows :meth:`.DispatchModel.re_summary` and
   derived outputs to include a ``redispatch_cost_fom`` column.
*  :class:`.DispatchModel` now contains examples as doctests.
*  :meth:`.DispatchModel.plot_all_years` to create daily redispatch plot faceted by
   month and year.
*  :meth:`.DispatchModel.dispatchable_summary` now includes mmbtu and co2 data for
   historical, redispatch, and avoided column groupings. These metrics are based on
   ``heat_rate`` and ``co2_factor`` columns in ``dispatchable_cost``, these columns are
   optional.
*  Updates to :class:`.DispatchModel` to work with the new simpler, cleaner
   :class:`.DataZip`.


Bug Fixes
^^^^^^^^^
*  Fixed an issue in :func:`.engine.dispatch_engine` where a storage resource's state of
   charge would not be carried forward if it wasn't charged or discharged in that
   hour.
*  Fixed a bug where storage metrics in :meth:`.DispatchModel.system_level_summary`
   were :class:`numpy.nan` because selecting of data from ``storage_specs`` returned
   a :class:`pandas.Series` rather than a :class:`int` or :class:`float`. Further, in
   cases of division be zero in these calculations, the result is now 0 rather than
   :class:`numpy.nan`. Tests now make sure that no new :class:`numpy.nan` show up.
*  Fixed a bug in :meth:`.DispatchModel.dispatchable_summary` where ``pct_replaced``
   would be :class:`numpy.nan` because of division by zero in these calculations, the
   result is now 0 rather than :class:`numpy.nan`. Tests now make sure that no new
   :class:`numpy.nan` show up.
*  Fixed an issue where :meth:`.DispatchModel.full_output` and methods that use it,
   i.e. :meth:`.DispatchModel.plot_output` improperly aggregated
   :attr:`.DispatchModel.system_data` when ``freq`` was not 'YS'.
*  Fixed an issue where :meth:`.DispatchModel.full_output` didn't properly show
   ``Curtailment`` and ``Storage``.

Known Issues
^^^^^^^^^^^^
*   The storage in DC-coupled RE+Storage system can be charged by either the grid or
    excess RE that would have been curtailed because of the size of the inverter. It is
    not possible to restrict grid charging in these systems. It is also not possible to
    charge storage rather than export to the grid when RE output can fit through the
    inverter.
*   It is possible that output from DC-coupled RE+Storage facilities during some hours
    will exceed the system's inverter capacity because when we discharge these storage
    facilities, we do not know how much 'room' there is in the inverter because we do
    not know the RE-side's output. This issue is now in some sense compounded when
    ``interconnect_mw`` is less than ``capacity_mw``.
*   :class:`.DataZip` are effectively immutable once they are created so the ``a`` mode
    is not allowed and the ``w`` mode is not allowed on existing files. This is because
    it is not possible to overwrite or remove a file already in a
    :class:`zipfile.ZipFile`. That fact prevents us from updating metadata about
    :class:`pandas.DataFrame` that cannot be stored in the ``parquet`` itself. Ways of
    addressing this get messy and still wouldn't allow updating existing data without
    copying everything which a user can do if that is needed.


.. _release-v0-3-0:

---------------------------------------------------------------------------------------
0.3.0 (2022-10-08)
---------------------------------------------------------------------------------------

What's New?
^^^^^^^^^^^
*   :meth:`.DispatchModel.to_file` can create an output with summary
    outputs.
*   Adopting :mod:`.pandera` for metadata and validation using
    :class:`.Validator` to organize and specialize data input
    checking.
*   Adding cost component details and capacity data to
    :meth:`.DispatchModel.dispatchable_summary`.
*   We now automatically apply ``operating_date`` and ``retirement_date`` from
    :attr:`.DispatchModel.dispatchable_plant_specs` to
    :attr:`.DispatchModel.dispatchable_profiles` using
    :func:`.apply_op_ret_date`.
*   Added validation and processing for :attr:`.DispatchModel.re_plant_specs` and
    :attr:`.DispatchModel.re_profiles`, as well as :meth:`.DispatchModel.re_summary`
    to, when the data is provided create a summary of renewable operations analogous
    to :meth:`.DispatchModel.dispatchable_summary`.
*   Added :meth:`.DispatchModel.storage_summary` to create a summary of storage
    operations analogous to :meth:`.DispatchModel.dispatchable_summary`.
*   Added :meth:`.DispatchModel.full_output` to create the kind of outputs needed by
    Optimus and other post-dispatch analysis tools.
*   Added validation steps for each type of specs that raise an error when an
    operating_date is after the dispatch period which would otherwise result in
    dispatch errors.
*   New helpers (:meth:`.DataZip.dfs_to_zip` and :meth:`.DataZip.dfs_from_zip`) that
    simplify saving and reading in groups of :class:`pandas.DataFrame`.
*   Added plotting functions :meth:`.DispatchModel.plot_period` and
    :meth:`.DispatchModel.plot_year`.

Known Issues
^^^^^^^^^^^^
*   :meth:`.DispatchModel.re_summary` and :meth:`.DispatchModel.storage_summary` have
    null operations cost data.
*   There is still no nice way to include nuclear and hydro resources.
*   :meth:`.DispatchModel.plot_year` doesn't seem to really work. At all.


Bug Fixes
^^^^^^^^^
*   A validation check throws an error when ramp rates are zero which otherwise would
    prevent plant output from ever changing on a fresh dispatch.
*   Fixed a :exc:`TypeError` issue in :func:`.apply_op_ret_date` when some dates were
    inexplicably converted to :class:`int` rather than :class:`numpy.datetime64` by
    :meth:`pandas.DataFrame.to_numpy`.

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
