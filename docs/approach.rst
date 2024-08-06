=======================================================================================
Approach
=======================================================================================

In which we explain in greater detail how the thing works.

---------------------------------------------------------------------------------------
Preparing the input data
---------------------------------------------------------------------------------------
#. Data alignment and validation
#. Determining renewable AC profiles and excess for DC charging
#. Calculating net load


---------------------------------------------------------------------------------------
Dispatch logic
---------------------------------------------------------------------------------------
The dispatch logic is applied to each dispatchable resource (including storage) in each
hour sequentially in order to serve load in that hour. The only exception to this
process is the first hour in which historical dispatch is used and storage is charged
if there is excess renewable generation.

The general approach is to try and meet load with already operating generators, then
charge/discharge storage, then, if necessary, startup additional generators. In the
process we start with what we call the deficit, which begins as simply net load in the
hour, as we go through the procedure in an hour, we reduce the deficit as we dispatch
resources to serve it.

The following is the procedure applied in each hour, after the first. See
:func:`.dispatch_engine` for the full implementation.


1. Calculate provisional deficit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Before we do anything we adjust the deficit that will be used to drive the dispatch
logic. The goal of this process is to prepare storage for expected load over the next
24 hours, and note this the only place where the model can look into the future. We
implement this by determining the state of charge we would like
to be at given 'expected' near-term increase or decrease in net load, and then
adjusting the deficit in order to get closer to that reserve level.

The first part is to calculate what the target reserve level is for each storage
facility as follows, the target state of charge is simply the product of the reserve
and the storage facility's maximum state of charge. See :func:`.dynamic_reserve`.

    .. math::
       :label: reserve

          ramp &= \frac{max(load_{h+1}, ..., load_{h+24})}{load_h} - 1

          reserve &= 1 - e^{-coeff \times ramp}

Using these target states of charge we either increase the deficit so that we can run
dispatchable generators more in order to charge storage. Or we reduce the deficit so
that we run dispatchable generators less and are then able to draw down storage. If we
do have excess reserve, we only adjust the deficit in order to draw down storage to 2x
the reserve, not all the way down to the reserve. See
:func:`.adjust_for_storage_reserve`.

.. note::
   The coefficient in the formula above can be set manually or the model can try a
   number of values and select the one that produces the least unmet load.

2. Dispatch already operating generators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this step we set the output of all dispatchable generators that were operating in
the previous hour in order of ascending marginal cost. Once a generator's output is set
the amount of that output reduces the deficit. Once the deficit reaches zero,
subsequent generators may be turned off if allowed.

A generator's output in a given hour is determined both by system need but may also be
constrained by generator operating constraints which are, minimum uptime before output
can be reduced, ramp rate (which is assumed to be symmetric), and historical hourly
output. See :func:`.calculate_generator_output`. Which of these will be applied to a
given generator is configurable.

3. Charge storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Storage can be charged from two sources, either from DC-coupled renewables, or from
the grid at large. We assume that DC charging only occurs when coupled renewables
generate more than they their inverter capacity, i.e. there is no strategic charging or
ability to respond to level-system curtailment on the DC-side.

DC-charging is thus independent of grid conditions so any storage that is DC-coupled
will be charged in every hour if there is generation to do so.

Additionally, storage can be charged by the grid if there is excess generation in an
hour because of excess renewable generation (i.e. negative net load), because of an
increase in the provisional deficit to increase storage state of charge, or because
operating constraints prevented dispatchable generators from reducing output.
See :func:`.charge_storage`.

4. Discharge storage (part 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If there is still a deficit for the hour we want to discharge each storage facility
down to 2x its reserve in order to reduce the remaining deficit as much as possible.

5. Start offline generators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If a deficit remains after the first round of discharging storage, we start
dispatchable generators that were offline in the previous hour in order of ascending
marginal cost. A generator's output will be the lesser of the remaining defict, its
historical output, and its ramp rate in MW.

6. Discharge storage (part 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If there is still a deficit for the hour we discharge each storage facility
down to zero state of charge in order to reduce the remaining deficit as much as
possible.


---------------------------------------------------------------------------------------
Limitations
---------------------------------------------------------------------------------------

#. Place of storage in dispatch sequence means it reduces fossil dispatch more than it
   helps reliability. It can even make the latter worse by deferring fossil start-up
   until it is too late, i.e. make the required ramp steeper than the resources can
   meet.
#. Unintuitive behavior of more storage hurting reliability because of previous combined
   with storage able to discharge faster means fewer hours of deficits but larger
   deficits in those hours.
#. Storage reserves don't see DC-charging potential and so may run fossil to charge
   storage that would have been charged by DC. While true on the charging side,
   it is also true on the discharge side so we don't selectively deplete storage that
   can be recharged by expected captive generation.
#. We cannot start up fossil generators to build up reserve.
