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
#. The first hour
#. Dynamic reserves
#. Provisional deficit
#. Operating generators
#. Charge storage
#. Discharge storage (first round)
#. Start generators
#. Discharge storage (second round)


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
#. No ability to save state of charge for future needs or deliberately bank it by
   charging with fossil. This is absolutely required for storage to replace peakers.
