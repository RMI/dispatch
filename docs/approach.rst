=======================================================================================
Approach
=======================================================================================

In which we explain in greater detail how the thing works.




---------------------------------------------------------------------------------------
Issues in the current approach
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

---------------------------------------------------------------------------------------
Things to consider without adding look-aheads
---------------------------------------------------------------------------------------
#. Enforce minimum uptime for fossil generators. This will provide more opportunities
   for charging storage.
#. Segment dispatchable generators and storage to allow interspersing in the dispatch
   sequence.
