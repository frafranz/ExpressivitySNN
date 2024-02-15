# ExpressivitySNN - Linear Regions

This branch contains some additional functionality to distinguish and count linear regions.

**One of the main changes:** Both the event-based and the time-discretized implementation of spiking networks now not only return spike times, but also causality information for the spikes; for instance, whether an earlier spike arrived in time to influence one of the later spikes. Some of the methods we use to distinguish linear regions rely on this information.

The branch is still in a preliminary state and might be merged into master at a later point.
