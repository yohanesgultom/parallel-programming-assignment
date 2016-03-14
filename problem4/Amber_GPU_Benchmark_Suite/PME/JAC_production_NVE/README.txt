This is a more realistic version of the JAC (DHFR) benchmark
run with what would be typical NVE production parameters. The
SHAKE and DSUM_TOL tolerances have been increased by one order
of magnitude in order to ensure good energy conservation.

It uses shake with a 2fs timestep. It has a 8 angstrom cutoff,
runs in the NVE ensemble and writes to the output and trajectory
file every 1000 steps (2ps). Additionally it writes to the restart
file every 10000 steps. NSTLIM is set to 10,000 by default but 
should be increased for larger processor count runs.

For performance reasons it is set to produce NETCDF binary trajectory
files.

This is DHFR in TIP3P Water box - 23,558 atoms.

Run with e.g.:

mpirun -np 4 $AMBERHOME/exe/pmemd -O -i mdin -o mdout -p prmtop -c inpcrd

On 2 cpus of a Pentium-D 3.2GHz machine the timing is:

Master NonSetup CPU time:      1396.85 seconds

