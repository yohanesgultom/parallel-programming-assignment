This is a Myoglobin simulation, 2492 atoms in Generalized Born
implicit solvent, setup to have no cutoff for the nonbond
terms and 15 angstroms for the calculation of GB radii.

It uses shake with a 2fs timestep and runs for 25,000 steps (50ps).

It writes to the output and trajectory files every 1000 steps (2ps).
Additionally it writes to the restart file every 10000 steps.
NSTLIM should be increased for larger processor count runs.

The temperature used if 300K with temperature controlled using the
Langevin thermostat and a collision rate of 1.0 ps-1.

Note: ig=-1 is used to seed the random number generator with the wallclock
      time. This is ALWAYS recommended when running with ntt=3 or 2. To get
      reproducibility in the results, however, you should remove ig from the
      cntrl namelist.

Run with e.g.:

time mpirun -np 8 $AMBERHOME/exe/pmemd.MPI -O -i mdin -o mdout -p prmtop -c inpcrd

On 8 cpus of an Intel E5462  @ 2.80GHz:

Wallclock:    1094.10 seconds (3.95 ns/day)


