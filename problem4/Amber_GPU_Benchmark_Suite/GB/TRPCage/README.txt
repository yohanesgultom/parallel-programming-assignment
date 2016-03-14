This is a TRPCage simulation, 304 atoms in Generalized Born
implicit solvent, setup to have no cutoff for the nonbond
terms or for the calculation of GB radii. To speed this up
in practice one could probably reduce RGBMax to around 15 to 18
angstroms. The nonbond cutoff (cut) should probably not be changed.

It uses shake with a 2fs timestep and runs for 250,000 steps (500ps).

It writes to the output and trajectory files every 1000 steps (2ps).
Additionally it writes to the restart file every 50000 steps.
NSTLIM should be increased for larger processor count runs.

The temperature used if 325K to match the orignal Simmerling JACS 2002
paper. Temperature is controlled with the Berendsen thermostat, also 
matching the 2002 JACS paper.

Run with e.g.:

time mpirun -np 8 $AMBERHOME/exe/pmemd.MPI -O -i mdin -o mdout -p prmtop -c inpcrd

On 8 cpus of an Intel E5462  @ 2.80GHz:

Wallclock:    413.31 seconds (104.50 ns/day)


