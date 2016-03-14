This is a nucleosome simulation, 25095 atoms in Generalized Born
implicit solvent, setup to have no cutoff for the nonbond
terms and 15 angstroms for the calculation of GB radii.

It uses shake with a 2fs timestep and runs for 1,000 steps (2ps).

It writes to the output and trajectory files every 200 steps (0.4ps).
Additionally it writes to the restart file every 1000 steps.
NSTLIM should be increased for larger processor count runs.

The temperature used if 310K with temperature controlled using the
Berendsen thermostat.

Additionally the DNA is restrained using harmonic restraints.

Run with e.g.:

time mpirun -np 8 $AMBERHOME/exe/pmemd.MPI -O -i mdin -o mdout -p prmtop -c inpcrd -ref inpcrd

On 8 cpus of an Intel E5462  @ 2.80GHz:

Wallclock:    (0.051 ns/day)


