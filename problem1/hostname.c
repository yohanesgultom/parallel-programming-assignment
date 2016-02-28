#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
  int p, rank;
  char hostname[1024];
  hostname[1023] = '\0';

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  gethostname(hostname, 1023);
  printf("Hostname: %s\n", hostname);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
