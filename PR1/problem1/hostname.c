/**
 * Small program to print hostname of the MPI processor
 * By yohanes.gultom@gmail.com
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
  int i, j, p, rank;
  char hostname[1024];
  MPI_Status status;
  hostname[1023] = '\0';

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // printf("%d\n", p);

  // int row = 3;
  // int col = 3;
  // int m[row][col];
  // int n[col];
  // int c[col];
  //
  // if (rank == 0)
  // {
  //
  //     for(i = 0; i < row; i++)
  //     {
  //         for(j = 0; j < col; j++)
  //         {
  //             m[i][j] = i * col + j;
  //         }
  //     }
  //     for(i = 0; i < row; i++)
  //     {
  //         for(j = 0; j < col; j++)
  //         {
  //             printf("%d ", m[i][j]);
  //         }
  //         printf("\n");
  //     }
  //     printf("\n");
  //
  //     // send matrix
  //     MPI_Send(&m, row*col, MPI_INT, 1, 99, MPI_COMM_WORLD);
  //     // send row
  //   //   MPI_Send(&m[0], col, MPI_INT, 1, 99, MPI_COMM_WORLD);
  //   //   // send col
  //   //   int col_i = 0;
  //   //   for (i = 0; i < row; i++) {
  //   //       c[i] = m[i][col_i];
  //   //   }
  //   //   MPI_Send(&c[0], row, MPI_INT, 1, 99, MPI_COMM_WORLD);
  //   //   for(i = 0; i < row; i++)
  //   //   {
  //   //       printf("%d\n", c[i]);
  //   //   }
  // }
  // else if (rank == 1)
  // {
  //     MPI_Recv(&m, row*col, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);
  //   //   MPI_Recv(&n, col, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);
  //   //   MPI_Recv(&c, row, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);
  //   //   for(i = 0; i < row; i++)
  //   //   {
  //   //       printf("%d\n", c[i]);
  //   //   }
  //   //   printf("\n");
  // }

  gethostname(hostname, 1023);
  printf("Hostname: %s\n", hostname);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
