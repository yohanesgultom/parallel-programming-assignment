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

  int row = 3;
  int col = 3;
  int m[row][col];
  int n[col];
  // m = (int **)malloc(row * sizeof(int*));
  if (rank == 0)
  {
    //   for(i = 0; i < row; i++)
    //   {
    //       m[i]=(int *)malloc(col * sizeof(int));
    //   }

      for(i = 0; i < row; i++)
      {
          for(j = 0; j < col; j++)
          {
              m[i][j] = i * col + j;
          }
      }

    //   for(i = 0; i < row; i++)
    //   {
    //       for(j = 0; j < col; j++)
    //       {
    //           printf("%d ", m[i][j]);
    //       }
    //       printf("\n");
    //   }

      MPI_Send(&m, row*col, MPI_INT, 1, 99, MPI_COMM_WORLD);
      MPI_Send(&m[0], col, MPI_INT, 1, 99, MPI_COMM_WORLD);
  }
  else if (rank == 1)
  {
      MPI_Recv(&m, row*col, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);
      MPI_Recv(&n, col, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);
      for(i = 0; i < row; i++)
      {
          for(j = 0; j < col; j++)
          {
              printf("%d ", m[i][j]);
          }
          printf("\n");
      }
      printf("\n");
      for(i = 0; i < row; i++)
      {
          printf("%d ", n[i]);
      }
      printf("\n");
  }

  // gethostname(hostname, 1023);
  // printf("Hostname: %s\n", hostname);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
