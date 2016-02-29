#include"stdio.h"
#include"stdlib.h"
#include"mpi.h"

/**
 * Print multidimensional array
 */
void print_array(int **matrix, int row, int col)
{
  int i, j = 0;
  for(i = 0; i < row; i++)
  {
    for(j = 0; j < col; j++)
    {
      printf("%d\t",matrix[i][j]);
    }
    printf("\n");
  }
}

/**
 * multiply two arrays
 */
void multiply_two_arrays(int x, int n, int y, int size, int rank)
{
  int i, j, k, sum=0;
  int **matrix1;  //declared matrix1[x][n]
  int **matrix2; //declare matrix2[n][y]
  int **mat_res; //resultant matrix become mat_res[x][y]
  double t,tc;
  MPI_Status status;

  /*----------------------------------------------------*/
  //create array of pointers(Rows)
  matrix1 =(int **)malloc(x * sizeof(int*));
  matrix2 =(int **)malloc(n * sizeof(int*));
  mat_res=(int **)malloc(x * sizeof(int*));
  /*----------------------------------------------------*/

  /*--------------------------------------------------------------------------------*/
  //allocate memory for each Row pointer
  for(i = 0; i < x; i++)
  {
    matrix1[i]=(int *)malloc(n * sizeof(int));
    mat_res[i]=(int *)malloc(y * sizeof(int));
  }

  for(i = 0; i < n; i++)
  matrix2[i]=(int *)malloc(y*sizeof(int));
  /*--------------------------------------------------------------------------------*/

  for(i = 0; i < x; i++)
  {
    for(j = 0; j < n; j++)
    {
      matrix1[i][j] = 1; //initialize 1 to matrix1 for all processes
    }
  }

  for(i = 0; i < n; i++)
  {
    for(j = 0; j < y; j++)
    {
      matrix2[i][j] = 2;//initialize 2 to matrix2 for all processes
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) t -= MPI_Wtime();
  // divide the task in multiple processes
  // size = number of process.
  // -- if size == 1 (minimum) then that single process will calculate all row * column operations
  // -- if 1 < size < x then then each process will do (total process / size) operations. Each process will do rows * column with index = rank + size * iteration. eg: process rank n will do rows * column with index = { n, n+size, n+2*size, n+3*size + .. + n+i*size }
  // -- if size >= x then each process will only calculate one row * one column
  for(i = rank; i < x; i = i+size)
  {
    for(j = 0; j < y; j++)
    {
      sum=0;
      for(k = 0; k < n; k++)
      {
        sum = sum + matrix1[i][k] * matrix2[k][j];
      }
      mat_res[i][j] = sum;
    }
  }

  if(rank != 0)
  {
    for(i = rank; i < x; i = i+size)
    MPI_Send(&mat_res[i][0], y, MPI_INT, 0, 10+i, MPI_COMM_WORLD);//send calculated rows to process with rank 0
  }

  if(rank == 0)
  {
    tc -= MPI_Wtime();
    for(j = 1; j < size; j++)
    {
      for(i = j; i < x; i = i+size)
      {
        MPI_Recv(&mat_res[i][0], y, MPI_INT, j, 10+i, MPI_COMM_WORLD, &status);//receive calculated rows from respective process
      }
    }
    tc += MPI_Wtime();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0)
  {
    t += MPI_Wtime();
    // // print input matrices
    // print_array(matrix1, x, n);
    // printf("\n*\n\n");
    // print_array(matrix2, n, y);
    // // print result
    // printf("\n=\n\n");
    // print_array(mat_res, x, y);
    // printf("\nTime taken = %f seconds\n",result); //time taken
    //printf("%d\ta[%d][%d]*[%d][%d]\t%f\t%f\n", size, x, n, n, y, tc, t);
    // assuming square matrix
    printf("%d\t%d\t%f\t%f\n", size, x, tc, t);
  }

  // free memory
  for(i = 0; i < x; i++)
  {
    free(matrix1[i]);
    free(mat_res[i]);
    matrix1[i] = NULL;
    mat_res[i] = NULL;
  }
  for(i = 0; i < n; i++)
  {
      free(matrix2[i]);
      matrix2[i] = NULL;
  }
  free(matrix1);
  free(matrix2);
  free(mat_res);
  matrix1 = NULL;
  matrix2 = NULL;
  mat_res = NULL;
}

/**
 * main program
 */
int main(int argc , char **argv)
{
  int size,rank = 0;
  int x = atoi(argv[1]);
  int n = x;
  int y = 1;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  multiply_two_arrays(x, n, y, size, rank);

  MPI_Finalize();
  return 0;
}
