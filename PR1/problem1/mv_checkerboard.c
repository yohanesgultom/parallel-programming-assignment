/*
Matrix vector multiplication (checker-board)
Author: yohanes.gultom@gmail.com
*/

#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "math.h"

#define MASTER 0       /* id of the first process */
#define FROM_MASTER 1  /* setting a message type */
#define FROM_WORKER 2  /* setting a message type */

MPI_Status status;

void printmatrix(int row, int col, int matrix[row][col])
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

void printarray(int len, int a[len])
{
    int i = 0;
    for(i = 0; i < len; i++)
    {
        printf("%d\t",a[i]);
    }
	printf("\n");
}

void mv_checkerboard(int rows, int sm_rows, int numworkers, int procid) {
    double texec = 0, tcomm = 0;
    int worker, i, j = 0;

    int m[rows][rows];
    int v[rows][1];
    int mv[rows][1];
    int sm_m[sm_rows][sm_rows]; // small matrix
    int sm_v[sm_rows][1];
    int sm_mv[sm_rows][1];

    if (procid == MASTER) {

        // init
        for (i=0;i<rows;i++)
        {
            // v[i][0] = i;
            v[i][0] = 2;
            mv[i][0] = 0;
            for (j=0;j<rows;j++)
            {
                // m[i][j] = i*rows+j;
                m[i][j] = 1;
            }
        }

        // printmatrix(rows, rows, m);
        // printf("\n");
        // printmatrix(rows, 1, v);
        // printf("\n");

        // distribute m and v
        texec -= MPI_Wtime();
        for (worker = 1; worker <= numworkers; worker++) {
            // printf("worker:%d\n", worker);
            int row_start = ceil((worker-1) * sm_rows / rows) * sm_rows;
            // printf("%d row_start: %d\n", worker, row_start);
            int col_start = (worker-1) * sm_rows % rows;
            // printf("%d col_start: %d\n", worker, col_start);
            // printf("worker:%d\n", worker);
            for (i=0;i<sm_rows;i++)
            {
                for (j=0;j<sm_rows;j++)
                {
                    sm_m[i][j] = m[row_start+i][col_start+j];
                    // printf("%d sm_m[%d][%d] = m[%d][%d]\n", worker, i, j, (row_start+i), (col_start+j));
                }
                sm_v[i][0] = v[col_start+i][0];
                // printf("%d sm_v[%d][%d] = v[%d][%d]\n", worker, i, 0, (v_col_start+i), 0);
            }
            // printmatrix(sm_rows, sm_rows, sm_m);
            // printf("\n");
            tcomm -= MPI_Wtime();
            MPI_Send(&sm_m,sm_rows*sm_rows,MPI_INT,worker,FROM_MASTER,MPI_COMM_WORLD);
            MPI_Send(&sm_v,sm_rows,MPI_INT,worker,FROM_MASTER,MPI_COMM_WORLD);
            tcomm += MPI_Wtime();
            // printf("worker:%d\n", worker);
        }

        // get and merge result from workers
        for (worker = 1; worker <= numworkers; worker++) {
            int source = worker;
            int col_start = (worker-1) * sm_rows % rows;
			// receive tmp matrix from each process
            tcomm -= MPI_Wtime();
            MPI_Recv(&sm_mv,sm_rows,MPI_INT,source,FROM_WORKER,MPI_COMM_WORLD, &status);
            tcomm += MPI_Wtime();
            for (i=0;i<sm_rows;i++)
            {
                mv[col_start+i][0] += sm_mv[i][0];
            }
        }
        texec += MPI_Wtime();
        // printmatrix(rows, 1, mv);
        // printf("\n");
        printf("%d\t%d\t%lf\t%lf\n", numworkers+1, rows, tcomm, texec);

    } else { // == WORKERS
        MPI_Recv(&sm_m,sm_rows*sm_rows,MPI_INT,MASTER,FROM_MASTER,MPI_COMM_WORLD,&status);
        MPI_Recv(&sm_v,sm_rows,MPI_INT,MASTER,FROM_MASTER,MPI_COMM_WORLD,&status);

        // multiply
        for (i=0;i<sm_rows;i++)
        {
            int sum = 0;
            for (j=0;j<sm_rows;j++)
            {
                sum += sm_m[i][j] * sm_v[j][0];
            }
            sm_mv[i][0] = sum;
        }

        // if (procid == 4) {
        //     printmatrix(sm_rows, sm_rows, sm_m);
        //     printf("\n");
        //     printmatrix(sm_rows, 1, sm_v);
        //     printf("\n");
        //     printmatrix(sm_rows, 1, sm_mv);
        //     printf("\n");
        // }
        MPI_Send(&sm_mv,sm_rows,MPI_INT,MASTER,FROM_WORKER,MPI_COMM_WORLD);

    }
}

int main(int argc, char **argv) {

    int numprocs,   /* number of processes in partition */
    procid,         /* a process identifier */
    numworkers,     /* number of worker processes */
    rows;

    rows = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    numworkers = numprocs-1;

    // optimal data vs workers proportion
    double x = sqrt(rows * rows / (double)numworkers);
    if (ceil(x) != x) {
        if (procid == MASTER) printf("Number of matrix elements (%d) can't be evenly distributed to number of workers (%d)\n", (rows * rows), numworkers);
        MPI_Finalize();
        return 0;
    }

    mv_checkerboard(rows, ceil(x), numworkers, procid);
    MPI_Finalize();

    return 0;
} /* of main */
