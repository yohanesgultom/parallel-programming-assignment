/*
Matrix vector multiplication (row-wise)
Source: http://sites.google.com/site/heshanhome/resources/MPI_matrix_multiplication.c
*/

#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

#define MASTER 0       /* id of the first process */
#define FROM_MASTER 1  /* setting a message type */
#define FROM_WORKER 2  /* setting a message type */

MPI_Status status;

// void printmatrix(int row, int col, int **matrix)
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

void multiply_two_arrays(int NRA, int NCA, int NCB, int numworkers, int procid) {

    int source,         /* process id of message source */
    dest,           /* process id of message destination */
    nbytes,         /* number of bytes in message */
    mtype,          /* message type */
    rows,           /* rows of A sent to each worker */
    averow, extra, offset,
    i, j, k, count;

    int a[NRA][NCA],   /* matrix A to be multiplied */
    b[NCA][NCB],   /* matrix B to be multiplied */
    c[NRA][NCB];   /* result matrix C */
    // int **a, **b, **c;
    double tstart, tend, tcomm = 0;


    /******* master process ***********/
    if (procid == MASTER) {

        // inits
        for (i=0; i<NRA; i++)
            for (j=0; j<NCA; j++)
                a[i][j]= 1;
        // printmatrix(NRA, NCA, a);

        for (i=0; i<NCA; i++)
            for (j=0; j<NCB; j++)
                b[i][j]= 2;
        // printmatrix(NCA, NCB, b);

        /* send matrix data to the worker processes */
        tstart = MPI_Wtime();
        averow = NRA/numworkers;
        extra = NRA%numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        tcomm -= MPI_Wtime();
        for (dest=1; dest<=numworkers; dest++) {
            rows = (dest <= extra) ? averow+1 : averow;
            MPI_Send(&offset,1,MPI_INT,dest,mtype,MPI_COMM_WORLD);
            MPI_Send(&rows,1,MPI_INT,dest,mtype,MPI_COMM_WORLD);
            count = rows*NCA;
            MPI_Send(&a[offset][0],count,MPI_INT,dest,mtype,MPI_COMM_WORLD);
            count = NCA*NCB;
            MPI_Send(&b,count,MPI_INT,dest,mtype,MPI_COMM_WORLD);
            offset = offset + rows;
        }
        tcomm += MPI_Wtime();

        /* wait for results from all worker processes */
        mtype = FROM_WORKER;
        tcomm -= MPI_Wtime();
        for (i=1; i<=numworkers; i++) {
            source = i;
            MPI_Recv(&offset,1,MPI_INT,source,mtype,MPI_COMM_WORLD, &status);
            MPI_Recv(&rows,1,MPI_INT,source,mtype,MPI_COMM_WORLD, &status);
            count = rows*NCB;
            MPI_Recv(&c[offset][0],count,MPI_INT,source,mtype,MPI_COMM_WORLD, &status);
        }
        tcomm += MPI_Wtime();

        // printmatrix(NRA, NCB, c);

        tend = MPI_Wtime();
        printf("\n%d\t%d\t%lf\t%lf\n", numworkers+1, NRA, tcomm, (tend - tstart));

    } /* end of master */

    /************ worker process *************/
    if (procid > MASTER) {

        mtype = FROM_MASTER;
        source = MASTER;

        MPI_Recv(&offset,1,MPI_INT,source,mtype,MPI_COMM_WORLD,&status);
        MPI_Recv(&rows,1,MPI_INT,source,mtype,MPI_COMM_WORLD,&status);
        count = rows*NCA;
        MPI_Recv(&a,count,MPI_INT,source,mtype,MPI_COMM_WORLD,&status);
        count = NCA*NCB;
        MPI_Recv(&b,count,MPI_INT,source,mtype,MPI_COMM_WORLD,&status);

        for (k=0; k<NCB; k++) {       /* multiply our part */
            for (i=0; i<rows; i++) {
                c[i][k] = 0.0;
                for (j=0; j<NCA; j++){
                    c[i][k] = c[i][k] + a[i][j] * b[j][k];
                }
            }
        }

        mtype = FROM_WORKER;
        MPI_Send(&offset,1,MPI_INT,MASTER,mtype,MPI_COMM_WORLD);
        MPI_Send(&rows,1,MPI_INT,MASTER,mtype,MPI_COMM_WORLD);
        MPI_Send(&c,rows*NCB,MPI_INT,MASTER,mtype,MPI_COMM_WORLD);

    }
}

int main(int argc, char **argv) {

    int numprocs,   /* number of processes in partition */
    procid,         /* a process identifier */
    numworkers,     /* number of worker processes */
    NRA, NCA, NCB;

    NRA = atoi(argv[1]);
    NCA = NRA;
    NCB = 1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    numworkers = numprocs-1;

    multiply_two_arrays(NRA, NCA, NCB, numworkers, procid);

    MPI_Finalize();

    return 0;
} /* of main */
