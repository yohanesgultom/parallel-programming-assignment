/*
Matrix vector multiplication (column-wise)
Author: yohanes.gultom@gmail.com
Baseline: http://sites.google.com/site/heshanhome/resources/MPI_matrix_multiplication.c
*/

#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

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

void multiply_two_arrays(int NRA, int NCA, int NCB, int numworkers, int procid) {

    int source,         /* process id of message source */
    dest,           /* process id of message destination */
    nbytes,         /* number of bytes in message */
    mtype,          /* message type */
    cols,           /* cols of A sent to each worker */
    avecol, extra, offset,
    i, j, k, n, count;

    int a[NRA][NCA],   /* matrix A to be multiplied */
    b[NCA][NCB],   /* matrix B to be multiplied */
	c_tmp[NRA][NCB],   /* tmp matrix before sum */
    c[NRA][NCB];   /* result matrix C */
    // int **a, **b, **c;
    double texec = 0, tcomm = 0;


    /******* master process ***********/
    if (procid == MASTER) {

        // inits
        for (i=0; i<NRA; i++)
            for (j=0; j<NCA; j++)
                a[i][j]= 1;

        for (i=0; i<NCA; i++)
            for (j=0; j<NCB; j++)
                b[i][j]= 2;

		for (i=0; i<NRA; i++)
            for (j=0; j<NCB; j++)
                c[i][j]= 0;

        /* send matrix data to the worker processes */
        texec -= MPI_Wtime();
        avecol = NCA/numworkers;
        extra = NCA%numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        for (dest=1; dest<=numworkers; dest++) {
            cols = (dest <= extra) ? avecol+1 : avecol;
			// printf("dest: %d cols: %d\n", dest, cols);
            tcomm -= MPI_Wtime();
            MPI_Send(&cols,1,MPI_INT,dest,mtype,MPI_COMM_WORLD);
            tcomm += MPI_Wtime();
            count = cols*NRA;
			// cols to array
			int tmp[count];
			k = 0;
			for (i = offset; i < offset + cols; i++) {
				for (j = 0; j < NRA; j++) {
					tmp[k] = a[j][i];
					k++;
				}
			}
			// printarray(count, tmp);
			// send columns
            tcomm -= MPI_Wtime();
            MPI_Send(&tmp,count,MPI_INT,dest,mtype,MPI_COMM_WORLD);
			// send a row
            MPI_Send(&b[0],NCB,MPI_INT,dest,mtype,MPI_COMM_WORLD);
            tcomm += MPI_Wtime();
            offset = offset + cols;
        }

        /* wait for results from all worker processes */
        mtype = FROM_WORKER;
        for (i = 1; i <= numworkers; i++) {
            source = i;
			// receive tmp matrix from each process
            tcomm -= MPI_Wtime();
            MPI_Recv(&c_tmp,NRA*NCB,MPI_INT,source,mtype,MPI_COMM_WORLD, &status);
            tcomm += MPI_Wtime();
			// add matrix
			int j, k = 0;
			for(j = 0; j < NRA; j++)
			{
				for(k = 0; k < NCB; k++)
				{
					c[j][k] += c_tmp[j][k];
				}
			}
        }

		// printmatrix(NRA, NCB, c);

        texec += MPI_Wtime();
        printf("%d\t%d\t%lf\t%lf\n", numworkers+1, NRA, tcomm, texec);

    } /* end of master */

    /************ worker process *************/
    if (procid > MASTER) {

        mtype = FROM_MASTER;
        source = MASTER;

        MPI_Recv(&cols,1,MPI_INT,source,mtype,MPI_COMM_WORLD,&status);
		// printf("proc:%d cols:%d\n", procid, cols);
        count = cols*NRA;
		int tmp[count];
        MPI_Recv(&tmp,count,MPI_INT,source,mtype,MPI_COMM_WORLD,&status);
		// printf("tmp %d: ", procid);
		// printarray(count, tmp);
		int tmp_row[NCB];
        MPI_Recv(&tmp_row,NCB,MPI_INT,source,mtype,MPI_COMM_WORLD,&status);
		// printf("tmp_row %d: ", procid);
		// printarray(NCB, tmp_row);

		// init with zeros
		for(i = 0; i < NRA; i++)
        {
            for(j = 0; j < NCB; j++)
            {
                c[i][j] = 0;
            }
        }

		// if (procid == 1) {
		// 	printarray(count, tmp);
		// 	printarray(NCB, tmp_row);
		// 	printmatrix(NRA, NCB, c);
		// }

		// multiply and add (if more than one col)
		for (n = 0; n < cols; n++) {
			int start = n * NRA;
			for (i = 0; i < NRA; i++) {
				for (j = 0; j < NCB; j++) {
					c[i][j] = c[i][j] + tmp[start + i] * tmp_row[j];
					// if (procid == 1) printf("c[%d][%d]=%d\n", i, j, c[i][j]);
				}
			}
		}

		// if (procid == 1) {
		// 	printmatrix(NRA, NCB, c);
		// }

        mtype = FROM_WORKER;
        MPI_Send(&c,NRA*NCB,MPI_INT,MASTER,mtype,MPI_COMM_WORLD);

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
