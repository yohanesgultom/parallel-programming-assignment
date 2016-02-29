/* 
 *	Matrix - Vector Multiplication, Version 2
 *      Columnwise block-striped decomposition
 */

// Source: http://www.math.utep.edu/Faculty/pmdelgado2/courses/parallel/ch08/mvmult2.c

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "MyMPI.h"

// change the next two lines to adapt code to different data types
// Ex: int, float, double, etc...

typedef float dtype;
#define mpitype MPI_FLOAT

int main (int argc, char **argv)
{
// Note:  A * x = b
dtype **A; 		// the input matrix A
dtype *x;  		// the input vector x
dtype *b;  		// the output vector b
dtype *b_part_out;	// Partial sums, sent
dtype *b_part_in;	// Partial sums, received
int *cnt_out;		// number of elements sent to each processor
int *cnt_in;		// # of elements received by each processor
int *disp_out;		// indices of sent elements
int *disp_in;		// indices of received elements
int i, j;  		// loop indices
int id;			// process id #
int p;			// total # of processes
int local_els;		// Columns of matrix A and elements of vector x
			// held by this process.
int m,n;		// # of rows, columns, respectively, of matrix A
int nprime;		// size of the vector
dtype *storage;		// This process's portion of matrix A.

MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &p);
MPI_Comm_rank(MPI_COMM_WORLD, &id);

read_col_striped_matrix(argv[1], (void ***) &A, (void **) &storage,
		        mpitype, &m, &n, MPI_COMM_WORLD);
if (!id) printf("A=\n");
print_col_striped_matrix((void **) A, mpitype, m,n, MPI_COMM_WORLD);

read_block_vector(argv[2], (void **) &x, mpitype, &nprime, MPI_COMM_WORLD);
if (!id) printf("x=\n");
print_block_vector( (void *) x, mpitype, nprime, MPI_COMM_WORLD);

// Now, each process multiplies its columns of A with vector x, resulting
// in a partial sum of the product 'b'.

b_part_out=(dtype *) my_malloc(id, n * sizeof(dtype));
local_els = BLOCK_SIZE(id,p,n); // divide columns to each of the 
				// processors


for (i=0; i<n; i++) {
	b_part_out[i] = 0.0;
	for (j=0; j<local_els; j++)
		b_part_out[i] += A[i][j] * x[j];	
}



create_mixed_xfer_arrays(id, p, n, &cnt_out, &disp_out);
create_uniform_xfer_arrays(id, p, n, &cnt_in, &disp_in);

b_part_in = (dtype *) my_malloc(id, p*local_els*sizeof(dtype));
MPI_Alltoallv (b_part_out, cnt_out, disp_out, mpitype, b_part_in, 
		cnt_in, disp_in, mpitype, MPI_COMM_WORLD);

b= (dtype *) my_malloc(id, local_els * sizeof(dtype));
for (i=0; i<local_els; i++) {
	b[i]=0.0;
	for (j=0; j<p; j++)
		b[i] += b_part_in[i +j*local_els];
}

if (!id) printf("A x = b\nb=\n");
print_block_vector ((void *) b, mpitype, n, MPI_COMM_WORLD);

MPI_Finalize();
return 0;
}

