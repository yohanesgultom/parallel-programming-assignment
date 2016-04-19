#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define MIN( x, y ) ((x) < (y) ? x : y)

int main( int argc, char *argv[] )
{
    int SIZE, i, j, row, numworkers, numsent, sender;
    double dotp, comm_time = 0, exec_time = 0;
    if ( argc != 3 ) {
        printf( "usage: %s <number of workers> <square matrix/vector row size>\n", argv[0]);
    } else {
        numworkers = atoi( argv[1] );
        SIZE = atoi( argv[2] );
    }

    exec_time -= MPI_Wtime();
    double a[SIZE][SIZE], b[SIZE], c[SIZE];
    MPI_Status status;
    MPI_Comm workercomm;

    MPI_Init( &argc, &argv );
    comm_time -= MPI_Wtime();
    MPI_Comm_spawn( "worker", MPI_ARGV_NULL, numworkers, MPI_INFO_NULL, 0, MPI_COMM_SELF, &workercomm, MPI_ERRCODES_IGNORE );
    comm_time += MPI_Wtime();

    /* initialize a and b */
    for (i = 0; i < SIZE; i++ )
        for ( j = 0; j < SIZE; j++ )
            a[i][j] = ( double ) 1;

    for (i = 0; i < SIZE; i++ )
        b[i] = ( double ) 2;

    /* send SIZE to each worker */
    comm_time -= MPI_Wtime();
    MPI_Bcast( &SIZE, 1, MPI_INT, MPI_ROOT, workercomm );

    /* send b to each worker */
    MPI_Bcast( b, SIZE, MPI_DOUBLE, MPI_ROOT, workercomm );

    /* send a's row to each process */
    numsent = 0;
    for ( i = 0; i < MIN( numworkers, SIZE ); i++ ) {
	    MPI_Send( a[i], SIZE, MPI_DOUBLE, i, i+1, workercomm );
        numsent++;
    }
    comm_time += MPI_Wtime();

    //printf("numsent: %d SIZE: %d\n", numsent, SIZE);

    /* receive dot products back from workers */
    for ( i = 0; i < SIZE; i++ ) {
        comm_time -= MPI_Wtime();
        MPI_Recv( &dotp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, workercomm, &status );
        comm_time += MPI_Wtime();
        sender = status.MPI_SOURCE;
        row = status.MPI_TAG;
        c[i] = dotp;
        /* send another row back to this worker if there is one */
        if ( numsent < SIZE ) {
            comm_time -= MPI_Wtime();
            MPI_Send( a[numsent], SIZE, MPI_DOUBLE, sender, numsent+1, workercomm );
            comm_time += MPI_Wtime();
            numsent++;
        } else { /* no more work */
            comm_time -= MPI_Wtime();
            MPI_Send( MPI_BOTTOM, 0, MPI_DOUBLE, sender, 0, workercomm );
            comm_time += MPI_Wtime();
        }
    }

    /* print matrix, vector & result */
    /*
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            printf("%f ", a[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    for (i = 0; i < SIZE; i++)
        printf("%f\n", b[i]);
    printf("\n");
    for (i = 0; i < SIZE; i++)
        printf("%f\n", c[i]);
    */

    MPI_Finalize();
    exec_time += MPI_Wtime();

    printf("%d\t%f\t%f\n", numworkers, comm_time, exec_time);
    return 0;
}
