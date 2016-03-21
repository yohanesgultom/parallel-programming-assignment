#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define MIN( x, y ) ((x) < (y) ? x : y)

int main( int argc, char *argv[] )
{
    int SIZE, i, j, row, numworkers, numsent, sender;
    double dotp, exec_time = 0;
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
    MPI_Comm_spawn( "worker", MPI_ARGV_NULL, numworkers, MPI_INFO_NULL, 0, MPI_COMM_SELF, &workercomm, MPI_ERRCODES_IGNORE );

    /* initialize a and b */
    for (i = 0; i < SIZE; i++ )
        for ( j = 0; j < SIZE; j++ )
            a[i][j] = ( double ) 1;

    for (i = 0; i < SIZE; i++ )
        b[i] = ( double ) 2;

    /* send SIZE to each worker */
    MPI_Bcast( &SIZE, 1, MPI_INT, MPI_ROOT, workercomm );

    /* send b to each worker */
    MPI_Bcast( b, SIZE, MPI_DOUBLE, MPI_ROOT, workercomm );

    /* send a's row to each process */
    for ( i = 1; i <= MIN( numworkers, SIZE ); i++ ) {
        MPI_Send( a[i-1], SIZE, MPI_DOUBLE, (i-1), i, workercomm );
        numsent++;
    }

    /* receive dot products back from workers */
    for ( i = 1; i <= SIZE; i++ ) {
        MPI_Recv( &dotp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, workercomm, &status );
        sender = status.MPI_SOURCE;
        row = status.MPI_TAG;
        c[row] = dotp;
        /* send another row back to this worker if there is one */
        if ( numsent < SIZE ) {
            MPI_Send( a[numsent], SIZE, MPI_DOUBLE, sender, numsent + 1, workercomm );
            numsent++;
        } else { /* no more work */
            MPI_Send( MPI_BOTTOM, 0, MPI_DOUBLE, sender, 0, workercomm );
        }
    }

    // /* print matrix, vector & result */
    // for (i = 0; i < SIZE; i++) {
    //     for (j = 0; j < SIZE; j++) {
    //         printf("%f ", a[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // for (i = 0; i < SIZE; i++)
    //     printf("%f\n", b[i]);
    // printf("\n");
    // for (i = 0; i < SIZE; i++)
    //     printf("%f\n", c[i]);

    MPI_Finalize();
    exec_time += MPI_Wtime();

    printf("%d\t%f\n", numworkers, exec_time);
    return 0;
}
