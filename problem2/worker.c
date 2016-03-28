#include <mpi.h>
#include <stdio.h>

int main( int argc, char *argv[] )
{
    int numprocs, myrank;
    int i, row, SIZE;
    double dotp;
    MPI_Status status;
    MPI_Comm parentcomm;

    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &numprocs );

    MPI_Comm_get_parent( &parentcomm );
    MPI_Comm_rank( parentcomm, &myrank );
    // printf("worker %d is up!\n", myrank);

    MPI_Bcast( &SIZE, 1, MPI_INT, 0, parentcomm );
    // printf("worker %d get SIZE: %d\n", myrank, SIZE);

    double b[SIZE], c[SIZE];
    MPI_Bcast( b, SIZE, MPI_DOUBLE, 0, parentcomm );

    if ( myrank < SIZE ) {
        MPI_Recv( c, SIZE, MPI_DOUBLE, 0, MPI_ANY_TAG, parentcomm, &status );
        while ( status.MPI_TAG > 0 ) {
            row = status.MPI_TAG - 1;
            dotp = 0.0;
            for ( i = 0; i < SIZE; i++ )
                dotp += c[i] * b[i];
            MPI_Send( &dotp, 1, MPI_DOUBLE, 0, row, parentcomm );
            MPI_Recv( c, SIZE, MPI_DOUBLE, 0, MPI_ANY_TAG, parentcomm, &status );
	    // printf("p %d receive MPI_TAG %d\n", myrank, status.MPI_TAG);
        }
    }

    MPI_Comm_free( &parentcomm );
    MPI_Finalize( );

    // printf("worker %d out\n", myrank);
    return 0;
}
