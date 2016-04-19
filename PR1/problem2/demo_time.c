#include <stdio.h>
#include <mpi.h>
#include <math.h>

#define TRUE 1
#define FALSE 0

int main(int argc, char *argv[]) {
    int i, j, rank, np, sq, left, right, up, down;
    MPI_Comm vu;
    int dim[2],period[2],reorder;
    int coord[2],id;
    double exec_time = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (np < 10) {
        if (rank == 0) printf("np must be > 9!\n");
        MPI_Finalize();
        return 0;
    }

    if(rank==0) {
        exec_time -= MPI_Wtime();
    }

    sq = ceil(sqrt(np));
    dim[1]= np / sq;
    dim[0]= np / dim[1];
    period[0]=TRUE; period[1]=FALSE;
    reorder=TRUE;

    // create cartesian topology
    MPI_Cart_create(MPI_COMM_WORLD,2,dim,period,reorder,&vu);
    // if(rank==0) {
    //     printf("%d x %d\n", dim[0], dim[1]);
    //     printf("MPI_Cart_rank:\n");
    //     for (i=0;i<dim[1];i++) {
    //         for (j=0;j<dim[0];j++) {
    //             coord[1]=i;
    //             coord[0]=j;
    //             // get processor rank/id by coordinates
    //             MPI_Cart_rank(vu,coord,&id);
    //             printf("(%d,%d)->%d\t", coord[0], coord[1], id);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    MPI_Barrier(MPI_COMM_WORLD);

    // if(rank==5){
    //     // get coordinates of a process by rank
    //     MPI_Cart_coords(vu,rank,2,coord);
    //     printf("P:%d coordinates (MPI_Cart_coords) are (%d,%d)\n",rank,coord[0],coord[1]);
    // }
    // if(rank==9){
    //     // get left and right neighbors' ranks
    //     MPI_Cart_shift(vu,0,1,&left,&right);
    //     // get up and down neighbors' ranks
    //     MPI_Cart_shift(vu,1,1,&up,&down);
    //     printf("P:%d neighbors (MPI_Cart_shift) are r: %d d:%d 1:%d u:%d\n",rank,right,down,left,up);
    // }
    MPI_Finalize();
    if(rank==0) {
        exec_time += MPI_Wtime();
        printf("%d\t%f\n", np, exec_time);
    }
    return 0;
}
