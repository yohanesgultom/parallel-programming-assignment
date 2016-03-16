#include "mpi.h"
#include <stdio.h>

#define SIZE 16
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

int main(argc,argv)
int argc;
char *argv[];  {

int numtasks, rank, source, dest, outbuf, i, tag=1, 
   inbuf[4]={MPI_PROC_NULL,MPI_PROC_NULL,MPI_PROC_NULL,MPI_PROC_NULL}, 
   nbrs[4], dims[2]={4,4}, 
   periods[2]={0,0}, reorder=0, coords[2];

  MPI_Request reqs[8];
  MPI_Status stats[8];
  MPI_Comm cartcomm;

   /*----------------*/
  /* Initialize MPI */
 /*----------------*/

  MPI_Init(&argc,&argv);

   /*-------------------------------------------------------*/                                                                     
  /* Get the size of the MPI_COMM_WORLD communicator group */
 /*-------------------------------------------------------*/

  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  if (numtasks == SIZE) {

   /*---------------------------------------------------------------------*/
  /* Make a new communicator to which 2-D Cartesian topology is attached */
 /*---------------------------------------------------------------------*/

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartcomm);

   /*------------------------------------------*/
  /* Get my rank in the cartcomm communicator */
 /*------------------------------------------*/

    MPI_Comm_rank(cartcomm, &rank);

   /*--------------------------------------------------------------------*/
  /* Determine process coords in cartesian topology given rank in group */
 /*--------------------------------------------------------------------*/

    MPI_Cart_coords(cartcomm, rank, 2, coords);

   /*--------------------------------------------------------------------*/
  /* Obtain the shifted source and destination ranks in both directions */ 
 /*--------------------------------------------------------------------*/
    MPI_Cart_shift(cartcomm, 0, 1, &nbrs[UP], &nbrs[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &nbrs[LEFT], &nbrs[RIGHT]);

    outbuf = rank;

    for (i=0; i<4; i++) {
      dest = nbrs[i];
      source = nbrs[i];

   /*----------------------------------------------*/
  /* send messages to the four adjacent processes */
 /*----------------------------------------------*/

      MPI_Isend(&outbuf, 1, MPI_INT, dest, tag, 
               MPI_COMM_WORLD, &reqs[i]);

   /*---------------------------------------------------*/
  /* receive messages from the four adjacent processes */
 /*---------------------------------------------------*/

      MPI_Irecv(&inbuf[i], 1, MPI_INT, source, tag, 
               MPI_COMM_WORLD, &reqs[i+4]);
    }

   /*------------------------------------------------*/
  /* Wait for all 8 communication tasks to complete */
 /*------------------------------------------------*/

    MPI_Waitall(8, reqs, stats);
   
    printf("rank = %2d coords = %2d%2d neighbors(u,d,l,r) = %2d %2d %2d %2d\n",
        rank,coords[0],coords[1],nbrs[UP],nbrs[DOWN],nbrs[LEFT],nbrs[RIGHT]);
    printf("rank = %2d                   inbuf(u,d,l,r) = %2d %2d %2d %2d\n",
        rank,inbuf[UP],inbuf[DOWN],inbuf[LEFT],inbuf[RIGHT]);
  }
  else
    printf("Must specify %d processors. Terminating.\n",SIZE);

   /*--------------*/
  /* Finalize MPI */
 /*--------------*/
   
  MPI_Finalize();

}
