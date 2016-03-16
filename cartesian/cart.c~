#define NO_OF_PROCESSES 4
#define MESH_SIZE 2
#define NDIMENSIONS 2 

typedef struct {
   int      Size;     /* The number of processors. (Size = q_proc*q_proc)
  */
   int      p_proc;        /* The number of processors in a row (column).
*/
   int      Row;      /* The mesh row this processor occupies.        */
   int      Col;      /* The mesh column this processor occupies.     */
   int      MyRank;     /* This processors unique identifier.           */
   MPI_Comm Comm;     /* Communicator for all processors in the mesh. */
   MPI_Comm Row_comm; /* All processors in this processors row   .    */
   MPI_Comm Col_comm; /* All processors in this processors column.    */
} MESH_INFO_TYPE;
