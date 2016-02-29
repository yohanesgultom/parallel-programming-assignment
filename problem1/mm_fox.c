/* fox.c -- uses Fox's algorithm to multiply two square matrices
 *
 * Yohanes: modified to read matrix size (n) from cli argument
 * http://web.mst.edu/~ercal/387/MPI/ppmpi_c/chap07/fox.c
 *
 * Input:
 *     n: global order of matrices
 *     A,B: the factor matrices
 * Output:
 *     C: the product matrix
 *
 * Notes:
 *     1.  Assumes the number of processes is a perfect square
 *     2.  The array member of the matrices is statically allocated
 *     3.  Assumes the global order of the matrices is evenly
 *         divisible by sqrt(p).
 *
 * See Chap 7, pp. 113 & ff and pp. 125 & ff in PPMPI
 */

 /*
 * Comments and changes made by students of MIPT
 * Ablyazov N.A.	ablyazov@mail.ru
 * Makarov O.S.		gsvoleg@mail.ru
 * 		2003
 */
#include <stdio.h>
#include "mpi.h"
#include <math.h>

#include <stdlib.h>
#include <time.h>


typedef struct {
    int       p;         /* Total number of processes    */
    MPI_Comm  comm;      /* Communicator for entire grid */
    MPI_Comm  row_comm;  /* Communicator for my row      */
    MPI_Comm  col_comm;  /* Communicator for my col      */
    int       q;         /* Order of grid                */
    int       my_row;    /* My row number                */
    int       my_col;    /* My column number             */
    int       my_rank;   /* My rank in the grid comm     */
} GRID_INFO_T;


#define MAX 65536
#define ConstVal 3 //Value witch filled the matrix
#define MaxInt 2147483648

typedef struct {
    int     n_bar;
#define Order(A) ((A)->n_bar)
    float  entries[MAX];
#define Entry(A,i,j) (*(((A)->entries) + ((A)->n_bar)*(i) + (j))) //Adress arithmetics
} LOCAL_MATRIX_T;

/* Function Declarations */
LOCAL_MATRIX_T*  Local_matrix_allocate(int n_bar);	//Memory for matrix
void Free_local_matrix(LOCAL_MATRIX_T** local_A);	//free memory
void Read_matrix(char* prompt, LOCAL_MATRIX_T* local_A,	GRID_INFO_T* grid, int n);
void Print_matrix(char* title, LOCAL_MATRIX_T* local_A, GRID_INFO_T* grid, int n);
void Set_to_zero(LOCAL_MATRIX_T* local_A); //Fills the matrix with zeroes
void Local_matrix_multiply(LOCAL_MATRIX_T* local_A,	LOCAL_MATRIX_T* local_B, LOCAL_MATRIX_T* local_C);
void Build_matrix_type(LOCAL_MATRIX_T* local_A);	//creation of new MPI type for matrix
MPI_Datatype local_matrix_mpi_t;	//New MPI type for matrix

LOCAL_MATRIX_T*  temp_mat;
/*void             Print_local_matrices(char* title, LOCAL_MATRIX_T* local_A,
                     GRID_INFO_T* grid);
*/
/*********************************************************/
int main(int argc, char* argv[]) {
    int              p;			//number of processes
    int              q;			//sqrt(p)
    int              my_rank;		//rank of process
    GRID_INFO_T      grid;		//Info of process
    LOCAL_MATRIX_T*  local_A;
    LOCAL_MATRIX_T*  local_B;
    LOCAL_MATRIX_T*  local_C;
    int              n;			//order of matrix
    int              n_bar;		//order of submatrix
    double	     begin_time, end_time, interval, comm_time;

    void Setup_grid(GRID_INFO_T*  grid);				//setup of Info of process
    void Fox(int n, GRID_INFO_T* grid, LOCAL_MATRIX_T* local_A,	LOCAL_MATRIX_T* local_B, LOCAL_MATRIX_T* local_C, double* comm_time);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);		// Ehh

    q=(int) sqrt((double) p);

    if ( (q * q) == p) {  	//p==q*q else exit!! (Perfect square)

      Setup_grid(&grid);		//setup of Info of process

      n = atoi(argv[1]);
      if (my_rank == 0) {
         // printf("%d\n",my_rank);
          // printf("What's the order of the matrices?\n");
          // fflush(stdout);
          // scanf("%d", &n);
          //beginning time
          begin_time = MPI_Wtime();
      }

      //Sending size to all processes
      if (my_rank == 0) comm_time -= MPI_Wtime();
      MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (my_rank == 0) comm_time += MPI_Wtime();

      n_bar = n/grid.q; //size of submatrix

      local_A = Local_matrix_allocate(n_bar);             //memory for submatrix in each process
      Order(local_A) = n_bar;
      Read_matrix("Enter A", local_A, &grid, n);          //Only 0-th rank do this
      // Print_matrix("We read A =", local_A, &grid, n);     //Only 0-th rank do this

      local_B = Local_matrix_allocate(n_bar);             //memory for submatrix in each process
      Order(local_B) = n_bar;
      Read_matrix("Enter B", local_B, &grid, n);          //Only 0-th rank do this
      // Print_matrix("We read B =", local_B, &grid, n);     //Only 0-th rank do this

      Build_matrix_type(local_A);                         //struct type for submatrix (MPI)
      temp_mat = Local_matrix_allocate(n_bar);            //memory for Temp matrix

      local_C = Local_matrix_allocate(n_bar);             //memory for result matrix
      Order(local_C) = n_bar;                             //size of C
      Fox(n, &grid, local_A, local_B, local_C, &comm_time);           //Block-algoritm of FOX

      // Print_matrix("The product is", local_C, &grid, n);

      //ending time
      end_time = MPI_Wtime();
      if (begin_time < end_time){
    	   interval = end_time - begin_time;
    	} else{
    	   interval = MaxInt - begin_time + end_time;
    	}
      if (my_rank == 0) {
         printf("%d\t%d\t%f\t%f\n", p, n, comm_time, interval);
      }

      Free_local_matrix(&local_A);
      Free_local_matrix(&local_B);
      Free_local_matrix(&local_C);
    } else{	//p!=q*q else exit!! (Perfect square)
      if (my_rank == 0) {
        printf("Error: number of processes must be perfect square\n");
        printf("Now exit\n");
        fflush(stdout);
      }
    }
    MPI_Finalize();
    return 0;
}  /* main */


/**
 * out: GRID_INFO_T*  grid
 */
void Setup_grid(GRID_INFO_T*  grid) {
    int old_rank;
    int dimensions[2];
    int wrap_around[2];
    int coordinates[2];
    int free_coords[2];

    /* Set up Global Grid Information */
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->p));
    MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);

    /* We assume p is a perfect square */
    grid->q = (int) sqrt((double) grid->p);
    dimensions[0] = dimensions[1] = grid->q;

    /* We want a circular shift in second dimension. */
    /* Don't care about first                        */
    //For Fox`s algorithm

    wrap_around[0] = wrap_around[1] = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, wrap_around, 1, &(grid->comm));
    MPI_Comm_rank(grid->comm, &(grid->my_rank));
    MPI_Cart_coords(grid->comm, grid->my_rank, 2, coordinates);
    grid->my_row = coordinates[0];
    grid->my_col = coordinates[1];

    /* Set up row communicators */
    free_coords[0] = 0;
    free_coords[1] = 1;
    MPI_Cart_sub(grid->comm, free_coords, &(grid->row_comm));

    /* Set up column communicators */
    free_coords[0] = 1;
    free_coords[1] = 0;
    MPI_Cart_sub(grid->comm, free_coords, &(grid->col_comm));
} /* Setup_grid */


/**
 * input: int n, GRID_INFO_T* grid, LOCAL_MATRIX_T*  local_A, LOCAL_MATRIX_T*  local_B
 * out: LOCAL_MATRIX_T* local_C
 */
void Fox(int n, GRID_INFO_T* grid, LOCAL_MATRIX_T* local_A, LOCAL_MATRIX_T* local_B, LOCAL_MATRIX_T* local_C, double* comm_time) {

    LOCAL_MATRIX_T*  temp_A; /* Storage for the sub-    */
                             /* matrix of A used during */
                             /* the current stage       */
    int              stage;
    int              bcast_root;
    int              n_bar;  /* n/sqrt(p)               */
    int              source;
    int              dest;
    MPI_Status       status;

    n_bar = n/grid->q;
    Set_to_zero(local_C);

    /* Calculate addresses for circular shift of B */
    source = (grid->my_row + 1) % grid->q;              // (i+1) mod q
    dest = (grid->my_row + grid->q - 1) % grid->q;      // (i+q-1) mod q
    /*
    //debug
    printf("my_row = %d\n",grid->my_row);
    printf("my_col = %d\n",grid->my_col);
    printf("source = %d\n",source);
    printf("dest = %d\n",dest);
    printf("my_rank = %d\n",grid->my_rank);
    fflush(stdout);//debug!!*/

    /* Set aside storage for the broadcast block of A */
    temp_A = Local_matrix_allocate(n_bar);

    for (stage = 0; stage < grid->q; stage++) {
        bcast_root = (grid->my_row + stage) % grid->q;  //(i+k) mod q (k~)
        if (bcast_root == grid->my_col) {               //k~ == j
            if (grid->my_rank == 0) *comm_time -= MPI_Wtime();
            // BroadCasting of Aij (submatrix)
            MPI_Bcast(local_A, 1, local_matrix_mpi_t, bcast_root, grid->row_comm);
            if (grid->my_rank == 0) *comm_time += MPI_Wtime();
             //C[i,j]=C[i,j] + (A[i,k~] * B[k~,j])
            Local_matrix_multiply(local_A, local_B, local_C);
        } else {
            //recieving to temp_A?
            if (grid->my_rank == 0) *comm_time -= MPI_Wtime();
            MPI_Bcast(temp_A, 1, local_matrix_mpi_t,  bcast_root, grid->row_comm);
            if (grid->my_rank == 0) *comm_time += MPI_Wtime();
            //Why do they multiplying?
            Local_matrix_multiply(temp_A, local_B, local_C);
        }
        //Sends and receives using a single buffer
        if (grid->my_rank == 0) *comm_time -= MPI_Wtime();
        MPI_Sendrecv_replace(local_B, 1, local_matrix_mpi_t, dest, 0/* sendtag*/, source, 0/*recvtag*/, grid->col_comm, &status);
        if (grid->my_rank == 0) *comm_time += MPI_Wtime();
    } /* for */

} /* Fox */


/*********************************************************/
LOCAL_MATRIX_T* Local_matrix_allocate(int local_order) {
    LOCAL_MATRIX_T* temp;

    temp = (LOCAL_MATRIX_T*) malloc(sizeof(LOCAL_MATRIX_T));
    return temp;
}  /* Local_matrix_allocate */


/*********************************************************/
void Free_local_matrix(LOCAL_MATRIX_T** local_A_ptr  /* in/out */) {
    free(*local_A_ptr);
}  /* Free_local_matrix */


/*********************************************************/
/* Read and distribute matrix:
 *     foreach global row of the matrix,
 *         foreach grid column
 *             read a block of n_bar floats on process 0
 *             and send them to the appropriate process.
 */
void Read_matrix(char* prompt   /* in  */, LOCAL_MATRIX_T* local_A  /* out */, GRID_INFO_T* grid/* in  */, int n /* in  */) {
    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        dest;
    int        coords[2];
    float*     temp;
    MPI_Status status;

    if (grid->my_rank == 0) {
        //printf("%d\n", grid->my_rank);
        temp = (float*) malloc(Order(local_A)*sizeof(float));   //vector-stroka?
        // printf("%s\n", prompt);
        // fflush(stdout);
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/Order(local_A);
            coords[0] = grid_row;
            //debug
            // printf("grid_row=%d\n",grid_row);

            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &dest);
                //debug
                /*printf("coords[0]=%d\n",coords[0]);
                printf("coords[1]=%d\n",coords[1]);
                printf("dest=%d\n",dest);
                fflush(stdout);//debug!!*/
                if (dest == 0) {
                    for (mat_col = 0; mat_col < Order(local_A); mat_col++)
                        //scanf("%f", (local_A->entries)+mat_row*Order(local_A)+mat_col);
                        *((local_A->entries)+mat_row*Order(local_A)+mat_col) = ConstVal;
                } else {
                    for(mat_col = 0; mat_col < Order(local_A); mat_col++)
                        //scanf("%f", temp + mat_col);
                        *(temp + mat_col) = ConstVal;
                    MPI_Send(temp, Order(local_A), MPI_FLOAT, dest, 0, grid->comm);
                }
            }
        }
        free(temp);
    } else {
        for (mat_row = 0; mat_row < Order(local_A); mat_row++) {
          MPI_Recv(&Entry(local_A, mat_row, 0), Order(local_A), MPI_FLOAT, 0, 0, grid->comm, &status);
        }
    }

}  /* Read_matrix */


/*********************************************************/
void Print_matrix(char* title/* in  */, LOCAL_MATRIX_T*  local_A  /* out */, GRID_INFO_T* grid/* in  */, int n/* in  */) {
    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        source;
    int        coords[2];
    float*     temp;
    MPI_Status status;

    if (grid->my_rank == 0) {
        temp = (float*) malloc(Order(local_A)*sizeof(float));
        //printf("%s\n", title);
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/Order(local_A);
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &source);
                if (source == 0) {
                    for(mat_col = 0; mat_col < Order(local_A); mat_col++)
                       ;// printf("%4.1f ", Entry(local_A, mat_row, mat_col));
                } else {
                    MPI_Recv(temp, Order(local_A), MPI_FLOAT, source, 0, grid->comm, &status);
                    //for(mat_col = 0; mat_col < Order(local_A); mat_col++);//printf("%4.1f ", temp[mat_col]);
                }
            }
            //;printf("\n");
        }
        free(temp);
    } else {
        for (mat_row = 0; mat_row < Order(local_A); mat_row++)
            MPI_Send(&Entry(local_A, mat_row, 0), Order(local_A), MPI_FLOAT, 0, 0, grid->comm);
    }

}  /* Print_matrix */


/*********************************************************/
void Set_to_zero(LOCAL_MATRIX_T*  local_A  /* out */) {
    int i, j;
    for (i = 0; i < Order(local_A); i++)
        for (j = 0; j < Order(local_A); j++)
            Entry(local_A,i,j) = 0.0;
}  /* Set_to_zero */


/*********************************************************/
void Build_matrix_type(LOCAL_MATRIX_T*  local_A  /* in */) {
    MPI_Datatype  temp_mpi_t;
    int           block_lengths[2];
    MPI_Aint      displacements[2];
    MPI_Datatype  typelist[2];
    MPI_Aint      start_address;
    MPI_Aint      address;

    MPI_Type_contiguous(Order(local_A)*Order(local_A), MPI_FLOAT, &temp_mpi_t); //array[0..n-1,0..n-1] of float

    block_lengths[0] = block_lengths[1] = 1;

    typelist[0] = MPI_INT; //2 elements: 1) int, 2) array[...] of float
    typelist[1] = temp_mpi_t;

    MPI_Get_address(local_A, &start_address); //adress of submatrix - start
    MPI_Get_address(&(local_A->n_bar), &address); //adress of Size of submatrix
    displacements[0] = address - start_address; //between n_bar and beginning (sometimes <>0 )

    MPI_Get_address(local_A->entries, &address); //adress of entries
    displacements[1] = address - start_address; //between entries and beginning

    //new struct type
    MPI_Type_create_struct(2, block_lengths, displacements, typelist, &local_matrix_mpi_t);
    MPI_Type_commit(&local_matrix_mpi_t); //return type?
}  /* Build_matrix_type */

/*********************************************************/
void Local_matrix_multiply(LOCAL_MATRIX_T*  local_A  /* in  */, LOCAL_MATRIX_T*  local_B  /* in  */, LOCAL_MATRIX_T*  local_C  /* out */) {
    int i, j, k;
    for (i = 0; i < Order(local_A); i++)
        for (j = 0; j < Order(local_A); j++)
            for (k = 0; k < Order(local_B); k++)
                Entry(local_C,i,j) = Entry(local_C,i,j)
                    + Entry(local_A,i,k)*Entry(local_B,k,j);

}  /* Local_matrix_multiply */

/*********************************************************/
