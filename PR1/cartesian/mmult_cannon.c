/*
******************************************************************** 

                Example 22 (mm_mult_cannon.c)

  Objective           : Matrix Matrix multiplication
                        (Using Cartesian Topology, CANNON Algorithm)

  Input               : Read files (mdata1.inp) for first input matrix 
                        and (mdata2.inp) for second input matrix 

  Output              : Result of matrix matrix multiplication on Processor 0.

  Necessary Condition : Number of Processes should be less than
                        or equal to 8. Matrices A and B should be
                        equally striped. that is Row size and
                        Column size should be properly divisible 
                        by Number of processes used.

********************************************************************
*/



#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "cart.h

/* Communication block set up for mesh toplogy */
void SetUp_Mesh(MESH_INFO_TYPE *);

main (int argc, char *argv[])
{

 int istage,irow,icol,jrow,jcol,iproc,jproc,index,Proc_Id,Root =0;
 int A_Bloc_MatrixSize, B_Bloc_MatrixSize;
 int NoofRows_A, NoofCols_A, NoofRows_B, NoofCols_B;
 int NoofRows_BlocA, NoofCols_BlocA, NoofRows_BlocB, NoofCols_BlocB;
 int Local_Index, Global_Row_Index, Global_Col_Index;
 int Matrix_Size[4];
 int source, destination, send_tag, recv_tag, Bcast_root;

 float **Matrix_A, **Matrix_B, **Matrix_C;
 float *A_Bloc_Matrix, *B_Bloc_Matrix, *C_Bloc_Matrix, *Temp_BufferA;

 float *MatA_array, *MatB_array, *MatC_array;

 FILE *fp;
 int	 MatrixA_FileStatus = 1, MatrixB_FileStatus = 1;

 MESH_INFO_TYPE grid;
 MPI_Status status; 

 /* Initialising */
 MPI_Init (&argc, &argv);

 /* Set up the MPI_COMM_WORLD and CARTESIAN TOPOLOGY */
  SetUp_Mesh(&grid);
  

 /* Reading Input */
  if (grid.MyRank == Root){
     if ((fp = fopen ("mdata1.inp", "r")) == NULL){
			MatrixA_FileStatus = 0;
     } else {
	printf("mdata1\n");
}

  if(MatrixA_FileStatus != 0) {
 	fscanf(fp, "%d %d\n", &NoofRows_A, &NoofCols_A);
 	Matrix_Size[0] = NoofRows_A;
     	Matrix_Size[1] = NoofCols_A;

     	Matrix_A = (float **) malloc (NoofRows_A * sizeof(float *));
     for (irow = 0; irow < NoofRows_A; irow++){
	Matrix_A[irow] = (float *) malloc(NoofCols_A * sizeof(float));
     	for (icol = 0; icol < NoofCols_A; icol++)
	  fscanf(fp, "%f", &Matrix_A[irow][icol]);
     }
  fclose(fp);
 }

     if((fp = fopen ("mdata2.inp", "r")) == NULL){
	MatrixB_FileStatus = 0;
     } else {
	printf("mdata2\n");
}

  if(MatrixB_FileStatus != 0) {
     	fscanf(fp, "%d %d\n", &NoofRows_B, &NoofCols_B);
   	Matrix_Size[2] = NoofRows_B;
    	Matrix_Size[3] = NoofCols_B;

     	Matrix_B = (float **) malloc (NoofRows_B * sizeof(float *));
    for(irow = 0; irow < NoofRows_B; irow++){
    Matrix_B[irow] = (float *) malloc(NoofCols_B * sizeof(float *));
      for(icol = 0; icol < NoofCols_B; icol++)
       fscanf(fp, "%f", &Matrix_B[irow][icol]);
    }
    fclose(fp);
  }

 } /* MyRank == Root */

 /*  Send Matrix Size to all processors  */
 MPI_Barrier(grid.Comm);

 MPI_Bcast (&MatrixA_FileStatus, 1, MPI_INT, 0, grid.Comm);
 if(MatrixA_FileStatus == 0) {
  if(grid.MyRank == 0) printf("Can't open input file for Matrix A ..");
    MPI_Finalize();
    exit(-1);
 }
 MPI_Bcast (&MatrixB_FileStatus, 1, MPI_INT, 0, grid.Comm);
 if(MatrixB_FileStatus == 0) {
 if(grid.MyRank == 0) printf("Can't open input file for Matrix B ..");
    MPI_Finalize();
    exit(-1);
}

 MPI_Bcast (Matrix_Size, 4, MPI_INT, 0, grid.Comm);

 NoofRows_A = Matrix_Size[0];
 NoofCols_A = Matrix_Size[1];
 NoofRows_B = Matrix_Size[2];
 NoofCols_B = Matrix_Size[3];

 if(NoofCols_A != NoofRows_B){
  MPI_Finalize();
  if(grid.MyRank == Root){
    printf("Matrices Dimensions incompatible for Multiplication");
  }
  exit(-1);
 }

 if( NoofRows_A % grid.p_proc != 0 || NoofCols_A % grid.p_proc != 0 ||
   NoofRows_B % grid.p_proc != 0 || NoofCols_B % grid.p_proc != 0){
     
  MPI_Finalize();
  if(grid.MyRank == Root){
     printf("Matrices can't be divided among processors equally");
  }
  exit(-1);
 }

  NoofRows_BlocA = NoofRows_A / grid.p_proc;
  NoofCols_BlocA = NoofCols_A / grid.p_proc;

  NoofRows_BlocB = NoofRows_B / grid.p_proc;
  NoofCols_BlocB = NoofCols_B / grid.p_proc;

  A_Bloc_MatrixSize = NoofRows_BlocA * NoofCols_BlocA;
  B_Bloc_MatrixSize = NoofRows_BlocB * NoofCols_BlocB;

  /* Memory allocating for Bloc Matrices */
 A_Bloc_Matrix = (float *) malloc (A_Bloc_MatrixSize * sizeof(float));
 B_Bloc_Matrix = (float *) malloc (B_Bloc_MatrixSize * sizeof(float));

 /* memory for arrangmeent of the data in one dim. arrays before MPI_SCATTER */
 MatA_array =(float *)malloc(sizeof(float) * NoofRows_A * NoofCols_A);
 MatB_array =(float *)malloc(sizeof(float) * NoofRows_B * NoofCols_B);

 /*Rearrange the input matrices in one dim arrays by approriate order*/
 if (grid.MyRank == Root) {

 /* Rearranging Matrix A*/
     for (iproc = 0; iproc < grid.p_proc; iproc++){
        for (jproc = 0; jproc < grid.p_proc; jproc++){
	  Proc_Id = iproc * grid.p_proc + jproc;
          for (irow = 0; irow < NoofRows_BlocA; irow++){
	     Global_Row_Index = iproc * NoofRows_BlocA + irow;
	     for (icol = 0; icol < NoofCols_BlocA; icol++){
	          Local_Index  = (Proc_Id * A_Bloc_MatrixSize) + 
			      (irow * NoofCols_BlocA) + icol;
		  Global_Col_Index = jproc * NoofCols_BlocA + icol;
	          MatA_array[Local_Index] = Matrix_A[Global_Row_Index][Global_Col_Index];
	      }
	   }
         }
       }
    
 /* Rearranging Matrix B*/
     for (iproc = 0; iproc < grid.p_proc; iproc++){
        for (jproc = 0; jproc < grid.p_proc; jproc++){
	  Proc_Id = iproc * grid.p_proc + jproc;
          for (irow = 0; irow < NoofRows_BlocB; irow++){
       	    Global_Row_Index = iproc * NoofRows_BlocB + irow;
	    for (icol = 0; icol < NoofCols_BlocB; icol++){
	      Local_Index = (Proc_Id * B_Bloc_MatrixSize) + 
			      (irow * NoofCols_BlocB) + icol;
	      Global_Col_Index = jproc * NoofCols_BlocB + icol;
	      MatB_array[Local_Index] = Matrix_B[Global_Row_Index][Global_Col_Index];
	     }
	   }
         }
       }

  } /* if loop ends here */


  MPI_Barrier(grid.Comm);

  /* Scatter the Data  to all processes by MPI_SCATTER */
 MPI_Scatter (MatA_array, A_Bloc_MatrixSize, MPI_FLOAT, A_Bloc_Matrix ,
            A_Bloc_MatrixSize , MPI_FLOAT, 0, grid.Comm);

 MPI_Scatter (MatB_array, B_Bloc_MatrixSize, MPI_FLOAT, B_Bloc_Matrix,
            B_Bloc_MatrixSize, MPI_FLOAT, 0, grid.Comm);


 /* Do initial arrangement of Matrices */

 if(grid.Row !=0){
    source   = (grid.Col + grid.Row) % grid.p_proc;
    destination = (grid.Col + grid.p_proc - grid.Row) % grid.p_proc; 
    recv_tag =0;
    send_tag = 0;
    MPI_Sendrecv_replace(A_Bloc_Matrix, A_Bloc_MatrixSize, MPI_FLOAT,
    destination, send_tag, source, recv_tag, grid.Row_comm, &status);
  }
  if(grid.Col !=0){
    source   = (grid.Row + grid.Col) % grid.p_proc;
    destination = (grid.Row + grid.p_proc - grid.Col) % grid.p_proc; 
    recv_tag =0;
    send_tag = 0;
    MPI_Sendrecv_replace(B_Bloc_Matrix, B_Bloc_MatrixSize, MPI_FLOAT,
    destination,send_tag, source, recv_tag, grid.Col_comm, &status);
  }

  /* Allocate Memory for Bloc C Array */
 C_Bloc_Matrix = (float *) malloc (NoofRows_BlocA * NoofCols_BlocB * sizeof(float));
  for(index=0; index<NoofRows_BlocA*NoofCols_BlocB; index++)
     C_Bloc_Matrix[index] = 0;

 /* The main loop */

 send_tag = 0;
 recv_tag = 0;
 for(istage=0; istage<grid.p_proc; istage++){
   index=0;
   for(irow=0; irow<NoofRows_BlocA; irow++){
      for(icol=0; icol<NoofCols_BlocB; icol++){
	  for(jcol=0; jcol<NoofCols_BlocA; jcol++){
 C_Bloc_Matrix[index] += A_Bloc_Matrix[irow*NoofCols_BlocA + jcol] *
         B_Bloc_Matrix[jcol*NoofCols_BlocB + icol];
	  }
	  index++;
       }
      }
  /* Move Bloc of Matrix A by one position left with wraparound */
  source   = (grid.Col + 1) % grid.p_proc;
  destination = (grid.Col + grid.p_proc - 1) % grid.p_proc; 
  MPI_Sendrecv_replace(A_Bloc_Matrix, A_Bloc_MatrixSize, MPI_FLOAT,
  destination,send_tag, source, recv_tag, grid.Row_comm, &status);

 /* Move Bloc of Matrix B by one position upwards with wraparound */
  source   = (grid.Row + 1) % grid.p_proc;
  destination = (grid.Row + grid.p_proc - 1) % grid.p_proc; 
  MPI_Sendrecv_replace(B_Bloc_Matrix, B_Bloc_MatrixSize, MPI_FLOAT,
  destination, send_tag, source, recv_tag, grid.Col_comm, &status);
  }


 /* Memory for output global matrix in the form of array  */
 if(grid.MyRank == Root) 
 MatC_array = (float *) malloc (sizeof(float) * NoofRows_A * NoofCols_B);

 MPI_Barrier(grid.Comm);

 /* Gather output block matrices at processor 0 */
 MPI_Gather (C_Bloc_Matrix, NoofRows_BlocA * NoofCols_BlocB, MPI_FLOAT,
 MatC_array,NoofRows_BlocA*NoofCols_BlocB, MPI_FLOAT, Root, grid.Comm);

 /* Memory for output global array for OutputMatrix_C after rearrangement */
  if(grid.MyRank == Root){
  Matrix_C = (float **) malloc (NoofRows_A * sizeof(float *));
  for(irow=0; irow<NoofRows_A ;irow++)
     Matrix_C[irow] = (float *) malloc(NoofCols_B * sizeof(float));
  }

  /* Rearranging the output matrix in a array by approriate order  */
  if (grid.MyRank == Root) {
     for (iproc = 0; iproc < grid.p_proc; iproc++){
        for (jproc = 0; jproc < grid.p_proc; jproc++){
	  Proc_Id = iproc * grid.p_proc + jproc;
	        for (irow = 0; irow < NoofRows_BlocA; irow++){
		  Global_Row_Index = iproc * NoofRows_BlocA + irow;
	           for (icol = 0; icol < NoofCols_BlocB; icol++){
      Local_Index = (Proc_Id * NoofRows_BlocA * NoofCols_BlocB) + 
		     (irow * NoofCols_BlocB) + icol;
		     Global_Col_Index = jproc * NoofCols_BlocB + icol;
		       Matrix_C[Global_Row_Index][Global_Col_Index] = MatC_array[Local_Index];
	           }
	        }
		  }
	  }
 printf ("-----------MATRIX MULTIPLICATION RESULTS --------------\n");
    printf(" Processor %d, Matrix A : Dimension %d * %d : \n",
    grid.MyRank, NoofRows_A, NoofCols_A);
    for(irow = 0; irow < NoofRows_A; irow++) {
       for(icol = 0; icol < NoofCols_A; icol++)
        printf ("%7.3f ", Matrix_A[irow][icol]);
        printf ("\n");
    }
    printf("\n");

    printf("Processor %d, Matrix B : Dimension %d * %d : \n",
				grid.MyRank, NoofRows_B, NoofCols_B);
    for(irow = 0; irow < NoofRows_B; irow++){
       for(icol = 0; icol < NoofCols_B; icol++)
	       printf("%7.3f ", Matrix_B[irow][icol]);
       printf("\n");
    }
    printf("\n");
    
    printf("Processor %d, Matrix C : Dimension %d * %d : \n",
				grid.MyRank, NoofRows_A, NoofCols_B);
    for(irow = 0; irow < NoofRows_A; irow++){
       for(icol = 0; icol < NoofCols_B; icol++)
		    printf("%7.3f ",Matrix_C[irow][icol]);
       printf("\n");
    }

    for(irow=0; irow<NoofRows_A; irow++)
		 for(icol=0; icol<NoofCols_B; icol++){
			 Matrix_C[irow][icol] = 0;
			 for(jrow=0; jrow<NoofRows_B; jrow++)
			 Matrix_C[irow][icol] += Matrix_A[irow][jrow] * Matrix_B[jrow][icol];
		 }

    printf("Serial results\n");
    for(irow = 0; irow < NoofRows_A; irow++){
       for(icol = 0; icol < NoofCols_B; icol++)
		    printf("%7.3f ",Matrix_C[irow][icol]);
       printf("\n");
    }

  }
  MPI_Finalize();
 }

 /* Function : Finds communication information suitable to mesh topology  */
 /*            Create Cartesian topology in two dimnesions                */

 void SetUp_Mesh(MESH_INFO_TYPE *grid) {

   int Periods[2];      /* For Wraparound in each dimension.*/
   int Dimensions[2];   /* Number of processors in each dimension.*/
   int Coordinates[2];  /* processor Row and Column identification */
   int Remain_dims[2];      /* For row and column communicators */


 /* MPI rank and MPI size */
   MPI_Comm_size(MPI_COMM_WORLD, &(grid->Size));
   MPI_Comm_rank(MPI_COMM_WORLD, &(grid->MyRank));

 /* For square mesh */
   grid->p_proc = (int)sqrt((double) grid->Size);             
	if(grid->p_proc * grid->p_proc != grid->Size){
	 MPI_Finalize();
	 if(grid->MyRank == 0){
	 printf("Number of Processors should be perfect square\n");
	 }
		 exit(-1);
	}

   Dimensions[0] = Dimensions[1] = grid->p_proc;

   /* Wraparound mesh in both dimensions. */
   Periods[0] = Periods[1] = 1;    

   /*  Create Cartesian topology  in two dimnesions and  Cartesian 
       decomposition of the processes   */
   MPI_Cart_create(MPI_COMM_WORLD, NDIMENSIONS, Dimensions, Periods, 1, &(grid->Comm));
   MPI_Cart_coords(grid->Comm, grid->MyRank, NDIMENSIONS, Coordinates);

   grid->Row = Coordinates[0];
   grid->Col = Coordinates[1];

 /*Construction of row communicator and column communicators 
 (use cartesian row and columne machanism to get Row/Col Communicators)  */

   Remain_dims[0] = 0;            
   Remain_dims[1] = 1; 

 /*The output communicator represents the column containing the process */
   MPI_Cart_sub(grid->Comm, Remain_dims, &(grid->Row_comm));
   
   Remain_dims[0] = 1;
   Remain_dims[1] = 0;

 /*The output communicator represents the row containing the process */
   MPI_Cart_sub(grid->Comm, Remain_dims, &(grid->Col_comm));
}




