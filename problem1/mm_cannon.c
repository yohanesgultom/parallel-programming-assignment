/**
Yohanes: modified to read matrix size (n) from cli argument & write to stdout instead of file
Original source: https://paralelos2008.googlecode.com/svn-history/r57/trunk/MPI-Paralelos/MPI-Paralelos/CANNON-MPI.c
**/

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
//#define n 800
#include <math.h>

int mi_id, rank_envio, n;
int coords[2], dims[2], periodos[2], coords_envio[2];

void llenarMatriz(float m[n][n])
{
  static float k=0;
  int i, j;
  for (i=0; i<n; i++)
  for (j=0; j<n; j++)
  m[i][j] = k++;
}

void imprimirMatriz(float m[n][n])
{
  int i, j = 0;
  for (i=0; i<n; i++) {
    printf("\n\t| ");
    for (j=0; j<n; j++)
    printf("%2f ", m[i][j]);
    printf("|");
  }
}
void imprimirSubMatriz(float m[3][3])
{
  int i, j = 0;
  for (i=0; i<3; i++) {
    printf("\n\t| ");
    for (j=0; j<3; j++)
    printf("%2f ", m[i][j]);
    printf("|");
  }
}
int main(int argc, char** argv) {
  /*DEFINICIONES DE DATOS E INICIALIZACIONES*/
  int mi_fila, mi_columna, fila_recepcion, col_recepcion;
  double timeIni, timeFin, timeComm;
  int rank_envio,size,destino,fuente, valor_matriz;
  int i,j,k,l, cont_fila, cont_columna, ciclos;
  FILE *fp;
  MPI_Status statusA;
  MPI_Status statusB;
  MPI_Status statusC;

  n = atoi(argv[1]); // read matrix size from cli argument
  float A[n][n], B[n][n], C[n][n];

  MPI_Comm comm2d;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&mi_id);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  double m = pow((double)size,(double) 1/2);
  if (n % (int)m !=0 )
  {
    printf("Por favor corra con una cantidad de procesos multiplo de %d.\n", n);fflush(stdout);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int tam_subM = n/m;
  //printf("EL VALOR M ES: %d",(int) m);
  //printf("EL VALOR TAMSUBM ES: %d",(int) tam_subM);
  dims[0]=dims[1]=(int) m;
  periodos[0]=periodos[1]=1;

  float subm_A[tam_subM][tam_subM];
  float subm_B[tam_subM][tam_subM];
  float subm_C[tam_subM][tam_subM];
  //float subm_C_aux[tam_subM][tam_subM];
  /*EN ESTE CASO SOLO ES DE DOS DIMENSIONES*/
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodos, 0, &comm2d);

  //printf(" cart create \n");
  /* Obtiene mi nuevo id en 2D */
  MPI_Comm_rank(comm2d, &mi_id);
  //printf(" comm rank \n");
  /* Obtiene mis coordenadas */
  MPI_Cart_coords(comm2d, mi_id, 2, coords);
  //printf(" CART COORDS\n");

  mi_fila = coords[0];
  mi_columna= coords[1];

  //    printf(" mi fila %d \n", mi_fila);
  //    printf(" mi columna %d \n", mi_columna);
  //    printf(" MI RANKING %d \n", mi_id);

  /*inicializamos submatriz C*/
  for (i=0; i<tam_subM; i++){
    for (j=0; j<tam_subM; j++){
      subm_C[i][j] =0;
    }
  }
  for (k=0; k<tam_subM; k++){
    for (l=0;l < tam_subM; l++){
      subm_A[k][l]=1;
      subm_B[k][l]=2;
    }
  }

  timeIni = MPI_Wtime();
  if(mi_id == 0)
  {

    /*Ahora basicamente lo que hacemos es enviar a cada proceso
    una parte de A y B que es la que les corresponde, enviamos a cada uno ya con la distribuci�n
    inicial para que puedan empezar multiplicando*/
    /*RESUELVO LA PARTE DE LA MATRIZ QUE SE ME QUEDO
    EN EL PROCESO 0*/

    for (ciclos = 0; ciclos < m; ciclos++) {
      for (i = 0; i < tam_subM; i++) {
        for (j = 0; j < tam_subM; j++) {
          for (k = 0;  k < tam_subM; k++) {
            subm_C[i][j] += subm_A[i][k] * subm_B[k][j];
          }
        }
      }
      timeComm -= MPI_Wtime();
      MPI_Cart_shift(comm2d,1,-1,&fuente,&destino);
      MPI_Sendrecv_replace(subm_A,(tam_subM*tam_subM),MPI_FLOAT,destino,1,fuente,1,comm2d,&statusA);

      MPI_Cart_shift(comm2d,0,-1,&fuente,&destino);
      MPI_Sendrecv_replace(subm_B,(tam_subM*tam_subM),MPI_FLOAT,destino,2,fuente,2,comm2d,&statusB);
      timeComm += MPI_Wtime();
    }
    //printf("PROCESO 0 MATRIZ C FINAL:\n");
    //imprimirSubMatriz(subm_C);

  } else {

    /*INICIALIZO LOS SUBBLOQUES DE MATRIZ QUE ME CORRESPONDE*/

    for (ciclos = 0; ciclos < m; ciclos++) {
      for (i = 0; i < tam_subM; i++) {
        for (j = 0; j < tam_subM; j++) {
          for (k = 0;  k < tam_subM; k++) {
            subm_C[i][j] += subm_A[i][k] * subm_B[k][j];
          }
        }
      }
      MPI_Cart_shift(comm2d,1,-1,&fuente,&destino);
      MPI_Sendrecv_replace(subm_A,(tam_subM*tam_subM),MPI_FLOAT,destino,1,fuente,1,comm2d,&statusA);

      MPI_Cart_shift(comm2d,0,-1,&fuente,&destino);
      MPI_Sendrecv_replace(subm_B,(tam_subM*tam_subM),MPI_FLOAT,destino,2,fuente,2,comm2d,&statusB);
    }
    //printf("MI ID--> %d\n", mi_id);
    //imprimirSubMatriz(subm_C);
  }
  if(mi_id == 0){
    //printf("AK EMPIEZO A IMPRIMIR\n");
    timeFin = MPI_Wtime();

    // if((fp=freopen("ResultadosCannon.txt", "w" ,stdout))==NULL) {
    //     printf("NO PUEDO ABRIR ARCHIVO.\n");
    //     exit(-1);
    // }

    //printf("IMPRESION FINAL DE LA MATRIZ C\n.");
    //printf("TIEMPO TARDADO---> %f segundos\n", timeFin-timeIni);
    //imprimirMatriz(C);
    //fclose(fp);
    //imprimirMatriz(C);

    printf("%d\t%d\t%f\t%f\n", size, n, timeComm, timeFin-timeIni);

  }

  MPI_Finalize();
}
