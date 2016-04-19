/****************************************************************
* gd2gemm.C							*
*								*
* Author: Stephen Fink						*
*****************************************************************/

#include "kelp.h"
#include "dock.h"
#include "dGrid.h"
#include "timer.h"

/****************************************************************
* void g2dgemm(GridBlock2<double>& C, const Region2& Rc,	*
*	       GridBlock2<double>& A, const Region2& Ra,	*
*	       GridBlock2<double>& B, const Region2& Rb		*
*	       const double& alpha, const double& beta)		*
*								*
* blas level 3: C[Rc] = beta*C[Rc] + alpha*A[Ra]*B[Rb]		*
* where A[Ra] and B[Rb] are matrices				*
*								*
* serial routine: calls serial dgemm				*
*****************************************************************/
#define f_dgemm FORTRAN_NAME(dgemm_, DGEMM, dgemm)
 
extern "C" {
void f_dgemm(char *transa, char* transb, int* m, int* n, int *k,
	      const double *alpha, double *a, int *lda, double *b, int *ldb, 
	      const double *beta, double *c, int *ldc);
}
					 

void g2dgemm(GridBlock2<double>& C, const Region2& Rc,
     GridBlock2<double>& A, const Region2& Ra,
     GridBlock2<double>& B, const Region2& Rb,
     const double& alpha, const double& beta)
{ 

   // assert that the Regions conform 
   assert (Ra.extents(0) == Rc.extents(0));
   assert (Ra.extents(1) == Rb.extents(0));
   assert (Rb.extents(1) == Rc.extents(1));
   assert (SubRegion(Ra,A.region()));
   assert (SubRegion(Rb,B.region()));
   assert (SubRegion(Rc,C.region()));

   char no = 'N';
   int m = Rc.extents(0);
   int n = Rc.extents(1);
   int k = Ra.extents(1);
   int lda = A.extents(0);
   int ldb = B.extents(0);
   int ldc = C.extents(0);

   STATS_START(STATS_DGEMM);
   f_dgemm(&no,&no,&m,&n,&k,&alpha,A.ptr(Ra.lower()),&lda,
	   B.ptr(Rb.lower()),&ldb,&beta,C.ptr(Rc.lower()),&ldc);
   STATS_STOP(STATS_DGEMM);
}
