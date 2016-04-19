#include "summa.h"
#include "kelp.h"
#include "dock.h"
#include "dGrid.h"
#include "timer.h"
#include <stream.h>
#include <iostream.h>


/****************************************************************
* void verify(dGrid2<double>& C, const Region2& Rc,		*
*    dGrid2<double>& A, const Region2& Ra,			*
*    dGrid2<double>& B, const Region2& Rb, int panelSize)	*
*								*
* check the results with a serial matrix multiplication		*
* this is super-inefficient, but who cares?			*
*****************************************************************/
void verify(dGrid2<double>& C, const Region2& Rc,
   dGrid2<double>& A, const Region2& Ra,
   dGrid2<double>& B, const Region2& Rb, int panelSize)
{
   const double alpha = 1.0;
   const double beta = 0.0;

   pdgemm(C,C.domain(),A,A.domain(),B,B.domain(),alpha,beta,panelSize);

   replicatedGrid2<double> result(C.domain());
   replicatedGrid2<double> tempC(C.domain());
   replicatedGrid2<double> tempA(A.domain());
   replicatedGrid2<double> tempB(B.domain());

   tempC.copyOnIntersection(C);
   tempB.copyOnIntersection(B);
   tempA.copyOnIntersection(A);

   if (!mpMyID()) {
     g2dgemm(result(0),result(0).region(),
	     tempA(0),tempA(0).region(),
	     tempB(0),tempB(0).region(),
	     alpha,beta);
     int success = TRUE;

     for_point_2(p,result(0))
	if (result(0)(p) != tempC(0)(p)) success = FALSE;
     end_for

     if (success) {
         OUTPUT("verification succeeded" << endl);
     }
     else {
         OUTPUT("verification FAILED" << endl);
     }

   }

}
