/****************************************************************
* summa.C							*
*								*
* Author: Stephen Fink						*
*****************************************************************/

#include "summa.h"
#include "kelp.h"
#include "dock.h"
#include "dGrid.h"
#include "timer.h"
#include <stream.h>
#include <iostream.h>

Statistics StatMaster;


/****************************************************************
* void InitGrid(dGrid2<double>& A)				*
*								*
* initialize values in a Distributed array			*
*****************************************************************/
void InitGrid(dGrid2<double>& A)
{
   for_all(i,A)
     for_point_2(p,A(i))
      A(i)(p) = p(0) + p(1);
     end_for
   end_for_all
}
void verify(dGrid2<double>& C, const Region2& Rc,
   dGrid2<double>& A, const Region2& Ra,
   dGrid2<double>& B, const Region2& Rb, int panelSize);

/****************************************************************
* main()							*
*								*
* main() takes one argument: N: the number of points		*
* along each axis						*
*****************************************************************/
int noComm;
int main(int argc,char **argv)
{
   MPI_Init(&argc,&argv);
   InitKeLP(argc,argv);
   try {
   CollectiveGroupInit();
   int N,pRow,pCol,nIters,printResult,panelSize,checkAnswer;



   StatMaster.init(StatsNames,NUM_STATS);

   cmdLine(argc,argv,N,pRow,pCol,panelSize,nIters,printResult,checkAnswer,noComm);
   Region2 domain(1,1,N,N);
   OUTPUT("SUMMA : " << N << " x " << N << endl);
   OUTPUT("processor Array : " << pRow << " x " << pCol << endl);
   OUTPUT("panel size : " << panelSize << endl);
   if (noComm)
	OUTPUT("COMMUNICATION SHUT OFF" << endl);
 
   // set up pseudo block-cyclie distribution
   Processors2 P(Region2(1,1,pRow,pCol));
   OUTPUT(P);
   Decomposition2 D(domain);
   D.distribute(BLOCK2,BLOCK2,P);
   OUTPUT(D);

   // initialize data
//   dGrid2<double> A(D);
   dGrid2<double> A;
   A.instantiate(D);
   InitGrid(A);
//   dGrid2<double> B(D);
   dGrid2<double> B;
   B.instantiate(D);
   InitGrid(B);
//   dGrid2<double> C(D);
   dGrid2<double> C;
   C.instantiate(D);
   Region2 RC = C.domain();
   Region2 prc = C.pRegion(RC);
   InitGrid(C);

   if (checkAnswer) {
     verify(C,C.domain(),A,A.domain(),B,B.domain(),panelSize);
     InitGrid(C);
   }

   if (printResult) cout << A;
   if (printResult) cout << B;
   if (printResult) cout << C;


   // one iteration for startup
   pdgemm(C,C.domain(),A,A.domain(),B,B.domain(),1.0,0.0,panelSize);
   InitGrid(C);

   STATS_RESET();
   mpBarrier();

   for (int i=1; i<=nIters; i++) {
     STATS_START(STATS_ITERATION);
     pdgemm(C,C.domain(),A,A.domain(),B,B.domain(),1.0,0.0,panelSize);
     STATS_STOP(STATS_ITERATION);
   }

   if (printResult) cout << C;

   STATS_REPORT(STATS_MAX);
   double time = StatMaster.Time(STATS_ITERATION);
   double Nd = (double)N;
   double flops = Nd * Nd * Nd * 2.0 * (double)nIters;
   OUTPUT ("MFLOPS: " << flops/time/1000000.0 << endl);

   }
   catch (KelpErr & ke) {
     ke.abort();
   }

   MPI_Finalize();
   return(0);
//   ExitKeLP();
//   return(0);
}
