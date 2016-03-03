/****************************************************************
* pdgemm.C							*
*								*
* Author: Stephen Fink						*
*****************************************************************/

#include "summa.h"
#include "kelp.h"
#include "dock.h"
#include "dGrid.h"
#include "timer.h"

/****************************************************************
* void pdgemm(dGrid2<double>& C, const Region2& Rc,		*
*	      dGrid2<double>& A, const Region2& Ra,		*
*	      dGrid2<double>& B, const Region2& Rb,		*
*	      const double& alpha, const double& beta,		*
*	      const int panelSize)				*
*								*
* blas level 3: C[Rc] = beta*C[Rc] + alpha*A[Ra]*B[Rb]		*
* where A[Ra] and B[Rb] are distributed matrices		*
*								*
* algorithm uses a series of block outer products: like		*
* van de Giejn's SUMMA						*
*								*
* Note: due to my laziness, this routine only works if A,B,and C*
* all live in the same index space. i.e., make sure 		*
* Ra.lower() == Rb.lower() == Rc.lower()			*
* Maybe I'll generalize this some other time, probably by 	*
* aliasing the data into the correct index space 		*
****************************************************************/
extern int noComm;
void pdgemm(dGrid2<double>& C, const Region2& Rc,
            dGrid2<double>& A, const Region2& Ra,
	    dGrid2<double>& B, const Region2& Rb,
	    const double& alpha, const double& beta,
	    const int panelSize)
{ 
   // assert that the Regions conform 
   assert (Ra.extents(0) == Rc.extents(0));
   assert (Ra.extents(1) == Rb.extents(0));
   assert (Rb.extents(1) == Rc.extents(1));
   assert (SubRegion(Ra,A.domain()));
   assert (SubRegion(Rb,B.domain()));
   assert (SubRegion(Rc,C.domain()));
   
   const int b = panelSize;
   int alreadyCalledDgemm = FALSE;
   const int nPanels = ((Ra.extents(1)-1)/b)+1;
   Processors2 virtualRows(Region2(1,1,nPanels,1));
   Processors2 virtualCols(Region2(1,1,1,nPanels));


   // compute the block vectors of the nPanels outer products 
   Decomposition2 blockColsOfA(Ra);
   blockColsOfA.distribute(BLOCK1,BLOCK1,virtualCols);
   Decomposition2 blockRowsOfB(Rb);
   blockRowsOfB.distribute(BLOCK1,BLOCK1,virtualRows);

   // compute the global regions of domains corresponding to 
   // processor rows and columns of C
   FloorPlan2 pRowsOfC(C.pExtents(0));
   int cRow = C.pLower(0);

   CollectiveGroup *cRowGroup = new CollectiveGroup[C.pExtents(0)];
   CollectiveGroup *cColGroup = new CollectiveGroup[C.pExtents(1)];

   int ii = 0;
   for_1(i,pRowsOfC)
     const int row1 = C(Point2(cRow,C.pLower(1))).lower(0);
     const int row2 = C(Point2(cRow++,C.pLower(1))).upper(0);
     pRowsOfC.setregion(i,Region2(row1,Rc.lower(1),row2,Rc.upper(1)));
     cRowGroup[ii++] = C.pGroup(pRowsOfC(i));
   end_for

   FloorPlan2 pColsOfC(C.pExtents(1));
   int cCol = C.pLower(1);
   ii = 0;
   for_1(i,pColsOfC)
     const int col1 = C(Point2(C.pLower(0),cCol)).lower(1);
     const int col2 = C(Point2(C.pLower(0),cCol++)).upper(1);
     pColsOfC.setregion(i,Region2(Rc.lower(0),col1,Rc.upper(0),col2));
     cColGroup[ii++] = C.pGroup(pColsOfC(i));
   end_for

   replicatedGrid2<double> *AA = new replicatedGrid2<double>[C.pExtents(0)];
   replicatedGrid2<double> *BB = new replicatedGrid2<double>[C.pExtents(1)];

   Region2 pRegionRc = C.pRegion(Rc);


   for_1(i,blockColsOfA)
      // multiply a block Col of A times a block Row of B
      int ii = 0;
      for_1(j,pRowsOfC)
	 // determine the section of A to broadcast to this pRow
	 Region2 R = pRowsOfC(j) * blockColsOfA(i);

	 // set up storage for the broadcast on each processor in the
	 // block row of C
	 AA[ii].resize(R,cRowGroup[ii]);
	 ii++;
      end_for

      ii = 0;
      for_1(j,pColsOfC)
	 // determine the section of B to broadcast to this pCol
	 Region2 R = pColsOfC(j) * blockRowsOfB(i);

	 // set up storage for the broadcast on each processor in the
	 // block row of C
	 BB[ii].resize(R,cColGroup[ii]);
	 ii++;
      end_for

      // broadcast the blocks
      STATS_START(STATS_COMM);
      if (!noComm){
	  int k;
	  for (k=0; k<C.pExtents(0); k++) {
	    AA[k].ringCopyOnIntersection(A,A(A.pOwner(AA[k].domainLower())).owner());
	  }
	  for (k=0; k<C.pUpper(1); k++) {
	    BB[k].ringCopyOnIntersection(B,B(B.pOwner(BB[k].domainLower())).owner());
	  } 
       }
	  STATS_STOP(STATS_COMM);

      // call local dgemm to multiply the blocks
      double beta1 = (alreadyCalledDgemm) ? 1.0 : beta;
      for_all_dgrid_point_2(p,C,pRegionRc)
	 const int myid = mpMyID();
	 int prow = C.pIndex(0,C(p).lower(0)) - C.pLower(0);
	 int pcol = C.pIndex(1,C(p).lower(1)) - C.pLower(1);
         g2dgemm(C(p),C(p).region(),
		 AA[prow](AA[prow].pMap(myid)),AA[prow].domain(),
		 BB[pcol](BB[pcol].pMap(myid)),BB[pcol].domain(),
	         alpha,beta1);
      end_for_all
      alreadyCalledDgemm = TRUE;

   end_for
   
   delete [] AA; delete [] BB;
   delete [] cRowGroup; delete [] cColGroup;
}
