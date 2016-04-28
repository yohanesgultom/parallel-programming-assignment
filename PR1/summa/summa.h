#ifndef _defined_summa_h
#define _defined_summa_h

/****************************************************************
*	summa.h							*
*								*
*	Author: Stephen Fink					*
*****************************************************************/

#include <stream.h>
#include <iomanip.h>
#include "kelp.h"
#include "dGrid2.h"

void pdgemm(dGrid2<double>& C, const Region2& Rc,
     dGrid2<double>& A, const Region2& Ra,
     dGrid2<double>& B, const Region2& Rb,
     const double& alpha, const double& beta,
     const int panelSize);

void g2dgemm(GridBlock2<double>& C, const Region2& Rc,
     GridBlock2<double>& A, const Region2& Ra,
     GridBlock2<double>& B, const Region2& Rb,
     const double& alpha, const double& beta);

void cmdLine(int argc, char **argv, int& N, int& pRow, int& pCol,
	   int& panelSize, int& niter, int& print, int &check, int &noComm);

#define ERROR (-1)
#define OUTPUT(X) {if (!mpMyID()) cout << X << flush; }

#endif
