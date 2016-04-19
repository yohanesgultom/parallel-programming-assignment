/****************************************************************
*  cmdLine.C							*
*								*
*  Parse command line						*
*								*
*  Author: Stephen Fink						*
****************************************************************/

#include <stdlib.h>
#include <string.h>
#include "kelp.h"

// Default values

const int def_N = 8;
const int def_pRow = 1;
const int def_pCol = 1;
const int def_panelSize = 1;
const int def_niter = 1;
const int def_print = FALSE;
const int def_check = FALSE;

#define MATCH(s) (!strcmp(argv[arg], (s)))
void PrintUsage(const char *const program, const char *const option);

/****************************************************************
* void cmdLine(int argc, char **argv, int& N, int& pRow, 	*
*	       int &pCol, int& niter, int& print, int &check)	*
* Parses the parameters from argv 				*
*								*
*	-n	<integer>	matrix size			*
*	-prow   <integer>	prow, processor array		*
*	-pcol   <integer>	pcol, processor array		*
*	-panel	<integer>	summa panel size		*
*	-i	<integer>	# iterations 			*
*	-p			print the results		*
*	-check			check the results		*
*	-nocomm			shut off communication 		*
*****************************************************************/

void cmdLine(int argc, char **argv, int& N, int& pRow, int& pCol,
		int& panelSize, int& niter, int& print, int &check,
		int& noComm)
{
   int argcount = argc;

#ifdef P4
  argcount -= 2;       // P4 passes extra arguments
#endif

   // Fill in default values
   N = def_N;
   pRow = def_pRow;
   pCol = def_pCol;
   panelSize = def_panelSize;
   niter = def_niter;
   print = def_print;
   check = def_check;
   noComm = FALSE;

   if (!mpMyID()) {
     // Parse the command line arguments -- if something is not kosher, die
     for (int arg = 1; arg < argcount; arg++)
     {
      if      (MATCH("-n"))  N = atoi(argv[++arg]);
      else if (MATCH("-prow"))  pRow = atoi(argv[++arg]);
      else if (MATCH("-pcol"))  pCol = atoi(argv[++arg]);
      else if (MATCH("-i"))  niter = atoi(argv[++arg]);
      else if (MATCH("-panel"))  panelSize = atoi(argv[++arg]);
      else if (MATCH("-p"))  print = TRUE;
      else if (MATCH("-check"))  check = TRUE;
      else if (MATCH("-nocomm"))  noComm = TRUE;
      else PrintUsage(argv[0], argv[arg]);
     }
    }
   // MPI problem: only root process gets argv correctly with P4
   // must broadcast arguments to others

   MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&pRow,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&pCol,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&panelSize,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&niter,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&print,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&check,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&noComm,1,MPI_INT,0,MPI_COMM_WORLD);
}

/****************************************************************
* void PrintUsage(const char *const program, 			*
*		  const char *const option)			*
*								*
* Prints out the command line options 				*
*****************************************************************/

void PrintUsage(const char *const program, const char *const option)
{
  if (mpMyID() == 0) {
      cerr << endl << program << ": error in argument `" << option << "'\n";
      cerr << "\t-n        <integer>  problem domain size\n";
      cerr << "\t-prow     <integer>  processor array size\n";
      cerr << "\t-pcol     <integer>  processor array size\n";
      cerr << "\t-panel    <integer>  panel size\n";
      cerr << "\t-i        <integer>  # iterations\n";
      cerr << "\t-p        	      print matrices\n";
      cerr << "\t-check        	      check result\n";
      cerr << "\t-nocomm              shut off communcation\n";
      cerr.flush();
     }
   MPI_Finalize();
   exit(0);
}
