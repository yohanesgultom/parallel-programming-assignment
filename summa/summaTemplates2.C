#ifdef __GNUC__



#define DEFINE_TEMPLATES
#pragma implementation "Array1.h"
#pragma implementation "Array2.h"
#pragma implementation "Array3.h"

#pragma implementation "msg.h"

#include "msg.h"
#include "Array1.h"
#include "Array2.h"
#include "Array3.h"


#endif

#ifdef XLC


#include "msg.h"
#include "msg.C"
#include "Array1.h"
#include "Array1.C"
#include "Array2.h"
#include "Array2.C"
#include "Array3.h"
#include "Array3.C"


#endif

#ifdef SGI



#include "Grid2.h"
#include "Grid2.C"
#include "XArray2.h"
#include "XArray2.C"


#include "GridBlock2.h"
#include "GridBlock2.C"
#include "dGrid2.h"
#include "dGrid2.C"
#include "replicatedGrid2.h"
#include "replicatedGrid2.C"
#include "Mover2.h"
#include "Mover2.C"
#include "VectorMover2.h"
#include "VectorMover2.C"



#include "Grid3.h"
#include "Grid3.C"
#include "XArray3.h"
#include "XArray3.C"
#include "GridBlock3.h"
#include "GridBlock3.C"
#include "dGrid3.h"
#include "dGrid3.C"
#include "replicatedGrid3.h"
#include "replicatedGrid3.C"
#include "Mover3.h"
#include "Mover3.C"
#include "VectorMover3.h"
#include "VectorMover3.C"


#pragma instantiate Grid2<double>
#pragma instantiate GridBlock2<double>
#pragma instantiate dGrid2<double>
#pragma instantiate replicatedGrid2<double>
#pragma instantiate Mover2<GridBlock2<double>, double>
#pragma instantiate VectorMover2<GridBlock2<double>, double>
#pragma instantiate XArray2<GridBlock2<double> >

#pragma instantiate Grid3<double>
#pragma instantiate GridBlock3<double>
#pragma instantiate dGrid3<double>
#pragma instantiate replicatedGrid3<double>
#pragma instantiate Mover3<GridBlock3<double>, double>
#pragma instantiate VectorMover3<GridBlock3<double>, double>
#pragma instantiate XArray3<GridBlock3<double> >

ostream& operator << (ostream& o, const dGrid2<double>& G);
ostream& operator << (ostream& o, const GridBlock2<double>& G);

ostream& operator << (ostream& o, const dGrid3<double>& G);
ostream& operator << (ostream& o, const GridBlock3<double>& G);

#endif
