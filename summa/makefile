#TOP = /u/ac/baden/lib/kelp1.2.95/kelp
#TOP = /projects/paci/pte/carte/kelp/kelp1.2.95_sp2_e/kelp
TOP = /projects/paci/pte/carte/kelp/kelp1.2.95_sp2/kelp
#TOP = /net/picard/disk1/software/kelp1.2.95/kelp
include $(TOP)/arch

#BLAS_LIB_DIR  = -L/net/locutus/disk1/software/blas 

USR_INCLUDE = -I$(TOP)/../apps/dock -I$(TOP)/../apps/dgrid
USR_LINK    = -L$(TOP)/../apps/dock -L$(TOP)/../apps/dgrid \
	      $(BLAS_LIB_DIR) -ldgrid -lessl -ldock
#	      $(BLAS_LIB_DIR) -ldgrid -lblas -ldock

app:		summa

OBJS = 		summa.o cmdLine.o verify.o pdgemm.o 	\
		g2dgemm.o summaTemplates.o summaTemplates2.o

summa:		$(OBJS)
		$(C++LINK) $(C++FLAGS) -o $@ \
		$(OBJS) \
		-L$(LIBRARY) $(LDLIBS) 
#		-lF77 -lsunmath $(LDLIBS)

clean:	
	$(RM) *.o summa PI*;
	$(RM) core;
	$(RM) mpirun;
	$(RM) -r Templates.DB ptrepository;
