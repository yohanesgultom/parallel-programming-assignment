#!/bin/bash

#This benchmark script runs AMBER benchmarks on 1 to 8 K10 GPUs.
#It assumes AMBERHOME is set correctly.
#
#You will need to adjust this as necessary for different GPU and CPU counts / combinations.
#

echo ""
echo "JAC_PRODUCTION_NVE - 23,558 atoms PME"
echo "-------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd PME/JAC_production_NVE
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x K10: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1K10 -inf mdinfo.1K10 -x mdcrd.1K10 -r restrt.1K10
grep "ns/day" mdinfo.1K10 | tail -n1

echo -n "       2 x K10: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2K10 -inf mdinfo.2K10 -x mdcrd.2K10 -r restrt.2K10
grep "ns/day" mdinfo.2K10 | tail -n1

echo -n "       3 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3K10 -inf mdinfo.3K10 -x mdcrd.3K10 -r restrt.3K10
grep "ns/day" mdinfo.3K10 | tail -n1

echo -n "       4 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4K10 -inf mdinfo.4K10 -x mdcrd.4K10 -r restrt.4K10
grep "ns/day" mdinfo.4K10 | tail -n1

echo -n "       6 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
mpirun -np 6 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.6K10 -inf mdinfo.6K10 -x mdcrd.6K10 -r restrt.6K10
grep "ns/day" mdinfo.6K10 | tail -n1

echo -n "       8 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mpirun -np 8 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.8K10 -inf mdinfo.8K10 -x mdcrd.8K10 -r restrt.8K10
grep "ns/day" mdinfo.8K10 | tail -n1


echo ""
echo "JAC_PRODUCTION_NPT - 23,558 atoms PME"
echo "-------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../JAC_production_NPT
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x K10: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1K10 -inf mdinfo.1K10 -x mdcrd.1K10 -r restrt.1K10
grep "ns/day" mdinfo.1K10 | tail -n1

echo -n "       2 x K10: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2K10 -inf mdinfo.2K10 -x mdcrd.2K10 -r restrt.2K10
grep "ns/day" mdinfo.2K10 | tail -n1

echo -n "       3 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3K10 -inf mdinfo.3K10 -x mdcrd.3K10 -r restrt.3K10
grep "ns/day" mdinfo.3K10 | tail -n1

echo -n "       4 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4K10 -inf mdinfo.4K10 -x mdcrd.4K10 -r restrt.4K10
grep "ns/day" mdinfo.4K10 | tail -n1

echo -n "       6 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
mpirun -np 6 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.6K10 -inf mdinfo.6K10 -x mdcrd.6K10 -r restrt.6K10
grep "ns/day" mdinfo.6K10 | tail -n1

echo -n "       8 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mpirun -np 8 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.8K10 -inf mdinfo.8K10 -x mdcrd.8K10 -r restrt.8K10
grep "ns/day" mdinfo.8K10 | tail -n1


echo ""
echo "FACTOR_IX_PRODUCTION_NVE - 90,906 atoms PME"
echo "-------------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../FactorIX_production_NVE
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x K10: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1K10 -inf mdinfo.1K10 -x mdcrd.1K10 -r restrt.1K10
grep "ns/day" mdinfo.1K10 | tail -n1

echo -n "       2 x K10: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2K10 -inf mdinfo.2K10 -x mdcrd.2K10 -r restrt.2K10
grep "ns/day" mdinfo.2K10 | tail -n1

echo -n "       3 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3K10 -inf mdinfo.3K10 -x mdcrd.3K10 -r restrt.3K10
grep "ns/day" mdinfo.3K10 | tail -n1

echo -n "       4 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4K10 -inf mdinfo.4K10 -x mdcrd.4K10 -r restrt.4K10
grep "ns/day" mdinfo.4K10 | tail -n1

echo -n "       6 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
mpirun -np 6 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.6K10 -inf mdinfo.6K10 -x mdcrd.6K10 -r restrt.6K10
grep "ns/day" mdinfo.6K10 | tail -n1

echo -n "       8 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mpirun -np 8 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.8K10 -inf mdinfo.8K10 -x mdcrd.8K10 -r restrt.8K10
grep "ns/day" mdinfo.8K10 | tail -n1

echo ""
echo "FACTOR_IX_PRODUCTION_NPT - 90,906 atoms PME"
echo "-------------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../FactorIX_production_NPT
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x K10: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1K10 -inf mdinfo.1K10 -x mdcrd.1K10 -r restrt.1K10
grep "ns/day" mdinfo.1K10 | tail -n1

echo -n "       2 x K10: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2K10 -inf mdinfo.2K10 -x mdcrd.2K10 -r restrt.2K10
grep "ns/day" mdinfo.2K10 | tail -n1

echo -n "       3 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3K10 -inf mdinfo.3K10 -x mdcrd.3K10 -r restrt.3K10
grep "ns/day" mdinfo.3K10 | tail -n1

echo -n "       4 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4K10 -inf mdinfo.4K10 -x mdcrd.4K10 -r restrt.4K10
grep "ns/day" mdinfo.4K10 | tail -n1

echo -n "       6 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
mpirun -np 6 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.6K10 -inf mdinfo.6K10 -x mdcrd.6K10 -r restrt.6K10
grep "ns/day" mdinfo.6K10 | tail -n1

echo -n "       8 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mpirun -np 8 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.8K10 -inf mdinfo.8K10 -x mdcrd.8K10 -r restrt.8K10
grep "ns/day" mdinfo.8K10 | tail -n1


echo ""
echo "CELLULOSE_PRODUCTION_NVE - 408,609 atoms PME"
echo "--------------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../Cellulose_production_NVE
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x K10: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1K10 -inf mdinfo.1K10 -x mdcrd.1K10 -r restrt.1K10
grep "ns/day" mdinfo.1K10 | tail -n1

echo -n "       2 x K10: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2K10 -inf mdinfo.2K10 -x mdcrd.2K10 -r restrt.2K10
grep "ns/day" mdinfo.2K10 | tail -n1

echo -n "       3 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3K10 -inf mdinfo.3K10 -x mdcrd.3K10 -r restrt.3K10
grep "ns/day" mdinfo.3K10 | tail -n1

echo -n "       4 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4K10 -inf mdinfo.4K10 -x mdcrd.4K10 -r restrt.4K10
grep "ns/day" mdinfo.4K10 | tail -n1

echo -n "       6 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
mpirun -np 6 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.6K10 -inf mdinfo.6K10 -x mdcrd.6K10 -r restrt.6K10
grep "ns/day" mdinfo.6K10 | tail -n1

echo -n "       8 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mpirun -np 8 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.8K10 -inf mdinfo.8K10 -x mdcrd.8K10 -r restrt.8K10
grep "ns/day" mdinfo.8K10 | tail -n1


echo ""
echo "CELLULOSE_PRODUCTION_NPT - 408,609 atoms PME"
echo "--------------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../Cellulose_production_NPT
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x K10: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1K10 -inf mdinfo.1K10 -x mdcrd.1K10 -r restrt.1K10
grep "ns/day" mdinfo.1K10 | tail -n1

echo -n "       2 x K10: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2K10 -inf mdinfo.2K10 -x mdcrd.2K10 -r restrt.2K10
grep "ns/day" mdinfo.2K10 | tail -n1

echo -n "       3 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3K10 -inf mdinfo.3K10 -x mdcrd.3K10 -r restrt.3K10
grep "ns/day" mdinfo.3K10 | tail -n1

echo -n "       4 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4K10 -inf mdinfo.4K10 -x mdcrd.4K10 -r restrt.4K10
grep "ns/day" mdinfo.4K10 | tail -n1

echo -n "       6 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
mpirun -np 6 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.6K10 -inf mdinfo.6K10 -x mdcrd.6K10 -r restrt.6K10
grep "ns/day" mdinfo.6K10 | tail -n1

echo -n "       8 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mpirun -np 8 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.8K10 -inf mdinfo.8K10 -x mdcrd.8K10 -r restrt.8K10
grep "ns/day" mdinfo.8K10 | tail -n1


echo ""
echo "TRPCAGE_PRODUCTION - 304 atoms GB"
echo "---------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../../GB/TRPCage
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU -ref inpcrd
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x K10: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1K10 -inf mdinfo.1K10 -x mdcrd.1K10 -r restrt.1K10 -ref inpcrd
grep "ns/day" mdinfo.1K10 | tail -n1

echo -n "       2 x K10: N/A"
echo -n "       3 x K10: N/A"
echo -n "       4 x K10: N/A"
echo -n "       6 x K10: N/A"
echo -n "       8 x K10: N/A"

echo ""
echo "MYOGLOBIN_PRODUCTION - 2,492 atoms GB"
echo "-------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../myoglobin
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU -ref inpcrd
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x K10: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1K10 -inf mdinfo.1K10 -x mdcrd.1K10 -r restrt.1K10 -ref inpcrd
grep "ns/day" mdinfo.1K10 | tail -n1

echo -n "       2 x K10: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2K10 -inf mdinfo.2K10 -x mdcrd.2K10 -r restrt.2K10 -ref inpcrd
grep "ns/day" mdinfo.2K10 | tail -n1

echo -n "       3 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3K10 -inf mdinfo.3K10 -x mdcrd.3K10 -r restrt.3K10 -ref inpcrd
grep "ns/day" mdinfo.3K10 | tail -n1

echo -n "       4 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4K10 -inf mdinfo.4K10 -x mdcrd.4K10 -r restrt.4K10 -ref inpcrd
grep "ns/day" mdinfo.4K10 | tail -n1

echo -n "       6 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
mpirun -np 6 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.6K10 -inf mdinfo.6K10 -x mdcrd.6K10 -r restrt.6K10 -ref inpcrd
grep "ns/day" mdinfo.6K10 | tail -n1

echo -n "       8 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mpirun -np 8 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.8K10 -inf mdinfo.8K10 -x mdcrd.8K10 -r restrt.8K10 -ref inpcrd
grep "ns/day" mdinfo.8K10 | tail -n1


echo ""
echo "NUCLEOSOME_PRODUCTION - 25,095 atoms GB"
echo "---------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../nucleosome
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU -ref inpcrd
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x K10: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1K10 -inf mdinfo.1K10 -x mdcrd.1K10 -r restrt.1K10 -ref inpcrd
grep "ns/day" mdinfo.1K10 | tail -n1

echo -n "       2 x K10: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2K10 -inf mdinfo.2K10 -x mdcrd.2K10 -r restrt.2K10 -ref inpcrd
grep "ns/day" mdinfo.2K10 | tail -n1

echo -n "       3 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3K10 -inf mdinfo.3K10 -x mdcrd.3K10 -r restrt.3K10 -ref inpcrd
grep "ns/day" mdinfo.3K10 | tail -n1

echo -n "       4 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4K10 -inf mdinfo.4K10 -x mdcrd.4K10 -r restrt.4K10 -ref inpcrd
grep "ns/day" mdinfo.4K10 | tail -n1

echo -n "       6 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
mpirun -np 6 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.6K10 -inf mdinfo.6K10 -x mdcrd.6K10 -r restrt.6K10 -ref inpcrd
grep "ns/day" mdinfo.6K10 | tail -n1

echo -n "       8 x K10: "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mpirun -np 8 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.8K10 -inf mdinfo.8K10 -x mdcrd.8K10 -r restrt.8K10 -ref inpcrd
grep "ns/day" mdinfo.8K10 | tail -n1

