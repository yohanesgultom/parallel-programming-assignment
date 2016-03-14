#!/bin/bash

#This benchmark script runs AMBER benchmarks on 1 to 4 GPUs.
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
echo -n "       1 x GTX680: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GTX680 -inf mdinfo.1GTX680 -x mdcrd.1GTX680 -r restrt.1GTX680
grep "ns/day" mdinfo.1GTX680 | tail -n1

echo -n "       2 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2GTX680 -inf mdinfo.2GTX680 -x mdcrd.2GTX680 -r restrt.2GTX680
grep "ns/day" mdinfo.2GTX680 | tail -n1

echo -n "       3 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3GTX680 -inf mdinfo.3GTX680 -x mdcrd.3GTX680 -r restrt.3GTX680
grep "ns/day" mdinfo.3GTX680 | tail -n1

echo -n "       4 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4GTX680 -inf mdinfo.4GTX680 -x mdcrd.4GTX680 -r restrt.4GTX680
grep "ns/day" mdinfo.4GTX680 | tail -n1

echo ""
echo "JAC_PRODUCTION_NPT - 23,558 atoms PME"
echo "-------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../JAC_production_NPT
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x GTX680: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GTX680 -inf mdinfo.1GTX680 -x mdcrd.1GTX680 -r restrt.1GTX680
grep "ns/day" mdinfo.1GTX680 | tail -n1

echo -n "       2 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2GTX680 -inf mdinfo.2GTX680 -x mdcrd.2GTX680 -r restrt.2GTX680
grep "ns/day" mdinfo.2GTX680 | tail -n1

echo -n "       3 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3GTX680 -inf mdinfo.3GTX680 -x mdcrd.3GTX680 -r restrt.3GTX680
grep "ns/day" mdinfo.3GTX680 | tail -n1

echo -n "       4 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4GTX680 -inf mdinfo.4GTX680 -x mdcrd.4GTX680 -r restrt.4GTX680
grep "ns/day" mdinfo.4GTX680 | tail -n1


echo ""
echo "FACTOR_IX_PRODUCTION_NVE - 90,906 atoms PME"
echo "-------------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../FactorIX_production_NVE
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x GTX680: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GTX680 -inf mdinfo.1GTX680 -x mdcrd.1GTX680 -r restrt.1GTX680
grep "ns/day" mdinfo.1GTX680 | tail -n1

echo -n "       2 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2GTX680 -inf mdinfo.2GTX680 -x mdcrd.2GTX680 -r restrt.2GTX680
grep "ns/day" mdinfo.2GTX680 | tail -n1

echo -n "       3 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3GTX680 -inf mdinfo.3GTX680 -x mdcrd.3GTX680 -r restrt.3GTX680
grep "ns/day" mdinfo.3GTX680 | tail -n1

echo -n "       4 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4GTX680 -inf mdinfo.4GTX680 -x mdcrd.4GTX680 -r restrt.4GTX680
grep "ns/day" mdinfo.4GTX680 | tail -n1


echo ""
echo "FACTOR_IX_PRODUCTION_NPT - 90,906 atoms PME"
echo "-------------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../FactorIX_production_NPT
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x GTX680: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GTX680 -inf mdinfo.1GTX680 -x mdcrd.1GTX680 -r restrt.1GTX680
grep "ns/day" mdinfo.1GTX680 | tail -n1

echo -n "       2 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2GTX680 -inf mdinfo.2GTX680 -x mdcrd.2GTX680 -r restrt.2GTX680
grep "ns/day" mdinfo.2GTX680 | tail -n1

echo -n "       3 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3GTX680 -inf mdinfo.3GTX680 -x mdcrd.3GTX680 -r restrt.3GTX680
grep "ns/day" mdinfo.3GTX680 | tail -n1

echo -n "       4 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4GTX680 -inf mdinfo.4GTX680 -x mdcrd.4GTX680 -r restrt.4GTX680
grep "ns/day" mdinfo.4GTX680 | tail -n1


echo ""
echo "CELLULOSE_PRODUCTION_NVE - 408,609 atoms PME"
echo "--------------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../Cellulose_production_NVE
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x GTX680: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GTX680 -inf mdinfo.1GTX680 -x mdcrd.1GTX680 -r restrt.1GTX680
grep "ns/day" mdinfo.1GTX680 | tail -n1

echo -n "       2 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2GTX680 -inf mdinfo.2GTX680 -x mdcrd.2GTX680 -r restrt.2GTX680
grep "ns/day" mdinfo.2GTX680 | tail -n1

echo -n "       3 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3GTX680 -inf mdinfo.3GTX680 -x mdcrd.3GTX680 -r restrt.3GTX680
grep "ns/day" mdinfo.3GTX680 | tail -n1

echo -n "       4 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4GTX680 -inf mdinfo.4GTX680 -x mdcrd.4GTX680 -r restrt.4GTX680
grep "ns/day" mdinfo.4GTX680 | tail -n1


echo ""
echo "CELLULOSE_PRODUCTION_NPT - 408,609 atoms PME"
echo "--------------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../Cellulose_production_NPT
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x GTX680: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GTX680 -inf mdinfo.1GTX680 -x mdcrd.1GTX680 -r restrt.1GTX680
grep "ns/day" mdinfo.1GTX680 | tail -n1

echo -n "       2 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2GTX680 -inf mdinfo.2GTX680 -x mdcrd.2GTX680 -r restrt.2GTX680
grep "ns/day" mdinfo.2GTX680 | tail -n1

echo -n "       3 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3GTX680 -inf mdinfo.3GTX680 -x mdcrd.3GTX680 -r restrt.3GTX680
grep "ns/day" mdinfo.3GTX680 | tail -n1

echo -n "       4 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4GTX680 -inf mdinfo.4GTX680 -x mdcrd.4GTX680 -r restrt.4GTX680
grep "ns/day" mdinfo.4GTX680 | tail -n1


echo ""
echo "TRPCAGE_PRODUCTION - 304 atoms GB"
echo "---------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../../GB/TRPCage
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU -ref inpcrd
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x GTX680: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GTX680 -inf mdinfo.1GTX680 -x mdcrd.1GTX680 -r restrt.1GTX680 -ref inpcrd
grep "ns/day" mdinfo.1GTX680 | tail -n1

echo -n "       2 x GTX680: N/A"
echo -n "       3 x GTX680: N/A"
echo -n "       4 x GTX680: N/A"

echo ""
echo "MYOGLOBIN_PRODUCTION - 2,492 atoms GB"
echo "-------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../myoglobin
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU -ref inpcrd
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x GTX680: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GTX680 -inf mdinfo.1GTX680 -x mdcrd.1GTX680 -r restrt.1GTX680 -ref inpcrd
grep "ns/day" mdinfo.1GTX680 | tail -n1

echo -n "       2 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2GTX680 -inf mdinfo.2GTX680 -x mdcrd.2GTX680 -r restrt.2GTX680 -ref inpcrd
grep "ns/day" mdinfo.2GTX680 | tail -n1

echo -n "       3 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3GTX680 -inf mdinfo.3GTX680 -x mdcrd.3GTX680 -r restrt.3GTX680 -ref inpcrd
grep "ns/day" mdinfo.3GTX680 | tail -n1

echo -n "       4 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4GTX680 -inf mdinfo.4GTX680 -x mdcrd.4GTX680 -r restrt.4GTX680 -ref inpcrd
grep "ns/day" mdinfo.4GTX680 | tail -n1


echo ""
echo "NUCLEOSOME_PRODUCTION - 25,095 atoms GB"
echo "---------------------------------------"
echo ""
echo -n "CPU code 16 cores: "
cd ../nucleosome
mpirun -np 16 $AMBERHOME/bin/pmemd.MPI -O -o mdout.16CPU -inf mdinfo.16CPU -x mdcrd.16CPU -r restrt.16CPU -ref inpcrd
grep "ns/day" mdinfo.16CPU | tail -n1
echo -n "       1 x GTX680: "
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GTX680 -inf mdinfo.1GTX680 -x mdcrd.1GTX680 -r restrt.1GTX680 -ref inpcrd
grep "ns/day" mdinfo.1GTX680 | tail -n1

echo -n "       2 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.2GTX680 -inf mdinfo.2GTX680 -x mdcrd.2GTX680 -r restrt.2GTX680 -ref inpcrd
grep "ns/day" mdinfo.2GTX680 | tail -n1

echo -n "       3 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2
mpirun -np 3 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.3GTX680 -inf mdinfo.3GTX680 -x mdcrd.3GTX680 -r restrt.3GTX680 -ref inpcrd
grep "ns/day" mdinfo.3GTX680 | tail -n1

echo -n "       4 x GTX680: "
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 $AMBERHOME/bin/pmemd.cuda.MPI -O -o mdout.4GTX680 -inf mdinfo.4GTX680 -x mdcrd.4GTX680 -r restrt.4GTX680 -ref inpcrd
grep "ns/day" mdinfo.4GTX680 | tail -n1


