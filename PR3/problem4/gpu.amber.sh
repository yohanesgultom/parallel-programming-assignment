#!/bin/bash
echo ""
echo "JAC_PRODUCTION_NVE - 23,558 atoms PME"
echo "-------------------------------------"
cd Amber_GPU_Benchmark_Suite/PME/JAC_production_NVE
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GT940M -inf mdinfo.1GT940M -x mdcrd.1GT940M -r restrt.1GT940M
grep "ns/day" mdinfo.1GT940M | tail -n1

echo ""
echo "FACTOR_IX_PRODUCTION_NVE - 90,906 atoms PME"
echo "-------------------------------------------"
cd ../FactorIX_production_NVE
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GT940M -inf mdinfo.1GT940M -x mdcrd.1GT940M -r restrt.1GT940M
grep "ns/day" mdinfo.1GT940M | tail -n1

echo ""
echo "CELLULOSE_PRODUCTION_NVE - 408,609 atoms PME"
echo "--------------------------------------------"
cd ../Cellulose_production_NVE
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GT940M -inf mdinfo.1GT940M -x mdcrd.1GT940M -r restrt.1GT940M
grep "ns/day" mdinfo.1GT940M | tail -n1

echo ""
echo "TRPCAGE_PRODUCTION - 304 atoms GB"
cd ../../GB/TRPCage
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GT940M -inf mdinfo.1GT940M -x mdcrd.1GT940M -r restrt.1GT940M -ref inpcrd
grep "ns/day" mdinfo.1GT940M | tail -n1

echo ""
echo "MYOGLOBIN_PRODUCTION - 2,492 atoms GB"
echo "-------------------------------------"
cd ../myoglobin
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GT940M -inf mdinfo.1GT940M -x mdcrd.1GT940M -r restrt.1GT940M -ref inpcrd
grep "ns/day" mdinfo.1GT940M | tail -n1


echo ""
echo "NUCLEOSOME_PRODUCTION - 25,095 atoms GB"
echo "---------------------------------------"
cd ../nucleosome
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.1GT940M -inf mdinfo.1GT940M -x mdcrd.1GT940M -r restrt.1GT940M -ref inpcrd
grep "ns/day" mdinfo.1GT940M | tail -n1
