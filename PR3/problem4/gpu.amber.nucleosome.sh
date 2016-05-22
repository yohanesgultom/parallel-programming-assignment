#!/bin/bash

echo ""
echo "NUCLEOSOME_PRODUCTION - 25,095 atoms GB"
echo "---------------------------------------"
cd Amber14_Benchmark_Suite/GB/nucleosome
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.GPU -inf mdinfo.GPU -x mdcrd.GPU -r restrt.GPU -ref inpcrd
tail -100f mdinfo.GPU
