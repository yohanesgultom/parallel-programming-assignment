#!/bin/bash

echo ""
echo "CELLULOSE_PRODUCTION_NVE - 408,609 atoms PME"
echo "--------------------------------------------"
cd Amber_GPU_Benchmark_Suite/PME/Cellulose_production_NVE
export CUDA_VISIBLE_DEVICES=0
$AMBERHOME/bin/pmemd.cuda -O -o mdout.GPU -inf mdinfo.GPU -x mdcrd.GPU -r restrt.GPU
tail -100f mdinfo.GPU
