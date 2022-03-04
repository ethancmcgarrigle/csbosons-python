#!/bin/bash
#PBS -q batch
#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00
#PBS -V
#PBS -j oe
#PBS -N __jobname__-2
######################################
######################################

cd $PBS_O_WORKDIR

python3 toy_model_noPsi.py 
