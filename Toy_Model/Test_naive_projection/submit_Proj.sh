#!/bin/bash
#PBS -q batch
#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00
#PBS -V
#PBS -j oe
#PBS -N __P_CL_test
######################################
######################################

inputfile=params.yml
#outputfile=output.out
src=~/Projected_SDEs/Toy_Model/src
######################################

cd $PBS_O_WORKDIR
outdir=${PBS_O_WORKDIR}
rundir=${outdir}
cd $PBS_O_WORKDIR

python3 ${src}/toy_model_Projected.py ${inputfile} 
