#!/bin/bash
#PBS -m be
#PBS -q jumbo

export OMP_NUM_THREADS=1

cd "/home/clustor2/ma/n/np923"
source firedrake/bin/activate

cd "/home/clustor2/ma/n/np923/fem_neural_operator"

# Parameters
project_dir="/home/clustor2/ma/n/np923/fem_neural_operator"
nu=0.01
N=8192
samples=40
seed=${BATCH} # Environment variable
batch=${BATCH}


python3 fem_neural_operator/burgers/burgers_generator.py ${project_dir} ${nu} ${N} ${samples} ${seed} ${batch}

