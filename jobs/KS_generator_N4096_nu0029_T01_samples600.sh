#!/bin/bash
#PBS -m be
#PBS -q jumbo

export OMP_NUM_THREADS=1

cd "/home/clustor2/ma/n/np923"
source firedrake/bin/activate

cd "/home/clustor2/ma/n/np923/fem_neural_operator"

# Parameters
project_dir="/home/clustor2/ma/n/np923/fem_neural_operator"
N=4096
samples=600
batch=${BATCH}
T=0.1


python3 playground/KS/KS_generator.py ${project_dir} ${N} ${T} ${samples} ${batch}

