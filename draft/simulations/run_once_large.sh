#! /bin/bash

source ~/.bashrc

# run_once <seed> <dataset> <algo> <rho>

seed=$1
dataset=$2
algo=$3
rho=$4

conda activate cpcn

folder="${dataset}_${algo}_large_rho${rho}"

mkdir -p ${folder}/log

./run_large_linear_simulation.py \
    . \
    ../hyper/ \
    $dataset \
    $algo \
    $rho \
    $seed \
    > "${folder}/log/out_${seed}.txt" \
    2> "${folder}/log/err_${seed}.txt"
