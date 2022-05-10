#! /bin/bash

source ~/.bashrc

# run_once <seed> <dataset> <algo> <arch> <rho>

seed=$1
dataset=$2
algo=$3
arch=$4
rho=$5

conda activate cpcn

folder="${dataset}_${algo}_${arch}"

./run_linear_simulation.py \
    . \
    ../hyper/ \
    $dataset \
    $algo \
    $arch \
    $rho \
    $seed \
    > "${folder}/log/out_${seed}.txt" \
    2> "${folder}/log/err_${seed}.txt"