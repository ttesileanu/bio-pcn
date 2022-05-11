#! /bin/bash

source ~/.bashrc

# run_once <seed> <dataset> <algo> <arch> <rho>

seed=$1
dataset=$2
algo=$3
arch=$4
rho=$5

trials=100

folder="${dataset}_${algo}_${arch}"

conda activate cpcn

./optimize_linear.py \
    "${folder}/hyper_${seed}.pkl" \
    $dataset \
    $algo \
    $arch \
    $rho \
    $trials \
    $seed \
    > "${folder}/log/out_${seed}.txt" \
    2> "${folder}/log/err_${seed}.txt"
