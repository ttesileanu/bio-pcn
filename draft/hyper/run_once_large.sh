#! /bin/bash

source ~/.bashrc

# run_once_large <seed> <dataset> <algo> <rho>

seed=$1
dataset=$2
algo=$3
rho=$4

trials=100

folder="${dataset}_${algo}_large_rho${rho}"

mkdir -p ${folder}/log

conda activate cpcn

./optimize_large_linear.py \
    "${folder}/hyper_${seed}.pkl" \
    $dataset \
    $algo \
    $rho \
    $trials \
    $seed \
    > "${folder}/log/out_${seed}.txt" \
    2> "${folder}/log/err_${seed}.txt"
