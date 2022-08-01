#! /bin/bash

source ~/.bashrc

# run_once_large <seed> <dataset>

seed=$1
dataset=$2
algo=biopcn

hidden1=50
hidden2=5
arch="many_${hidden1}_${hidden2}"

trials=250
folder="${dataset}_${algo}_large"

mkdir -p ${folder}/log
outfile="${folder}/hyper_${seed}.pkl"

if [ -f "${outfile}" ]; then
    echo "output file ${outfile} exists! quitting!"
    exit 1
fi

conda activate cpcn

./optimize_net.py \
    "${outfile}" \
    $dataset \
    $algo \
    $arch \
    $trials \
    $seed \
    --n-batches 500 \
    --n-rep 5 \
    --rho-sweep 0.005 5.0 \
    --per-layer-rho \
    --lr 1e-4 0.1 \
    --lr-decay 1e-5 0.01 \
    --Q-lrf 1.0 50.0 \
    --Wa-lrf 0.3 4.0 \
    > "${folder}/log/out_${seed}.txt" \
    2> "${folder}/log/err_${seed}.txt"
