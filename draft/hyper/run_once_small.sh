#! /bin/bash

source ~/.bashrc

# run_once <seed> <dataset> <algo> <rho>

seed=$1
dataset=$2
algo=$3
rho=$4
arch="small"

trials=100
if [ "${algo}" = "wb" ]; then
    folder="${dataset}_${algo}_small"
    rho_args=""
else
    folder="${dataset}_${algo}_small_rho${rho}"
    rho_args="--rho ${rho}"
fi

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
    $rho_args \
    > "${folder}/log/out_${seed}.txt" \
    2> "${folder}/log/err_${seed}.txt"
