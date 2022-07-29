#! /bin/bash

source ~/.bashrc

# run_once_lfw <seed> <algo> <rho>

seed=$1
algo=$2
rho=$3

dataset="lfw"

arch="many_20_5_20"

trials=100
if [ "${algo}" = "wb" ]; then
    folder="${dataset}_${algo}_${arch}"
    rho_args=""
else
    all_rho="${rho}"
    folder="${dataset}_${algo}_${arch}_rho${all_rho}"
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
    --n-batches 300 \
    --n-rep 5 \
    --lr 0.01 0.20 \
    --lr-decay 1e-6 1e-4 \
    --Q-lrf 0.1 10.0 \
    --Wa-lrf 0.3 4.0 \
    > "${folder}/log/out_${seed}.txt" \
    2> "${folder}/log/err_${seed}.txt"
