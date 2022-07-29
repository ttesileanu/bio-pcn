#! /bin/bash

source ~/.bashrc

# run_once_fashion_mnist <seed> <algo> <hidden1> <hidden2> <rho1> <rho2>

seed=$1
algo=$2
hidden1=$3
hidden2=$4
rho1=$5
rho2=$6

dataset="cifar10"

arch="many_${hidden1}_${hidden2}"

trials=500
if [ "${algo}" = "wb" ]; then
    folder="${dataset}_${algo}_${arch}"
    rho_args=""
else
    all_rho="${rho1}_${rho2}"
    folder="${dataset}_${algo}_${arch}_rho${all_rho}"
    rho_args="--rho ${rho1} ${rho2}"
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
    --lr 1e-3 0.02 \
    --lr-decay 1e-5 4e-3 \
    --Q-lrf 0.1 50.0 \
    --Wa-lrf 0.3 4.0 \
    > "${folder}/log/out_${seed}.txt" \
    2> "${folder}/log/err_${seed}.txt"
