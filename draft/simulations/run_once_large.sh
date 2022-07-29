#! /bin/bash

source ~/.bashrc

# run_once_large <seed> <dataset> <algo> <rho1> <rho2>

seed=$1
dataset=$2
algo=$3
rho1=$4
rho2=$5

hidden1=50
hidden2=5
arch="many_${hidden1}_${hidden2}"

if [ "${algo}" = "wb" ]; then
    folder="${dataset}_${algo}_large"
    rho_args=""
else
    all_rho="${rho1}_${rho2}"
    folder="${dataset}_${algo}_large_rho${all_rho}"
    rho_args="--rho ${rho1} ${rho2}"
fi

mkdir -p ${folder}/log
outfile="${folder}/history_${seed}.pkl"

if [ -f "${outfile}" ]; then
    echo "output file ${outfile} exists! quitting!"
    exit 1
fi

conda activate cpcn

./run_simulation.py \
    --arch-alias "large" \
    ${rho_args} -- \
    . \
    ../hyper/ \
    $dataset \
    $algo \
    $arch \
    $seed \
    > "${folder}/log/out_${seed}.txt" \
    2> "${folder}/log/err_${seed}.txt"
