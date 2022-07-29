#! /bin/bash

source ~/.bashrc

# run_once_extra <dataset> <seed> <algo> <hidden1> <hidden2> <rho1> <rho2>

dataset=$1
seed=$2
algo=$3
hidden1=$4
hidden2=$5
rho1=$6
rho2=$7

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

./run_simulation.py \
    ${rho_args} -- \
    . \
    ../hyper/ \
    $dataset \
    $algo \
    $arch \
    $seed \
    > "${folder}/log/out_${seed}.txt" \
    2> "${folder}/log/err_${seed}.txt"
