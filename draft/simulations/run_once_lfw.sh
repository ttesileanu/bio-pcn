#! /bin/bash

source ~/.bashrc

# run_once_extra <seed> <algo> <rho>

seed=$1
algo=$2
rho=$3

dataset="lfw"

arch="many_20_5_20"

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
