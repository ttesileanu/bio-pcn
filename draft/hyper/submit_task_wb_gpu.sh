#! /bin/bash

module load slurm
module load disBatch

sbatch -p gpu -N4 --ntasks-per-node=4 -c1 --gres=gpu:4 --wrap "disBatch.py -g task_file_wb_gpu.db"
