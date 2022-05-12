#! /bin/bash

module load slurm
module load disBatch

sbatch -p ccn --constraint=skylake -n 120 --ntasks-per-node 40 disBatch task_file_wb.db
