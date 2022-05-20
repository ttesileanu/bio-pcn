#! /bin/bash
for i in `seq -w 0 39`
do
    ./run_large_linear_simulation.py . ../hyper/ ${1} ${i}
done
