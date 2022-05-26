#! /bin/bash
./run_multi_arch.py --lr-scale 0.1 . ../hyper/mnist_biopcn_large_small_rho1.0/ mnist biopcn two 1.0 123
./run_multi_arch.py --lr-scale 0.3 . ../hyper/mnist_biopcn_large_small_rho1.0/ mnist biopcn two-big 1.0 123
./run_multi_arch.py . ../hyper/mnist_biopcn_large_small_rho1.0/ mnist biopcn two-bigger 1.0 123
./run_multi_arch.py . ../hyper/mnist_biopcn_large_small_rho1.0/ mnist biopcn two-biggest 1.0 123

./run_multi_arch.py . ../hyper/mmill_biopcn_large_rho1.0/ mmill biopcn two 1.0 123
./run_multi_arch.py . ../hyper/mmill_biopcn_large_rho1.0/ mmill biopcn two-big 1.0 123
./run_multi_arch.py . ../hyper/mmill_biopcn_large_rho1.0/ mmill biopcn two-bigger 1.0 123
./run_multi_arch.py . ../hyper/mmill_biopcn_large_rho1.0/ mmill biopcn two-biggest 1.0 123
