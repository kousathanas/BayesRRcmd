#!/bin/bash

# To be run within a SLURM allocation (salloc)

module purge
module load intel intel-mpi intel-mkl boost eigen zlib
module list

EXE=./src/mpi_gibbs

# COMPILATION
cd ./src
B='-B'
B=''
make $B -f Makefile || exit 1;
cd ..

# DATASET
datadir=./test/data
dataset=uk10k_chr1_1mb
sparsedir=$datadir
sparsebsn=$dataset
phen=test

#datadir=/scratch/orliac/testM100K_N5K_missing
#dataset=memtest_M100K_N5K_missing0.01
#phen=memtest_M100K_N5K_missing0.01

#datadir=/scratch/orliac/testN500K
#dataset=testN500K
#phen=$dataset

N=1
TPN=10
BPR=1

echo
echo Convert BED to SPARSE
echo
srun -N $N --ntasks-per-node=$TPN $EXE --bed-to-sparse --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --blocks-per-rank $BPR --sparse-dir $sparsedir --sparse-basename $sparsebsn
