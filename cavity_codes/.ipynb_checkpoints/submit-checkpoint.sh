#!/bin/bash

#SBATCH --partition=shared
#
#SBATCH --job-name=rafel
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.err
#
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --exclude=tur[000-026]
##SBATCH --exclude=ampt[000-020]
#SBATCH --time=36:00:00


mpirun -n 128 python -u dfl_cbxfel_mpi.py
