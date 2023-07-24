#!/bin/bash

#SBATCH --partition=shared
#
#SBATCH --job-name=rafel
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30g
#SBATCH --exclude=tur[000-026]
##SBATCH --exclude=ampt[000-020]
#SBATCH --time=36:00:00


export PYTHONPATH="/sdf/group/beamphysics/jytang/cavity_alignment/cavity_codes:$PYTHONPATH"


python -u /sdf/group/beamphysics/jytang/cavity_alignment/cavity_codes/track_misalignment.py
