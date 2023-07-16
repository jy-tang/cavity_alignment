#!/bin/bash

#SBATCH --partition=shared
#
#SBATCH --job-name=recirculation
#SBATCH --output=output_c-%j.txt
#SBATCH --error=output_c-%j.err
#
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --exclude=ampt000,ampt001,ampt002,ampt003,ampt004,ampt005,ampt006,ampt007,ampt008,ampt009,ampt010,ampt011,ampt012,ampt013,ampt014,ampt015,ampt016,ampt017,ampt018,ampt019,ampt020,psc000,psc001,psc002,psc003,psc004,psc005,psc006,psc007,psc008,psc009,tur000,tur001,tur002,tur003,tur004,tur005,tur006,tur007,tur008,tur009,tur010,tur011,tur012,tur013,tur014,tur015,tur016,tur017,tur018,tur019,tur020,tur021,tur022,tur023,tur024,tur025,tur026,volt000,volt001,volt002,volt003,volt004,volt005
#SBATCH --time=36:00:00


mpirun -n 128 python -u /sdf/group/beamphysics/jytang/genesis/CBXFEL/cavity_codes/dfl_cbxfel_stats.py
