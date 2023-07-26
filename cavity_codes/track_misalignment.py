from __future__ import print_function
import numpy as np
import time, os, sys
from run_recirculation_sdf import start_recirculation

from run_mergeFiles_sdf import *
from rfp2 import *
import subprocess
from time import sleep

import sys, os, csv, copy,shutil

import time
import gc
from numpy.random import normal

roll_error = 2.0e-6

nRoundtrips = 19         # number of iteration

ncar = 181
dgrid = 540e-6
w0 =40e-6
trms = 10e-15
peak_power = 10e9


xlamds = 1.261043e-10
zsep = 50
c_speed  = 299792458
nslice = 2000
isradi = 1
npadt = (4096 - nslice//isradi)//2
npad1 = (1025-ncar)//2
npadx = [int(npad1), int(npad1) + 1]
dt = xlamds*zsep/c_speed

nx_padded = ncar + int(npadx[0]) + int(npadx[1])
dx = 2. * dgrid / ncar
Dkx = 2. * np.pi / dx
dkx = Dkx/nx_padded
dtheta = dkx * xlamds / 2. / np.pi

n_theta_shift = int(np.round(roll_error/dtheta))
real_theta_shift = n_theta_shift*dtheta

print(n_theta_shift)
print("real shift", real_theta_shift)

root_dir = '/sdf/group/beamphysics/jytang/cavity_alignment/'
folder_name = 'data2'

nametag = 'test1'


with open(folder_name + '/'+nametag+'_recirc.txt', "w") as myfile:
    myfile.write("Round energy/uJ peakpower/GW tmean/fs trms/fs  tfwhm/fs xmean/um xrms/um  xfwhm/um xmean/um  yrms/um yfwhm/um \n")
with open(folder_name + '/'+nametag+'_transmit.txt', "w") as myfile:
    myfile.write("Round energy/uJ peakpower/GW tmean/fs trms/fs  tfwhm/fs  xmean/um xrms/um  xfwhm/um ymean/um yrms/um yfwhm/um \n")

    
# do recirculation on all workers
t0 = time.time()
jobid = start_recirculation(zsep = zsep, ncar = ncar, dgrid = dgrid, nslice = nslice, xlamds=xlamds, w0= w0,  trms=trms,  peak_power = peak_power,          # dfl params
                             npadt = npadt, Dpadt = 0, npadx = npadx,isradi = isradi,       # padding params
                             l_undulator = 0, l_cavity = 65.2, w_cavity = 0.6, d1 = 20e-6, d2 = 500e-6, # cavity params
                              verboseQ = 1, # verbose params
                             nRoundtrips = nRoundtrips,               # recirculation params
                             readfilename = None ,
                            n_theta_shift1 = 0, n_theta_shift2 =0, n_theta_shift3 = 0, n_theta_shift4 = 0,   #yaw angle error on each of the mirror
                            x1 = 0, x2 = 0,                   # displacement error of the t
                             workdir = root_dir + '/' + folder_name + '/' , saveFilenamePrefix = nametag)
    
all_done([jobid])
    
print('It takes ', time.time() - t0, ' seconds to finish recirculation.')
        
# merge files for each roundtrip on nRoundtrips workers, with larger memory
t0 = time.time()
jobid = start_mergeFiles(nRoundtrips =nRoundtrips, workdir = root_dir + '/' + folder_name + '/', saveFilenamePrefix=nametag, dgrid = dgrid, dt = dt, Dpadt = 0)
    
all_done([jobid])
print('It takes ', time.time() - t0, ' seconds to finish merging files.')
    
    

