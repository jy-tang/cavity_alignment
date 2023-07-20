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



nRoundtrips = 19         # number of iteration

ncar = 181
dgrid = 540e-6
w0 =40e-6
xlamds = 1.261043e-10
zsep = 140
c_speed  = 299792458
nslice = 1024
isradi = 1
npadt = (4096 - nslice//isradi)//2
npad1 = (256-ncar)//2
npadx = [int(npad1), int(npad1) + 1]
dt = xlamds*zsep/c_speed

root_dir = '/sdf/group/beamphysics/jytang/cavity_alignment/'
folder_name = 'data1'

nametag = 'test1'


with open(folder_name + '/'+nametag+'_recirc.txt', "w") as myfile:
    myfile.write("Round energy/uJ peakpower/GW tmean/fs trms/fs  tfwhm/fs xmean/um xrms/um  xfwhm/um xmean/um  yrms/um yfwhm/um \n")
with open(folder_name + '/'+nametag+'_transmit.txt', "w") as myfile:
    myfile.write("Round energy/uJ peakpower/GW tmean/fs trms/fs  tfwhm/fs  xmean/um xrms/um  xfwhm/um ymean/um yrms/um yfwhm/um \n")

    
# do recirculation on all workers
t0 = time.time()
jobid = start_recirculation(zsep = zsep, ncar = ncar, dgrid = dgrid, nslice = nslice, xlamds=xlamds, w0=31.8e-6,  trms=10.e-15,  peak_power = 1e10,          # dfl params
                             npadt = npadt, Dpadt = 0, npadx = npadx,isradi = isradi,       # padding params
                             l_undulator = 0, l_cavity = 65.2, w_cavity = 0.6, d1 = 20e-6, d2 = 500e-6, # cavity params
                              verboseQ = 1, # verbose params
                             nRoundtrips = nRoundtrips,               # recirculation params
                             readfilename = None ,
                            delta1 = 0, delta2 =0, delta3 = 0, delta4 = 0,   #yaw angle error on each of the mirror
                            x1 = 0, x2 = 0,                   # displacement error of the t
                             workdir = root_dir + '/' + folder_name + '/' , saveFilenamePrefix = nametag)
    
all_done([jobid])
    
print('It takes ', time.time() - t0, ' seconds to finish recirculation.')
        
# merge files for each roundtrip on nRoundtrips workers, with larger memory
t0 = time.time()
jobid = start_mergeFiles(nRoundtrips =nRoundtrips, workdir = root_dir + '/' + folder_name + '/', saveFilenamePrefix=nametag, dgrid = dgrid, dt = dt, Dpadt = 0)
    
all_done([jobid])
print('It takes ', time.time() - t0, ' seconds to finish merging files.')
    
    

