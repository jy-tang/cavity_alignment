from __future__ import print_function
from run_recirculation_sdf import start_recirculation
from run_mergeFiles_sdf import *
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
from rfp2 import *
import subprocess
from time import sleep
import SDDS
from sdds2genesis import match_to_FODO,sdds2genesis
import sys, os, csv, copy,shutil
from scipy.interpolate import interpn, interp2d
import time
import gc

ncar = 181
dgrid = 540e-6
w0 = 40e-6

nslice = 1058
h_Plank = 4.135667696e-15
c_speed  = 299792458   
xlamds = 1.261043e-10
isradi = 1
zsep = 40
dt = xlamds*zsep * max(1,isradi) /c_speed

npadt = (8196 - nslice//isradi)//2
npad1 = (256-ncar)//2
npadx = [int(npad1), int(npad1) + 1]
l_cavity = 32*3.9
l_undulator = 149
w_cavity = 1
nRoundtrips = 9
nametag = 'n0'


root_dir = '/sdf/group/beamphysics/jytang/genesis/CBXFEL/'
folder_name = '/test/'
misalignQ = False
roughnessQ = False
theta = 200e-9
surface_filename = 'cavity_codes/synthetic_SPring-8'

# prepare error on crystals
if misalignQ:
    M1 = theta*np.random.randn(1)[0] * 2*np.pi/xlamds
    M2 = theta*np.random.randn(1)[0]* 2*np.pi/xlamds
    M3 = theta*np.random.randn(1)[0] * 2*np.pi/xlamds
    M4 = theta*np.random.randn(1)[0] * 2*np.pi/xlamds
else:
    M1 = M2 = M3 = M4 = 0

if roughnessQ:
    nx = ny = ncar
    nx_padded = ncar + int(npadx[0]) + int(npadx[1])
    dx = 2. * dgrid / ncar
    # get x, y coordinates
    xs = (np.arange(nx_padded) - np.floor(nx_padded/2))*dx
    ys = (np.arange(ny) - np.floor(ny/2))*dx
    
    
    fname = root_dir + surface_filename + '_001.txt'
    C1h = np.loadtxt(fname)
    C1h *= 1e-10
    xa = np.linspace(-500e-6, 500e-6, C1h.shape[1])
    ya = np.linspace(-500e-6, 500e-6, C1h.shape[0])
    
    f = interp2d(xa, ya, C1h, kind='cubic')
    C1 = f(xs, ys)
    C1 = C1.T
    
    fname = root_dir + surface_filename + '_002.txt'
    C2h = np.loadtxt(fname)
    C2h *= 1e-10
    f = interp2d(xa, ya, C2h, kind='cubic')
    C2 = f(xs, ys)
    C2 = C2.T
    
    fname = root_dir + surface_filename + '_003.txt'
    C3h = np.loadtxt(fname)
    C3h *= 1e-10
    f = interp2d(xa, ya, C3h, kind='cubic')
    C3 = f(xs, ys)
    C3 = C3.T
    
    fname = root_dir + surface_filename + '_004.txt'
    C4h = np.loadtxt(fname)
    C4h *= 1e-10
    f = interp2d(xa, ya, C4h, kind='cubic')
    C4 = f(xs, ys)
    C4 = C4.T

else:
    C1 = C2 = C3 = C4 = None
    
with open(root_dir + '/' + folder_name + '/'+nametag+'_recirc.txt', "w") as myfile:
    myfile.write("Round energy/uJ peakpower/GW trms/fs  tfwhm/fs xrms/um  xfwhm/um yrms/um yfwhm/um \n")
with open(root_dir + '/' + folder_name + '/'+nametag+'_transmit.txt', "w") as myfile:
    myfile.write("Round energy/uJ peakpower/GW trms/fs  tfwhm/fs xrms/um  xfwhm/um yrms/um yfwhm/um \n")
    
# do recirculation on all workers
t0 = time.time()
jobid = start_recirculation(zsep = zsep, ncar = ncar, dgrid = dgrid, nslice = nslice, xlamds=xlamds,           # dfl params
                                 npadt = npadt, Dpadt = 0, npadx = npadx,isradi = isradi,       # padding params
                                 l_undulator = l_undulator, l_cavity = l_cavity, w_cavity = 1, d1 = 50e-6, d2 = 200e-6, # cavity params
                                  verboseQ = 1, # verbose params
                                 nRoundtrips = nRoundtrips,               # recirculation params
                                misalignQ = misalignQ, M1 = M1, M2 = M2, M3 = M3, M4 = M4,        # misalignment parameter
                             roughnessQ = roughnessQ, C1 = C1, C2 = C2, C3 = C3, C4 = C4, 
                                 readfilename = root_dir +'/data_long2/tap0.03_K1.172_n9.out.dfl' , 
                                 seedfilename = 'n0_seed_init.dfl',
                                       workdir = root_dir + '/' + folder_name + '/' , saveFilenamePrefix = nametag)
    
all_done([jobid])
print('It takes ', time.time() - t0, ' seconds to finish recirculation.')
    
    
    # merge files for each roundtrip on nRoundtrips workers, with larger memory
t0 = time.time()
jobid = start_mergeFiles(nRoundtrips =nRoundtrips, workdir = root_dir + '/' + folder_name + '/', saveFilenamePrefix=nametag, dgrid = dgrid, dt = dt, Dpadt = 0)
    
all_done([jobid])
print('It takes ', time.time() - t0, ' seconds to finish merging files.')
