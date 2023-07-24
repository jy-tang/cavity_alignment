from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
from run_mirror_test import start_testMirror_stats
from run_genesis_sdf import *
from run_mergeFiles_sdf import *
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
w0 =40e-6
xlamds = 1.261043e-10
zsep = 140
c_speed  = 299792458
nslice = 1024
isradi = 1
npadt = (8196 - nslice//isradi)//2
npad1 = (256 - ncar)//2
npadx = [int(npad1), int(npad1) + 1]
dt = xlamds*zsep/c_speed
prad0 = 2e8
nametag = 't4'

root_dir = '/sdf/group/beamphysics/jytang/genesis/CBXFEL/'
folder_name = 'testMirror'

misalignQ = False
roughnessQ = False
theta = 800e-9
surface_filename = 'cavity_codes/synthetic_SPring-8'
# prepare error on crystals
# prepare error on crystals
if misalignQ:
    M1 = 2*theta*np.random.randn(1)[0] * 2*np.pi/xlamds
    M2 = 2*theta*np.random.randn(1)[0]* 2*np.pi/xlamds
    M3 = 2*theta*np.random.randn(1)[0] * 2*np.pi/xlamds
    M4 = 2*theta*np.random.randn(1)[0] * 2*np.pi/xlamds
    #400
    M1 = 64321.5573049364
    M2 = 9001.244586578989
    M3 = -45935.776549619804
    M4 = -8309.768204209093
    
    #600
    #M1 = 33510.93232610154 
    #M2 = -76538.80633504392 
    #M3 = -99136.96869056678
    #M4 = -95777.98526072034
    print('Misalignment on each mirror')
    print(M1, M2, M3, M4)
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
    print('Apply surface roughness')

else:
    C1 = C2 = C3 = C4 = None




    

with open(folder_name + '/'+nametag+'_recirc.txt', "w") as myfile:
    myfile.write("Round energy/uJ peakpower/GW tmean/fs trms/fs  tfwhm/fs xmean/um xrms/um  xfwhm/um xmean/um  yrms/um yfwhm/um \n")
with open(folder_name + '/'+nametag+'_transmit.txt', "w") as myfile:
    myfile.write("Round energy/uJ peakpower/GW tmean/fs trms/fs  tfwhm/fs  xmean/um xrms/um  xfwhm/um ymean/um yrms/um yfwhm/um \n")

   
     #---------------------Make Preparation---------------------------------------------------#
    # make a file to record the energy and size of radiation
    #if k ==0:
    #    with open(folder_name+"/" +record_init_name, "w") as myfile:
    #        myfile.write("energy/uJ peakpower/GW trms/fs  tfwhm/fs xrms/um  xfwhm/um yrms/um yfwhm/um \n")
    #    with open(folder_name+"/" +record_recir_name, "w") as myfile:
    #        myfile.write("energy/uJ peakpower/GW trms/fs  tfwhm/fs xrms/um  xfwhm/um yrms/um yfwhm/um \n")
    #    with open(folder_name+"/" +record_extract_name, "w") as myfile:
    #        myfile.write("energy/uJ peakpower/GW trms/fs  tfwhm/fs xrms/um  xfwhm/um yrms/um yfwhm/um \n")
    
    #   submit genesis job 
     
    #-------------------Prepare Seed Beam----------------------------------------------------#


        

t0 = time.time()
    #simulation (change dfl filename)
jobid, sim_name = start_simulation(folder_name = folder_name, dKbyK = 0.01,undKs = 1.172,und_period = 0.026,und_nperiods=130, nslice = nslice, zsep = zsep,
                                           nametag = nametag,gamma0 = np.around(8000./0.511,3), 
                                           Nf=17, Nt=15, emitnx = 0.3e-6, emitny = 0.3e-6,
                                           pulseLen = 60e-15, sigma = 20e-15, chirp = 0, Ipeak = 2e3,
                                           xlamds = xlamds,
                                           ipseed=np.random.randint(10000), prad0 = prad0)

all_done([jobid])

print('It takes ', time.time() - t0, ' seconds to finish Genesis. Start recirculation')



    
# do recirculation on all workers
t0 = time.time()
jobid = start_testMirror_stats(zsep = zsep, ncar = ncar, dgrid = dgrid, nslice = nslice, xlamds=xlamds,           # dfl params
                                 npadt = npadt, Dpadt = 0, npadx = npadx,isradi = isradi,       # padding params
                                  d = 100e-6,  # cavity params
                                  verboseQ = 1, # verbose params
                                 
                                misalignQ = misalignQ, M = M1, 
                             roughnessQ = roughnessQ, C = C1, 
                                 readfilename = root_dir + '/'+folder_name+'/'+sim_name + '.out.dfl' , 
                                       workdir = root_dir + '/' + folder_name + '/' , saveFilenamePrefix = nametag)
    
all_done([jobid])
print('It takes ', time.time() - t0, ' seconds to finish recirculation.')
  
    
# merge files for each roundtrip on nRoundtrips workers, with larger memory
t0 = time.time()
jobid = start_mergeFiles(nRoundtrips =0, workdir = root_dir + '/' + folder_name + '/', saveFilenamePrefix=nametag, dgrid = dgrid, dt = dt, Dpadt = 0)
    
all_done([jobid])
print('It takes ', time.time() - t0, ' seconds to finish merging files.')

