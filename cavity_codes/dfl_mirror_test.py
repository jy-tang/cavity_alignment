#Author:
#    J. Tang

import numpy as np
import time, os, sys
from rfp2 import *
from Bragg_mirror import *
import time
import psutil
import pickle
from pathlib import Path
import gc

def propagate_slice_kspace(field, z, xlamds, kx, ky):
    H = np.exp(-1j*xlamds*z*(kx**2 + ky**2)/(4*np.pi))
    return field*H

def Bragg_mirror_reflect(ncar, dgrid, xlamds, nslice, dt, npadx=[0, 0], 
                         verboseQ = True,  d = 50e-6):
    t0 = time.time()
    
    h_Plank = 4.135667696e-15;      # Plank constant [eV-sec]
    c_speed  = 299792458;           # speed of light[m/sec]
    
    # get photon energy coordinate
    hw0_eV = h_Plank * c_speed / xlamds
    Dhw_eV = h_Plank / dt 
    dhw_eV = Dhw_eV / (nslice - 1.)
    eph = hw0_eV + Dhw_eV / 2. * np.linspace(-1.,1.,nslice)
    
    # get transverse angle coordinate
    theta_0 = 45.0*np.pi/180.
    dx = 2. * dgrid / ncar
    Dkx = 2. * np.pi / dx
    Dtheta = Dkx * xlamds / 2. / np.pi
    theta = theta_0 + Dtheta / 2. * np.linspace(-1.,1.,ncar+int(npadx[0]) + int(npadx[1]))


   
    R0H = Bragg_mirror_reflection(eph, theta, d).T
    
    R00 = Bragg_mirror_transmission(eph, theta, d).T
        
    if verboseQ: print('took',time.time()-t0,'seconds to calculate Bragg filter')
        
    
    return R0H, R00


def propagate_slice_1mirror(fld_slice, npadx,     # fld slice in spectral space, (Ek, x, y)
                             R00_slice, R0H_slice,     # Bragg reflection information
                             lambd_slice, kx_mesh, ky_mesh, xmesh, ymesh, #fld slice information
                             Ldrift,          # Add a drift
                             n_theta_shift,  # Angular misalignment
                             verboseQ): 
    
    # propagate one slice, reflect and transmit through a Bragg Mirror
    # take a slice in real space, unpadded, return a slice in real space, unpadded
        
    # pad in x
    if np.sum(npadx) > 0:
        fld_slice = pad_dfl_slice_x(fld_slice, npadx)
   
    # fft to kx, ky space
    t0 = time.time()
    fld_slice = np.fft.fftshift(fft2(fld_slice), axes=(0,1))
    if verboseQ: print('took',time.time()-t0,'seconds for fft over x, y')
        
    # reflect from mirror
    fld_slice = np.einsum('i,ij->ij',R0H_slice,fld_slice)
    
    # add error of M1 in k space
    if n_theta_shift != 0:
        fld_slice = np.roll(fld_slice, n_theta_shift, axis = 0)

    # drift
    if Ldrift > 0:
        fld_slice = propagate_slice_kspace(field=fld_slice, z=Ldrift, xlamds=lambd_slice, kx=kx_mesh, ky=ky_mesh)

    # recirculation finished, ifft to real space
    fld_slice = ifft2(np.fft.ifftshift(fld_slice))

   
        
    # unpad in x
    if np.sum(npadx) > 0:
        fld_slice = unpad_dfl_slice_x(fld_slice,  npadx)




    return fld_slice

def propagate_1mirror_mpi(zsep, ncar, dgrid, nslice, xlamds=1.261043e-10,           # dfl params
                             npadt = 0, Dpadt = 0, npadx = [0,0],isradi = 1,       # padding params
                             d = 100e-6,  # crystal thickness
                             Ldrift = 0,
                             verboseQ = 1,    # verbose params
                             n_theta_shift = 0,        # misalign params
                             readfilename = None, saveFilenamePrefix = None, workdir = None):        # read and write
    
    t00 = time.time()
    
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    
    
    h_Plank = 4.135667696e-15      # Plank constant [eV-sec]
    c_speed  = 299792458           # speed of light[m/sec]
    
    dt = xlamds*zsep * max(1,isradi) /c_speed
    
    nslice_padded = nslice//max(1,isradi) + 2*int(npadt)
    nx = ny = ncar
    nx_padded = ncar + int(npadx[0]) + int(npadx[1])

    #-------------------------------
    # get coordinates after padding
    #-------------------------------

    # get photon energy coordinate
    hw0_eV = h_Plank * c_speed / xlamds
    Dhw_eV = h_Plank / dt 
    eph = hw0_eV + Dhw_eV / 2. * np.linspace(-1.,1., nslice_padded)
    lambd = h_Plank*c_speed/eph

    # get kx,ky coordinates
    dx = 2. * dgrid / ncar
    Dkx = 2. * np.pi / dx


    kx = Dkx/ 2. * np.linspace(-1.,1.,nx_padded)
    ky = Dkx/ 2. * np.linspace(-1.,1.,ny)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    kx_mesh = kx_mesh.T
    ky_mesh = ky_mesh.T

    # get x, y coordinates
    xs = (np.arange(nx_padded) - np.floor(nx_padded/2))*dx
    ys = (np.arange(ny) - np.floor(ny/2))*dx
    xmesh, ymesh = np.meshgrid(xs, ys)
    xmesh = xmesh.T
    ymesh = ymesh.T

    #----------------------------
    # get Bragg mirror response
    #----------------------------    

    R0H, R00 = Bragg_mirror_reflect(ncar = ncar, dgrid = dgrid, xlamds = xlamds, nslice = nslice_padded, dt = dt, npadx=npadx,  verboseQ = True,  d = d)   #first mirror
   
    
    
    #-------------------------------------------------------------------------------------------
    # read or make field on root node
    #------------------------------------------------------------------------------------------- 
    if not workdir:
        workdir = '.'
    
    if not saveFilenamePrefix:
        saveFilenamePrefix   = 'test'

    if readfilename == None:
        # make a new field
        t0 = time.time()
        fld = make_gaus_beam(ncar= ncar, dgrid=dgrid, w0=40e-6, dt=dt, nslice=nslice, trms=20.e-15)
        fld *= np.sqrt(1e9/np.max(np.sum(np.abs(fld)**2, axis = (1,2))))
        print('took',time.time()-t0,'seconds total to make field with dimensions',fld.shape)
        fld = fld[::isradi,:,:]
        print("fld shape after downsample ", fld.shape)
    else:
        # read dfl file on disk
        print('Reading in',readfilename)
        t0 = time.time()
        fld = read_dfl(readfilename, ncar=ncar,conjugate_field_for_genesis=False, swapxyQ=False) # read the field from disk
        print('took',time.time()-t0,'seconds total to read in and format the field with dimensions',fld.shape)
        fld = fld[:nslice,:,:]
        print('fld shape after truncation ', fld.shape)
        fld = fld[::isradi,:,:]
        print("fld shape after downsample ", fld.shape)

        


    #init_field_info = [energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm]

    #--------------------------------------------------
    # fft in time domain to get spectral representaion
    #--------------------------------------------------
    # pad field in time
    if int(npadt) > 0:
        fld = pad_dfl_t(fld, [int(npadt),int(npadt)])
        if verboseQ: print('Padded field in time by',int(npadt),'slices (',dt*int(npadt)*1e15,'fs) at head and tail')
    #nslice_padded, nx, ny = fld.shape
    if verboseQ:
        print("after padding, fld shape " + str(fld.shape))
    # fft
    t0 = time.time()
    fld = np.fft.fftshift(fft(fld, axis=0), axes=0)
    if verboseQ: print('took',time.time()-t0,'seconds for fft over t')

    #---------------------------------------------------------------------------------------------------
    # propagate slice by slice 
    #---------------------------------------------------------------------------------------------------
    
    
    
    # first round from Undstart to Undend
    t0 = time.time()
    for k in range(fld.shape[0]):
        #if k%50 == 0:    
            #print("worker " + str(rank) + " finished "+str(np.round(k/fld_block.shape[0],2)*100) + " % of the job")
        
        # take the frequency slice
        fld_slice = np.squeeze(fld_block[k, :, :])
        
        # take the reflectivity and transmission slice
        R00_slice = np.squeeze(R00[k, :])
        R0H_slice = np.squeeze(R0H[k, :])
        lambd_slice = lambd[k]
        
       
        
        # propagate the slice from und end to und start
        fld_slice = propagate_slice_1mirror(fld_slice = fld_slice, npadx = npadx,
                             R00_slice = R00_slice, R0H_slice = R0H_slice,
                             lambd_slice = lambd_slice, kx_mesh = kx_mesh, ky_mesh = ky_mesh, xmesh = xmesh, ymesh = ymesh, Ldrift = Ldrift,
                             n_theta_shift = n_theta_shift,       # misalignment parameter
                             verboseQ = False)
       
        # record the current slice
        fld[k,:, :] = fld_slice



    # ----------------------
    # ifft to time domain
    # ----------------------
    t0 = time.time()
    fld = ifft(np.fft.ifftshift(fld, axes=0), axis=0)
    if verboseQ: print('took', time.time() - t0, 'seconds for ifft over t')

    # ----------------
    # Dpadt in time
    # ----------------
    if int(Dpadt) > 0:

        fld = unpad_dfl_t(fld, [int(Dpadt), int(Dpadt)])
        print("shape of fld after unpadding is ", fld.shape)

        if verboseQ: print('Removed padding of ', dt * int(npadt) * 1e15, 'fs in time from head and tail of field')
    
    

    #-----------------------------------------
    #  write results
    #-----------------------------------------
    # write field to disk
    if seedfilename != None:
        print('Writing seed file to',seedfilename)
            #writefilename = readfilename + 'r'
        write_dfl(fld, seedfilename,conjugate_field_for_genesis = False,swapxyQ=False)
    print('It takes ' + str(time.time() - t00) + ' seconds to finish the recirculation.')





