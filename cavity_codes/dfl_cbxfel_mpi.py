import numpy as np
import time, os, sys
from rfp2 import *
from Bragg_mirror import *
import time
from mpi4py import MPI
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


def propagate_slice(fld_slice, npadx,     # fld slice in spectral space, (Ek, x, y)
                             R00_slice, R0H_slice, R00_slice_2, R0H_slice_2,    # Bragg reflection information
                             l_cavity, l_undulator, w_cavity,  # cavity parameter
                             lambd_slice, kx_mesh, ky_mesh, xmesh, ymesh, #fld slice information
                             roundtripQ,               # recirculation parameter
                             n_theta_shift1 = 0, n_theta_shift2 =0, n_theta_shift3 = 0, n_theta_shift4 = 0,   #yaw angle error on each of the mirror
                             x1 = 0, x2 = 0,                   # displacement error of the two lenses
                             verboseQ = False):
    
    # propagate one slice around the cavity
    # take a slice in real space, unpadded, return a slice in real space, unpadded
    # pycharm test
    
     # focal length of the lens
    flens1 = (l_cavity + w_cavity)/2
    flens2 = (l_cavity + w_cavity)/2
    
    #flens1 = float('inf')
    #flens2 = float('inf')
    
    # propagation length in cavity
    z_und_start = (l_cavity - l_undulator)/2
    z_und_end = z_und_start + l_undulator
    

        
    # pad in x
    if np.sum(npadx) > 0:
        fld_slice = pad_dfl_slice_x(fld_slice, npadx)
   
    # fft to kx, ky space
    t0 = time.time()
    fld_slice = np.fft.fftshift(fft2(fld_slice), axes=(0,1))
    if verboseQ: print('took',time.time()-t0,'seconds for fft over x, y')
    
    # if a roundtrip, propagate from UNDSTART to UNDEND
    if roundtripQ:
        fld_slice = propagate_slice_kspace(field = fld_slice, z = l_undulator, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
    # drift from UNDEND to M1
    Ldrift = l_cavity - z_und_end
        
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)

    
    # trasmission through M1
    fld_slice_transmit = np.einsum('i,ij->ij',R00_slice,fld_slice)
        
    # reflect from M1
    fld_slice = np.einsum('i,ij->ij',R0H_slice,fld_slice)
    
    # Add yaw error of M1
    if n_theta_shift1 != 0:
        print("Applying angular error on M1")
        fld_slice = np.roll(fld_slice, n_theta_shift1, axis=0)
        
    # drift to the lens
    Ldrift = w_cavity/2
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
        
    # lens
    f = flens1
    #ifft to the real space
    fld_slice = ifft2(np.fft.ifftshift(fld_slice))
    #apply intracavity focusing CRL, with displacement d1
    fld_slice *= np.exp(-1j*np.pi/(f*lambd_slice)*((xmesh - x1)**2 + ymesh**2))
    #fft to kx, ky space, check it!!!!
    fld_slice = np.fft.fftshift(fft2(fld_slice))
   
        
        
    # drift to M2
    Ldrift = w_cavity/2
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)

    # reflect from M2
    fld_slice = np.einsum('i,ij->ij',np.flip(R0H_slice_2),fld_slice)

    # Add yaw error of M2
    if n_theta_shift2 != 0:
        print("Applying angular error on M2")
        fld_slice = np.roll(fld_slice, n_theta_shift2, axis=0)


    # drift to M3
    Ldrift = l_cavity
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)

    # reflect from M3
    fld_slice = np.einsum('i,ij->ij',R0H_slice_2,fld_slice)

    # Add yaw error of M3
    if n_theta_shift3 != 0:
        print("Applying angular error on M3")
        fld_slice = np.roll(fld_slice, n_theta_shift3, axis=0)



    # drift to lens
    Ldrift = w_cavity/2
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
    # lens
    f = flens2
    #ifft to the real space
    fld_slice = ifft2(np.fft.ifftshift(fld_slice))
    #apply intracavity focusing CRL
    fld_slice *= np.exp(-1j*np.pi/(f*lambd_slice)*((xmesh - x2)**2 + ymesh**2))
    #fft to kx, ky space, check it!!!!
    fld_slice = np.fft.fftshift(fft2(fld_slice))
  
    # drift to M4
    Ldrift = w_cavity/2
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
    # reflect from M4
    fld_slice = np.einsum('i,ij->ij',np.flip(R0H_slice_2),fld_slice)

    # Add yaw error of M4
    if n_theta_shift4 != 0:
        print("Applying angular error on M4")
        fld_slice = np.roll(fld_slice, n_theta_shift4, axis=0)


    # drift to undulator start
    Ldrift = z_und_start
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
    # recirculation finished, ifft to real space
    fld_slice = ifft2(np.fft.ifftshift(fld_slice))
    fld_slice_transmit = ifft2(np.fft.ifftshift(fld_slice_transmit))
   
        
    # unpad in x
    if np.sum(npadx) > 0:
        fld_slice = unpad_dfl_slice_x(fld_slice,  npadx)
        fld_slice_transmit = unpad_dfl_slice_x(fld_slice_transmit,  npadx)

    
    return fld_slice, fld_slice_transmit

def recirculate_to_undulator_mpi(zsep, ncar, dgrid, nslice, xlamds=1.261043e-10,  w0=40e-6,  trms=10.e-15,  peak_power = 1e10,    # dfl params
                             npadt = 0, Dpadt = 0, npadx = [0,0],isradi = 1,       # padding params
                             l_undulator = 32*3.9, l_cavity = 149, w_cavity = 1, d1 = 100e-6, d2 = 100e-6,  # cavity params
                             verboseQ = 1,    # verbose params
                             nRoundtrips = 0,                     # recirculation params
                             n_theta_shift1 = 0, n_theta_shift2 =0, n_theta_shift3 = 0, n_theta_shift4 = 0,   #yaw angle error on each of the mirror
                             x1 = 0, x2 = 0,                   # displacement error of the two lenses
                             readfilename = None, workdir = None, saveFilenamePrefix = None):        # read and write
    
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
    
    
    #------------------------------
    # get the size of each sub-task
    #------------------------------
    ave, res = divmod(nslice_padded, nprocs)
    count = [ave + 1 if p < res else ave for p in range(nprocs)]
    count = np.array(count)
    count_sum = [sum(count[:p]) for p in range(nprocs)]
    displ = [sum(count[:p])*nx*ny for p in range(nprocs)]
        
    if rank == 0:
        print("Input data split into vectors of sizes %s" %count)
        print("Input data split with displacements of %s" %displ)
        print("Input data split with displacements in row of %s" %count_sum)
    
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

    R0H, R00 = Bragg_mirror_reflect(ncar = ncar, dgrid = dgrid, xlamds = xlamds, nslice = nslice_padded, dt = dt, npadx=npadx, 
                             verboseQ = True,  d = d1)   #first mirror
    R0H_2, R00_2 = Bragg_mirror_reflect(ncar = ncar, dgrid = dgrid, xlamds = xlamds, nslice = nslice_padded, dt = dt, npadx=npadx, 
                             verboseQ = True,  d = d2)   #2-4 mirror
    
    
    #-------------------------------------------------------------------------------------------
    # read or make field on root node
    #------------------------------------------------------------------------------------------- 
    if not workdir:
        workdir = '.'
    
    if not saveFilenamePrefix:
        saveFilenamePrefix = 'test'
    
    if rank == 0:                    
        if readfilename == None:
            # make a new field
            t0 = time.time()
            fld = make_gaus_beam(ncar= ncar, dgrid=dgrid, w0=w0, dt=dt, nslice=nslice, trms=trms)
            fld *= np.sqrt(peak_power/np.max(np.sum(np.abs(fld)**2, axis = (1,2))))
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

        

        #energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm = fld_info(fld, dgrid = dgrid, dt=dt, verbose = True)

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
        
    #-------------------------------------------------------------------------------------------
    # create variables on other nodes
    #------------------------------------------------------------------------------------------- 
    else:
        fld = None
        # initialize count on worker processes
        #count = np.zeros(nprocs, dtype=np.int)
        #displ = None
    
    # broadcast count
    #comm.Bcast(count, root=0)
    #comm.Barrier()
    
    #---------------------------------------------------------------------------------------------
    # scatter fld along first axis to all workers
    #---------------------------------------------------------------------------------------------
    
    # initialize recvbuf on all processes
    recvbuf_real  = np.zeros((count[rank],nx,ny))
    recvbuf_imag  = np.zeros((count[rank],nx,ny))
    
    comm.Scatterv([np.ascontiguousarray(np.real(fld)), count*nx*ny, displ, MPI.DOUBLE], recvbuf_real, root=0)
    comm.Barrier()
    
    comm.Scatterv([np.ascontiguousarray(np.imag(fld)), count*nx*ny, displ, MPI.DOUBLE], recvbuf_imag, root=0)
    comm.Barrier()
    
    #print('After Scatterv, process {} has real data shape :'.format(rank), recvbuf_real.shape)
    #print('After Scatterv, process {} has imag data shape :'.format(rank), recvbuf_imag.shape)
    
    comm.Barrier()
    
    # put together the real and imag data
    fld_block = recvbuf_real + 1j* recvbuf_imag 
    fld_block_transmit = np.zeros(fld_block.shape, dtype = complex)
    
    
    recvbuf_real = None
    recvbuf_imag = None
    
    
    #---------------------------------------------------------------------------------------------------
    # propagate slice by slice
    #---------------------------------------------------------------------------------------------------
    
    # first round from Undstart to Undend
    t0 = time.time()
    for k in range(fld_block.shape[0]):   
        #if k%50 == 0:    
            #print("worker " + str(rank) + " finished "+str(np.round(k/fld_block.shape[0],2)*100) + " % of the job")
        
        # take the frequency slice
        fld_slice = np.squeeze(fld_block[k, :, :])
        
        # take the reflectivity and transmission slice
        ind0 = count_sum[rank]
        R00_slice = np.squeeze(R00[ind0+k, :])
        R0H_slice = np.squeeze(R0H[ind0+k, :])
        R00_slice_2 = np.squeeze(R00_2[ind0+k, :])
        R0H_slice_2 = np.squeeze(R0H_2[ind0+k, :])
        lambd_slice = lambd[ind0+k]
        
       
        
        # propagate the slice from und end to und start
        fld_slice, fld_slice_transmit = propagate_slice(fld_slice = fld_slice, npadx = npadx,     
                             R00_slice = R00_slice, R0H_slice = R0H_slice,  R00_slice_2 = R00_slice_2, R0H_slice_2 = R0H_slice_2,    
                             l_cavity = l_cavity, l_undulator = l_undulator, w_cavity = w_cavity,  
                             lambd_slice = lambd_slice, kx_mesh = kx_mesh, ky_mesh = ky_mesh, xmesh = xmesh, ymesh = ymesh, 
                             roundtripQ = False,
                             n_theta_shift1 = n_theta_shift1, n_theta_shift2 =n_theta_shift2, n_theta_shift3 = n_theta_shift3, n_theta_shift4 = n_theta_shift4,  #yaw angle error on each of the mirror
                             x1 = x1, x2 = x2, verboseQ = False)
       
        # record the current slice
        fld_block[k,:, :] = fld_slice
        fld_block_transmit[k, :, :] = fld_slice_transmit
    
    pickle.dump(fld_block, open(workdir + '/'+saveFilenamePrefix +"_block"+str(rank)+"_round0.p", "wb" ) )
    pickle.dump(fld_block_transmit, open(workdir + '/'+saveFilenamePrefix +"_block_transmit_"+str(rank)+"_round0.p", "wb" ) )
    
    #For additional roundtrips
    for l in range(nRoundtrips):
        for k in range(fld_block.shape[0]):
            # take the frequency slice
            fld_slice = np.squeeze(fld_block[k, :, :])
        
            # take the reflectivity and transmission slice
            ind0 = count_sum[rank]
            R00_slice = np.squeeze(R00[ind0+k, :])
            R0H_slice = np.squeeze(R0H[ind0+k, :])
            lambd_slice = lambd[ind0+k]
        
       
        
           # propagate the slice from und start to und start
            fld_slice, fld_slice_transmit = propagate_slice(fld_slice = fld_slice, npadx = npadx,
                                                            R00_slice = R00_slice, R0H_slice = R0H_slice, R00_slice_2 = R00_slice_2, R0H_slice_2 = R0H_slice_2,
                                                            l_cavity = l_cavity, l_undulator = l_undulator, w_cavity = w_cavity,
                                                            lambd_slice = lambd_slice, kx_mesh = kx_mesh, ky_mesh = ky_mesh, xmesh = xmesh, ymesh = ymesh,
                                                            roundtripQ = True,
                                                            n_theta_shift1=n_theta_shift1,
                                                            n_theta_shift2=n_theta_shift2,
                                                            n_theta_shift3=n_theta_shift3,
                                                            n_theta_shift4=n_theta_shift4, # yaw angle error on each of the mirror
                                                            verboseQ = False)
       
            # record the current slice
            fld_block[k,:, :] = fld_slice
            fld_block_transmit[k, :, :] = fld_slice_transmit
        
        pickle.dump(fld_block, open(workdir + '/'+saveFilenamePrefix + "_block"+str(rank)+"_round"+str(l+1)+".p", "wb" ) ) 
        pickle.dump(fld_block_transmit, open(workdir + '/'+saveFilenamePrefix +"_block_transmit_"+str(rank)+"_round"+str(l+1) + ".p", "wb" ) )
            
        
       
    if rank < nprocs - 1:
        print('slice #', count_sum[rank], ' to slice #', count_sum[rank+1], ' finished by worker ' ,rank, ', took',time.time()-t0, 'seconds')
    else:
        print('slice #', count_sum[rank], ' to slice #', np.sum(count),  ' finished by worker ', rank, ', took',time.time()-t0, 'seconds')
    
    
    comm.Barrier()
    
    gc.collect()
    

params_dic = pickle.load( open( "params.p", "rb" ) )        
recirculate_to_undulator_mpi(**params_dic) 
        
#if __name__ == '__main__':
#    ncar = 181
#    dgrid = 540e-6
#    w0 =40e-6
#    xlamds = 1.261043e-10
#    zsep = 80
#    c_speed  = 299792458
#    nslice = 1780
#    isradi = 2
#    npadt = (8192 - nslice//isradi)//2
#    npad1 = (1024-ncar)//2
#    npadx = [int(npad1), int(npad1) + 1]
#    verbosity = True
    
        
#    recirculate_to_undulator_mpi(zsep = zsep, ncar = ncar, dgrid = dgrid, nslice = nslice, xlamds=1.261043e-10,           # dfl params
#                                 npadt = npadt, Dpadt = 0, npadx = npadx,isradi = isradi,       # padding params
#                                 l_undulator = 32*3.9, l_cavity = 149, w_cavity = 1,  # cavity params
#                                  verboseQ = 1, # verbose params
#                                 nRoundtrips = 10,               # recirculation params
#                                 readfilename ='/sdf/home/j/jytang/beamphysics/genesis/CBXFEL/data_long/tap0.03_K1.172_nt28_nf4.out.dfl', 
#                                       workdir = '/sdf/home/j/jytang/beamphysics/genesis/CBXFEL/data_long/', saveFilenamePrefix = 'n0')
    
    

