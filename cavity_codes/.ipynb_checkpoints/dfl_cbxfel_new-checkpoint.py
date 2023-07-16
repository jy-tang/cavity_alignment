import numpy as np
import time, os, sys
from cavity_codes.rfp2 import *
from cavity_codes.Bragg_mirror import *
import time
import pickle

def propagate_slice_kspace(field, z, xlamds, kx, ky):
    H = np.exp(-1j*xlamds*z*(kx**2 + ky**2)/(4*np.pi))
    return field*H

def Bragg_mirror_reflect(ncar, dgrid, xlamds, nslice, dt, npadx=0, 
                         verboseQ = True, showPlotQ = False, xlim = None, ylim = None):
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
    theta = theta_0 + Dtheta / 2. * np.linspace(-1.,1.,ncar+2*int(npadx))
    

   
    R0H = Bragg_mirror_reflection(eph, theta).T
    
    R00 = Bragg_mirror_transmission(eph, theta).T
        
    if verboseQ: print('took',time.time()-t0,'seconds to calculate Bragg filter')


    if showPlotQ:  # plot reflectivity and transmission
        
        # axes
        pi = np.pi; ncontours = 100
        thetaurad = 1e6*(theta-pi/4)
        Eph,Thetaurad = np.meshgrid(eph,thetaurad);
        
        
        # contour plots vs hw and kx
        absR2 = np.abs(R0H.T)**2
        absRT = np.abs(R00.T)**2
        print('absR2.shape =',absR2.shape)
        print('np.sum(np.isnan(absR2.reshape(-1))) =',np.sum(np.isnan(absR2.reshape(-1))))
        print('np.sum(absR2.reshape(-1)>0) =',np.sum(absR2.reshape(-1)>0))
        
        
        plt.figure(1)
        plt.contourf(Eph,Thetaurad,absR2,ncontours, label='reflectivity')
        #plt.contour(np.fft.fftshift(Eph),np.fft.fftshift(Thetaurad),absR2,3)
        plt.ylabel('Angle - 45 deg (urad)')
        plt.xlabel('Photon energy (eV)')
        
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        
        plt.colorbar()
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        plt.figure(2)
        plt.contourf(Eph,Thetaurad,absRT,ncontours, label='reflectivity')
        #plt.contour(np.fft.fftshift(Eph),np.fft.fftshift(Thetaurad),absR2,3)
        plt.ylabel('Angle - 45 deg (urad)')
        plt.xlabel('Photon energy (eV)')

        
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        
        plt.colorbar()
        plt.legend()
        plt.tight_layout()
        plt.show()
      
        
        # slice plot vs hw along kx=0
        x, _  = Eph.shape
        plt.figure(3)
        plt.plot(Eph[x//2,:],np.abs(R0H.T[x//2, :])**2,label='reflectivity')
        plt.plot(Eph[x//2, :],np.abs(R00.T[x//2, :])**2,label='transmission')
        plt.title(['Angle = 45 deg'])
        plt.xlabel('Photon energy (eV)')
        if xlim: 
            plt.xlim(xlim)
        plt.ylim([0,1])
        plt.legend()
        plt.tight_layout()
        plt.show()
        #fwhm1 = half_max_x(Eph[cut], np.abs(R.T[cut])**2)
        #print("FWHM is" + str(fwhm1) + 'eV')
        
    
    return R0H, R00


def propagate_slice(fld_slice, npadx,     # fld slice in spectral space, (Ek, x, y)
                             R00_slice, R0H_slice,     # Bragg reflection information
                             l_cavity, l_undulator, w_cavity,  # cavity parameter
                             lambd_slice, kx_mesh, ky_mesh):  #fld slice information
    
    # propagate one slice from Und end to Und start
    # take a slice in real space, unpadded, return a slice in real space, unpadded
    
     # focal length of the lens
    flens1 = (l_cavity + w_cavity)/2
    flens2 = (l_cavity + w_cavity)/2
    
    # propagation length in cavity
    z_und_start = (l_cavity - l_undulator)/2
    z_und_end = z_und_start + l_undulator
    

        
    # pad in x
    if npadx > 0:
        fld_slice = pad_dfl_slice_x(fld_slice, [int(npadx),int(npadx)])
        
    # fft to kx, ky space
    t0 = time.time()
    fld_slice = np.fft.fftshift(fft2(fld_slice), axes=(0,1))
    if verboseQ: print('took',time.time()-t0,'seconds for fft over x, y')
    
        
    # drift from undulator to M1
    Ldrift = l_cavity - z_und_end
        
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
    
        
    # reflect from M1
    fld_slice = np.einsum('i,ij->ij',R0H_slice,fld_slice)
    # trasmission through M1
    #fld_slice = np.einsum('i,ij->ij',R00_slice,fld_slice)
        
        
    # drift to the lens
    Ldrift = w_cavity/2
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
        
    # lens
    f = flens1
    #ifft to the real space
    fld_slice = ifft2(np.fft.ifftshift(fld_slice))
    #apply intracavity focusing CRL
    fld_slice *= np.exp(-1j*np.pi/(f*lambd_slice)*(xmesh**2 + ymesh**2))
    #fft to kx, ky space, check it!!!!
    fld_slice = np.fft.fftshift(fft2(fld_slice))
        
        
        
    # drift to M2
    Ldrift = w_cavity/2
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
        
    # reflect from M2
    fld_slice = np.einsum('i,ij->ij',np.flip(R0H_slice),fld_slice)
        
    # drift to M3
    Ldrift = l_cavity
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
    # reflect from M3
    fld_slice = np.einsum('i,ij->ij',np.flip(R0H_slice),fld_slice)
        
    # drift to lens
    Ldrift = w_cavity/2
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
    # lens
    f = flens2
    #ifft to the real space
    fld_slice = ifft2(np.fft.ifftshift(fld_slice))
    #apply intracavity focusing CRL
    fld_slice *= np.exp(-1j*np.pi/(f*lambd_slice)*(xmesh**2 + ymesh**2))
    #fft to kx, ky space, check it!!!!
    fld_slice = np.fft.fftshift(fft2(fld_slice))
        
    # drift to M4
    Ldrift = w_cavity/2
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
    # reflect from M4
    fld_slice = np.einsum('i,ij->ij',np.flip(R0H_slice),fld_slice)
        
    # drift to undulator start
    Ldrift = z_und_start
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
    # recirculation finished, ifft to real space
    fld_slice = ifft2(np.fft.ifftshift(fld_slice))
        
        
    # unpad in x
    if npadx > 0:
        fld_slice = unpad_dfl_slice_x(fld_slice,  [int(npadx),int(npadx)])

    return fld_slice

def recirculate_to_undulator(zsep, ncar, dgrid, xlamds=1.261043e-10,           # dfl params
                             npadt = 0, Dpadt = 0, npadx = 0,isradi = 1,       # padding params
                             l_undulator = 32*3.9, l_cavity = 149, w_cavity = 1,  # cavity params
                             showPlotQ = False, savePlotQ = False, verboseQ = 1, # plot params
                             roundripQ = False, nRoundtrips = 0,               # recirculation params
                             readfilename = None, writefilename = None):        # read and write
    
    t00 = time.time()
    
    h_Plank = 4.135667696e-15      # Plank constant [eV-sec]
    c_speed  = 299792458           # speed of light[m/sec]
    
    dt = xlamds*zsep * max(1,isradi) /c_speed
    #-------------------------------------------------------------------------------------------
    # read or make field 
    #------------------------------------------------------------------------------------------- 
    if readfilename == None:
        saveFilenamePrefix = 'test'
    else:
        saveFilenamePrefix = readfilename
        
    if readfilename == None:
        # make a new field
        t0 = time.time()
        fld = make_gaus_beam(ncar= ncar, dgrid=dgrid, w0=40e-6, dt=dt, nslice=1024, trms=10.e-15)
        print('took',time.time()-t0,'seconds total to make field with dimensions',fld.shape)
    
    else:
        # read dfl file on disk
        print('Reading in',readfilename)
        t0 = time.time()
        fld = read_dfl(readfilename, ncar=ncar,conjugate_field_for_genesis=False, swapxyQ=False) # read the field from disk
        print('took',time.time()-t0,'seconds total to read in and format the field with dimensions',fld.shape)
        fld = fld[::isradi,:,:]
        print("The shape before padding is ", fld.shape)
    
    if showPlotQ:
        # plot the imported field
        plot_fld_marginalize_t(fld, dgrid, dt=dt, saveFilename=saveFilenamePrefix+'_init_xy.png',showPlotQ=showPlotQ, savePlotQ = savePlotQ) 
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2, saveFilename=saveFilenamePrefix+'_init_tx.png',showPlotQ=showPlotQ, savePlotQ = savePlotQ)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1, saveFilename=saveFilenamePrefix+'_init_ty.png',showPlotQ=showPlotQ, savePlotQ = savePlotQ)
        plot_fld_power(fld, dt=dt, saveFilename=saveFilenamePrefix+'_init_t.png',showPlotQ=showPlotQ, savePlotQ = savePlotQ)
    
    
    energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm = fld_info(fld, dgrid = dgrid, dt=dt)
    
    init_field_info = [energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm]
    
    #--------------------------------------------------------------------------------------------------
    # fft in time domain to get spectral representaion
    #--------------------------------------------------------------------------------------------------
    # pad field in time
    if int(npadt) > 0:
        fld = pad_dfl_t(fld, [int(npadt),int(npadt)])
        if verboseQ: print('Padded field in time by',int(npadt),'slices (',dt*int(npadt)*1e15,'fs) at head and tail')
    nslice_padded, _, _ = fld.shape
    
    # plot the field after padding
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) 
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) 
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1)
    
    # fft
    t0 = time.time()
    fld = np.fft.fftshift(fft(fld, axis=0), axes=0)
    if verboseQ: print('took',time.time()-t0,'seconds for fft over t')
    
    #--------------------------------------------------------------------------
    # get coordinates after padding
    #---------------------------------------------------------------------------
    
    # get photon energy coordinate
    hw0_eV = h_Plank * c_speed / xlamds
    Dhw_eV = h_Plank / dt 
    eph = hw0_eV + Dhw_eV / 2. * np.linspace(-1.,1., nslice_padded)
    lambd = h_Plank*c_speed/eph
    
    # get kx,ky coordinates
    dx = 2. * dgrid / ncar
    Dkx = 2. * np.pi / dx
    kx = Dkx/ 2. * np.linspace(-1.,1.,ncar+2*int(npadx))
    ky = Dkx/ 2. * np.linspace(-1.,1.,ncar)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    kx_mesh = kx_mesh.T
    ky_mesh = ky_mesh.T
    
    # get x, y coordinates
    xs = (np.arange(ncar+2*int(npadx)) - np.floor((ncar+2*int(npadx))/2))*dx
    ys = (np.arange(ncar) - np.floor(ncar/2))*dx
    xmesh, ymesh = np.meshgrid(xs, ys)
    xmesh = xmesh.T
    ymesh = ymesh.T
    
    #---------------------------------------------------------------------------------------------------
    # get Bragg mirror response
    #---------------------------------------------------------------------------------------------------
    
    # focal length of the lens
    flens1 = (l_cavity + w_cavity)/2
    flens2 = (l_cavity + w_cavity)/2
    
    # propagation length in cavity
    z_und_start = (l_cavity - l_undulator)/2
    z_und_end = z_und_start + l_undulator
    
    
    # get Bragg mirror response matrix
    R0H, R00 = Bragg_mirror_reflect(ncar = ncar, dgrid = dgrid, xlamds = xlamds, nslice = nslice_padded, dt = dt, npadx=npadx, 
                         verboseQ = True, showPlotQ = showPlotQ, xlim = [9831,9833], ylim = [-10, 10])
    
    #---------------------------------------------------------------------------------------------------
    # propagate through cavity to return to undulator
    # TODO: parallelize, roundtrip, angular error, wavefront distort
    #---------------------------------------------------------------------------------------------------
    # propagate slice by slice
    for k in range(nslice_padded):   
        print('start to propagate slice ' + str(k))
        # take the frequency slice
        fld_slice = np.squeeze(fld[k, :, :])
        
        # pad in x
        if npadx > 0:
            fld_slice = pad_dfl_slice_x(fld_slice, [int(npadx),int(npadx)])
        
        # fft to kx, ky space
        t0 = time.time()
        fld_slice = np.fft.fftshift(fft2(fld_slice), axes=(0,1))
        if verboseQ: print('took',time.time()-t0,'seconds for fft over x, y')
        
        # take the reflectivity and transmission slice
        R00_slice = np.squeeze(R00[k, :])
        R0H_slice = np.squeeze(R0H[k, :])
        
        # drift from undulator to M1
        Ldrift = l_cavity - z_und_end
        
        fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd[k], kx = kx_mesh, ky = ky_mesh)
        
    
        
        # reflect from M1
        fld_slice = np.einsum('i,ij->ij',R0H_slice,fld_slice)
        # trasmission through M1
        #fld_slice = np.einsum('i,ij->ij',R00_slice,fld_slice)
        
        
        # drift to the lens
        Ldrift = w_cavity/2
        fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd[k], kx = kx_mesh, ky = ky_mesh)
        
        
        # lens
        f = flens1
        #ifft to the real space
        fld_slice = ifft2(np.fft.ifftshift(fld_slice))
        #apply intracavity focusing CRL
        fld_slice *= np.exp(-1j*np.pi/(f*lambd[k])*(xmesh**2 + ymesh**2))
        #fft to kx, ky space, check it!!!!
        fld_slice = np.fft.fftshift(fft2(fld_slice))
        
        
        
        # drift to M2
        Ldrift = w_cavity/2
        fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd[k], kx = kx_mesh, ky = ky_mesh)
        
        
        # reflect from M2
        fld_slice = np.einsum('i,ij->ij',np.flip(R0H_slice),fld_slice)
        
        # drift to M3
        Ldrift = l_cavity
        fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd[k], kx = kx_mesh, ky = ky_mesh)
        
        # reflect from M3
        fld_slice = np.einsum('i,ij->ij',np.flip(R0H_slice),fld_slice)
        
        # drift to lens
        Ldrift = w_cavity/2
        fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd[k], kx = kx_mesh, ky = ky_mesh)
        
        # lens
        f = flens2
        #ifft to the real space
        fld_slice = ifft2(np.fft.ifftshift(fld_slice))
        #apply intracavity focusing CRL
        fld_slice *= np.exp(-1j*np.pi/(f*lambd[k])*(xmesh**2 + ymesh**2))
        #fft to kx, ky space, check it!!!!
        fld_slice = np.fft.fftshift(fft2(fld_slice))
        
        # drift to M4
        Ldrift = w_cavity/2
        fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd[k], kx = kx_mesh, ky = ky_mesh)
        
        # reflect from M4
        fld_slice = np.einsum('i,ij->ij',np.flip(R0H_slice),fld_slice)
        
        # drift to undulator start
        Ldrift = z_und_start
        fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd[k], kx = kx_mesh, ky = ky_mesh)
        
        # recirculation finished, ifft to real space
        fld_slice = ifft2(np.fft.ifftshift(fld_slice))
        
        
        # unpad in x
        if npadx > 0:
            fld_slice = unpad_dfl_slice_x(fld_slice,  [int(npadx),int(npadx)])
        
        # record
        fld[k,:, :] = fld_slice
        
    #--------------------------------------------------------------------------------------------------
    # ifft to time domain
    #--------------------------------------------------------------------------------------------------
    t0 = time.time()
    fld = ifft(np.fft.ifftshift(fld,axes = 0), axis=0)
    if verboseQ: print('took',time.time()-t0,'seconds for ifft over t')
    
    #--------------------------------------------------------------------------------------------------
    # Dpadt in time
    #--------------------------------------------------------------------------------------------------
    if int(Dpadt) > 0:
        
        fld = unpad_dfl_t(fld, [int(Dpadt), int(Dpadt)])
        print("shape of fld after unpadding is ", fld.shape)
 
        if verboseQ: print('Removed padding of ',dt*int(npadt)*1e15,'fs in time from head and tail of field')
        
    
    
    #--------------------------------------------------------------------------------------------------   
    # plot the final result and write results
    #--------------------------------------------------------------------------------------------------
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1)
    
    # write field to disk
    if readfilename != None and writefilename != None:
        print('Writing to',writefilename)
            #writefilename = readfilename + 'r'
        write_dfl(fld, writefilename,conjugate_field_for_genesis = False,swapxyQ=False)
        
    print('It takes ' + str(time.time() - t00) + ' seconds to finish the recirculation.')    
            
    return fld 
        
        
        
    
    