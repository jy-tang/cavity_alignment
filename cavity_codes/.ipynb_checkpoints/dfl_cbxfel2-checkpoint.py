from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
from rfp2 import *
from Bragg_mirror import *
import SDDS
from sdds2genesis import match_to_FODO,sdds2genesis
import sys, os, csv, copy,shutil
#from get_memory import *
import time

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    x1 = lin_interp(x, y, zero_crossings_i[0], half),
    x2 = lin_interp(x, y, zero_crossings_i[1], half)
    return x2 - x1


def Bragg_mirror_reflect(fld, ncar, dgrid, xlamds, dt, npadx=0, showPlotQ=False, reflectQ=True, verboseQ=False, timeDomainQ=True, undo_slippageQ=False, kxspace_inQ=False, kxspace_outQ=False, slice_processing_relative_power_threshold=0, fliplr = False):
       
    #print("npadx = " + str(npadx))
    # dt is sample time of field
    h_Plank = 4.135667696e-15;      # Plank constant [eV-sec]
    c_speed  = 299792458;           # speed of light[m/sec]
    nslice = fld.shape[0]
    hw0_eV = h_Plank * c_speed / xlamds
    Dhw_eV = h_Plank / dt; dhw_eV = Dhw_eV / (nslice - 1.)
    eph0 = hw0_eV + Dhw_eV / 2. * np.linspace(-1.,1.,nslice)
    eph = np.fft.ifftshift(eph0)
    theta_0 = 45.0*np.pi/180.
    dx = 2. * dgrid / ncar
    Dkx = 2. * np.pi / dx
    Dtheta = Dkx * xlamds / 2. / np.pi
    theta0 = theta_0 + Dtheta / 2. * np.linspace(-1.,1.,ncar)
    theta = np.fft.ifftshift(theta0)

    # go to frequency domain
    if timeDomainQ: # we are in the time domain so go to frequency domain
        t0 = time.time()
        #fld = np.fft.fftshift(fft(fld, axis=0), axes=0)
        fld = fft(fld, axis=0)
        if verboseQ: print('took',time.time()-t0,'seconds for fft over t')
    # process only frequency slices with power
    if slice_processing_relative_power_threshold > 0:
        t0 = time.time()
        fld0 = fld
        omega_slice_selection = frequency_slice_selection(fld, slice_processing_relative_power_threshold, verboseQ=verboseQ)
        fld = fld[omega_slice_selection]
        eph = eph[omega_slice_selection]
        eph0 = eph0[np.fft.fftshift(omega_slice_selection)]
        if verboseQ: print('took',time.time()-t0,'seconds for selecting only',len(fld),'slices with power / max(power) >',slice_processing_relative_power_threshold,'for processing')
    # pad in x
    if int(npadx) > 0:
        if not kxspace_inQ and not kxspace_outQ:
            fld = pad_dfl_x(fld, [int(npadx),int(npadx)])
            # adjust resolution
            theta0 = theta_0 + Dtheta / 2. * np.linspace(-1.,1.,ncar+2*int(npadx))
            theta = np.fft.ifftshift(theta0)
        else:
            print('ERROR - Bragg_mirror_reflect: Cannot pad in x unless both kxspace_inQ (',kxspace_inQ,') and kxspace_outQ (',kxspace_outQ,') are False')
        
    # go to reciprocal space
    if not kxspace_inQ:
        t0 = time.time()
        #fld = np.fft.fftshift(fft(fld, axis=1), axes=1)
        fld = fft(fld, axis=1)
        if verboseQ: print('took',time.time()-t0,'seconds for fft over x')

    if showPlotQ:
        t0 = time.time()
        spectrum = np.sum(np.sum(np.abs(fld)**2, axis=1), axis=1)
        if verboseQ: print('took',time.time()-t0,'seconds to calculate spectrum')
    #if showPlotQ:
        #plt.plot(eph, spectrum)
        #plt.xlabel('Photon energy (eV)'); plt.ylabel('Spectral intensity')
        #plt.tight_layout(); plt.show()

    t0 = time.time()
    if reflectQ: 
        R = Bragg_mirror_reflection(eph, theta, undo_slippageQ=undo_slippageQ).T
        ylabel = 'Bragg diffraction intensity'
    else:
        R = Bragg_mirror_transmission(eph, theta).T
        ylabel = 'Forward diffraction intensity'
    if verboseQ: print('took',time.time()-t0,'seconds to calculate Bragg filter')
        
    if fliplr:
        R = np.fliplr(R)

    if showPlotQ:
        
        # axes
        pi = np.pi; ncontours = 100
        thetaurad = 1e6*(theta-pi/4)
        #print('angles span: ',min(thetaurad), max(thetaurad), 'urad')
        #print('photon energies span span: ',min(eph), max(eph),'eV')
        Eph,Thetaurad = np.meshgrid(eph,thetaurad);
        
        # moments
        iptf = np.fft.fftshift(spectrum)
        intensity_profile = np.fft.fftshift(np.sum(np.abs(fld)**2,axis=2).T)
        ipxf = np.sum(intensity_profile,axis=1) # might have the axes flipped here
        eph_mean = np.dot(eph0,iptf) / np.sum(iptf)
        eph_rms = np.sqrt(np.dot(eph0**2,iptf) / np.sum(iptf) - eph_mean**2)
        eph_lim = eph_mean + eph_rms * np.array([-1,1])
        thetaurad_mean = np.dot(thetaurad,ipxf) / np.sum(ipxf)
        thetaurad_rms = np.sqrt(np.dot(thetaurad**2,ipxf) / np.sum(ipxf) - thetaurad_mean**2)
        thetaurad_lim = thetaurad_mean + thetaurad_rms * np.array([-1,1])
        
        # contour plots vs hw and kx
        absR2 = np.fft.fftshift(np.abs(R.T)**2)
        print('absR2.shape =',absR2.shape)
        print('np.sum(np.isnan(absR2.reshape(-1))) =',np.sum(np.isnan(absR2.reshape(-1))))
        print('np.sum(absR2.reshape(-1)>0) =',np.sum(absR2.reshape(-1)>0))
        
        #extent=[min(eph),max(eph),min(thetaurad),max(thetaurad)]
        #aspect = (extent[1]-extent[0])/(extent[3]-extent[2])
        #plt.imshow(absR2,extent=extent,aspect=aspect, label='filter')
        #plt.colorbar(); plt.legend(); plt.tight_layout(); plt.show()
        
        #extent=[min(eph_lim),max(eph_lim),min(thetaurad_lim),max(thetaurad_lim)]
        #aspect = (extent[1]-extent[0])/(extent[3]-extent[2])
        #plt.imshow(absR2,extent=extent,aspect=aspect, label='filter')
        #plt.colorbar(); plt.legend(); plt.tight_layout(); plt.show()
        
        #plt.contourf(Eph,Thetaurad,absR2,ncontours, label='filter')
        #plt.colorbar(); plt.legend(); plt.tight_layout(); plt.show()
        
        #plt.contourf(Eph,Thetaurad,absR2,10, label='filter')
        #plt.colorbar(); plt.legend(); plt.tight_layout(); plt.show()
        
        #plt.contourf(Eph,Thetaurad,absR2,3, label='filter')
        #plt.colorbar(); plt.legend(); plt.tight_layout(); plt.show()
        
        plt.contourf(np.fft.fftshift(Eph),np.fft.fftshift(Thetaurad),absR2,ncontours, label='filter')
        plt.contour(np.fft.fftshift(Eph),np.fft.fftshift(Thetaurad),intensity_profile,5, label='radiation')
        plt.contour(np.fft.fftshift(Eph),np.fft.fftshift(Thetaurad),absR2,3)
        plt.ylabel('Angle - 45 deg (urad)')
        plt.xlabel('Photon energy (eV)')
        plt.title(ylabel); #plt.xlim(eph_lim); #plt.ylim(thetaurad_lim)
        plt.xlim([9831,9833])#plt.xlim(eph_lim);
        #plt.ylim(thetaurad_lim)
        plt.ylim([-10,10])
        
        #plt.xlim([9750,9900])
        plt.colorbar(); plt.legend(); plt.tight_layout(); plt.show()
      
        
        # slice plot vs hw along kx=0
        cut = Thetaurad == 0;
        plt.plot(np.fft.fftshift(Eph[cut]),np.fft.fftshift(np.abs(R.T[cut]))**2,label='filter')
        plt.plot(np.fft.fftshift(eph),np.fft.fftshift(spectrum/np.max(spectrum)),dashes=[2, 1],label='radiation')
        plt.title(['Angle = 45 deg'])
        plt.xlabel('Photon energy (eV)')
        plt.ylabel(ylabel); #plt.xlim(eph_lim)
        plt.xlim([9831,9833])
        #plt.xlim([9750,9900])
        plt.ylim([0,1])
        plt.legend(); plt.tight_layout(); plt.show()
        fwhm1 = half_max_x(np.fft.fftshift(Eph[cut]),np.fft.fftshift(np.abs(R.T[cut]))**2)
        print("FWHM is" + str(fwhm1) + 'eV')
        
        import pickle
        temp = (np.fft.fftshift(Eph[cut]), np.fft.fftshift(np.abs(R.T[cut]))**2)
        if reflectQ:
            outfile_name = "Brag_reflection"
        else:
            outfile_name = "Brag_transmission"
            
        outfile = open(outfile_name,'wb')
        pickle.dump(temp,outfile)
        outfile.close()

    # apply effect of mirror to field
    t0 = time.time()
    fld = np.einsum('ij,ijk->ijk',R,fld)
    if verboseQ: print('took',time.time()-t0,'seconds to apply Bragg filter')

    # return to real space
    if not kxspace_outQ:
        t0 = time.time()
        #fld = ifft(np.fft.ifftshift(fld, axes=1), axis=1)
        fld = ifft(fld, axis=1)
        if verboseQ: print('took',time.time()-t0,'seconds for ifft over x')
    # unpad in x
    if int(npadx) > 0 and not kxspace_inQ and not kxspace_outQ:
        fld = unpad_dfl_x(fld, [int(npadx),int(npadx)])
    # release processing mask
    if slice_processing_relative_power_threshold > 0:
        t0 = time.time()
        fld0 *= 0. # clear field
        fld0[omega_slice_selection] = fld # overwrite field with processed field
        fld = fld0
        if verboseQ: print('took',time.time()-t0,'seconds to release selection for',len(fld),'slices with power / max(power) >',slice_processing_relative_power_threshold,'for processing')
    if timeDomainQ: # we were in the time domain so return to time domain
        t0 = time.time()
        #fld = ifft(np.fft.ifftshift(fld, axes=0), axis=0)
        fld = ifft(fld, axis=0)
        if verboseQ: print('took',time.time()-t0,'seconds for ifft over t')
    
    return fld


def Bragg_mirror_transmit(fld, ncar, dgrid, xlamds, dt, npadx=0, showPlotQ=False, verboseQ=False, timeDomainQ=True, undo_slippageQ=False, kxspace_inQ=False, kxspace_outQ=False, slice_processing_relative_power_threshold=0):
    return Bragg_mirror_reflect(fld, ncar, dgrid, xlamds, dt, npadx=npadx, showPlotQ=showPlotQ, reflectQ=False, verboseQ=verboseQ, timeDomainQ=timeDomainQ, undo_slippageQ=undo_slippageQ, kxspace_inQ=kxspace_inQ, kxspace_outQ=kxspace_outQ, slice_processing_relative_power_threshold=slice_processing_relative_power_threshold)

# select a contiguous and symmetric region of the fft of the fld to process
def frequency_slice_selection(fld, slice_processing_relative_power_threshold = 1e-6, verboseQ=False):
        #slice_processing_relative_power_threshold = 1e-6 # only propagate slices where there's beam (0 disables)
        pows = np.sum(np.abs(fld)**2,axis=(1,2))
        omega_slice_selection = pows >= np.max(pows) * slice_processing_relative_power_threshold
        omega_slice_selection = np.fft.fftshift(omega_slice_selection)
        # find contiguous region containing beam
        bslo = omega_slice_selection.argmax()
        bshilo=np.flip(omega_slice_selection).argmax()
        bslo = max([bslo,bshilo])
        bshi=len(pows)-1-bslo
        omega_slice_selection = np.arange(len(pows))
        omega_slice_selection = (omega_slice_selection >= bslo) & (omega_slice_selection <= bshi)
        omega_slice_selection = np.fft.ifftshift(omega_slice_selection)
        if verboseQ:
            u0 = np.sum(pows); u1 = np.sum(pows[omega_slice_selection])
            print('INFO: frequency_slice_selection - Fraction of power lost is',1.-u1/u0,'for slice_processing_relative_power_threshold of',slice_processing_relative_power_threshold)
        return omega_slice_selection


# from undulator exit, through ring cavity, to undulator entrance
def cavity_return_to_undulator(fld,ncar, dgrid,  zsep, l_undulator, xlamds=1.261043e-10, l_cavity = 149, w_cavity = 1,
                               isradi = 1, skipTimeFFTsQ = 1, skipSpaceFFTsQ = 1, 
                               npadt = 0, Dpadt = 0, npadx = 0,  unpadtQ = True,
                               slice_processing_relative_power_threshold = 0,tjitter_rms = 5,
                               showPlotQ = 0, savePlotQ = 0, verbosity = 1):

    # npadt: total number of padding in t
    # Dpadt: npadt - Dpadt = nslice(1-1/isradi) . If isradi =1, Dpad = npadt 
    
    t0total = time.time()
    dt = xlamds * zsep * max(1,isradi) / 299792458
    
    flens1 = (l_cavity + w_cavity)/2
    flens2 = (l_cavity + w_cavity)/2
    
    
    z_und_start = (l_cavity - l_undulator)/2
    z_und_end = z_und_start + l_undulator
    
    # pad field in time
    if int(npadt) > 0:
        fld = pad_dfl_t(fld, [int(npadt),int(npadt)])
        if verbosity: print('Padded field in time by',int(npadt),'slices (',dt*int(npadt)*1e15,'fs) at head and tail')
    
    # plot the field
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field
    
    if skipTimeFFTsQ:
        timeDomainQ = 0
        t0 = time.time()
        #fld = np.fft.fftshift(fft(fld, axis=0), axes=0)
        fld = fft(fld, axis=0)
        if verbosity: print('took',time.time()-t0,'seconds for fft over t')
        undo_slippageQ = 1
    else:
        timeDomainQ = 1
        undo_slippageQ = 1

    # drift from undulator to first mirror
    Ldrift = l_cavity - z_und_end
    fld = rfp(fld, xlamds, dgrid, A=1, B=Ldrift, D=1, ncar=ncar, cutradius=0, dgridout=-1, kxspace_inQ=0, kxspace_outQ = skipSpaceFFTsQ, slice_processing_relative_power_threshold=slice_processing_relative_power_threshold)
    #fld = rfp_old(fld, xlamds, dgrid, A=1, B=8.706, D=1, ncar=ncar, cutradius=0, dgridout=1.001*dgrid, verboseQ=verbosity>2)
    #fld = rfp(fld, xlamds, dgrid, A=1, B=8.706, D=1, ncar=ncar, cutradius=0, dgridout=1.001*dgrid)
    # plot the field
    if verbosity: print('Field after',Ldrift,'m drift from undulator to first mirror')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field

        
    # reflect from the first Bragg mirror
    fld = Bragg_mirror_reflect(fld, ncar, dgrid, xlamds, dt, npadx=npadx, showPlotQ=showPlotQ, verboseQ=1, timeDomainQ=timeDomainQ, undo_slippageQ=undo_slippageQ, kxspace_inQ = skipSpaceFFTsQ, slice_processing_relative_power_threshold=slice_processing_relative_power_threshold)
    # plot the field
    if verbosity: print('Field after 1st Bragg mirror')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field


    # 0.5 m drift, 61 m focal length lens, 0.5 m drift
    #R = matprod([Rdrift(0.5),Rlens(61),Rdrift(0.5)])
    Ldrift1 = w_cavity/2; flens = flens1; Ldrift2 = w_cavity/2
    R = matprod([Rdrift(Ldrift1),Rlens(flens),Rdrift(Ldrift2)])
    fld = rfp(fld, xlamds, dgrid, A=R[0,0], B=R[0,1], D=R[1,1], ncar=ncar, cutradius=0, dgridout=-1, slice_processing_relative_power_threshold=slice_processing_relative_power_threshold)
    # plot the field
    if verbosity: print('Field after',Ldrift1,'m drift,',flens,'m focal length lens, and',Ldrift2,'m drift')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field


    # reflect from the second Bragg mirror
    fld = Bragg_mirror_reflect(fld, ncar, dgrid, xlamds, dt, npadx=npadx, showPlotQ=showPlotQ, verboseQ=1, timeDomainQ=timeDomainQ, undo_slippageQ=undo_slippageQ, kxspace_outQ = skipSpaceFFTsQ, slice_processing_relative_power_threshold=slice_processing_relative_power_threshold,fliplr = True)
    # plot the field
    if verbosity: print('Field after 2nd Bragg mirror')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field


    # drift 79 m
    Ldrift = l_cavity
    fld = rfp(fld, xlamds, dgrid, A=1, B=Ldrift, D=1, ncar=ncar, cutradius=0, dgridout=-1, kxspace_inQ = skipSpaceFFTsQ, kxspace_outQ = skipSpaceFFTsQ, slice_processing_relative_power_threshold=slice_processing_relative_power_threshold)
    # plot the field
    if verbosity: print('Field after',Ldrift,'m drift')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field


    # reflect from the third Bragg mirror
    fld = Bragg_mirror_reflect(fld = fld, ncar = ncar, dgrid = dgrid,xlamds =xlamds,dt = dt, showPlotQ=showPlotQ, verboseQ=1, timeDomainQ=timeDomainQ, undo_slippageQ=undo_slippageQ, kxspace_inQ = skipSpaceFFTsQ)
    # plot the field
    if verbosity: print('Field after 3rd Bragg mirror')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field


    # 0.5 m drift, 61 m focal length lens, 0.5 m drift
    Ldrift1 = w_cavity; flens = flens2; Ldrift2 = w_cavity
    R = matprod([Rdrift(Ldrift1),Rlens(flens),Rdrift(Ldrift2)])
    fld = rfp(fld, xlamds, dgrid, A=R[0,0], B=R[0,1], D=R[1,1], ncar=ncar, cutradius=0, dgridout=-1, slice_processing_relative_power_threshold=slice_processing_relative_power_threshold)
    # plot the field
    if verbosity: print('Field after',Ldrift1,'m drift,',flens,'m focal length lens, and',Ldrift2,'m drift')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1)


    # reflect from the fourth Bragg mirror
    fld = Bragg_mirror_reflect(fld, ncar, dgrid, xlamds, dt, npadx=npadx, showPlotQ=showPlotQ, verboseQ=1, timeDomainQ=timeDomainQ, undo_slippageQ=undo_slippageQ, kxspace_outQ = skipSpaceFFTsQ, slice_processing_relative_power_threshold=slice_processing_relative_power_threshold, fliplr = True)
    # plot the field
    if verbosity: print('Field after 4th Bragg mirror')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1)
        
    # drift 3.5 m
    Ldrift = z_und_start
    fld = rfp(fld, xlamds, dgrid, A=1, B=Ldrift, D=1, ncar=ncar, cutradius=0, dgridout=-1, kxspace_inQ = skipSpaceFFTsQ, slice_processing_relative_power_threshold=slice_processing_relative_power_threshold)
    # plot the field
    # plot the field
    if verbosity: print('Field drifted',Ldrift,'m from mirror to undulator start')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1)
        
    if skipTimeFFTsQ:
        t0 = time.time()
        #fld = np.fft.ifftshift(ifft(fld, axis=0), axes=0)
        fld = ifft(fld, axis=0)
        if verbosity: print('took',time.time()-t0,'seconds for ifft over t')
        timeDomainQ = 1
        undo_slippageQ = 0
        
    time_jitter = 0.   
     # upad field in time
    if int(Dpadt) > 0 and unpadtQ:
        
        time_jitter = np.random.normal(scale = tjitter_rms)
        print("apply time jitter "+str(time_jitter)+"fs")
        time_jitter *= 1e-15
        nshift = int(np.floor(time_jitter/dt))
        fld = unpad_dfl_t(fld, [int(Dpadt+nshift), int(Dpadt-nshift)])
        print("shape of fld after unpadding is ", fld.shape)
 
        if verbosity: print('Removed padding of ',dt*int(npadt)*1e15,'fs in time from head and tail of field')
        #if showPlotQ:
            #plot_fld_marginalize_t(fld, dgrid)
            #plot_fld_slice(fld, dgrid, dt=dt, slice=-2)
            #plot_fld_slice(fld, dgrid, dt=dt, slice=-1)

    print('Finished! It took',time.time()-t0total,'seconds total time to track radiation from undulator exit to undulator start')
    # plot the final result
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2)
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1)
    
    

    return fld, time_jitter
    
def cavity_exit_from_mirror(fld, ncar, dgrid, zsep, l_undulator, xlamds=1.261043e-10, l_cavity = 149, w_cavity = 1, 
                               isradi = 1, skipTimeFFTsQ = 1, skipSpaceFFTsQ = 1, showPlotQ = 0, savePlotQ = 0, verbosity = 1,
                           npadx = 0, npadt = 0,  unpadtQ = True,  slice_processing_relative_power_threshold = 0):

    dt = xlamds * zsep * max(1,isradi) / 299792458


    t0total = time.time()

    z_und_start = (l_cavity - l_undulator)/2
    z_und_end = z_und_start + l_undulator
    
     # pad field in time
    if int(npadt) > 0:
        fld = pad_dfl_t(fld, [int(npadt),int(npadt)])
        if verbosity: print('Padded field in time by',int(npadt),'slices (',dt*int(npadt)*1e15,'fs) at head and tail')

    # drift from undulator to first mirror
    Ldrift = l_cavity - z_und_end
    fld = rfp(fld, xlamds, dgrid, A=1, B=Ldrift, D=1, ncar=ncar, cutradius=0, dgridout=-1, kxspace_inQ=0, kxspace_outQ=skipSpaceFFTsQ, slice_processing_relative_power_threshold=slice_processing_relative_power_threshold)
    # plot the field
    if verbosity: print('Field after',Ldrift,'m drift from undulator to first mirror')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field


    # reflect from the Bragg mirror
    fld = Bragg_mirror_transmit(fld, ncar, dgrid, xlamds, dt, npadx=npadx, showPlotQ=showPlotQ, verboseQ=1, timeDomainQ=1, kxspace_inQ=skipSpaceFFTsQ, kxspace_outQ=0, slice_processing_relative_power_threshold=slice_processing_relative_power_threshold)
    # plot the field
    if verbosity: print('Field transmitted through 1st Bragg mirror')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1) # plot the imported field

    if int(npadt) > 0 and unpadtQ:
        fld = unpad_dfl_t(fld, [npadt, npadt])
        print("shape of fld after unpadding is ", fld.shape)
    
    return fld


########################################################


def recirculate_to_undulator(zsep, ncar, dgrid, l_undulator, xlamds=1.261043e-10,
                             npadt = 0, Dpadt = 0, npadx = 0, unpadtQ = True,
                             tjitter_rms = 5, slice_processing_relative_power_threshold = 0,
                                 isradi = 1, l_cavity = 149, w_cavity = 1,
                                 skipTimeFFTsQ = 1, skipSpaceFFTsQ = 1,
                                 showPlotQ = 0, savePlotQ = 0, verbosity = 1,
                                 readfilename = None, writefilename = None):

    dt = xlamds*zsep * max(1,isradi) / 299792458

    if readfilename == None:
        saveFilenamePrefix = 'test'
    else:
        saveFilenamePrefix = readfilename

    if readfilename == None:
        # make a new field
        t0 = time.time()
        dt = xlamds*zsep* max(1,isradi) / 299792458
        ncar = 201; dgrid =750e-6; dt*=1.; fld = make_gaus_beam(ncar= ncar, dgrid=dgrid, w0=80e-6, dt=dt, nslice=4096, trms=3.e-15)
        #settings['ncar']=121; settings['dgrid']=500e-6; dt*=10.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=1024, trms=3.e-15)
        #settings['ncar']=121; settings['dgrid']=200e-6; dt*=100.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=128, trms=3.e-15)
        print('took',time.time()-t0,'seconds total to make field with dimensions',fld.shape)
    else:
        ## import a field from a file on disk
        ##readfilename = '/nfs/slac/g/beamphysics/jytang/genesis/lasershaping2/cavity/flattop_flatbeam/gen_tap0.021_K1.1742_s0_a.out.dfl'
        #readfilename = '/gpfs/slac/staas/fs1/g/g.beamphysics/jytang/genesis/lasershaping2/cavity/flattop_flatbeam/gen_tap0.021_K1.1742_s0_a.out.dfl'
        ##readfilename = '/u/ra/jduris/code/genesis_dfl_tools/rfp_radiation_field_propagator/myfile.dfl'
        ##readfilename = sys.argv[1]
        print('Reading in',readfilename)
        t0 = time.time()
        fld = read_dfl(readfilename, ncar=ncar,conjugate_field_for_genesis = True) # read the field from disk
        print('took',time.time()-t0,'seconds total to read in and format the field with dimensions',fld.shape)
        fld = fld[::isradi,:,:]
        print("The shape before padding is ", fld.shape)
 
        
    # plot initial beam
    #plot_fld_marginalize_3(fld, dgrid=settings['dgrid'], dt=dt, title='Initial beam')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid, dt=dt, saveFilename=saveFilenamePrefix+'_init_xy.png',showPlotQ=showPlotQ) # plot the imported field
        energy_uJ,trms, tfwhm, xrms, xfwhm = plot_fld_slice(fld, dgrid, dt=dt, slice=-2, saveFilename=saveFilenamePrefix+'_init_tx.png',showPlotQ=showPlotQ) # plot the imported field
        _,_, _, yrms, yfwhm = plot_fld_slice(fld, dgrid, dt=dt, slice=-1, saveFilename=saveFilenamePrefix+'_init_ty.png',showPlotQ=showPlotQ) # plot the imported field
        plot_fld_power(fld, dt=dt, saveFilename=saveFilenamePrefix+'_init_t.png',showPlotQ=showPlotQ) # plot the imported field
    
    energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm = fld_info(fld, dgrid = dgrid, dt=dt)
    
    init_field_info = [energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm]

    # propagate through cavity to return to undulator
    fld, time_jitter = cavity_return_to_undulator(fld = fld,ncar = ncar, dgrid = dgrid,  zsep = zsep,
                                     l_undulator = l_undulator, xlamds=xlamds, l_cavity = l_cavity, w_cavity = w_cavity,
                                     isradi = isradi, skipTimeFFTsQ = skipTimeFFTsQ, skipSpaceFFTsQ = skipSpaceFFTsQ,
                                     npadt = npadt, Dpadt = Dpadt, npadx = npadx, unpadtQ = unpadtQ,
                                     slice_processing_relative_power_threshold = slice_processing_relative_power_threshold,
                                     tjitter_rms = tjitter_rms,
                                     showPlotQ = showPlotQ, savePlotQ =savePlotQ, verbosity = verbosity)
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid, dt=dt, saveFilename=saveFilenamePrefix+'_recirculated_xy.png',showPlotQ=showPlotQ) # plot the imported field
        energy_uJ,trms, tfwhm, xrms, xfwhm = plot_fld_slice(fld, dgrid, dt=dt, slice=-2,saveFilename=saveFilenamePrefix+'_recirculated_tx.png',showPlotQ=showPlotQ) # plot the imported field
        _,_, _, yrms, yfwhm = plot_fld_slice(fld, dgrid, dt=dt, slice=-1,saveFilename=saveFilenamePrefix+'_recirculated_ty.png',showPlotQ=showPlotQ) # plot the imported field
        plot_fld_power(fld, dt=dt, saveFilename=saveFilenamePrefix+'_recirculated_t.png',showPlotQ=showPlotQ) # plot the imported field
    
    energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm = fld_info(fld, dgrid = dgrid, dt=dt)
    
    return_field_info = [energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm]

  
    #if 0:
        #plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        #plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
    # drift 10 m
    #fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=10, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    #plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_recirculated+10m_xy.png',showPlotQ=showPlotQ) # plot the imported field
    #fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=10, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    #plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_recirculated+20m_xy.png',showPlotQ=showPlotQ) # plot the imported field
    #fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=10, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    #plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_recirculated+30m_xy.png',showPlotQ=showPlotQ) # plot the imported field
        #fld = rfp(fld, xlamds, dgrid, A=1, B=30, D=1, ncar=ncar, cutradius=0, dgridout=-1)
        #plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        #fld = rfp(fld, xlamds, dgrid, A=1, B=30, D=1, ncar=ncar, cutradius=0, dgridout=-1)
        #plot_fld_marginalize_t(fld, dgrid) # plot the imported field

    ## write field to disk
    if readfilename != None and writefilename != None:
        print('Writing to',writefilename)
            #writefilename = readfilename + 'r'
        write_dfl(fld, writefilename,conjugate_field_for_genesis = True)

        #try:
        #    print('Writing to',settings['writefilename'])
            #writefilename = readfilename + 'r'
        #    write_dfl(settings['writefilename'], fld)
        #except:
        #    print('ERROR: Could not write field to file',settings['writefilename'])

    ## transmit through mirror
    #fld = cavity_return_to_undulator(fld, settings)

    return fld, init_field_info, return_field_info, time_jitter


def recirculate_roundtrip(zsep, ncar, dgrid, l_undulator, xlamds=1.261043e-10,
                             npadt = 0, Dpadt = 0, npadx = 0, unpadtQ = True,
                             tjitter_rms = 5, slice_processing_relative_power_threshold = 0,
                                 isradi = 1, l_cavity = 149, w_cavity = 1,
                                 skipTimeFFTsQ = 1, skipSpaceFFTsQ = 1,
                                 showPlotQ = 0, savePlotQ = 0, verbosity = 1,
                                 readfilename = None, writefilename = None):

    dt = xlamds*zsep * max(1,isradi) / 299792458

    if readfilename == None:
        saveFilenamePrefix = 'test'
    else:
        saveFilenamePrefix = readfilename

    if readfilename == None:
        # make a new field
        t0 = time.time()
        dt = xlamds*zsep* max(1,isradi) / 299792458
        ncar = 201; dgrid =750e-6; dt*=1.; fld = make_gaus_beam(ncar= ncar, dgrid=dgrid, w0=80e-6, dt=dt, nslice=4096, trms=3.e-15)
        #settings['ncar']=121; settings['dgrid']=500e-6; dt*=10.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=1024, trms=3.e-15)
        #settings['ncar']=121; settings['dgrid']=200e-6; dt*=100.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=128, trms=3.e-15)
        print('took',time.time()-t0,'seconds total to make field with dimensions',fld.shape)
    else:
        ## import a field from a file on disk
        ##readfilename = '/nfs/slac/g/beamphysics/jytang/genesis/lasershaping2/cavity/flattop_flatbeam/gen_tap0.021_K1.1742_s0_a.out.dfl'
        #readfilename = '/gpfs/slac/staas/fs1/g/g.beamphysics/jytang/genesis/lasershaping2/cavity/flattop_flatbeam/gen_tap0.021_K1.1742_s0_a.out.dfl'
        ##readfilename = '/u/ra/jduris/code/genesis_dfl_tools/rfp_radiation_field_propagator/myfile.dfl'
        ##readfilename = sys.argv[1]
        print('Reading in',readfilename)
        t0 = time.time()
        fld = read_dfl(readfilename, ncar=ncar,conjugate_field_for_genesis = True) # read the field from disk
        print('took',time.time()-t0,'seconds total to read in and format the field with dimensions',fld.shape)
        fld = fld[::isradi,:,:]
        print("The shape before padding is ", fld.shape)
 
        
    # plot initial beam
    #plot_fld_marginalize_3(fld, dgrid=settings['dgrid'], dt=dt, title='Initial beam')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid, dt=dt, saveFilename=saveFilenamePrefix+'_init_xy.png',showPlotQ=showPlotQ) # plot the imported field
        energy_uJ,trms, tfwhm, xrms, xfwhm = plot_fld_slice(fld, dgrid, dt=dt, slice=-2, saveFilename=saveFilenamePrefix+'_init_tx.png',showPlotQ=showPlotQ) # plot the imported field
        _,_, _, yrms, yfwhm = plot_fld_slice(fld, dgrid, dt=dt, slice=-1, saveFilename=saveFilenamePrefix+'_init_ty.png',showPlotQ=showPlotQ) # plot the imported field
        plot_fld_power(fld, dt=dt, saveFilename=saveFilenamePrefix+'_init_t.png',showPlotQ=showPlotQ) # plot the imported field
    
    energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm = fld_info(fld, dgrid = dgrid, dt=dt)
    init_field_info = [energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm]
    
    # drift a undulator distance
    Ldrift = l_undulator
    fld = rfp(fld, xlamds, dgrid, A=1, B=Ldrift, D=1, ncar=ncar, cutradius=0, dgridout=-1, kxspace_inQ=0, kxspace_outQ=0, slice_processing_relative_power_threshold=slice_processing_relative_power_threshold)
    
    # propagate through cavity to return to undulator
    fld, time_jitter = cavity_return_to_undulator(fld = fld,ncar = ncar, dgrid = dgrid,  zsep = zsep,
                                     l_undulator = l_undulator, xlamds=xlamds, l_cavity = l_cavity, w_cavity = w_cavity,
                                     isradi = isradi, skipTimeFFTsQ = skipTimeFFTsQ, skipSpaceFFTsQ = skipSpaceFFTsQ,
                                     npadt = npadt, Dpadt = Dpadt, npadx = npadx, unpadtQ = unpadtQ,
                                     slice_processing_relative_power_threshold = slice_processing_relative_power_threshold,
                                     tjitter_rms = tjitter_rms,
                                     showPlotQ = showPlotQ, savePlotQ =savePlotQ, verbosity = verbosity)
    
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid, dt=dt, saveFilename=saveFilenamePrefix+'_recirculated_xy.png',showPlotQ=showPlotQ) # plot the imported field
        energy_uJ,trms, tfwhm, xrms, xfwhm = plot_fld_slice(fld, dgrid, dt=dt, slice=-2,saveFilename=saveFilenamePrefix+'_recirculated_tx.png',showPlotQ=showPlotQ) # plot the imported field
        _,_, _, yrms, yfwhm = plot_fld_slice(fld, dgrid, dt=dt, slice=-1,saveFilename=saveFilenamePrefix+'_recirculated_ty.png',showPlotQ=showPlotQ) # plot the imported field
        plot_fld_power(fld, dt=dt, saveFilename=saveFilenamePrefix+'_recirculated_t.png',showPlotQ=showPlotQ) # plot the imported field
    
    energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm = fld_info(fld, dgrid = dgrid, dt=dt)
    return_field_info = [energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm]

  
    #if 0:
        #plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        #plot_fld_slice(fld, dgrid, dt=dt, slice=-2) # plot the imported field
    # drift 10 m
    #fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=10, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    #plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_recirculated+10m_xy.png',showPlotQ=showPlotQ) # plot the imported field
    #fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=10, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    #plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_recirculated+20m_xy.png',showPlotQ=showPlotQ) # plot the imported field
    #fld = rfp(fld, settings['xlamds'], settings['dgrid'], A=1, B=10, D=1, ncar=settings['ncar'], cutradius=0, dgridout=-1)
    #plot_fld_marginalize_t(fld, settings['dgrid'],saveFilename=saveFilenamePrefix+'_recirculated+30m_xy.png',showPlotQ=showPlotQ) # plot the imported field
        #fld = rfp(fld, xlamds, dgrid, A=1, B=30, D=1, ncar=ncar, cutradius=0, dgridout=-1)
        #plot_fld_marginalize_t(fld, dgrid) # plot the imported field
        #fld = rfp(fld, xlamds, dgrid, A=1, B=30, D=1, ncar=ncar, cutradius=0, dgridout=-1)
        #plot_fld_marginalize_t(fld, dgrid) # plot the imported field

    ## write field to disk
    if readfilename != None and writefilename != None:
        print('Writing to',writefilename)
            #writefilename = readfilename + 'r'
        write_dfl(fld, writefilename,conjugate_field_for_genesis =True)

        #try:
        #    print('Writing to',settings['writefilename'])
            #writefilename = readfilename + 'r'
        #    write_dfl(settings['writefilename'], fld)
        #except:
        #    print('ERROR: Could not write field to file',settings['writefilename'])

    ## transmit through mirror
    #fld = cavity_return_to_undulator(fld, settings)

    return fld, init_field_info, return_field_info, time_jitter


def extract_from_cavity(zsep, ncar, dgrid, l_undulator, xlamds=1.261043e-10,
                        npadx = 0, npadt =0, slice_processing_relative_power_threshold = 0,
                        isradi = 1, l_cavity = 149, w_cavity = 1,unpadtQ = True,
                        skipTimeFFTsQ = 1, skipSpaceFFTsQ = 1, 
                        showPlotQ = 0, savePlotQ = 0, verbosity = 1,
                        readfilename = None, writefilename = None):
    
    dt = xlamds* zsep * max(1,isradi) / 299792458

    
    if readfilename == None:
        saveFilenamePrefix = 'test'
    else:
        saveFilenamePrefix = readfilename

    if readfilename == None:
        # make a new field
        t0 = time.time()
        dt = xlamds* zsep * max(1,isradi) / 299792458
        ncar=201; dgrid=750e-6; dt*=1.; fld = make_gaus_beam(ncar=ncar, dgrid=dgrid, w0=80e-6, dt=dt, nslice=4096, trms=3.e-15)
        #settings['ncar']=121; settings['dgrid']=500e-6; dt*=10.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=1024, trms=3.e-15)
        #settings['ncar']=121; settings['dgrid']=200e-6; dt*=100.; fld = make_gaus_beam(ncar=settings['ncar'], dgrid=settings['dgrid'], w0=80e-6, dt=dt, nslice=128, trms=3.e-15)
        print('took',time.time()-t0,'seconds total to make field with dimensions',fld.shape)
    else:
        ## import a field from a file on disk
        ##readfilename = '/nfs/slac/g/beamphysics/jytang/genesis/lasershaping2/cavity/flattop_flatbeam/gen_tap0.021_K1.1742_s0_a.out.dfl'
        #readfilename = '/gpfs/slac/staas/fs1/g/g.beamphysics/jytang/genesis/lasershaping2/cavity/flattop_flatbeam/gen_tap0.021_K1.1742_s0_a.out.dfl'
        ##readfilename = '/u/ra/jduris/code/genesis_dfl_tools/rfp_radiation_field_propagator/myfile.dfl'
        ##readfilename = sys.argv[1]
        print('Reading in',readfilename)
        t0 = time.time()
        fld = read_dfl(readfilename, ncar=ncar,conjugate_field_for_genesis = False) # read the field from disk
        print('took',time.time()-t0,'seconds total to read in and format the field with dimensions',fld.shape)
        fld = fld[::isradi,:,:]
        print("The shape before padding is ", fld.shape)

    # plot initial beam
    #plot_fld_marginalize_3(fld, dgrid=settings['dgrid'], dt=dt, title='Initial beam')
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid, dt=dt, saveFilename=saveFilenamePrefix+'_init_xy.png',showPlotQ=showPlotQ) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-2, saveFilename=saveFilenamePrefix+'_init_tx.png',showPlotQ=showPlotQ) # plot the imported field
        plot_fld_slice(fld, dgrid, dt=dt, slice=-1, saveFilename=saveFilenamePrefix+'_init_ty.png',showPlotQ=showPlotQ) # plot the imported field
        plot_fld_power(fld, dt=dt, saveFilename=saveFilenamePrefix+'_init_t.png',showPlotQ=showPlotQ) # plot the imported field
    
    fld_info(fld, dgrid = dgrid, dt=dt)

    # propagate through cavity to return to undulator
    fld = cavity_exit_from_mirror(fld = fld, ncar = ncar, dgrid = dgrid, zsep = zsep, l_undulator = l_undulator, xlamds=xlamds, l_cavity = l_cavity, w_cavity = w_cavity, isradi = isradi, skipTimeFFTsQ = skipTimeFFTsQ, skipSpaceFFTsQ = skipSpaceFFTsQ, showPlotQ =  showPlotQ, savePlotQ = savePlotQ, verbosity = verbosity, npadx = npadx, npadt = npadt,
                                  slice_processing_relative_power_threshold = slice_processing_relative_power_threshold,unpadtQ=unpadtQ)
    
    if showPlotQ:
        plot_fld_marginalize_t(fld, dgrid, dt=dt,saveFilename=saveFilenamePrefix+'_extracted_xy.png',showPlotQ=showPlotQ) # plot the imported field
        energy_uJ, trms, tfwhm, xrms, xfwhm = plot_fld_slice(fld, dgrid, dt=dt, slice=-2,saveFilename=saveFilenamePrefix+'_extracted_tx.png',showPlotQ=showPlotQ) # plot the imported field
        _, _, _, yrms, yfwhm = plot_fld_slice(fld, dgrid, dt=dt, slice=-1,saveFilename=saveFilenamePrefix+'_extracted_ty.png',showPlotQ=showPlotQ) # plot the imported field
        plot_fld_power(fld, dt=dt, saveFilename=saveFilenamePrefix+'_extracted_t.png',showPlotQ=showPlotQ) # plot the imported field
    
    energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm = fld_info(fld, dgrid = dgrid, dt=dt)
    
    
    extracted_field_info = [energy_uJ, maxpower,  trms, tfwhm, xrms, xfwhm, yrms, yfwhm]
    
    ## write field to disk
    if readfilename != None and writefilename != None:
        print('Writing to',writefilename)
            #writefilename = readfilename + 'r'
        write_dfl(fld, writefilename)
    
    return fld, extracted_field_info


def add_energy_jitter(folder_name, readname = 'HXRSTART.out', cut_range =  [-0.4e-13,0.5e-13], showPlotQ = False, jitter_rms = 0.003/100):# energy jitter
    mc2_eV = 0.511e6
    filename = folder_name + '/'+readname
    fnhead = '.'.join(filename.split('.')[:-1])
    ff = SDDS.readSDDS(folder_name + '/'+readname)
    parameters, bunches = ff.read()
    print('number of bunches: ', bunches.shape[0],' \t particles in bunch 0: ', bunches.shape[1])
    d = bunches[0].T # particle data
    npart = bunches.shape[1] # number of macroparticles
    charge_pC = parameters[0]['Charge'] * 1e12
    gam0 = np.mean(d[5,:])
    Ejitter = np.random.normal(scale = jitter_rms)
    print("appy energy jitter " + str(Ejitter*100)+'%')
    d[5,:] += gam0*Ejitter     # add energy jitter
    d[4,:] -= np.mean(d[4,:])


    cut_range =  cut_range

    t_min  = cut_range[0]
    t_max =  cut_range[1]
        
    d_cut = d[:,(d[4,:]>=t_min)&(d[4,:]<=t_max)]
    d_cut[4,:] -= np.mean(d_cut[4,:])
    del d
    
    d_match, average_beamsize_core, emittance_normalized_core= match_to_FODO(d_cut, d_cut, L_quad=10*0.026, L_drift=150*0.026, g_quad=14.584615)
    
    del d_cut
    
    genesis_ndmax = int(1e6)
    # select particles randomly but in order
    npart_match = np.shape(d_match)[1]
    cut_ndmax = np.arange(npart_match); np.random.shuffle(cut_ndmax); cut_ndmax = cut_ndmax < genesis_ndmax
    indicies_ndmax = np.arange(npart_match)[cut_ndmax]
    beam_sel_charge = charge_pC * npart_match / npart*1e-12
    #print(keep_charge_fraction, '=?', 1. * npart_match / npart)
    if showPlotQ:
        plt.figure()
        ax = plt.gca()
        ax_I = ax.twinx()
        h, xedges, yedges, image = ax.hist2d(d_match[4,:],d_match[5,:] * mc2_eV,bins=500); 
        ax.set_xlabel('Time/s'); ax.set_ylabel('Energy/eV'); 
         # current profile 
        xcoords = 0.5*(xedges[1:]+xedges[:-1])
        ycoords = np.sum(h,axis=1) / np.sum(h) * beam_sel_charge / np.mean(np.diff(xcoords))/1e3
    
        ax_I.plot(xcoords, ycoords, label='All');
   
        ax_I.set_ylabel('Current/kA)'); 
        plt.show()
        plt.close()
    
    header = ["# sdds2genesis from elegant matched for undulator line "]
    header += ["? version = 1.0"]
    header += ["? charge =   " + str(beam_sel_charge)]
    header += ["? size =   " + str(genesis_ndmax)]
    header += ["? COLUMNS X XPRIME Y YPRIME T GAMMA"]
    
    #distfilename = fnhead+'_matchproj.dist'
    distfilename = folder_name +  '/HXRSTART_matchproj.dist'
    with open(distfilename, 'w') as distfile:
        for line in header:
            distfile.write(line+'\n')
            
        writer = csv.writer(distfile, delimiter=" ") # genesis requires space delimiters but can also do "\t "
        for i in indicies_ndmax:
            writer.writerow(d_match[:6,i])
    
    print('INFO: Wrote ',distfilename)
    return Ejitter