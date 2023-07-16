from mpi4py import MPI
import pickle
import time
import numpy as np
import gc
from rfp2 import *
from pathlib import Path
import os

def merge_files(nRoundtrips, workdir, saveFilenamePrefix, dgrid, dt, Dpadt):
    
    t0 = time.time()
    
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    nblocks = 128
    
    # merge recirculation and transmission blocks of each roundtrip
    if nRoundtrips >= 0:
        ave2, res2 = divmod(nRoundtrips + 1, nprocs)
        count2 = [ave2 + 1 if p < res2 else ave2 for p in range(nprocs)]
        
        if rank == 0:
            print(count2)
        
        count2 = np.array(count2)
        count_sum2 = [sum(count2[:p+1]) for p in range(nprocs)]
        
        if rank == 0:
            print(count_sum2)
        
        if count2[rank] > 0:
            if rank == 0:
                Round_range = range(0, count_sum2[0])
            else:
                Round_range = range(count_sum2[rank-1], count_sum2[rank])
            for Round in Round_range:
                #--------------------for recirculation files ------------------------------
                field_t = []
                for block in range(nblocks):
                    loadname = workdir + '/'+saveFilenamePrefix + "_block"+str(block)+"_round"+str(Round)+".p"
                    field_t.append(pickle.load( open(loadname , "rb" )))
                field_t = np.concatenate(field_t, axis = 0)
                
                # fft in time and unpad
                field_t = ifft(np.fft.ifftshift(field_t,axes = 0), axis=0)
                if int(Dpadt) > 0:
                    field_t = unpad_dfl_t(field_t, [int(Dpadt), int(Dpadt)])
                
                
                energy_uJ, maxpower, tmean, trms, tfwhm, xmean, xrms, xfwhm, ymean, yrms, yfwhm = fld_info(field_t, dgrid = dgrid, dt=dt)
                return_field_info = [Round, energy_uJ, maxpower, tmean, trms, tfwhm, xmean, xrms, xfwhm, ymean, yrms, yfwhm]
               
                with open(workdir + '/'+saveFilenamePrefix+'_recirc.txt', "a") as myfile:
                    myfile.write(" ".join(str(item) for item in return_field_info))
                    myfile.write("\n")
              
                
                #write to disk
                #writefilename = workdir + '/' + saveFilenamePrefix + "_field_round" + str(Round) + '.dfl'
                #write_dfl(field_t, writefilename,conjugate_field_for_genesis = False,swapxyQ=False)
                
                #--------------------for transmission files ------------------------------
                
                field_t = []
                for block in range(nblocks):
                    loadname = workdir + '/'+saveFilenamePrefix + "_block_transmit_"+str(block)+"_round"+str(Round)+".p"
                    field_t.append(pickle.load( open(loadname , "rb" )))
                field_t = np.concatenate(field_t, axis = 0)
                
                # fft in time and unpad
                field_t = ifft(np.fft.ifftshift(field_t,axes = 0), axis=0)
                if int(Dpadt) > 0:
                    field_t = unpad_dfl_t(field_t, [int(Dpadt), int(Dpadt)])
                
                
                energy_uJ, maxpower, tmean, trms, tfwhm, xmean, xrms, xfwhm, ymean, yrms, yfwhm = fld_info(field_t, dgrid = dgrid, dt=dt)
                return_field_info = [Round, energy_uJ, maxpower, tmean, trms, tfwhm, xmean, xrms, xfwhm, ymean, yrms, yfwhm]
               
                with open(workdir + '/'+saveFilenamePrefix+'_transmit.txt', "a") as myfile:
                    myfile.write(" ".join(str(item) for item in return_field_info))
                    myfile.write("\n")
              
                
                #write to disk
                if Round == 0:
                    writefilename =workdir + '/'+ saveFilenamePrefix+"_field_transmit_round" + str(Round) + '.dfl'
                    write_dfl(field_t, writefilename,conjugate_field_for_genesis = False,swapxyQ=False)
                
                
                
                del field_t
                gc.collect()
                
                

    
    comm.Barrier()
    
    if rank == 0:
        print("It takes " + str(time.time()-t0) + "seconds to finish merging files")        
        
        # delete block files
        for filename in Path(workdir).glob("*.p"):
            filename.unlink()
        


params_dic = pickle.load( open( "merge_params.p", "rb" ) )        
merge_files(**params_dic) 