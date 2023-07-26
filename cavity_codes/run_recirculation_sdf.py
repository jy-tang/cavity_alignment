import numpy as np
import pickle
import os
import subprocess
from time import sleep

def submit_recirculation(shell_script = 'RunRecirculation.sh'):
    cmd = 'sbatch '+ shell_script
    x = subprocess.check_output(cmd.split())
    y = x.split()
    print(x)
    return y[-1]


def all_done(jid):
    flag = [False for i in range(len(jid))]
    all_done_flag = False
    while not all_done_flag:
        sleep(30)
        count = 0
        for id in jid:
            ret2=subprocess.getoutput("squeue -u jytang")
            if ret2.count(str(int(id))) < 1:
                flag[count] = True
            count +=1
        all_done_flag = all(flag)
        print("job "  + str(jid[0]) + " is running")
    print('all done!')



def start_recirculation(zsep, nslice, npadt, npadx, nRoundtrips, 
                        readfilename,  workdir, saveFilenamePrefix,
                        ncar = 181 , dgrid =540e-6,  xlamds=1.261043e-10,   w0=40e-6,  trms=10.e-15,  peak_power = 1e10,
                       Dpadt = 0, isradi = 1,       # padding params
                       l_undulator = 32*3.9, l_cavity = 149, w_cavity = 1, d1 = 100e-6, d2 = 100e-6, # cavity params
                       n_theta_shift1 = 0, n_theta_shift2 =0, n_theta_shift3 = 0, n_theta_shift4 = 0,   #yaw angle error on each of the mirror
                       x1 = 0, x2 = 0,                   # displacement error of the two lenses
                       verboseQ = 1):
    
    param_dic = locals()
    pickle.dump(param_dic, open( workdir + "/params.p", "wb" ) )
    
    os.system('cp  cavity_codes/RunRecirculation.sh ' + workdir)
    os.system('cp  cavity_codes/dfl_cbxfel_mpi.py ' + workdir)
    
    root_dir = os.path.realpath(os.path.curdir)
    os.chdir(workdir)
    jobid = submit_recirculation(shell_script = 'RunRecirculation.sh')
    os.chdir(root_dir)
    return jobid


