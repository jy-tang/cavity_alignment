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

def start_recirculation_stats(zsep, nslice, npadt, npadx, nRoundtrips, 
                        readfilename, seedfilename, workdir, saveFilenamePrefix,
                        ncar = 181 , dgrid =540e-6,  xlamds=1.261043e-10,   
                       Dpadt = 0, isradi = 1,       # padding params
                       l_undulator = 12*3.9, l_cavity = 149, w_cavity = 1, d1 = 100e-6, d2 = 100e-6, # cavity params
                        misalignQ = False, M1 = 0, M2 = 0, M3 = 0, M4 = 0,        # misalignment parameter
                             roughnessQ = False, C1 = None, C2 = None, C3 = None, C4 = None, 
                    verboseQ = 1):
    
    param_dic = locals()
    pickle.dump(param_dic, open( workdir + "/params.p", "wb" ) )
    
    os.system('cp  cavity_codes/RunRecirculation_stats.sh ' + workdir)
    os.system('cp  cavity_codes/dfl_cbxfel_stats.py ' + workdir)
    
    root_dir = os.path.realpath(os.path.curdir)
    os.chdir(workdir)
    jobid = submit_recirculation(shell_script = 'RunRecirculation_stats.sh')
    os.chdir(root_dir)
    return jobid


def start_recirculation(zsep, nslice, npadt, npadx, nRoundtrips, 
                        readfilename, seedfilename, workdir, saveFilenamePrefix,
                        ncar = 181 , dgrid =540e-6,  xlamds=1.261043e-10,   
                       Dpadt = 0, isradi = 1,       # padding params
                       l_undulator = 32*3.9, l_cavity = 149, w_cavity = 1, d1 = 100e-6, d2 = 100e-6, # cavity params
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


def start_recirculation_newconfig(zsep, nslice, npadt, npadx, nRoundtrips, 
                        readfilename, seedfilename, workdir, saveFilenamePrefix,
                        ncar = 181 , dgrid =540e-6,  xlamds=1.261043e-10,   
                       Dpadt = 0, isradi = 1,       # padding params
                       l_undulator = 32*3.9, l_cavity = 149, w_cavity = 1, d1 = 100e-6, d2 = 100e-6, # cavity params
                    verboseQ = 1):
    
    param_dic = locals()
    pickle.dump(param_dic, open( workdir + "/params.p", "wb" ) )
    
    os.system('cp  cavity_codes/RunRecirculation_newconfig.sh ' + workdir)
    os.system('cp  cavity_codes/dfl_cbxfel_new_config.py ' + workdir)
    
    root_dir = os.path.realpath(os.path.curdir)
    os.chdir(workdir)
    jobid = submit_recirculation(shell_script = 'RunRecirculation_newconfig.sh')
    os.chdir(root_dir)
    return jobid


def start_recirculation_4lens(zsep, nslice, npadt, npadx, nRoundtrips, 
                        readfilename, seedfilename, workdir, saveFilenamePrefix,
                        ncar = 181 , dgrid =540e-6,  xlamds=1.261043e-10,   
                       Dpadt = 0, isradi = 1,       # padding params
                       l_undulator = 32*3.9, l_cavity = 149, w_cavity = 1, d1 = 100e-6, d2 = 100e-6, # cavity params
                    verboseQ = 1):
    
    param_dic = locals()
    pickle.dump(param_dic, open( workdir + "/params.p", "wb" ) )
    
    os.system('cp  cavity_codes/RunRecirculation_4lens.sh ' + workdir)
    os.system('cp  cavity_codes/dfl_cbxfel_4lens.py ' + workdir)
    
    root_dir = os.path.realpath(os.path.curdir)
    os.chdir(workdir)
    jobid = submit_recirculation(shell_script = 'RunRecirculation_4lens.sh')
    os.chdir(root_dir)
    return jobid