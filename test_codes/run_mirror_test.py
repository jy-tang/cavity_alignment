import numpy as np
import pickle
import os
import subprocess
from time import sleep

def submit_recirculation(shell_script = 'RunMirrorTest.sh'):
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

def start_testMirror_stats(zsep, nslice, npadt, npadx, 
                        readfilename, workdir, saveFilenamePrefix,
                        ncar = 181 , dgrid =540e-6,  xlamds=1.261043e-10,   
                       Dpadt = 0, isradi = 1,       # padding params
                       d = 100e-6,# cavity params
                        misalignQ = False, M = 0,       # misalignment parameter
                             roughnessQ = False, C = None,
                    verboseQ = 1):
    
    param_dic = locals()
    pickle.dump(param_dic, open( workdir + "/params.p", "wb" ) )
    
    os.system('cp  cavity_codes/RunMirrorTest.sh ' + workdir)
    os.system('cp  cavity_codes/dfl_mirror_test.py ' + workdir)
    
    root_dir = os.path.realpath(os.path.curdir)
    os.chdir(workdir)
    jobid = submit_recirculation(shell_script = 'RunMirrorTest.sh')
    os.chdir(root_dir)
    return jobid


