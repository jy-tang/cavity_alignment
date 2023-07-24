import numpy as np
import pickle
import os
import subprocess
from time import sleep

def submit_mergeFiles(shell_script = 'RunMergeFiles.sh'):
    cmd = 'sbatch '+ shell_script
    print(cmd)
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

def start_mergeFiles(nRoundtrips, workdir, saveFilenamePrefix, dgrid, dt, Dpadt):
    
    param_dic = locals()
    pickle.dump(param_dic, open( workdir + "/merge_params.p", "wb" ) )
    
    os.system('cp  cavity_codes/RunMergeFiles.sh ' + workdir)
    os.system('cp  cavity_codes/merge_files_mpi.py ' + workdir)
    
    root_dir = os.path.realpath(os.path.curdir)
    os.chdir(workdir)
    jobid = submit_mergeFiles()
    os.chdir(root_dir)
    return jobid