import os
import sys
from sets import Set
def processCrash(filename):
    log = open(filename,"r")
    logs = log.readlines()
    new_logs = []
    for log in logs:
        if "['info" not in log and "activated" not in log and "incorrectly" not in log and "starts" not in log and "New value" not in log:
            pass
        else:
            new_logs.append(log)
    flag = 0
    buf = []
    for log in new_logs:
        buf.append(log)
        if "incorrectly" in log:
                print "#######"
                print buf
        if "starts" in log:
            buf = []
               
processCrash("matrixMul_matrixMulCUDA.log")
#processCrash("libor_Pathcalc_Portfolio_KernelGPU.log.part2")
            
