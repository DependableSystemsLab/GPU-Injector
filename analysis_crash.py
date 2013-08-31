import os
import sys
from sets import Set
def processCrash(filename):
    log = open(filename,"r")
    logs = log.readlines()
    new_logs = []
    for log in logs:
        if "['info" not in log and "activated" not in log and "EXCEPTION" not in log and "starts" not in log and "New value" not in log and "crashed" not in log:
            pass
        else:
            new_logs.append(log)
    flag = 0
    buf = []
    for log in new_logs:
        if flag == 0:
            buf.append(log)
        if "crashed" in log:
                print "#######"
                if len(buf) > 0:
                    print buf
                buf = []
                flag = 1
        if "starts" in log:
            buf = []
            flag = 0   
processCrash("matrixMul_matrixMul_kernel.log.part1")
processCrash("matrixMul_matrixMul_kernel.log.part2")
processCrash("matrixMul_matrixMul_kernel.log.part3")
processCrash("matrixMul_matrixMul_kernel.log.part4")
processCrash("matrixMul_matrixMul_kernel.log.part5")
processCrash("matrixMul_matrixMul_kernel.log.part6")
processCrash("matrixMul_matrixMul_kernel.log.part7")
#processCrash("libor_Pathcalc_Portfolio_KernelGPU.log.part2")
            
