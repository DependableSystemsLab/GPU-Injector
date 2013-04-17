import os
import sys
import configure
import subprocess 
import pexpect

CUDA_GDB_EXPECT = "\(cuda-gdb\)"
cmd = "cuda-gdb "+configure.binary_path
argument = configure.parameter
cuda_gdb_p = pexpect.spawn(cmd)
cuda_gdb_p.expect(CUDA_GDB_EXPECT)
cuda_gdb_p.sendline("run "+argument)
cuda_gdb_p.expect(CUDA_GDB_EXPECT)
logs = cuda_gdb_p.before
print logs

with open(configure.profile_file,"a") as profile:
    log = logs.split("\r\n")
    for line in log:
        
        if configure.kernel in line and "Launch" in line:
            profile.write("INFO "+line+"\n")
cuda_gdb_p.close()
  
