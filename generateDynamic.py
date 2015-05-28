import sys
import pexpect
import logging
import time 
import random
import configure
import faultInjection

logger = None
CUDA_GDB_PATH = "/usr/local/cuda-6.0/bin/cuda-gdb "
CUDA_GDB_EXPECT = "\(cuda-gdb\)"
BREAKPOINT="break"
RUN="run"
PRINT_PC = "print $pc"
BREAK_LOCATION = "matrixMul.cu:63" # this is application specific
NVCC_PATH = "/usr/local/cuda/bin/nvcc "
THREAD_CONTINUE_EXPECT = "---Type \<return\> to continue, or q \<return\> to quit---"


def main():
        global CUDA_GDB_PATH,RUN,NVCC_PATH,logger,CUDA_GDB_EXPECT,BREAK_LOCATION,THREAD_CONTINUE_EXPECT
	#-------------------------------------------------------------------------------------------------------------
	#if already complied no need these two lines.
        #child=pexpect.spawn(NVCC_PATH + configure.options +"-g -G " +configure.filename)	#first compile the application with all options required
	#child.expect(pexpect.EOF,timeout=None)						#expect command prompt				       
	#-------------------------------------------------------------------------------------------------------------------------
        child = pexpect.spawn(CUDA_GDB_PATH+" "+configure.binary_path)					 #to enter into cuda-gdb mode
    	child.maxread = 1000000
    	child.setecho(False)
    	child.expect(CUDA_GDB_EXPECT)  
    	child.sendline(BREAKPOINT+" "+BREAK_LOCATION)
    	child.expect(CUDA_GDB_EXPECT)
	wc = child.sendline(RUN)
    	resend = child.expect([CUDA_GDB_EXPECT,THREAD_CONTINUE_EXPECT])
    	if resend == 1:
       		child.sendline()
    	rawstr = child.before
	lines = rawstr.split("\r\n")
	print rawstr  # till here its ok , checked and verified.
        # Do processimg of rawstr to obtain the required parameters for line "INFO LAUNCH of CUDA Kernel 0 (matrixMulCUDA<32><<<(4,6,1),(32,32,1)>>>) on Device 0"
        for line in lines:
	    if "Switching focus" in line:
		line = line.lstrip().rstrip("\r\n")
            	temp1 = line.split(" ")
                items=len(temp1) # 20
		print items # 20
		kernel_id= (temp1[5].split(','))[0]  # to obtain 0 from 0,
		print "Kernel_id: "+str(kernel_id)
		device_id= (temp1[13].split(','))[0] # to obtain 0 from 0,
		print "Device_id: "+str(device_id)
	    elif "Breakpoint" and "<<<" and ">>>" in line:
		temp1 = line.split(" ")
		kernel_laucn_config="("+temp1[2]+")" # (matrixMulCUDA<32><<<(4,6,1),(32,32,1)>>>)
		print kernel_laucn_config
        # till here its ok , checked and verified.
        # Write to last line in log file
        with open(configure.profile_file, "a") as myfile:
             myfile.write("INFO [LAUNCH of CUDA Kernel "+str(kernel_id)+" "+kernel_laucn_config+ "on Device "+str(device_id)+"]")
	#find starting pc value
	child.sendline(PRINT_PC)
        child.expect(CUDA_GDB_EXPECT)
        pc_rawstr=child.before
        lines = pc_rawstr.split("\r\n")  # has two lines	
	start_pc_value=(lines[1].split(" "))[2] # second line has the $1 = 9049592
        #writing to config file
	with open("configure.py", "a") as my_file:
             my_file.write("startingpc = "+start_pc_value+"\n")
        faultInjection.fault_main()
        
