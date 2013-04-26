import sys
import pexpect
import logging
import random
import time
import configure
#---------------------
# System configuration
#---------------------
#BENCHMARK = "MAT"
logger = None
#------------------------
#CUDA-GDB commands
#------------------------
CUDA_GDB_PATH = "cuda-gdb"
BREAKPOINT = "break "
#BREAK_LOCATION = "matrixMul_kernel.cu:38"
BREAK_LOCATION = configure.startline
BREAK_LOCATION_2 = "bucket_query.cu:274"
BREAK_LOCATION_3 = "sort_scan.cu:120"
BREAK_LOCATION_4 = "bucket_query.cu:116"
BREAK_LOCATION_5 = "bucket_query.cu:970"
BREAK_LOCATION_7 = "bucket_query.cu:968"
BREAK_LOCATION_6 = "bucket_query.cu:138"


STEPI = "stepi"
NEXT = "n"
PC = "print $pc"
RUN = "run"
ARGUMENT= configure.parameter
CONTINUE = "continue"
CUDA_FUN_INFO = "cuda kernel block thread"
CUDA_THREAD_INFO = "info cuda threads"

DELETE_BREAKPOINT = "delete breakpoint 1"
QUIT= "quit"
KILL = "kill"
ENTER = ""
EXIT = "Program exited normally"
#------------------------
#Expect collection
#------------------------

CUDA_GDB_EXPECT = "\(cuda-gdb\)"
CUDA_SYN_EXPECT = "__syncthreads\(\)"
CUDA_SYN_EXPECT_2 = "__syncthreads\(\)"
PC_EXPECT = "="
CUDA_FUN_INFO_EXPECT = " "
THREAD_CONTINUE_EXPECT = "---Type \<return\> to continue, or q \<return\> to quit---"
THREAD_CONTINUE_EXPECT_WERIED = "---Type \<return\> to continue, or q \<return\> to quit---stepi"
NO_FOCUS = "Focus not set on any active CUDA kernel"

def profiler(path,trigger,trial):
    global CUDA_GDB_PATH, BREAKPOINT,BREAK_LOCATION,KILL,QUIT,DELETE_BREAKPOINT,CUDA_FUN_INFO,PC,RUN,CONTINUE,CUDA_THREAD_INFO,ENTER, BREAK_LOCATION_2, NEXT, CUDA_SYN_EXPECT_2
    global CUDA_GDB_EXPECT,PC_EXPECT,CUDA_FUN_INFO_EXPECT,THREAD_CONTINUE_EXPECT,CUDA_SYN_EXPECT,ARGUMENT,EXIT,NO_FOCUS,THREAD_CONTINUE_EXPECT_WERIED
    global logger
    global REAK_LOCATION_3, REAK_LOCATION_4, REAK_LOCATION_5,REAK_LOCATION_6
    cuda_gdb_p = pexpect.spawn(CUDA_GDB_PATH+" "+path)
    cuda_gdb_p.maxread = 100000
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)  
    #---------------
    # set breakpoint
    #---------------
    cuda_gdb_p.sendline(BREAKPOINT+" "+BREAK_LOCATION)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #cuda_gdb_p.sendline(BREAKPOINT+" "+BREAK_LOCATION_2)
    #cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #cuda_gdb_p.sendline(BREAKPOINT+" "+BREAK_LOCATION_3)
    #cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #cuda_gdb_p.sendline(BREAKPOINT+" "+BREAK_LOCATION_4)
    #cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #cuda_gdb_p.sendline(BREAKPOINT+" "+BREAK_LOCATION_7)
    #cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #cuda_gdb_p.sendline(BREAKPOINT+" "+BREAK_LOCATION_5)
    #cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #cuda_gdb_p.sendline(BREAKPOINT+" "+BREAK_LOCATION_6)
    #cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #---------------
    # run the program
    #---------------
    wc = cuda_gdb_p.sendline(RUN+ARGUMENT)
    resend = cuda_gdb_p.expect([CUDA_GDB_EXPECT,THREAD_CONTINUE_EXPECT])
    if resend == 1:
       cuda_gdb_p.sendline()
    rawstr = cuda_gdb_p.before
    # for debug
    #while "Kernel 59" not in rawstr:
    #    cuda_gdb_p.sendline(CONTINUE)
    #    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #    rawstr = cuda_gdb_p.before
    #    print rawstr
    lines = rawstr.split("\r\n")
    #for line in lines:
    #    if "<<<" and ">>>" in line:
    #        logger.info(line)
     
    #cuda_gdb_p.sendline(BREAKPOINT+" "+BREAK_LOCATION_5)
    #cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #cuda_gdb_p.sendline(CONTINUE)
    #cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    
    #------------------------------
    # check the current PC
    #------------------------------
    cuda_gdb_p.sendline(CUDA_FUN_INFO)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    logger.info("KERNEL INFO: "+cuda_gdb_p.before)
    cuda_gdb_p.sendline(PC)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    value = cuda_gdb_p.before.lstrip().rstrip("\r\n").split(PC_EXPECT)
    logger.info("PC is "+value[len(value)-1])
    target = ""
    temp = ""
    while EXIT not in target or EXIT not in temp or "is not being run" in target or "is not being run" in temp: 
        j = -1
        flag_step = 0
        flag_info = 0
        cuda_gdb_p.sendline(STEPI)
        while flag_step == 0:
            j = cuda_gdb_p.expect([CUDA_GDB_EXPECT,CUDA_SYN_EXPECT,THREAD_CONTINUE_EXPECT,THREAD_CONTINUE_EXPECT_WERIED,CUDA_SYN_EXPECT_2],timeout=60)
            target = cuda_gdb_p.before
            logger.info("in stepi "+target)
            if CUDA_SYN_EXPECT in target or CUDA_SYN_EXPECT_2 in target:
                    logger.info("Hit the barrier!")
                    cuda_gdb_p.sendline(NEXT)
                    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                    break 
            if j == 0:
                if NO_FOCUS in target and "Switching" not in target:
                    logger.info("CONTINUE THREADS to hit breakpoint again! -1")
                    cuda_gdb_p.sendline(CONTINUE)
                    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                    target = cuda_gdb_p.before
                    logger.info("target 1 "+target)
                    time.sleep(5)
                if CUDA_SYN_EXPECT in target:
                    logger.info("Hit the barrier!")
                    cuda_gdb_p.sendline(NEXT)
                    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                flag_step = 1
            elif j == 1:
                logger.info("Hit the barrier! - 1")
                cuda_gdb_p.sendline(NEXT)
                cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                flag_step = 1
            elif j == 2:
                cuda_gdb_p.sendline()
                logger.info("Send enter in stepi 1")
                flag_step = 1
            elif j == 3:
                cuda_gdb_p.sendline()
                logger.info("Send enter in stepi 2")
            elif j == 4:
                logger.info("Hit the barrier! - 2")
                cuda_gdb_p.sendline(NEXT)
                cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                flag_step = 1
            else:
                flag_step = 1
        i = -1
        cuda_gdb_p.sendline(CUDA_THREAD_INFO)
        while flag_info == 0:      
            i = cuda_gdb_p.expect([CUDA_GDB_EXPECT,THREAD_CONTINUE_EXPECT,THREAD_CONTINUE_EXPECT_WERIED,CUDA_SYN_EXPECT])
            temp = cuda_gdb_p.before
            logger.error("in threadinfo "+temp)
            if i == 0:
                if NO_FOCUS in temp and "Switching" not in temp:
                    logger.info("CONTINUE THREADS to hit breakpoint again! -3")
                    cuda_gdb_p.sendline(CONTINUE)
                    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                    temp = cuda_gdb_p.before
                    logger.info("temp 1 "+temp)
                    time.sleep(5)
                if CUDA_SYN_EXPECT in temp:
                    logger.info("Hit the barrier!")
                    cuda_gdb_p.sendline(NEXT)
                    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                flag_info = 1
                
            elif i == 1:
                
                cuda_gdb_p.sendline()
                logger.info("Send enter to continue!")
            elif i == 2:
                
                cuda_gdb_p.sendline()
                logger.info("Send enter to continue!")
            elif i == 3:
                logger.info("Hit the barrier! - 4")
                cuda_gdb_p.sendline(NEXT)
                cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                flag_info = 1 
            else :
                flag_info = 1
    cuda_gdb_p.sendline(QUIT)
    #cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #------------------------
    # get the target register
    #------------------------
def main():
    global logger
    logger = logging.getLogger(configure.benchmark+"profiler")
    hdlr = logging.FileHandler(configure.profile_file)
    formatter = logging.Formatter("%(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    for trial in range(1):
        profiler(configure.binary_path,0,trial)
        
main()
    
