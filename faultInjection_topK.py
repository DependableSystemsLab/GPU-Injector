import sys
import pexpect
import logging
import random
import re
from subprocess import Popen, PIPE, STDOUT
from datetime import datetime
import hashlib
from sets import Set
import time
import StringIO
import subprocess
import os
import signal
import configure
profile = {}
kernel_dim_map = {}

#---------------------
# System configuration
#---------------------
logger = None
#------------------------
#CUDA-GDB commands
#------------------------
CUDA_GDB_PATH = "cuda-gdb"
BREAKPOINT = "break "
BREAK_LOCATION = configure.startline
BREAK_LOCATION2 = "bucket_query.cu:970"
PRINT_PC = "print $pc"
CURRENT_INSTRUCTION = "display/i $pc"
NEXT_INSTRUCTION = "display/2i $pc"
STEPI = "stepi"
MODIFY_REGISTER = "print "
RUN = "run"
CONTINUE = "continue"
REGISTER = "info registers "
THREAD = "cuda kernel block thread"
DELETE_BREAKPOINT = "delete breakpoint 1"
DELETE_BREAKPOINT2 = "delete breakpoint 2"
DELETE_BREAKPOINT3 = "delete breakpoint 3"
QUIT= "quit"
KILL = "kill"
CUDA_KERNEL = "cuda kernel"
ARGUMENT = configure.parameter

#------------------------
#Expect collection
#------------------------

CUDA_GDB_EXPECT = "\(cuda-gdb\)"
BREAKPOINT_EXPECT = "Breakpoint *"
EOL_EXPECT = "\r\n"
MODIFY_REGISTER_EXPECT = "$1"
RUN_EXPECT = "Breakpoint 1"
CONTINUE_EXPECT = "Breakpoint 1"
CURRENT_INSTRUCTION_EXPECT = ">:"
THREAD_EXPECT = "kernel "
DELETE_BREAKPOINT_EXPECT = "\r\n"

#-------------------------
# Results
#-------------------------
CUDA_EXCEPTION = "CUDA_EXCEPTION_"
PASS = ""
SIGTRAP = "Program received signal SIGTRAP"
#--------------------
# BREAKPOINT format
#--------------------
# i.e. break file:line if thread = thread && block = block && $pc = 0xXX
# record format kernelnumber, blockx, blocky, blockz, threadx, thready, threadz, pc 

def killProcess(name):
    p = Popen(['ps','-A'],stdout=subprocess.PIPE)
    out, err = p.communicate()
    for line in out.splitlines():
            if name in line and "["+name+"]" not in line and "--color=auto" not in line :
                pid = int(line.split(None,1)[0])
                print str(pid)
                try:
                  os.kill(pid,signal.SIGKILL)
                  print "kill "+name
                except OSError,err:
                  print "No such process!"
            print line

def runDiff(cmd):
    p = Popen(cmd, shell=True, stdout= PIPE, stderr = PIPE)
    stdout,stderr = p.communicate()
    if len(stdout)!= 0 or len(stderr) != 0:
        return -1
    else:
        return 0

def checkFile(path):
    try:
       with open(path) as outfile:
           for item in outfile:
                if " CORRECT" in item:
                    return 0
           return -1
    except IOError:
       return -1
    
def runChecker(cmd,check_str):
    p = Popen(cmd,shell=True,stdout=PIPE,stderr = PIPE)
    stdout, stderr = p.communicate()
    print stdout
    print stderr
    if check_str in stdout:
        return 0;
    else:
        return -1;
    
class Symbols:
    
    def __init__(self):
        self.opcode = ""
        self.operand = []
        self.isRepetitive = 0
        self.isValid = 0

def determineIteration(ln,item,log):
    counter = 0
    lp_counter = 0
    for line in log:
        lp_counter = lp_counter +1
        if lp_counter > ln:
            break
        if cmp(line[0], item[0]) == 0 and cmp(line[1], item[1]) == 0 and cmp(line[2],item[2]) == 0 and cmp(line[3], item[3]) ==0 and cmp(line[6], item[6])  == 0  and cmp(line[5], item[5]) == 0:
            counter = counter +1
    return counter

        
def processLog(logfile):
    global profile
    global kernel_dim_map
    hashtable = Set()
    log = open(logfile,"r")
    flag = 0
    current_kernel = "";
    for line in log:
        if "Launch" in line and "Kernel " in line and "<<<" in line:
            line = line.lstrip().rstrip("\r\n")
            temp1 = line.split(" ")
            kernel_id = temp1[5]
            temp2 = temp1[6]
            temp3 = temp2.split("<<<")[1]
            temp4 = temp3.split(">>>")[0]
            temp5 = temp4.split("),(")
            temp6 = temp5[0]
            temp7 = temp5[1]
            temp6 = temp6+")"
            temp7 = "("+temp7
            dim_b = [int(i) for i in temp6[1:-1].split(',')]
            dim_t = [int(i) for i in temp7[1:-1].split(',')]
            if kernel_id not in kernel_dim_map:
                kernel_dim_map[kernel_id] = []
            kernel_dim_map[kernel_id].append(dim_b)
            kernel_dim_map[kernel_id].append(dim_t)
             
        if "Kernel " in line and "<<<" not in line:
            line = line.lstrip().rstrip("\r\n")
            kernel_id = line.split(" ")[1]
            if kernel_id not in profile:
                profile[kernel_id] = [] 
                current_kernel = kernel_id              
            flag = 1
        if flag == 1:
            if "Kernel " not in line:
                if re.search("\(\d*,\d*,\d*\)",line) and "focus" not in line and "Breakpoint" not in line and "kernel " not in line and "in tex" not in line and "info cuda threads block" not in line and "in matrixMul" not in line and "<<<" not in line:
                    print line
                    thread_from = ""
                    thread_end = ""
                    block_from = ""
                    block_end = ""
                    file_name = ""
                    count = ""
                    pc = ""
                    no_line = ""
                    temp = []
                    line = line.lstrip(" ").rstrip("\r\n")
                    item = line.split(" ")
                    for sub in item:
                        if sub != "":
                            temp.append(sub)
                    if temp[0] == "*":
                        block_from = temp[1]
                        thread_from = temp[2]
                        block_end = temp[3]
                        thread_end = temp[4]
                        count = temp[5]
                        pc = temp[6]
                        file_name = temp[7]
                        no_line = temp[8]
                    else :
                        block_from = temp[0]
                        thread_from = temp[1]
                        block_end = temp[2]
                        thread_end = temp[3]
                        count = temp[4]
                        pc = temp[5]
                        if len(temp) < 7:
                            temp.append("cmath")
                        file_name = temp[6]
                        if len(temp) > 7:
                            no_line = temp[7]
                    if int(count) <= 32 and no_line !='' and int(count) != 24  and file_name != "cmath" and file_name !="vector_functions.h" and file_name!="texture_fetch_functions.h" and  file_name != "device_functions.h" and file_name != "ci_include.h" and file_name!= "math_functions.h": 
                        
                        dim_b_from = [i for i in block_from[1:-1].split(',')]
                        dim_b_end = [i for i in block_end[1:-1].split(',')]
                        dim_t_from = [i for i in thread_from[1:-1].split(',')]
                        dim_t_end = [i for i in thread_end[1:-1].split(',')]
                        lis = []
                        lis.append(dim_b_from)
                        lis.append(dim_b_end)
                        lis.append(dim_t_from)
                        lis.append(dim_t_end)
                        lis.append(count)
                        lis.append(pc)
                        lis.append(file_name)
                        lis.append(no_line)
                        message = dim_b_from[0] + dim_b_from[1] + dim_b_from[2] + dim_b_end[0] + dim_b_end[1] + dim_b_end[2] + dim_t_from[0] + dim_t_from[1] + dim_t_from[2] + dim_t_end[0] + dim_t_end[1] + dim_t_end[2] + count + pc + file_name + no_line
                        m = hashlib.md5()
                        m.update(message)
                        hashtable.add(m.digest()) 
                        if m.digest() in hashtable:
                            profile[current_kernel].append(lis)
                            print lis

    duplicate = []
    print profile.keys()
    if configure.multiple_kernel == 1: 
        for item in configure.kernel_number:
            if item not in profile.keys():
                profile[item] = []
        for item in profile.keys():
            if len(profile[item]) != 0:
                duplicate = list(profile[item])
        print profile.keys()
        for item in profile.keys():
            if len(profile[item]) == 0:
                profile[item].extend(duplicate)
                print len(profile[item])

                        
def generateBreakpoint(breaklist,kernel_id):
    breakpoint = ""
    #get total number of threads in this span
    index = random.randint(0,int(breaklist[4])-1)
    gridx = int(kernel_dim_map[kernel_id][0][0])
    gridy = int(kernel_dim_map[kernel_id][0][1])
    gridz = int(kernel_dim_map[kernel_id][0][2])
    blockx = int(kernel_dim_map[kernel_id][1][0])
    blocky = int(kernel_dim_map[kernel_id][1][1])
    blockz = int(kernel_dim_map[kernel_id][1][2])
    if gridx > 16:
        gridx = 8
    bidx = random.randint(0,gridx-1)
    bidy = random.randint(0,gridy-1)
    if bidy > 2:
        bidy = 0
    bidz = random.randint(0,gridz-1)
    count = random.randint(0,(int(breaklist[4])-1))
    #tidx = random.randint(0,blockx-1)
    tidy = random.randint(0,blocky-1)
    tidz = random.randint(0,blockz-1)
    tidx = int(breaklist[2][0])+count
    #tidy = int(breaklist[2][1])+count
    #tidz = int(breaklist[2][2])
    breakstr = breaklist[6]+":"+breaklist[7]+" if blockIdx.x == "+str(bidx)+" && blockIdx.y == "+str(bidy)+" && blockIdx.z == "+str(bidz)+" && threadIdx.x == "+str(tidx)+ " && threadIdx.y == "+str(tidy)+" && threadIdx.z == "+str(tidz)
    print breakstr
    return breakstr
            
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def generateFaults(value):
    global logger
    if is_number(value) == False:
        logger.info("Non-numeric register!")
        return "Non-numeric"
    pos = random.randint(0,31)
    mask = (1 << pos)
    logger.info("New value is "+str(int(value)^mask))
    return str(int(value)^mask)

def getRegisterSymbols(instruction):
    global logger
    symbol = Symbols()
    if "EXIT" in instruction or instruction == "BRK":
        symbol.isValid = -1
        return symbol
    instruction = instruction.lstrip()
    instruction = instruction.rstrip("\r\n")
    logger.info("get symbol from "+instruction)
    
    #in case instructions like @P0 BRK
    if "," not in instruction:
        instruction = instruction+","
    ops = instruction.split(",")
    opcode = ops[0].split(" ")
    if instruction.startswith("@P") == True or instruction.startswith("@!P") == True : 
        if len(opcode) > 1:
          symbol.opcode = opcode[0]+opcode[1]
        else:
          symbol.opcode = opcode[0]
    else:
        symbol.opcode = opcode[0]
    if len(opcode) == 1:
        print opcode
        symbol.isValid = -1
        return symbol
    if "[" in opcode[len(opcode)-1]:
        sub = opcode[len(opcode)-1][opcode[len(opcode)-1].find("[")+1:opcode[len(opcode)-1].find("]")]
        if "+" in sub:
            suboperand = sub.split("+")
            symbol.operand.append(suboperand[0])
            if "R" in suboperand[1]:
                symbol.operand.append(suboperand[2])
        else:
            symbol.operand.append(sub)
    else:
        symbol.operand.append(opcode[len(opcode)-1])
    for i in range(1,len(ops)):
        ops[i] = ops[i].lstrip().rstrip();
        opsxs = ops[i].split(" ")
        for opsx in opsxs:
            if "R" in opsx:
                if "[" in opsx:
                    sub = opsx[opsx.find("[")+1:opsx.find("]")]
                    if "+" in sub:
                        suboperand = sub.split("+")
                        symbol.operand.append(suboperand[0])
                        if "R" in suboperand[1]:
                            symbol.operand.append(suboperand[2]) 
                    else:
                        symbol.operand.append(sub)
                else:
                    symbol.operand.append(opsx)
    logger.info("Symbol: "+symbol.opcode)
    list_temp = symbol.operand
    for j in range(0,len(symbol.operand)):
        if (list_temp[0] == symbol.operand[j] or list_temp[0] == symbol.operand[j]+".CC" or list_temp[0]+".CC" == symbol.operand[j]) and j != 0:
            symbol.isRepetitive = 1
        logger.info(symbol.operand[j])
    return symbol
        

def getTargetRegister(instruction):
    symbol = getRegisterSymbols(instruction)
    if symbol.isValid != 0:
        return "Invalid"
    if "LD" in symbol.opcode:
        return symbol.operand[1]                         
    else :
        if "." in symbol.operand[0]:
            return symbol.operand[0].split(".")[0]
        else:
            return symbol.operand[0]
        

def checkActivated(reg,instruction):
    symbol = getRegisterSymbols(instruction)
    logger.info("checking instruction "+instruction)
    if "ST" in symbol.opcode or "LD" in symbol.opcode:
        for operand in symbol.operand:
            if reg == operand:
                return 1
        return 0
    else:
        res = 0
        if symbol.isRepetitive == 1:
            if reg == symbol.operand[0] or reg+".CC" == symbol.operand[0]:
                return 0         
        for operand in symbol.operand:
            if operand == symbol.operand[0]:
                if operand == reg or operand == reg+".CC":
                    res = res + 1
            else:
                if operand == reg or operand == reg+".CC":
                    res = res + 2
        if res %2 == 1:
            return 2
        elif res > 0 and res%2 == 0:
            return 1
        else:
            return 0

    
def faultMain(path,trigger,trial,pc,kernel,iteration):
    global CUDA_GDB_PATH, BREAKPOINT,BREAK_LOCATION,CURRENT_INSTRUCTION,SETPI,MODIFY_REGISTER,REGISTER,THREAD,DELETE_BREAKPOINT,KILL,DELETE_BREAKPOINT2,BREAK_LOCATION2, DELETE_BREAKPOINT3
    global CUDA_GDB_EXPECT,BREAKPOINT_EXPECT,EOL_EXPECT,RUN_EXPECT,CONTINUE_EXPECT,CURRENT_INSTRUCTION_EXPECT,THREAD_EXPECT,DELETE_BREAKPOINT_EXPECT,PRINT_PC,ARGUMENT,NEXT_INSTRUCTION,SIGTRAP
    global logger
    assert_flag = 0
    cuda_gdb_p = pexpect.spawn(CUDA_GDB_PATH+" "+path)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #---------------
    # set breakpoint
    #---------------
    cuda_gdb_p.sendline(BREAKPOINT+" "+BREAK_LOCATION)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
#    cuda_gdb_p.sendline(BREAKPOINT+" "+BREAK_LOCATION2)
#    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #---------------
    # run the program
    #---------------
    cuda_gdb_p.sendline(RUN+ARGUMENT)
    j = cuda_gdb_p.expect([pexpect.TIMEOUT, CUDA_GDB_EXPECT],timeout=60)
    if j == 0:
        logger.info("Error happened! Terminated! 1")
        killProcess(configure.benchmark)
        cuda_gdb_p.terminate(force=True)
        cuda_gdb_p.close()
        return
    #---------------------
    # reset the breakpoint
    #---------------------
    #logger.info(trigger)
    rawstr = cuda_gdb_p.before
    print rawstr
    while "Kernel "+kernel not in rawstr:
        cuda_gdb_p.sendline(CONTINUE)
        cuda_gdb_p.expect(CUDA_GDB_EXPECT,timeout= 60)
        rawstr = cuda_gdb_p.before
        #logger.info(rawstr)
    # if the breakpoint is the same as initial breakpoint, do not delete 
    time.sleep(2)
    #cuda_gdb_p.sendline(BREAKPOINT+" "+BREAK_LOCATION2)
    #cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #print (cuda_gdb_p.before)
    #cuda_gdb_p.sendline(CONTINUE)
    #cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #print (cuda_gdb_p.before)
    if BREAK_LOCATION in trigger:
        pass
    elif BREAK_LOCATION2 in trigger:
        cuda_gdb_p.sendline(DELETE_BREAKPOINT)
        cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    else:
        cuda_gdb_p.sendline(DELETE_BREAKPOINT)
        cuda_gdb_p.expect(CUDA_GDB_EXPECT)
        #cuda_gdb_p.sendline(DELETE_BREAKPOINT2)
        #cuda_gdb_p.expect(CUDA_GDB_EXPECT)
        print trigger
        cuda_gdb_p.sendline(BREAKPOINT+" "+trigger)
        cuda_gdb_p.expect(CUDA_GDB_EXPECT)
        logger.info(cuda_gdb_p.before)
        #cuda_gdb_p.delaybeforesend = 0
        cuda_gdb_p.sendline(CONTINUE)
        j = cuda_gdb_p.expect([pexpect.TIMEOUT,CUDA_GDB_EXPECT],timeout=60)
        if j == 0:
            logger.info("Error happened ! Terminated! 2")
            killProcess(configure.benchmark)
            cuda_gdb_p.terminate(force=True)
            cuda_gdb_p.close()
            return 
        else :
            res_continue = cuda_gdb_p.before
            if "Program exited" in res_continue:
                logger.info("Cannot hit the breakpoint!")
                killProcess(configure.benchmark)
                cuda_gdb_p.terminate(force=True)
                cuda_gdb_p.close()
                return 
    # need to see if it is in the loop and jump over iterations
    logger.info("begin to see how many iterations we need to jump "+str(iteration))
    for iter in range(0,iteration-1):
        cuda_gdb_p.sendline(CONTINUE)
        j = cuda_gdb_p.expect([pexpect.TIMEOUT,CUDA_GDB_EXPECT],timeout=60)
        res = cuda_gdb_p.before
        logger.info(res)
        if j == 0:
            logger.info("Error happened ! Terminated! 5")
            killProcess(configure.benchmark)
            cuda_gdb_p.terminate(force=True)
            cuda_gdb_p.close()
            return 
        else :
            res_continue = cuda_gdb_p.before
            if "Program exited" in res_continue or "Switching " in res_continue:
                logger.info("Cannot hit the breakpoint!i 5")
                killProcess(configure.benchmark)
                cuda_gdb_p.terminate(force=True)
                cuda_gdb_p.close()
                return 
    i = 0
    counter_i= 0
    rand_counter = random.randint(0,configure.instruction_random)
    while i == 0:
        cuda_gdb_p.sendline(PRINT_PC)
        cuda_gdb_p.expect(CUDA_GDB_EXPECT,timeout=600)
        pcline = cuda_gdb_p.before
        logger.info(pcline)
        pcline = pcline.lstrip(" ").rstrip("\r\n")
        pcline_list = pcline.split(" ")
        pc_c = pcline_list[len(pcline_list)-1]
        logger.info("pc is "+str(pc))
        logger.info("new pc is "+str(pc_c))
        if "registers" in pc_c or is_number(pc_c) == False or "executing" in pc_c:
                logger.info("Cannot hit pc in stepi")
                killProcess(configure.benchmark)
                cuda_gdb_p.terminate(force=True)
                #cuda_gdb_p.close()
                return 
        if int(pc_c) != pc:
            cuda_gdb_p.sendline(STEPI)
            k = cuda_gdb_p.expect([pexpect.TIMEOUT,CUDA_GDB_EXPECT],timeout=120)
            if k == 0:
                logger.info("Cannot hit pc in stepi")
                killProcess(configure.benchmark)
                cuda_gdb_p.terminate(force=True)
                #cuda_gdb_p.close()
                return 
            res_pc = cuda_gdb_p.before
            logger.info(res_pc)
            if "Termination" in res_pc or "focus" in res_pc or "Focus" in res_pc or "Executing" in res_pc:
                logger.info("Cannot hit pc!")
                killProcess(configure.benchmark)
                cuda_gdb_p.terminate(force=True)
                #cuda_gdb_p.close()
                return
        else :
            i = 1
        counter_i = counter_i +1
        logger.info("counter is "+str(counter_i))
        if counter_i >= rand_counter+configure.instruction_counter:
            logger.info("Cannot hit pc! Pick next random instrcution!")
            i = 1
    #------------------------------
    # check the current instruction
    #------------------------------
    cuda_gdb_p.sendline(CURRENT_INSTRUCTION)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    curr = cuda_gdb_p.before
    #logger.info(curr)
    #------------------------
    # get the target register
    #------------------------
    target = curr.split(CURRENT_INSTRUCTION_EXPECT)
    tar_insn = target[len(target)-1].lstrip().rstrip("\r\n")
    reg = getTargetRegister(target[len(target)-1])
    logger.info("Inject into "+reg)
    if "R" not in reg and "SR" not in reg:
        logger.info("It is a invalid target! Exit")
        killProcess(configure.benchmark)
        cuda_gdb_p.terminate(force=True)
        #cuda_gdb_p.close()
        return 
    # if the instruction is memory instruction, the fault is considered activated immediatley. 
    symbol = getRegisterSymbols(target[len(target)-1])
    # dealing with predicate instuctions
    preDest = ""
    preDestValue = ""
    if "@P" in symbol.opcode or "@!P" in symbol.opcode:
        preDest = symbol.operand[0]
        cuda_gdb_p.sendline(MODIFY_REGISTER+"$"+preDest)
        cuda_gdb_p.expect(CUDA_GDB_EXPECT)
        preDestVList = cuda_gdb_p.before.lstrip().rstrip("\r\n").split("\t")
        preDestValue = preDestVList[len(preDestVList)-1]
    flag = 0
    mem_insn = ""
    mem_value_before = ""
    if "ST" in symbol.opcode or "LD" in symbol.opcode:
        flag = 1
        mem_insn = tar_insn
        cuda_gdb_p.sendline(REGISTER+reg)
        cuda_gdb_p.expect(CUDA_GDB_EXPECT)
        mem_value_before = cuda_gdb_p.before.lstrip().rstrip("\r\n").split("\t")
        logger.info(mem_value_before)
        #-----------------
        #inject the fault 
        #-----------------
        fault = generateFaults(mem_value_before[len(mem_value_before)-1])
        if fault == "Non-numeric":
            killProcess(configure.benchmark)
            cuda_gdb_p.terminate(force=True)
            cuda_gdb_p.close()
            return
        cuda_gdb_p.sendline(MODIFY_REGISTER+"$"+reg+" = "+str(fault))
        cuda_gdb_p.expect(CUDA_GDB_EXPECT)
        
    #need to obtain next instruction to make the last one get executed
    cuda_gdb_p.sendline(STEPI)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    res = cuda_gdb_p.before
    if CUDA_EXCEPTION in res:
              logger.info(res)
              logger.info("At trial "+str(trial) +" fault in register "+reg+" is activated")
              logger.info("At trial "+str(trial)+" fault in register "+reg+" crashed"+" latency is 0")
              killProcess(configure.benchmark)
              cuda_gdb_p.terminate(force=True)
              cuda_gdb_p.close()
              logger.info("Trail "+str(trial)+" finishes!")
              return
    if flag == 1 and symbol.isRepetitive != 1:
        cuda_gdb_p.sendline(MODIFY_REGISTER+"$"+reg+" = "+mem_value_before[len(mem_value_before)-1])
        cuda_gdb_p.expect(CUDA_GDB_EXPECT)
        logger.info("Memory instruction change reg "+reg+" back to "+mem_value_before[len(mem_value_before)-1])
    else:    
        instructions = res.split(CURRENT_INSTRUCTION_EXPECT)
        instruction = instructions[len(instructions)-1].lstrip().rstrip("\r\n")
        #get the value of the register
        cuda_gdb_p.sendline(REGISTER+reg)
        cuda_gdb_p.expect(CUDA_GDB_EXPECT)
        value_before = cuda_gdb_p.before.lstrip().rstrip("\r\n").split("\t")
        logger.info(value_before)
        if value_before == preDestValue:
            logger.info("Predicated instruction is not executed!")
            killProcess(configure.benchmark)
            cuda_gdb_p.terminate(force=True)
            cuda_gdb_p.close()
            return        
        #-----------------
        #inject the fault 
        #-----------------
        fault = generateFaults(value_before[len(value_before)-1])
        if fault == "Non-numeric":
            killProcess(configure.benchmark)
            cuda_gdb_p.terminate(force=True)
            cuda_gdb_p.close()
            return
        cuda_gdb_p.sendline(MODIFY_REGISTER+"$"+reg+" = "+str(fault))
        cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #-------------------------------
    #check if the fault is activated
    #-------------------------------
    isActivated = 0 # 0: not activated 1: activated 2: overwritten
    if flag == 1:
        isActivated = 1
        logger.info("At trial "+str(trial) +" fault in register "+reg+"is activated at instruction "+mem_insn)
    else :
        isActivated = checkActivated(reg,instruction)
        if isActivated == 1:
            logger.info("At trial "+str(trial) +" fault in register "+reg+"is activated at instruction "+instruction)
        if isActivated == 2:
            logger.info("At trial "+str(trial) +" fault in register "+reg+"is overwritten at instruction "+instruction)
    res = ""
    counter = 0
    last_inst = ""
    isPredicated = 0
    while isActivated == 0:
        if CUDA_EXCEPTION in res:
              logger.info(res)
              logger.info("At trial "+str(trial) +" fault in register "+reg+" is activated")
              logger.info("At trial "+str(trial)+" fault in register "+reg+" crashed"+" latency is 0")
              killProcess(configure.benchmark)
              cuda_gdb_p.terminate(force=True)
              #cuda_gdb_p.close()
              logger.info("Trail "+str(trial)+" finishes!")
              return
        if SIGTRAP in res:
              logger.info(res)
              logger.info("At trial "+str(trial) +" fault in register "+reg+" is activated")
              logger.info("At trial "+str(trial)+" fault in register "+reg+" trapped")
              killProcess(configure.benchmark)
              cuda_gdb_p.terminate(force=True)
              cuda_gdb_p.close()
              logger.info("Trail "+str(trial)+" finishes!")
              return
        if "Focus" not in res:
            cuda_gdb_p.sendline(STEPI)
            k = cuda_gdb_p.expect([pexpect.TIMEOUT,CUDA_GDB_EXPECT],timeout=300)
            if k == 0:
                logger.info("Error happened ! Terminated! 3")
                killProcess(configure.benchmark)
                cuda_gdb_p.terminate(force=True)
                cuda_gdb_p.close()
                return 
            res = cuda_gdb_p.before
            instructions = res.split(CURRENT_INSTRUCTION_EXPECT)
            instruction = instructions[len(instructions)-1].lstrip().rstrip("\r\n")
            symbol_check = getRegisterSymbols(instruction)
            if "@P" in symbol_check.opcode or "@!P" in symbol_check.opcode:
                preDest = symbol_check.operand[0]
                cuda_gdb_p.sendline(MODIFY_REGISTER+"$"+preDest)
                cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                preDestVList = cuda_gdb_p.before.lstrip().rstrip("\r\n").split("\t")
                preDestValue = preDestVList[len(preDestVList)-1]
                isPredicated = 1
                last_inst = instruction
            else:
                if isPredicated == 1:
                    cuda_gdb_p.sendline(MODIFY_REGISTER+"$"+preDest)
                    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                    preDestVList = cuda_gdb_p.before.lstrip().rstrip("\r\n").split("\t")
                    preDestValue_new = preDestVList[len(preDestVList)-1]
                    if preDestValue != preDestValue:
                        logger.info("Predicate inst is executed! Check")
                        isActivated = checkActivated(reg,last_inst)
                        if isActivated == 0:
                            isActivated = checkActivated(reg,instruction)
                    else:
                        logger.info("Predicate inst is not executed! Skip!")
                        isActivated = checkActivated(reg,instruction)
                    isPredicated = 0
                else:         
                    isActivated = checkActivated(reg,instruction)
            if isActivated == 2:
                logger.info("At trial "+str(trial) +" fault in register "+reg+" is overwritten at instruction "+instruction)
            if isActivated == 1:
                logger.info("At trial "+str(trial) +" fault in register "+reg+" is activated at instruction "+instruction)
        else:
            logger.info("At trial "+str(trial) +" fault in register "+reg+" is not observed at instruction "+instruction)
            isActivated = 3
        counter = counter + 1
        if counter >= 800:
            logger.info("At trial "+str(trial) +" fault in register "+reg+" is not observed")
            break
    cuda_gdb_p.sendline(DELETE_BREAKPOINT)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    cuda_gdb_p.sendline(DELETE_BREAKPOINT2)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    cuda_gdb_p.sendline(DELETE_BREAKPOINT3)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    start = datetime.now()
    cuda_gdb_p.sendline(CONTINUE)
    # 1 timeout 2 pass 3 exception
    try:
        i = cuda_gdb_p.expect([pexpect.TIMEOUT,CUDA_GDB_EXPECT],timeout=120)
        if i == 0:
            logger.info("At trial "+str(trial)+" fault in register "+reg+" is hang ")
            killProcess(configure.benchmark)
            cuda_gdb_p.terminate(force=True)
            #cuda_gdb_p.close()
            return
        elif i == 1:
            logs = cuda_gdb_p.before
            logger.info(logs)
            buf = StringIO.StringIO(logs)
            log = buf.readlines()
            for item in log:
                if "Caught" in item:
                    assert_flag = 1 
            #logger.info(logs)
            if CUDA_EXCEPTION in logs:
                logger.info(logs)
                end = datetime.now()
                latency = end-start
                logger.info("At trial "+str(trial)+" fault in register "+reg+" crashed"+" latency is "+str(latency.seconds)+"s "+str(latency.microseconds)+"micros")
                if assert_flag == 1:
                    logger.info("crash_and_asserted")
                killProcess(configure.benchmark)
                cuda_gdb_p.terminate(force=True)
                #cuda_gdb_p.close()
                return
            elif SIGTRAP in logs:
                logger.info(logs)
                end = datetime.now()
                latency = end-start
                logger.info("At trial "+str(trial)+" fault in register "+reg+" trapped")
                killProcess(configure.benchmark)
                cuda_gdb_p.terminate(force=True)
                cuda_gdb_p.close()
                return
            else: 
                #compare the results
                ret = checkFile(configure.outputfile)
                #ret = runChecker(configure.comparestring,configure.checkstring)
                if ret > 0:
                        logger.info("At trial "+str(trial)+" fault in register "+reg+" executed correctly")
                        if assert_flag == 1:
                            logger.info("correct_and_asserted")
                else :
                        logger.info("At trial "+str(trial)+" fault in register "+reg+" executed incorrectly")
                        if assert_flag == 1:
                            logger.info("incorrect_and_asserted")
            
    except pexpect.TIMEOUT:
        logger.info("At trial "+str(trial)+" fault in register "+reg+" is hang ")
        if assert_flag == 1:
            logger.info("hang_and_asserted")
        killProcess(configure.benchmark)
        cuda_gdb_p.terminate(force=True)
        cuda_gdb_p.close()
        return 
    logger.info("Trail "+str(trial)+" finishes!")
    logger.info("\n")
        
        

def main():
    global logger
    logger = logging.getLogger(configure.benchmark)
    hdlr = logging.FileHandler(configure.benchmark+"_"+configure.kernel+'.log')
    formatter = logging.Formatter("%(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    processLog(configure.profile_file)
    for trial in range(3000):
        #randomly generate a breakpoint
        num_kernels = len(profile.keys())
        random.seed()
        kernel = random.randint(0,num_kernels-1)
        #print kernel
        num_items = len(profile[str(profile.keys()[kernel])])
        item = random.randint(0,num_items-1)
        print item
        print profile[str(profile.keys()[kernel])][item]
        if profile[str(profile.keys()[kernel])][item][len(profile[str(profile.keys()[kernel])][item])-1] == "":
            if item != 0:
                item = item -1
            else:
                item = item +1
        iteration = determineIteration(item,profile[str(profile.keys()[kernel])][item],profile[str(profile.keys()[kernel])])
        breakpoint = generateBreakpoint(profile[str(profile.keys()[kernel])][item],str(profile.keys()[kernel]))
        pc = int(profile[str(profile.keys()[kernel])][item][5],0)
        logger.info("Trial "+str(trial)+" starts!")
        faultMain(configure.binary_path,breakpoint,trial,pc,str(profile.keys()[kernel]),iteration)
        #runDiff("rm "+configure.outputfile)
        time.sleep(5)

main()
    
