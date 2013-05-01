import os
import sys
from sets import Set

rec = {}
rec["EXCEPTION_5"] = []
rec["EXCEPTION_6"] = []
rec["EXCEPTION_10"] = []
rec["EXCEPTION_2"] = []
def processCrash(filename):
    log = open(filename,"r")
    logs = log.readlines()
    new_logs = []
    #rec = {}
    #rec["EXCEPTION_5"] = []
    #rec["EXCEPTION_6"] = []
    #rec["EXCEPTION_10"] = []
    #rec["EXCEPTION_2"] = []
    numloop = 0
    numbranch = 0
    numassign = 0
    incorrectswitch = 0
    loopswitch = 0
    branchswitch= 0
    assignswitch = 0
    loopassign = 0
    loopbranch = 0
    branchassign = 0
    num_all = 0
    lines = {}
    for log in logs:
        if "overwritten at instruction @!P0 SHL" not in log and "CUDA_EXCEPTION_" not in log and "latency" not in log and "starts" not in log and "Caught" not in log and "_and_" not in log:
            pass
        else:
            new_logs.append(log)
            
    for i in range(0,3000):
        print i
        flag = 0
        error = 0
        type_e = ""
        latency = ""
        for log in new_logs:
            if str(i)+" starts" in log:
                flag = 1
                results = []
            if flag == 1:
                if "overwritten at instruction @!P0 SHL" in log:
                    error = 1
                if "CUDA_EXCEPTION_" in log and "signal" in log:
                    type_e = log.split(" ")[3]
                if "latency" in log:
                    latencys = log.split(" ")
                    latency = latencys[len(latencys)-1]
                if "Caught" in log:
                    print log
                    details = log.split(" ")
                
                    if details[1] == "0":
                        loopswitch = 1
                    if details[1] == "1":
                        branchswitch = 1
                    if int(details[1]) >= 2 and int(details[1]) <= 9 :
                        assignswitch = 1
                    results.append(details)
                if "incorrect_and" in log:
                    incorrectswitch= 1
                    print results
                    for item in results:
                        if item[1] not in lines.keys():
                                lines[item[1]]= []
                                lines[item[1]].append(item[4])
                        else:
                            lines[item[1]].append(item[4])
                if str(i+1)+" starts" in log or i == 2999:

                    break
        if error == 0:
            for key in rec.keys():
                if key in type_e:
                    latency = latency.rstrip("\n")
                    if "micros" in latency:
                        latency_num = latency.split("micros")[0]
                        rec[key].append(latency_num)
                    else:
                        rec[key].append(latency)
        if incorrectswitch == 1:
            if loopswitch == 1:
                numloop = numloop +1
            if branchswitch == 1:
                numbranch = numbranch +1
            if assignswitch == 1:
                numassign = numassign +1
            if loopswitch == 1 and branchswitch == 1:
                loopbranch = loopbranch +1
            if loopswitch == 1 and assignswitch == 1:
                loopassign = loopassign +1
            if assignswitch == 1 and branchswitch == 1:
                branchassign = branchassign +1
            if loopswitch == 1 and  assignswitch == 1 and branchswitch == 1:
                num_all = num_all +1
        loopswitch = 0
        branchswitch = 0
        assignswitch = 0
        incorrectswitch = 0
    print "5: "+str(len(rec["EXCEPTION_5"]))
    print rec["EXCEPTION_5"]
    print "6: "+str(len(rec["EXCEPTION_6"]))
    print rec["EXCEPTION_6"]
    print "10: "+str(len(rec["EXCEPTION_10"]))
    print rec["EXCEPTION_10"]
    print "2: "+str(len(rec["EXCEPTION_2"]))
    print rec["EXCEPTION_2"]
    print "num of loop"+str(numloop)
    print "num of branch"+str(numbranch)
    print "num of assign"+str(numassign)
    print "num of loopbranch"+str(loopbranch)
    print "num of loopassign"+str(loopbranch)
    print "num of branchassign"+str(branchassign)
    print "num of all"+str(num_all)
    print lines
    for key in lines.keys():
        sort = {}
        for item in lines[key]:
            if item not in sort.keys():
                sort[item] = []
                sort[item].append(1)
            else: 
                sort[item].append(1)
        print key
        for item in sort.keys():
            print "#### line "+item.rstrip("\n")
            print "#### fequency "+str(len(sort[item]))+"\n"

processCrash("topK_rand.log.complete.node240")
