#configuration for the kernel 
startLine = "bucket_query.cu:140"
benchmark = "topK_rand"
kernel = "CompleteKthBucket"
#configuration for the profile
profile_file = benchmark+"_"+kernel+"_"+"profiler.log"

#configuration for the injection

instruction_counter = 50 
instruction_random = 50

#configuration for launching the benchmark

parameter = " 12 4096 0.5 1 1 /home/bo/topK_thrust/topK/data/testList_12 /home/bo/topK_thrust/topK/data/testListSorted_12 ALL_K CHECK 3589950516"
binary_path = "/home/bo/topK_thrust/topK/bin/linux/topK_rand"
outputfile = "/home/bo/topK_thrust/topK/output/beamOutput.txt"
