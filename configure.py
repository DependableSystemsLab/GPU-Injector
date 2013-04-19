#configuration for the kernel 
startline = "bucket_query.cu:9"
benchmark = "topK_rand"
kernel = ""
multiple_kernel = 0
kernel_number = ['86','87']
node = ""
#configuration for the profile
profile_file = benchmark+"_"+kernel+"_"+node+"_"+"profiler.log"
#profile_file = "spmvprofiler.log"

#configuration for the injection

instruction_counter = 50 
instruction_random = 50

#configuration for launching the benchmark

parameter = " 12 4096 0.5 1 1 /home/bo/topK_thrust/topK/data/testList_12 /home/bo/topK_thrust/topK/data/testListSorted_12 ALL_K CHECK 3589950516"
binary_path = "/home/bo/topK_thrust/topK/bin/linux/topK_rand"

#correctness check
outputfile = "/home/bo/topK_thrust/topK/output/beamOutput.txt"
comparestring = "/home/bo/parboil/benchmarks/spmv/tools/compare-output ~/parboil/datasets/spmv/small/output/1138_bus.mtx.out ./output"
checkstring = "Pass"
