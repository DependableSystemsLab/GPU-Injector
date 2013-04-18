#configuration for the kernel 
startline = "jds_kernels.cu:9"
benchmark = "spmv"
kernel = "spmv_jds"
multiple_kernel = 0
kernel_number = ['86','87']
#configuration for the profile
profile_file = benchmark+"_"+kernel+"_"+"profiler.log"
#profile_file = "spmvprofiler.log"

#configuration for the injection

instruction_counter = 50 
instruction_random = 50

#configuration for launching the benchmark

parameter = " -i ~/parboil/datasets/spmv/small/input/1138_bus.mtx,/home/bo/parboil/datasets/spmv/small/input/vector.bin -o output"
binary_path = "/home/bo/parboil/benchmarks/spmv/build/cuda_arch20/spmv"

#correctness check
outputfile = "/home/bo/topK_thrust/topK/output/beamOutput.txt"
comparestring = "/home/bo/parboil/benchmarks/spmv/tools/compare-output ~/parboil/datasets/spmv/small/output/1138_bus.mtx.out ./output"
checkstring = "Passed"
