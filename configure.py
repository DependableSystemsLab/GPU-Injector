#configuration for the kernel 
startline = "aesEncrypt256_kernel.h:61"
benchmark = "aes"
kernel = "aesEncrypt256"
multiple_kernel = 0
kernel_number = ['86','87']
#configuration for the profile
model = "telsa2070"  # GPU model for multiple GPUs
profile_file = benchmark+"_"+kernel+"_"+model+"_"+"profiler.log"
#profile_file = "sadprofiler.log.kernel2"

#configuration for the injection
#startingpc = 8502888  # the first pc from the profile, will be updated by Update.py
sm =14  # number of StreamingMultiprocessor

instruction_counter = 300 # better to keep it like this
instruction_random = 0  # better to keep it like this

#configuration for launching the benchmark
parameter = " e 256 ~/NVIDIA_GPU_Computing_SDK/C/src/AES/data/output.bmp ~/NVIDIA_GPU_Computing_SDK/C/src/AES/data/key256.txt"
#parameter = " -i ~/parboil/datasets/sad/default/input/reference.bin,/home/bo/parboil/datasets/sad/default/input/frame.bin -o output"
binary_path = "/home/bo/NVIDIA_GPU_Computing_SDK/C/bin/linux/debug/aescuda"

#correctness check
outputfile = "/home/bo/topK_thrust/topK/output/beamOutput.txt"
comparestring = "/home/bo/parboil/benchmarks/sad/tools/compare-output /home/bo/parboil/datasets/sad/default/output/out.bin ./output"
checkstring = "PASSED" #or "PASS" depending on version of cuda-gdb
