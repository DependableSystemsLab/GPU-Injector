#configuration for the kernel 
startline = "aesEncrypt256_kernel.h:61"
benchmark = "aes"
kernel = "aesEncrypt256"
multiple_kernel = 0
kernel_number = ['86','87'] # will also vary , user might have to change this
#configuration for the profile
model = "telsa2070"  # GPU model for multiple GPUs
profile_file = benchmark+"_"+kernel+"_"+model+"_"+"profiler.log"
#profile_file = "sadprofiler.log.kernel2"

#required parameters for Update.py
name="Matrix_Multiplication"
filename="matrixMul.cu"
#binary path will vary 
binary_path = "/home/sudeep/matrixMul" # where the binary executable of the application is located
 #if compiling is also done by Update.py then uncomment below line
 #options="-I /home/sudeep/common/inc " #include the appropriate libraries


#configuration for the injection
#startingpc = 8502888  # the first pc from the profile, will be updated by Update.py
sm =14  # number of StreamingMultiprocessor

instruction_counter = 300 # better to keep it like this
instruction_random = 0  # better to keep it like this

#configuration for launching the benchmark
parameter = " e 256 ~/NVIDIA_GPU_Computing_SDK/C/src/AES/data/output.bmp ~/NVIDIA_GPU_Computing_SDK/C/src/AES/data/key256.txt"
#parameter = " -i ~/parboil/datasets/sad/default/input/reference.bin,/home/bo/parboil/datasets/sad/default/input/frame.bin -o output"


#correctness check
outputfile = "/home/bo/topK_thrust/topK/output/beamOutput.txt"
comparestring = "/home/bo/parboil/benchmarks/sad/tools/compare-output /home/bo/parboil/datasets/sad/default/output/out.bin ./output"
#checkstring varies depending on 
checkstring = "PASSED" #or "PASS" depending on version of cuda-gdb
