
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_20 --ptxas-options=-v --use_fast_math

all: reduction reduction_blocks device_var

reduction:	reduction.cu Makefile
		nvcc reduction.cu -o reduction $(INC) $(NVCCFLAGS) $(LIB)

reduction_blocks:	reduction_blocks.cu Makefile
		nvcc reduction_blocks.cu -o reduction_blocks $(INC) $(NVCCFLAGS) $(LIB)

device_var:	device_var.cu Makefile
		nvcc -g device_var.cu -o device_var $(INC) $(NVCCFLAGS) $(LIB)

clean:
		rm -f reduction reduction_blocks

