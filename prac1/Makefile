
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

# NVCCFLAGS	:= -lineinfo -arch=sm_20 -DCUDA_DEVICE=0 --ptxas-options=-v --use_fast_math
NVCCFLAGS	:= -lineinfo -arch=sm_20 --ptxas-options=-v --use_fast_math

all:	prac1a prac1b

prac1a:	prac1a.cu Makefile
	nvcc prac1a.cu -o prac1a $(INC) $(NVCCFLAGS) $(LIB)

prac1b:	prac1b.cu Makefile
	nvcc prac1b.cu -o prac1b $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f prac1a prac1b
