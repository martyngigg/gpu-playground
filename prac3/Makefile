
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_20 --ptxas-options=-v --use_fast_math

both: 		laplace3d adi3d

laplace3d:	laplace3d.cu laplace3d_gold.cpp laplace3d_kernel.h Makefile
		nvcc laplace3d.cu laplace3d_gold.cpp -o laplace3d \
		$(INC) $(NVCCFLAGS) $(LIB)

adi3d:		adi3d.cu adi3d_gold.cpp adi3d_kernel.h Makefile
		nvcc adi3d.cu adi3d_gold.cpp -o adi3d \
		$(INC) $(NVCCFLAGS) $(LIB)

clean:
		rm -f laplace3d adi3d
