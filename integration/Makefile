CC := gcc
CXX := g++
COMMONCXXFLAGS := -Wall -Wextra
CXXFLAGS := $(COMMONCXXFLAGS) -std=c++0x
NVCC := nvcc
NVCCFLAGS := -lineinfo -arch=sm_20 --ptxas-options=-v --use_fast_math --compiler-options="$(COMMONCXXFLAGS)"
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

CPPSRCS := $(wildcard *.cpp)
CPPOBJS := $(patsubst %.cpp,%.o,$(CPPSRCS))
CUDASRCS := $(wildcard *.cu)
CUDAOBJS := $(patsubst %.cu,%.cuo,$(CUDASRCS))

all: integration_testsuite

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INC) -o $@ $< 

%.cuo: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INC) -o $@ $< 

integration_testsuite: $(CPPOBJS) $(CUDAOBJS)
	$(CXX) $(LIB) -o $@ $^

clean:
	rm -f integration_testsuite $(CPPOBJS) $(CUDAOBJS)
