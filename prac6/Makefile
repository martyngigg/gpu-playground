
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_20 --ptxas-options=-v --use_fast_math


all:	prac6 prac6a prac6b


main.o:	main.cpp Makefile
	g++ -c -fPIC -o main.o main.cpp

prac6.o:	prac6.cu Makefile
	nvcc prac6.cu -c -o prac6.o $(INC) $(NVCCFLAGS)

prac6.a:	prac6.cu Makefile
	nvcc prac6.cu -lib -o prac6.a $(INC) $(NVCCFLAGS)

prac6:	main.o prac6.o Makefile
	g++ -fPIC -o prac6 main.o prac6.o $(LIB)

prac6a:	main.cpp prac6.a Makefile
	g++ -fPIC -o prac6a main.o prac6.a $(LIB)


prac6b:	main.cpp prac6b.cu Makefile
	nvcc main.cpp prac6b.cu -o prac6b $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm prac6 prac6a prac6b *.o *.a
