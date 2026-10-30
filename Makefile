CXX_COMPLIER = g++
MKL_ROOT = /data/spack/opt/spack/linux-ubuntu24.04-icelake/gcc-15.2.0/intel-oneapi-mkl-2026.0.0-hfx23rl27uqjzo3hhabps7y6tbsaujgf/mkl/2026.0
CXX_FLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -fopenmp -I$(MKL_ROOT)/include
LINKER = g++
LINKER_FLAGS = -fopenmp -L$(MKL_ROOT)/lib -lmkl_rt -lm -ldl -Wl,-rpath,$(MKL_ROOT)/lib
INCLUDES = -I./include

# 必须用 main.cpp（题目规则），只编译 main.cpp + melspec.cpp
SRCS = src/main.cpp src/melspec.cpp
INCS = $(wildcard include/*.h)
OBJS = $(SRCS:.cpp=.o)
EXEC = melspec

.PHONY: all clean test format

all: $(EXEC)

$(EXEC): $(OBJS)
	$(LINKER) -o $@ $^ $(LINKER_FLAGS)

%.o: %.cpp
	$(CXX_COMPLIER) $(CXX_FLAGS) $(INCLUDES) -o $@ -c $<

clean:
	rm -f $(OBJS) $(EXEC)

test: $(EXEC)
	python3 ./validate.py ./$(EXEC)

format: $(SRCS) $(INCS)
	clang-format -i $^
