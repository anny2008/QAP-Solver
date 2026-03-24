# Makefile for m4 Volume Solver

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp



# Source files
SRC = m4_volume_solver.cpp VolVolume.cpp main.cpp
OBJ = m4_volume_solver.o VolVolume.o main.o
EXE = m4_volume_solver
STATIC_LIB = libm4_volume.a
LIB_OBJS = m4_volume_solver.o VolVolume.o


# Build rules

all: $(EXE) $(STATIC_LIB)



$(EXE): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm -lpthread

$(STATIC_LIB): $(LIB_OBJS)
	ar rcs $@ $^


clean:
	rm -f $(OBJ) $(EXE) $(STATIC_LIB)

.PHONY: all clean
