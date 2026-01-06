# Makefile for RTL1 Volume Solver

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp

# Include flags
INCFLAGS = -I$(VOL_INCDIR) -I../../include

# Source files
SRC = rtl1_volume_solver.cpp VolVolume.cpp
OBJ = rtl1_volume_solver.o VolVolume.o
EXE = rtl1_volume_solver

# Build rules
all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm -lpthread

clean:
	rm -f $(OBJ) $(EXE)

.PHONY: all clean
