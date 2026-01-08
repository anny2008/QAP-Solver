# Makefile for sfd Volume Solver

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp

# Source files
SRC = sfd_volume_solver.cpp VolVolume.cpp
OBJ = sfd_volume_solver.o VolVolume.o
EXE = sfd_volume_solver

# Build rules
all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm -lpthread

clean:
	rm -f $(OBJ) $(EXE)

.PHONY: all clean
