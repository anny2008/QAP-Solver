# Makefile for RTL1 SCIP Solver

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp

# Include and library flags
INCFLAGS = 
LIBFLAGS = -lscip -lpthread -ldl -lm

# Source and output
SRC = rtl1_solver.cpp
OBJ = rtl1_solver.o
EXE = rtl1_solver

# Build rules
all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBFLAGS)

$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(EXE)

.PHONY: all clean
