# Makefile for RLT1 Volume Solver

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp



# Source files
SRC = rlt1_volume_solver.cpp VolVolume.cpp main.cpp
OBJ = rlt1_volume_solver.o VolVolume.o main.o
EXE = rlt1_volume_solver
STATIC_LIB = librlt1_volume.a
LIB_OBJS = rlt1_volume_solver.o VolVolume.o


# Build rules

all: $(EXE) $(STATIC_LIB)



$(EXE): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm -lpthread

$(STATIC_LIB): $(LIB_OBJS)
	ar rcs $@ $^


clean:
	rm -f $(OBJ) $(EXE) $(STATIC_LIB)

.PHONY: all clean
