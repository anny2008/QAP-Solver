# Makefile for RLT1 SCIP Solver

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp

# Include and library flags
INCFLAGS = -I../../core
LIBFLAGS = -lscip -lpthread -ldl -lm -fopenmp


# Source and output
SRCS = m4_solver.cpp
OBJS = $(SRCS:.cpp=.o)
EXE = m4_solver
VOLUME_LIB = ../m4_volume/libm4_volume.a


# Build rules
all: $(EXE)


$(EXE): $(OBJS) $(VOLUME_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(VOLUME_LIB) $(LIBFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -c -o $@ $<


clean:
	rm -f $(OBJS) $(EXE)

.PHONY: all clean
