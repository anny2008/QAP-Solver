# Makefile for RTL1 SCIP Solver

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp

# Include and library flags
INCFLAGS = -I../../core -I../rtl1_volume
LIBFLAGS = -lscip -lpthread -ldl -lm -fopenmp

# Source and output
SRCS = rtl1_solver.cpp volume_relax.cpp ../rtl1_volume/VolVolume.cpp
OBJS = $(SRCS:.cpp=.o)
EXE = rtl1_solver

# Build rules
all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(EXE)

.PHONY: all clean
